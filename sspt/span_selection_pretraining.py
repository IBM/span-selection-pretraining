import logging
import pprint
import time
import torch

from sspt.rc_data import RCData

# FIXME: based on fork of old huggingface code
from pytorch_pretrained_bert.tokenization_offsets import BertTokenizer
from pytorch_pretrained_bert.hypers_rc import HypersRC
from pytorch_pretrained_bert.bert_trainer_apex import BertTrainer


logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train_on_file(hypers: HypersRC, batches, trainer: BertTrainer):
    if hypers.world_size > 1:
        num_batches = torch.tensor(len(batches), dtype=torch.long).to(hypers.device)
        torch.distributed.all_reduce(num_batches, torch.distributed.ReduceOp.MIN)
        batch_limit = num_batches.item()
        logger.warning(f'truncating from {len(batches)} to {batch_limit} batches on {hypers.global_rank}')
    else:
        batch_limit = len(batches)
    batch_count = 0

    for batch in batches:
        batch_count += 1
        if batch_count > batch_limit:
            break
        if hypers.n_gpu == 1:
            batch = tuple(t.to(hypers.device) for t in batch)  # multi-gpu does scattering itself
        input_ids, input_mask, segment_ids, start_positions, end_positions, answer_mask, answer_types = batch
        loss = trainer.model(input_ids, segment_ids, input_mask, start_positions, end_positions, answer_mask, answer_types)
        trainer.step_loss(loss)
        if not trainer.should_continue():
            break
    trainer.reset()
    trainer.train_stats.pause_timing()


def main():
    parser = BertTrainer.get_base_parser()

    # Other parameters
    parser.add_argument("--save_model", default=None, type=str, required=True,
                        help="Bert model checkpoint file")
    parser.add_argument("--load_model", default=None, type=str, required=False,
                        help="Bert model checkpoint file")
    parser.add_argument("--discard_optimizer_checkpoint", default=False, action='store_true',
                        help="discard checkpointed optimizer state from load_model")
    parser.add_argument("--train_dir", default=None, type=str,
                        help="rc_data json format for training. Must be a directory")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--time_limit", default=-1, type=int,
                        help="The maximum time to train in seconds")

    args = parser.parse_args()
    hypers = HypersRC(args)
    hypers.num_answer_categories = None
    hypers.num_train_steps = int(args.num_instances / args.train_batch_size)

    logger.info(pprint.pformat(vars(hypers), indent=4))
    logger.info('torch cuda version %s', torch.version.cuda)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'),
                                              add_special_tokens=['[BLANK]'])
    # TODO: use new-style of adding special tokens
    # tokenizer.add_special_tokens({additional_special_tokens: ['[BLANK]']})
    # model.resize_token_embeddings(len(tokenizer))

    checkpoint = None
    dataloader_checkpoint = None
    if args.load_model:
        logger.info('resuming from saved model %s', args.load_model)
        checkpoint = torch.load(args.load_model, map_location='cpu')
        if args.discard_optimizer_checkpoint:
            logger.info('not including optimizer state from %s', args.load_model)
            del checkpoint['optimizer']
        if 'dataloader' in checkpoint:
            dataloader_checkpoint = checkpoint['dataloader']

    # TODO: use optimizer-style instead
    trainer = BertTrainer(hypers, hypers.model_name, checkpoint, hypers_rc=hypers)

    loader = RCData(args.train_dir, tokenizer, dataloader_checkpoint,
                    hypers.global_rank, hypers.world_size, hypers.max_seq_length, hypers.doc_stride,
                    hypers.max_query_length, hypers.train_batch_size, hypers.fp16,
                    first_answer_only=False, include_source_info=False)

    while trainer.should_continue():
        start_time = time.time()
        # dataloader for one partial-epoch
        batches = loader.get_batches()
        # not a warning, just make sure to get from all nodes
        logger.warning(f"loading took {time.time()-start_time} secs on {hypers.global_rank}")

        train_on_file(hypers, batches, trainer)

        trainer.save(args.save_model, dataloader=loader.get_checkpoint_info())

        logger.info(f'One training block took {time.time()-start_time} secs')


if __name__ == "__main__":
    main()
