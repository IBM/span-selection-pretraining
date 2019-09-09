# coding=utf-8
# Copyright 2019 IBM
# based in part on the run_squad.py code from pytorch_pretrained_bert
#   Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
import random
import json
from typing import List
from util.line_corpus import jsonl_lines
import numpy as np
# FIXME: Tokenizer base
from pytorch_pretrained_bert.tokenization_offsets import BertTokenizer
import torch
import re, string
from enum import Enum, unique
import os
import glob
import math

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@unique
class AnswerType(Enum):
    """
    train_rc and rc_data have AnswerType baked in, but BertForQuestionMultiAnswering is general
    """
    yes = 0
    no = 1
    span = 2


def display_batch(batch, tokenizer: BertTokenizer):
    batch = tuple(t.numpy() if isinstance(t, torch.Tensor) else t for t in batch)
    if len(batch) == 10:
        input_ids, input_mask, segment_ids, start_positions, end_positions, answer_mask, answer_types, qids, passage_texts, token_offsets = batch
    else:
        input_ids, input_mask, segment_ids, start_positions, end_positions, answer_mask, answer_types = batch
        passage_texts, token_offsets = None, None
    batch_size = input_ids.shape[0]
    logger.info('='*80)
    for ii in range(batch_size):
        toks = tokenizer.convert_ids_to_tokens(input_ids[ii])
        segs = segment_ids[ii]
        amask = answer_mask[ii]
        num_ans = amask[amask == 0].shape[0]
        sp = start_positions[ii][:num_ans]
        ep = end_positions[ii][:num_ans]
        to_show = ''
        assert (0 in sp) == (0 in ep)
        if 0 in sp:
            if answer_types is not None and answer_types[ii] == AnswerType.yes.value:
                to_show += 'YES '
            elif answer_types is not None and answer_types[ii] == AnswerType.no.value:
                to_show += 'NO '
            else:
                to_show += 'NO ANS '
        else:
            assert answer_types is None or answer_types[ii] == AnswerType.span.value
            to_show += str(num_ans) + ' ANS '
        for ti in range(len(toks)):
            if ti > 0 and segs[ti] == 1 and segs[ti-1] == 0:
                to_show += '||| '
            if ti > 0 and ti in sp:
                to_show += '<<<'
            to_show += toks[ti]
            if ti > 0 and ti in ep:
                to_show += '>>>'
            to_show += ' '
        logger.info(to_show)
        if passage_texts:
            for s, e in zip(sp, ep):
                if s == 0:
                    continue
                logger.info("   "+passage_texts[ii][token_offsets[ii][s, 0]:token_offsets[ii][e, 1]])


def make_batch(features, fp16=False):
    batch_size = len(features)
    assert batch_size > 0
    # TODO: if we do single start/end style we should use a vector rather than matrix
    num_answers = [len(f.start_positions) for f in features]
    max_answers = max(num_answers)
    start_positions = np.zeros((batch_size, max_answers), dtype=np.long)
    end_positions = np.zeros((batch_size, max_answers), dtype=np.long)
    answer_mask = np.zeros((batch_size, max_answers), dtype=np.float32)
    # FIXME: version of make_batch that doesn't use the multi-answer style
    for i, num_ans in enumerate(num_answers):
        answer_mask[i, num_ans:] = 10000  # some large number, larger than any possible loss
        start_positions[i, 0:num_ans] = features[i].start_positions
        end_positions[i, 0:num_ans] = features[i].end_positions
    # do padding here
    max_input_ids = max(len(f.input_ids) for f in features)
    input_ids = []
    segment_ids = []
    input_mask = []
    for f in features:
        input_len = len(f.input_ids)
        input_ids.append(np.pad(f.input_ids, (0, max_input_ids-input_len), 'constant'))
        segment_ids.append(np.pad(f.segment_ids, (0, max_input_ids - input_len), 'constant'))
        input_mask.append(np.pad([1]*input_len, (0, max_input_ids - input_len), 'constant'))
    # we assume all features have an answer_type or none of them do
    if features[0].answer_type is not None:
        answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)
    else:
        answer_types = None
    # CONSIDER: try torch.int32 instead of long
    core_tensors = (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(input_mask, dtype=torch.long),
                    torch.tensor(segment_ids, dtype=torch.long),
                    torch.tensor(start_positions, dtype=torch.long), torch.tensor(end_positions, dtype=torch.long),
                    torch.tensor(answer_mask, dtype=torch.float16 if fp16 else torch.float32),
                    answer_types)
    if features[0].passage_text:
        return core_tensors + ([f.qid for f in features], [f.passage_text for f in features],
                               [f.token_offsets for f in features])
    else:
        return core_tensors


class RCExample(object):
    """
    A single training/test example for the RC dataset.
    For examples without an answer, the start and end position are empty lists
    """
    def __init__(self,
                 qid,
                 question,
                 passage,
                 start_positions=None,
                 end_positions=None,
                 answer_type: AnswerType=None,
                 passage_text=None,
                 passage_token_offsets=None):
        self.qid = qid
        self.question = question
        self.passage = passage
        self.start_positions = start_positions
        # NOTE: the end_positions give the index of the last token in the answer - not one past the end
        self.end_positions = end_positions
        self.answer_type = answer_type
        # for when we make predictions:
        self.passage_text = passage_text
        self.passage_token_offsets = passage_token_offsets

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # show the answer strings (use passage_token_offsets and passage_text)
        answer_strings = []
        for s, e in zip(self.start_positions, self.end_positions):
            answer_strings.append(self.passage_text[self.passage_token_offsets[s, 0]:self.passage_token_offsets[e, 1]])
        return f'id: {self.qid}, question: {str(self.question)}, passage: {str(self.passage)}, ' \
               f'starts: {str(self.start_positions)}, ends: {str(self.end_positions)}, ' \
               f'answer_type: {self.answer_type}, answer_strings: {str(answer_strings)}'


class InputFeatures(object):
    """A single set of features of data."""
    # CONSIDER: use __slots__
    def __init__(self,
                 input_ids,
                 segment_ids,
                 start_positions,
                 end_positions,
                 answer_type: AnswerType,
                 qid=None,
                 passage_text=None,
                 token_offsets=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        # NOTE: the end_positions give the index of the last token in the answer - not one past the end
        self.end_positions = end_positions
        self.answer_type = answer_type.value if answer_type else None
        self.qid = qid
        self.passage_text = passage_text
        self.token_offsets = token_offsets


def find_answer_starts(doc_toks, ans_toks):
    # CONSIDER: handle also Europe/European, portrait/portraits
    # also cases like South Korean/South Korea
    starts = []
    anslen = len(ans_toks)
    for s in range(len(doc_toks)-(anslen-1)):
        if ans_toks == doc_toks[s:s+anslen]:
            starts.append(s)
    return starts


def normalize(toks, filter_pattern):
    norm_to_orig = []
    normed = []
    for i, tok in enumerate(toks):
        if filter_pattern and filter_pattern.match(tok):
            continue
        normed.append(tok.lower())
        norm_to_orig.append(i)
    if filter_pattern and len(normed) == 0:
        return normalize(toks, None)
    assert len(normed) > 0
    assert len(normed) == len(norm_to_orig)
    return normed, norm_to_orig


def read_rc_examples(input_file, tokenizer: BertTokenizer, first_answer_only=False, include_source_info=False):
    """Read a RC jsonl file into a list of RCExample."""
    #filter_pattern = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))
    filter_pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
    examples = []
    impossible_count = 0
    qid2answers = dict() if include_source_info else None
    answer_type_stats = np.zeros(len(AnswerType), dtype=np.int32)
    for line in jsonl_lines(input_file):
        jobj = json.loads(line)
        qid = jobj["qid"]
        passage_orig_text = jobj['passage']
        passage_toks, passage_tok_offsets, passage_text = tokenizer.tokenize_offsets(passage_orig_text)
        if len(passage_toks) == 0:
            logger.info(f'bad passage: {passage_orig_text}')
            continue
        passage_tok_offsets = np.array(passage_tok_offsets, dtype=np.int32)
        norm_passage, norm_to_orig = normalize(passage_toks, filter_pattern)
        # TODO: we also need the passage normalized without filter_pattern, for use when the filter_pattern leaves the answer empty
        question_toks = tokenizer.tokenize(jobj["question"])
        if qid2answers is not None:
            if qid in qid2answers and qid2answers[qid] != jobj['answers']:
                raise ValueError('answers not consistent!')
            qid2answers[qid] = jobj['answers']
        # answer_type (span, yes, no)
        if 'answer_type' in jobj:
            answer_type = AnswerType[jobj['answer_type']]
            answer_type_stats[answer_type.value] += 1
        else:
            answer_type = None
        # if the answer_type is anything other than span, the 'answers' should be empty
        if answer_type is None or answer_type == AnswerType.span:
            answers = jobj["answers"]
        else:
            answers = []
        ans_starts = []
        ans_ends = []
        for ans in answers:
            ans_toks = tokenizer.tokenize(ans)
            if len(ans_toks) == 0 or sum([len(tok) for tok in ans_toks]) == 0:
                logger.info(f'bad answer for {qid}: "{ans}"')
                continue
            norm_ans, _ = normalize(ans_toks, filter_pattern)  # TODO: we need to know if we applied the filter or not so we can decide to use norm_passage with or without filter
            nstarts = find_answer_starts(norm_passage, norm_ans)
            starts = [norm_to_orig[s] for s in nstarts]
            ends = [norm_to_orig[s+len(norm_ans)-1] for s in nstarts]
            ans_starts.extend(starts)
            ans_ends.extend(ends)

        if (answer_type is None or answer_type == AnswerType.span) and len(ans_starts) == 0:
            # sample the impossible ones
            # if impossible_count < 10:
            #     logger.info(f'Impossible:\n   Question "{jobj["question"]}"\n'
            #                 f'   Passage "{passage_text}"\n   Answers {str(answers)}\n'
            #                 f'   Passage Tokens {str(norm_passage)}\n'
            #                 f'   Answer Tokens {[normalize(tokenizer.tokenize(ans), filter_pattern)[0] for ans in answers]}')
            impossible_count += 1
        # discard source information for training data to save some memory
        if not include_source_info:
            qid = None
            passage_text = None
            passage_tok_offsets = None
        if first_answer_only:
            ans_starts = ans_starts[:1]
            ans_ends = ans_ends[:1]
        example = RCExample(
            qid=qid,
            question=question_toks,
            passage=passage_toks,
            start_positions=ans_starts,
            end_positions=ans_ends,
            answer_type=answer_type,
            passage_text=passage_text,
            passage_token_offsets=passage_tok_offsets)
        examples.append(example)
    logger.info(f'from {input_file} loaded {impossible_count} impossible, {len(examples)} total')
    if answer_type_stats.sum() > 0:
        logger.info(f'Answer type statistics:')
        for at in AnswerType:
            logger.info(f'   {at.name} = {answer_type_stats[at.value]}')
    return examples, qid2answers


def convert_examples_to_features(examples: List[RCExample], tokenizer: BertTokenizer,
                                 max_seq_length, doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (example_index, example) in enumerate(examples):
        # logger.info(str(example))
        if len(example.question) > max_query_length:
            example.question = example.question[0:max_query_length]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(example.question) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(example.passage):
            length = len(example.passage) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(example.passage):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in example.question:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            doc_offset = len(tokens)
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length

            if example.passage_token_offsets is not None:
                token_offsets = np.zeros((doc_offset + doc_span.length, 2), dtype=np.int32)
                token_offsets[doc_offset:,:] = example.passage_token_offsets[doc_start:doc_end,:]
            else:
                token_offsets = None

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                tokens.append(example.passage[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            start_positions = []
            end_positions = []
            for s, e in zip(example.start_positions, example.end_positions):
                if s < doc_start or e >= doc_end:
                    continue
                start_positions.append(s-doc_start+doc_offset)
                end_positions.append(e-doc_start+doc_offset)

            is_impossible = False
            if len(start_positions) == 0:
                start_positions.append(0)
                end_positions.append(0)
                is_impossible = True

            features.append(
                InputFeatures(
                    input_ids=np.array(input_ids, dtype=np.int32),
                    segment_ids=np.array(segment_ids, dtype=np.int32),
                    start_positions=np.array(start_positions, dtype=np.int32),
                    end_positions=np.array(end_positions, dtype=np.int32),
                    answer_type=example.answer_type,
                    qid=example.qid,
                    passage_text=example.passage_text,
                    token_offsets=token_offsets))

    return features


class RCData:
    def __init__(self, input_dir, tokenizer, checkpoint_info,
                 world_rank, world_size, max_seq_length, doc_stride, max_query_length, gpu_batch_size,
                 fp16=False, first_answer_only=False, include_source_info=False):
        """
        must rebuild the cache after changing: tokenizer, max_seq_length, doc_stride, max_query_length
        :param input_dir:
        :param tokenizer:
        :param world_rank:
        :param world_size:
        :param max_seq_length:
        :param doc_stride:
        :param max_query_length:
        :param gpu_batch_size:
        """
        self.world_rank = world_rank
        self.world_size = world_size
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.gpu_batch_size = gpu_batch_size
        self.fp16 = fp16
        self.first_answer_only = first_answer_only
        self.include_source_info = include_source_info
        self.tokenizer = tokenizer
        self.input_dir = input_dir
        if checkpoint_info:
            self.on_epoch = checkpoint_info['on_epoch']
            self.completed_files = checkpoint_info['completed_files']
        else:
            self.on_epoch = 0
            self.completed_files = []
        self.input_files = []

    def build_features(self, input_file):
        examples, qid2gt = read_rc_examples(input_file, self.tokenizer,
                                            first_answer_only=self.first_answer_only,
                                            include_source_info=self.include_source_info)
        features = convert_examples_to_features(examples, self.tokenizer,
                                            self.max_seq_length, self.doc_stride, self.max_query_length)
        return features, qid2gt

    @classmethod
    def _str_file_list(cls, files):
        if len(files) > 10:
            prelen = len(os.path.commonprefix(files))
            return str(files[0:1] + [f[prelen:] for f in files[1:16]])
        else:
            return str(files)

    def _get_files(self, leftover_files=[]):
        logger.info('completed files = %s, count = %i',
                    str(self._str_file_list(self.completed_files)), len(self.completed_files))

        if not self.input_dir.endswith('/'):
            self.input_dir = self.input_dir + '/'
        self.input_files = glob.glob(self.input_dir + '**', recursive=True)
        # exclude directories
        self.input_files = [f for f in self.input_files if not os.path.isdir(f)]

        # exclude completed files, and leftover files (which we add at the start)
        self.input_files = [f for f in self.input_files if f not in self.completed_files and f not in leftover_files]

        self.input_files.sort()
        random.Random(123 * self.on_epoch).shuffle(self.input_files)
        # prepend files leftover from last epoch
        self.input_files = leftover_files + self.input_files

        logger.info('input files = %s, count = %i',
                    str(self._str_file_list(self.input_files)), len(self.input_files))

    def _get_input_file(self):
        if len(self.input_files) < self.world_size:
            self.on_epoch += 1
            self._get_files(leftover_files=self.input_files)
        used_files = []
        while len(used_files) < self.world_size:
            used_files.append(self.input_files.pop())
        self.completed_files.extend(used_files)
        return used_files[self.world_rank]

    def get_checkpoint_info(self):
        checkpoint_info = dict()
        checkpoint_info['on_epoch'] = self.on_epoch
        checkpoint_info['completed_files'] = self.completed_files
        return checkpoint_info

    def get_batches(self):
        input_file = self._get_input_file()
        features, _ = self.build_features(input_file)
        num_batches = int(math.ceil(len(features) / self.gpu_batch_size))
        random.Random(123 + 137 * self.on_epoch).shuffle(features)
        batches = []
        for bi in range(num_batches):
            batches.append(make_batch(features[bi::num_batches], fp16=self.fp16))
        return batches

    @classmethod
    def get_predicted_answer(cls, start, end, passage_text, token_offsets):
        if start > end or end >= len(token_offsets) or start < 0:
            return None
        soff = token_offsets[start][0]
        eoff = token_offsets[end][1]
        if soff >= eoff:
            return None
        assert eoff < len(passage_text)
        return passage_text[soff:eoff]

    @classmethod
    def get_predicted_answers(cls, answer_type_logits, start_logits, end_logits,
                              passage_text, token_offsets,
                              top_k, max_answer_length):
        """
        answer_type_logits is None if this does not include yes/no questions
        the logit arrays are numpy array
        """
        if answer_type_logits is not None:
            at = AnswerType(answer_type_logits.argsort()[-1])
            if at != AnswerType.span:
                return [(at.name, answer_type_logits[at.value])]
        qlen = 0
        # the question tokens have offsets (in the passage) of [0,0)
        for offsets in token_offsets:
            if offsets[1] == 0:
                qlen += 1
        total_len = len(token_offsets)
        span_scores = np.add.outer(start_logits, end_logits)
        valid_spans = []
        for s in range(qlen, total_len):
            for e in range(s, min(total_len, s + max_answer_length)):
                score = span_scores[s, e]
                valid_spans.append((s, e, score))
        valid_spans.sort(key=lambda tup: tup[2], reverse=True)
        scored_answers = []
        for span in valid_spans[:top_k]:
            answer = cls.get_predicted_answer(span[0], span[1], passage_text, token_offsets)
            scored_answers.append((answer, float(span[2])))  # float to make json serializable
        if len(scored_answers) == 0:
            scored_answers.append(('noanswer', 0.5))  # FIXME: make this an AnswerType
        return scored_answers
