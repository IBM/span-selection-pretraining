import en_core_web_sm
import glob
import json
import os
import random
import gzip
import argparse
import time
import logging
from util.reporting import Reporting

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    @classmethod
    def from_spacy(cls, id, contents, nlp):
        contents = contents.strip()
        d = Document(id, contents)
        doc = nlp(contents)
        for t in doc:
            d.tokens.append((t.idx, t.idx+len(t)))
            d.pos.append(t.tag_)
        for sent in doc.sents:
            d.sentences.append((d.tokens[sent.start][0], d.tokens[sent.end - 1][1]))
        for np in doc.noun_chunks:
            np_start = np.start
            # print(pos[np_start]+": "+np.text)
            if d.pos[np_start] in ['DT', 'PRP$']:
                np_start += 1
            if np_start < np.end:
                d.noun_chunks.append((d.tokens[np_start][0], d.tokens[np.end - 1][1]))
        for ent in doc.ents:
            d.entities.append((d.tokens[ent.start][0], d.tokens[ent.end - 1][1]))
        return d

    def __init__(self, id, contents, sentences=None, tokens=None, pos=None, noun_chunks=None, entities=None):
        self.id = id
        self.contents = contents  # contents instead of text for Anserini compatibility
        # these are lists of [start,end) character offsets
        self.sentences = sentences if sentences else []
        self.tokens = tokens if tokens else []
        self.pos = pos if pos else []
        self.noun_chunks = noun_chunks if noun_chunks else []
        self.entities = entities if entities else []

    def __str__(self):
        return self.id + '\n' + self.contents + \
               '\nSentences\n' + str([self.contents[s[0]:s[1]] for s in self.sentences]) + \
               '\nTokens\n' + str([self.contents[s[0]:s[1]] for s in self.tokens]) + \
               '\nNoun Chunks\n' + str([self.contents[s[0]:s[1]] for s in self.noun_chunks]) + \
               '\nEntities\n' + str([self.contents[s[0]:s[1]] for s in self.entities])


class DocumentCorpus:
    def __init__(self, directory, seed, world_rank, world_size, document_limit=0):
        self.directory = directory
        if not self.directory.endswith('/'):
            self.directory = self.directory + '/'
        self.seed = seed
        self.world_rank = world_rank
        self.world_size = world_size
        self.open_file_count = 30
        self.document_limit = document_limit

    @classmethod
    def open(cls, filename):
        if filename.endswith(".gz"):
            return gzip.open(filename, "rt", encoding='utf-8')
        else:
            return open(filename, "r", encoding='utf-8')

    def get_documents(self):
        """
        make one pass over the documents in the corpus, using only the fraction of files for our world_rank/world_size
        :return:
        """
        files = glob.glob(self.directory + '**', recursive=True)
        # exclude directories
        files = [f for f in files if not os.path.isdir(f)]

        # shuffle according to seed
        files.sort()
        random.Random(self.seed).shuffle(files)
        # take subset for our world_rank
        our_files = files[self.world_rank::self.world_size]
        file_iters = []
        file_names = []

        # open k files
        # read them line by line
        # open new file when one is finished
        document_count = 0
        while True:
            while len(file_iters) < self.open_file_count and len(our_files) > 0:
                fname = our_files.pop()
                file_names.append(fname)
                file_iters.append(self.open(fname))
            if len(file_iters) == 0:
                break
            fndx = random.randint(0, len(file_iters)-1)
            try:
                line = next(file_iters[fndx])
            except StopIteration:
                file_iters[fndx].close()
                logger.info(f"Finished {file_names[fndx]}")
                del file_iters[fndx]
                del file_names[fndx]
                continue
            yield line
            document_count += 1
            if 0 < self.document_limit <= document_count:
                break
        for file_iter in file_iters:
            file_iter.close()


class ToQuery:
    """
    Just contains the text of a sentence and the span to be used as a blank
    """
    def __init__(self, sentence, blank, docid):
        self.sentence = sentence
        self.blank = blank
        self.docid = docid

    def get_search_query(self):
        return self.sentence[:self.blank[0]] + self.sentence[self.blank[1]:]

    def get_answer(self):
        return self.sentence[self.blank[0]:self.blank[1]]

    def to_instance(self):
        # we indicate the blank with an easily approximated unicode char
        inst = dict()
        sent = self.sentence.replace('▁', '_')
        inst['sentence'] = sent[0:self.blank[0]] + '▁' + sent[self.blank[0]:self.blank[1]] + '▁' + sent[self.blank[1]:]
        inst['docId'] = self.docid
        return inst


class Writer:
    def __init__(self, out_dir, world_rank):
        self.insts_per_file = 50000
        self.query_file_limit = 50  # should be twice the maxOpenFiles of AsyncWriter.ShuffledQueries
        self.out = None
        self.file_count = 0
        self.file_insts_count = 0
        self.out_dir = out_dir
        self.world_rank = world_rank
        os.makedirs(out_dir, exist_ok=True)

    def write(self, to_query: ToQuery):
        # write to file
        if self.file_insts_count >= self.insts_per_file:
            self.close()
            self.file_insts_count = 0
        if self.out is None:
            # don't create if there are too many queries* in the outdir
            while len(glob.glob(f'{self.out_dir}/queries*.jsonl.gz')) >= self.query_file_limit:
                time.sleep(60)
            if len(glob.glob(f'{self.out_dir}/queries*.jsonl.gz')) < self.query_file_limit//2:
                self.insts_per_file = 5000  # this will happen when we are spinning up, lets create some files quickly
            else:
                self.insts_per_file = 50000
            self.file_count += 1
            self.out = gzip.open(f'{self.out_dir}/queries_{self.world_rank}_{self.file_count}.jsonl.gz.partial',
                                 'wt', encoding='utf-8')
        self.out.write(json.dumps(to_query.to_instance()) + '\n')
        self.file_insts_count += 1

    def close(self):
        if self.out is not None:
            self.out.close()
            os.rename(f'{self.out_dir}/queries_{self.world_rank}_{self.file_count}.jsonl.gz.partial',
                      f'{self.out_dir}/queries_{self.world_rank}_{self.file_count}.jsonl.gz')
            self.out = None

    def finish(self):
        self.close()


class SSPTInstanceGenerator:
    def __init__(self):
        self.min_length = 50
        self.max_length = 300

        self.min_answer_length = 4
        self.max_answer_length = 30

        self.token_select_prob = 0.2

    @staticmethod
    def is_open_class(tag):
        return tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')

    def get_possible_blanks(self, doc, sent):
        ents = [e for e in doc.entities if e[0] >= sent[0] and e[1] <= sent[1]]
        nps = [e for e in doc.noun_chunks if e[0] >= sent[0] and e[1] <= sent[1]]
        all_ents = ents + nps
        # also have a chance of selecting a token
        if len(all_ents) == 0 or random.random() < self.token_select_prob:
            all_ents = all_ents + [e for e, p in zip(doc.tokens, doc.pos)
                                   if e[0] >= sent[0] and e[1] <= sent[1] and self.is_open_class(p)]
        # filter answer by length
        all_ents = [e for e in all_ents if self.min_answer_length <= e[1] - e[0] <= self.max_answer_length]
        return all_ents

    def get_to_query(self, doc: Document):
        for sent in doc.sentences:
            slen = sent[1] - sent[0]
            if not self.min_length <= slen <= self.max_length:
                continue
            all_ents = self.get_possible_blanks(doc, sent)
            if len(all_ents) == 0:
                continue
            e = random.choice(all_ents)
            yield ToQuery(doc.contents[sent[0]:sent[1]], (e[0]-sent[0], e[1]-sent[0]), doc.id)


def main():
    """
    python sspt_gen_async.py \
      --doc_dir wikipassagesdir \
      --output ssptGen

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--doc_dir", default=None, type=str, required=True,
                        help="The passage document directory.")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="The SSPT output directory.")
    parser.add_argument("--document_limit", default=0, type=int, required=False,
                        help="The maximum number of documents to process.")
    # split into multiple processes to avoid spaCy memory leak (https://github.com/explosion/spaCy/issues/3618)
    parser.add_argument("--world_rank", default=0, type=int, required=False,
                        help="When process a large corpus we need to chop the processing up into multiple processes.")
    parser.add_argument("--world_size", default=1, type=int, required=False,
                        help="When process a large corpus we need to chop the processing up into multiple processes.")
    args = parser.parse_args()

    nlp = en_core_web_sm.load()
    dc = DocumentCorpus(args.doc_dir, 123, args.world_rank, args.world_size, document_limit=args.document_limit)
    sspt = SSPTInstanceGenerator()
    writer = Writer(f'{args.output}', args.world_rank)

    report = Reporting()
    for line in dc.get_documents():
        adoc = json.loads(line)
        doc = Document.from_spacy(adoc['id'], adoc['contents'], nlp)
        for tq in sspt.get_to_query(doc):
            if report.is_time():
                logger.info(f'{report.check_count} queries created, '
                            f'{report.check_count/(time.time() - report.start_time)} queries per second')
            writer.write(tq)
    writer.finish()
    logger.info(f'{report.check_count} queries created, '
                f'{report.check_count/(time.time() - report.start_time)} queries per second')


if __name__ == "__main__":
    main()
