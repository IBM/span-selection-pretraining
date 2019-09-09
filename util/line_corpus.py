import logging
import os
import glob
import gzip


logger = logging.getLogger(__name__)


def jsonl_lines(input_files, completed_files=None, limit=0, report_every=100000):
    return read_lines(jsonl_files(input_files, completed_files), limit=limit, report_every=report_every)


def jsonl_files(input_files, completed_files=None):
    return expand_files(input_files, '*.jsonl*', completed_files)


def expand_files(input_files, file_pattern='*', completed_files=None):
    """
    expand the list of files and directories
    :param input_files:
    :param file_pattern: glob pattern for recursive example '*.jsonl*' for jsonl and jsonl.gz
    :param completed_files: these will not be returned in the final list
    :return:
    """
    if type(input_files) is str:
        input_files = [input_files]
    # expand input files recursively
    all_input_files = []
    if completed_files is None:
        completed_files = []
    for input_file in input_files:
        if input_file in completed_files:
            continue
        if os.path.isdir(input_file):
            sub_files = glob.glob(input_file + "/**/" + file_pattern, recursive=True)
            sub_files = [f for f in sub_files if not os.path.isdir(f)]
            sub_files = [f for f in sub_files if f not in input_files and f not in completed_files]
            all_input_files.extend(sub_files)
        else:
            all_input_files.append(input_file)
    return all_input_files


def read_lines(input_files, limit=0, report_every=100000):
    """
    This takes a list of input files and iterates over the lines in them
    :param input_files: Directory name or list of file names
    :param completed_files: The files we have already processed; We won't read these again.
    :param limit: maximum number of examples to load
    :return:
    """
    count = 0
    for input_file in input_files:
        if input_file.endswith(".gz"):
            reader = gzip.open(input_file, "rt")
        else:
            reader = open(input_file, "r")
        with reader:
            for line in reader:
                yield line
                count += 1
                if count % report_every == 0:
                    logger.info(f'On line {count} in {input_file}')
                if 0 < limit <= count:
                    return
