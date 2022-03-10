import configparser
import linecache
import os
import re
import sys
import time
from os import listdir

import unicodedata
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import tracemalloc


def remove_non_ascii(words):
    new_words = []
    for word in words:
        # Normalise (normalize) unicode data in Python to remove umlauts, accents etc.
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)  # \w: an alphanumeric character; \s: a whitespace character
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords_set:
            new_words.append(word)
    return new_words


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words


def stem_words(words):
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def process_file(source_file):
    token = word_tokenize(source_file)
    token = normalize(token)
    token = stem_words(token)
    return token


def get_sorted_block():
    block_names = os.listdir(output_path)
    block_names_pos = [(block_name.split('.')[0], block_name) for block_name in block_names]
    sorted_block_jd = sorted(block_names_pos, key=lambda x: x[0])
    block_names = [name for id, name in sorted_block_jd]
    n_blocks = len(block_names)
    return block_names, n_blocks


def write_block_to_disk(blockIndex, dictBlock):
    try:
        del dictBlock['']
    except KeyError:
        pass

    # Sort dict block
    sortedDictBlock = list(sorted(dictBlock.items(), key=lambda t: t[0]))

    # Save dict block to disc
    fileName = '%s.txt' % blockIndex
    with open(output_path + os.path.sep + fileName, 'w') as out:
        for key, values in sortedDictBlock:
            out.write(str(key) + ',' + str(values)[1:-1].replace(' ', '') + '\n')


def merge_blocks(block1, block2):
    end_of_fine = False

    # Define files
    block1_file = open(output_path + os.path.sep + block1, 'rt', encoding='UTF-8')
    block2_file = open(output_path + os.path.sep + block2, 'rt', encoding='UTF-8')

    # Read first lines
    block1_line = block1_file.readline()[:-1].split(',')  # Delete \n
    block2_line = block2_file.readline()[:-1].split(',')  # Delete \n
    with open(output_path + os.path.sep + '_' + block1, 'a') as out:
        while not end_of_fine:
            # CASE 1: when keys are the same
            if block1_line[0] == block2_line[0]:
                block1_posting = [val for val in block1_line[1:]]
                block2_posting = [val for val in block2_line[1:]]
                merged_posting = block1_posting + block2_posting
                out.write(block1_line[0] + ',' + ','.join(merged_posting) + '\n')
                # Skip to Next  lines
                block1_line = block1_file.readline()[:-1].split(',')
                block2_line = block2_file.readline()[:-1].split(',')

            # CASE 2: when first block key is greater, second block comes first
            elif block1_line[0] > block2_line[0]:
                out.write(','.join(block2_line) + '\n')
                block2_line = block2_file.readline()[:-1].split(',')

            # CASE 3: when second block key is greater, first block comes first
            elif block1_line[0] < block2_line[0]:
                out.write(','.join(block1_line) + '\n')
                block1_line = block1_file.readline()[:-1].split(',')

            if block1_line[0] == '' and block2_line[0] != '':
                while not end_of_fine:
                    out.write(','.join(block2_line) + '\n')
                    block2_line = block2_file.readline()[:-1].split(',')
                    if block2_line[0] == '':
                        return

            if block1_line[0] != '' and block2_line[0] == '':
                while not end_of_fine:
                    out.write(','.join(block1_line) + '\n')
                    block1_line = block1_file.readline()[:-1].split(',')
                    if block1_line[0] == '':
                        return

            # End of merging
            if block1_line[0] == '' and block2_line[0] == '':
                break


def delete_merged_blocks(block1, block2):
    os.remove(output_path + os.path.sep + block1)
    os.remove(output_path + os.path.sep + block2)


def merge_all_blocks():
    merge_start = time.perf_counter()
    block_names, n_blocks = get_sorted_block()

    while n_blocks > 1:
        block_names, n_blocks = get_sorted_block()

        if n_blocks % 2 == 0:  # If number of blocks are even
            block_couples = [(i, i + 1) for i in range(0, n_blocks, 2)]
        else:
            block_couples = [(i, i + 1) for i in range(0, n_blocks - 1, 2)]

        for couple in block_couples:
            block1 = block_names[couple[0]]
            block2 = block_names[couple[1]]
            merge_blocks(block1, block2)  # Merge two blocks
            delete_merged_blocks(block1, block2)  # Delete blocks after merging
    print("Merge time: " + str(time.perf_counter() - merge_start) + " seconds")


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('%s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


overall_start = time.perf_counter()
tracemalloc.start()

config = configparser.ConfigParser()
config.read('file.ini')
input_path = config['default']['path']
block_size = int(config['default']['block_size'])
output_path = config['default']['output_path']

output_file_list = os.listdir(output_path)
for file in output_file_list:
    os.remove(output_path + os.path.sep + file)

stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()

file_list = os.listdir(input_path)

block_index = 0
dict_block = defaultdict(list)

for file in file_list:
    doc_id = file
    terms_in_doc = set()
    with open(input_path + os.path.sep + file, 'r', encoding='UTF-8') as f:
        lines = f.read()
        tokens = process_file(lines)
        for token in tokens:
            if token != '' and token not in terms_in_doc:
                dict_block[token].append(doc_id)
                terms_in_doc.add(token)
            if sys.getsizeof(dict_block) > block_size:
                write_block_to_disk(block_index, dict_block)
                dict_block = defaultdict(list)
                block_index += 1
    if len(dict_block) > 0:
        write_block_to_disk(block_index, dict_block)
    f.close()

print("Index time: " + str(time.perf_counter() - overall_start) + " seconds")
merge_all_blocks()
print("Run time: " + str(time.perf_counter() - overall_start) + " seconds")
display_top(tracemalloc.take_snapshot())
