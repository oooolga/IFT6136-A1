import ipdb
import numpy as np
import pickle

idx2word = open("./raw/vocabulary.txt", "r").read().splitlines()
vocab_size = len(idx2word)

def _freq(line):
    return int(line.split()[2])

def _word_id(line):
    return int(line.split()[1])-1

def _doc_id(line):
    return int(line.split()[0])-1

def _read_data(data_file, label_file):

    data_lines = open(data_file, "r").read().splitlines()
    label_lines = open(label_file, "r").read().splitlines()
    num_docs = _doc_id(data_lines[-1]) + 1
    assert num_docs == len(label_lines)

    data = np.zeros([num_docs, vocab_size])
    for line in data_lines:
        data[_doc_id(line), _word_id(line)] += _freq(line)

    ipdb.set_trace()
    label = np.array([int(line) for line in label_lines ])
    return data, label


_read_data("./raw/train.data", "./raw/train.label")

