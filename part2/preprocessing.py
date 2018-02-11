import ipdb
import numpy as np
import pickle
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0, help="0: not processing. 1: tf_idf. 2: standardization")
args = parser.parse_args()

idx2word = open("./raw/vocabulary.txt", "r").read().splitlines()
vocab_size = len(idx2word)

def _freq(line):
    return int(line.split()[2])

def _word_id(line):
    return int(line.split()[1])-1

def _doc_id(line):
    return int(line.split()[0])-1

def read_data(data_file, label_file):
    """
    :return: data [doc_size, vocab_size]. label [doc_size]
    """
    data_lines = open(data_file, "r").read().splitlines()
    label_lines = open(label_file, "r").read().splitlines()
    num_docs = _doc_id(data_lines[-1]) + 1
    assert num_docs == len(label_lines)

    data = np.zeros([num_docs, vocab_size])
    for line in data_lines:
        data[_doc_id(line), _word_id(line)] += _freq(line)
    label = np.array([int(line) for line in label_lines ])
    return data, label

train_data, train_label = read_data("./raw/train.data", "./raw/train.label")
test_data, test_label = read_data("./raw/test.data", "./raw/test.label")
num_doc = len(train_label)

if args.mode == 1:
    # compute IDF for each term
    for word in range(vocab_size):
        num_doc_has_term = 0
        for doc in range(num_doc):
            if train_data[doc, word] > 0:
                num_doc_has_term += 1
        idf = math.log(num_doc / float(num_doc_has_term))
        ipdb.set_trace()

        # modify train_data, test_data
        train_data[:, word] = train_data[:, word] * idf
        test_data[:, word] = test_data[:, word] * idf
    raise NotImplementedError
if args.mode == 2:
    raise NotImplementedError
else:
    pickle.dump({"train":(train_data, train_label), "test":(test_data, test_label)}, open("data.dat", "wb"))



