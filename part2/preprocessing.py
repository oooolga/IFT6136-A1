import ipdb
import numpy as np
import pickle
import argparse
import math
import os
import torch

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

    data = torch.zeros(num_docs, vocab_size)
    for line in data_lines:
        data[_doc_id(line), _word_id(line)] += _freq(line)
    label = np.array([int(line)-1 for line in label_lines ])
    return data, torch.LongTensor(label)

class data_iter:
    def __init__(self, data, label, bsz):
        self._data = data
        self._label = label
        self._bsz = bsz
        self._num_batch = math.ceil(len(label) / float(bsz))

    def __iter__(self):
        num_data = len(self._label)
        self._curr_batch = 0
        # shuffle data
        self._rand_indices = np.random.choice(num_data, num_data, False)
        return self

    def __next__(self):
        if self._curr_batch == self._num_batch:
            raise StopIteration
        indices = self._rand_indices[
                self._curr_batch*self._bsz:(self._curr_batch+1)*self._bsz
                ]
        indices = torch.LongTensor(indices)
        data_batch = self._data[indices]
        label_batch = self._label[indices]
        self._curr_batch += 1
        return data_batch, label_batch

def data_input(batch_size, mode=0):
    """
    mode 0: no preprocessing
    mode 1: tfidf
    mode 2: standardization
    """
    print("reading data...")
    train_data, train_label = read_data("./raw/train.data", "./raw/train.label")
    test_data, test_label = read_data("./raw/test.data", "./raw/test.label")
    if mode == 1:
        #ipdb.set_trace()
        if os.path.isfile("./idf.pkl"):
            print("loading idf...")
            idf = pickle.load(open("./idf.pkl", "rb"))
        else:
            print("computing idf...")
            ipdb.set_trace()
            corpus = torch.cat([train_data, test_data])
            idf = torch.sum((corpus > 0), 0)
            idf = -torch.log(idf / len(corpus))
            pickle.dump(idf, open("./idf.pkl", "wb"), protocol=4)
        print("computing tfidf...")
        train_data = train_data * idf
        test_data = test_data * idf

    if mode == 2:
        eps = 1e-5
        if os.path.isfile("./stats.pkl"):
            mean, sigma = pickle.load(open("./stats.pkl", 'rb'))
        else:
            mean = torch.mean(train_data, 0)
            sigma = torch.sqrt(torch.var(train_data, 0))
            pickle.dump(
                    (mean, sigma),
                    open('./stats.pkl', 'wb'),
                    protocol=4,
                    )
        train_data = (train_data - mean) / (sigma + eps)
        test_data = (test_data - mean) / (sigma + eps)


    train_iter = data_iter(train_data, train_label, batch_size)
    test_iter = data_iter(test_data, test_label, batch_size)
    return train_iter, test_iter


