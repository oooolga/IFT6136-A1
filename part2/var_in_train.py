import preprocessing
from preprocessing import data_input

import ipdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

#########CONFIG####################
mode = 1
lr = 0.1
bsz = 1
use_cuda = torch.cuda.is_available()
##################################

################DEFINITION#########
class MLP(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 100)
        self.linear1.bias.data.zero_()
        nn.init.xavier_uniform(
                self.linear1.weight.data,
                gain=nn.init.calculate_gain('relu')
                )

        self.linear2 = nn.Linear(100, num_classes)
        self.linear2.bias.data.zero_()
        nn.init.xavier_uniform(
                self.linear2.weight.data,
                )


    def forward(self, inputs):
        """
        inputs:[bsz, num_inputs]
        return: logprobs [bsz, num_classes]
        """
        out = F.relu(self.linear1(inputs))
        out = self.linear2(out)
        logprobs = F.log_softmax(out)
        return logprobs


########################################



###############batch_size 1 ########
start = time.time()
train_iter, _ = data_input(bsz, mode)
print("load data cost {0:.2f} sec".format(time.time()-start))

model = MLP(preprocessing.vocab_size, 20)
if use_cuda:
    model.cuda()

optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

######################################
import ipdb
losses = []
train_iter.__iter__()
data = torch.FloatTensor(bsz, preprocessing.vocab_size)
label = torch.LongTensor(bsz)
data = Variable(data)
label = Variable(label)

criterion = torch.nn.CrossEntropyLoss()

if use_cuda:
    data, label = data.cuda(), label.cuda()
for step in range(5000):
    try:
        _data, _label = train_iter.__next__()
    except StopIteration:
        train_iter.__iter__()
        _data, _label = train_iter.__next__()

    data.data.resize_(_data.size()).copy_(_data)
    label.data.resize_(_label.size()).copy_(_label)
    out = model(data)
    nll = criterion(out, label)
    optim.zero_grad()
    nll.backward()
    optim.step()
    print("step {} nll {:.2f}".format(step+1, nll.data[0]))
    losses.append(nll.data[0])

pickle.dump(losses, open("train_loss{}.pkl".format(bsz), "wb"))
