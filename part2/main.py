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

#########CONFIG####################
parser = argparse.ArgumentParser()
parser.add_argument("--bsz", default=200, type=int)
parser.add_argument("--mode", default=0, type=int)
parser.add_argument("--num_epochs", default=20, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--out", default="./result/", type=str)
parser.add_argument("--log_interval", default=5, type=int)
args = parser.parse_args()
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


def _to_torch(data, label):
    if use_cuda:
        data = Variable(torch.Tensor(data)).cuda()
        label = Variable(torch.LongTensor(label)).cuda()
    else:
        data = Variable(torch.Tensor(data))
        label = Variable(torch.LongTensor(label))
    return data, label


def eval_epoch(data_iter, model):
    model.eval()
    num_data = 0
    num_correct = 0
    for i, (data, label) in enumerate(data_iter):
        data, label = _to_torch(data, label)
        logprobs = model(data)

        num_data += data.size(0)
        pred = torch.max(logprobs, -1)[1]
        num_correct += (pred==label).sum().data[0]

    return num_correct*100 / float(num_data)


def train_epoch(data_iter, model, optim):
    model.train()
    num_data = 0
    num_correct = 0
    for i, (data, label) in enumerate(data_iter):
        data, label = _to_torch(data, label)
        logprobs = model(data)

        nll = F.nll_loss(logprobs, label)
        optim.zero_grad()
        nll.backward()
        optim.step()

        num_data += data.size(0)
        pred = torch.max(logprobs, -1)[1]
        num_correct += (pred==label).sum().data[0]

        if (i+1) % args.log_interval == 0:
            print("batch {} nll_loss {:.2f}, acc {:.2f}%".format(
                i+1, nll.data[0], num_correct*100/float(num_data)
                ))
    return num_correct*100 / float(num_data)



########################################

###############DATA, MODEL, OPTIM########
start = time.time()
train_iter, test_iter = data_input(args.bsz, args.mode)
print("load data cost {0:.2f} sec".format(time.time()-start))

model = MLP(preprocessing.vocab_size, 20)
if use_cuda:
    model.cuda()

optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

######################################
train_accs = []
test_accs = []
for epoch in range(args.num_epochs):
    print('-----------Epoch {}---------'.format(epoch))
    train_acc = train_epoch(train_iter, model, optim)
    test_acc = eval_epoch(test_iter, model)
    print("epoch {} train_acc {:.2f}% eval_acc {:.2f}%".format(
        epoch+1, train_acc, test_acc
        ))
    train_accs.append(train_acc)
    test_accs.append(test_acc)

if not os.path.isdir(args.out):
    os.mkdir(args.out)

plt.figure()
epochs = [ e+1 for e in range(args.num_epochs) ]
train_plt, = plt.plot(epochs, train_accs, "r-")
test_plt, = plt.plot(epochs, test_accs, "b-")
plt.legend([train_plt, test_plt], ["train acc", "test acc"])
plt.savefig(os.path.join(args.out, "acc.png"))
