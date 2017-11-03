import preprocessing
from preprocessing import data_input

import ipdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#########CONFIG####################
use_cuda = torch.cuda.is_available()
batch_size = 100
mode = 0 # preprocessing mode
num_epochs = 20
lr = 0.01
log_interval = 20
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


def train_epoch(data_iter, model, optim):
    model.train()
    num_data = 0
    num_correct = 0
    for i, (data, label) in enumerate(data_iter):
        data = Variable(torch.Tensor(data))
        label = Variable(torch.LongTensor(label))
        logprobs = model(data)

        nll = F.nll_loss(logprobs, label)
        optim.zero_grad()
        nll.backward()
        optim.step()

        num_data += data.size(0)
        pred = torch.max(logprobs, -1)[1]
        num_correct += (pred==label).sum().data[0]

        if (i+1) % log_interval == 0:
            print("batch {} nll_loss {%2f}, acc {%2f}".format(
                i+1, nll.data[0], num_correct/float(num_data)
                ))



########################################

###############DATA, MODEL, OPTIM########
start = time.time()
train_iter, test_iter = data_input(batch_size, mode)
print("load data cost {} sec".format(time.time()-start))

model = MLP(preprocessing.vocab_size, 20)
if use_cuda:
    model.cuda()

optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

######################################

for epoch in range(num_epochs):
    print('-----------Epoch {}---------'.format(epoch))
    train_epoch(train_iter, model, optim)
