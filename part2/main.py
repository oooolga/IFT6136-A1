from preprocessing import data_input
import ipdb
import time
import torch

class MLP(nn.module):
    def __init__():
        raise NotImplementedError

start = time.time()
train_iter, test_iter = data_input(1, mode=0)
print("load data cost {} sec".format(time.time()-start))

ipdb.set_trace()
for epoch in range(1):
    for i, (data, label) in enumerate(train_iter):
        print(i)


    for i, (data, label) in enumerate(test_iter):
        print(i)
