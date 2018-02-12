__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse, os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model import *
from train_valid import *

weight_init_methods = ['zero', 'normal', 'glorot']

def load_data(batch_size, test_batch_size):

	train_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True)

	return train_loader, test_loader