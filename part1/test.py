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

def load_data(batch_size, test_batch_size):

	train_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=batch_size, shuffle=True)

	return train_loader, test_loader

if __name__ == '__main__':
	pdb.set_trace()
	train_loader, test_loader = load_data(50, 1000)
	pdb.set_trace()