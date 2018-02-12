__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

class Net(nn.Module):

	def __init__(self, fc1_out_dim, fc2_out_dim, w_init_type):
		super(Net, self).__init__()

		in_dim = 784
		out_dim = 10

		self.fc1 = nn.Linear(in_dim, fc1_out_dim)
		self.fc2 = nn.Linear(fc1_out_dim, fc2_out_dim)
		self.fc3 = nn.Linear(fc2_out_dim, out_dim)

		self.weight_bias_init(w_init_type)

		print 'Number of features of the input data: {}'.format(in_dim)
		print 'Number of hidden units in layer 1: {}'.format(fc1_out_dim)
		print 'Number of hidden units in layer 2: {}'.format(fc1_out_dim)
		print 'Number of output classes: {}\n'.format(out_dim)

		print 'Total number of parameters: {}\n'.format(self.cal_total_num_parameters())


	def weight_bias_init(self, w_init_type):

		if w_init_type == 'zero':
			for p in self.parameters():
				torch.nn.init.constant(p, 0)

		if w_init_type == 'normal':
			torch.nn.init.normal(self.fc1.weight, 0, 1)
			torch.nn.init.normal(self.fc2.weight, 0, 1)
			torch.nn.init.normal(self.fc3.weight, 0, 1)

			torch.nn.init.constant(self.fc1.bias, 0)
			torch.nn.init.constant(self.fc2.bias, 0)
			torch.nn.init.constant(self.fc3.bias, 0)

		if w_init_type == 'glorot':
			torch.nn.init.xavier_uniform(self.fc1.weight, 1)
			torch.nn.init.xavier_uniform(self.fc2.weight, 1)
			torch.nn.init.xavier_uniform(self.fc3.weight, 1)

			torch.nn.init.constant(self.fc1.bias, 0)
			torch.nn.init.constant(self.fc2.bias, 0)
			torch.nn.init.constant(self.fc3.bias, 0)

		for p in self.parameters():
			p.requires_grad = True

	def cal_total_num_parameters(self):

		count = 0

		for p in self.parameters():
			count += p.view(-1).size(0)

		return count

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return F.log_softmax(self.fc3(x))