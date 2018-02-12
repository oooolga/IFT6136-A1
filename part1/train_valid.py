__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def _train(model, train_loader, optimizer, epoch):

	model.train()

	for batch_idx, (data, target) in enumerate(train_loader):

		data = data.view(-1, 784)

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		optimizer.zero_grad()
		output = model(data)

		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

	return loss


def run(model, train_loader, test_loader, total_epoch, lr, momentum, verbose=True):

	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	train_loss = []

	for epoch in range(1, total_epoch+1):
		loss = _train(model, train_loader, optimizer, epoch)
		
		train_loss.append(loss.data[0])
		print '[EPOCH:{}]\t loss={}'.format(epoch, loss.data[0])


