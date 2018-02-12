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

def _evaluate_data_set(model, data_loader, st):

	model.eval()
	total_loss, correct = 0, 0

	for batch_idx, (data, target) in enumerate(data_loader):

		data = data.view(-1, 784)

		data, target = Variable(data, volatile=True, requires_grad=False), \
					Variable(target, requires_grad=False)

		output = model(data)

		total_loss += F.nll_loss(output, target).data[0]

		_, predicted = torch.max(output.data, 1)

		correct += (predicted == target.data).sum()

	avg_loss = total_loss / float(len(data_loader))
	accuracy = correct / float(len(data_loader.dataset))

	return avg_loss, accuracy


def run(model, train_loader, test_loader, total_epoch, lr, momentum, result_path, verbose=True):

	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	train_loss, train_acc = [], []

	if not os.path.exists(result_path):
		try:
			os.makedirs(result_path)
		except OSError as e:
			print 'Error in making result directory.'
			exit()

	for epoch in range(1, total_epoch+1):
		_ = _train(model, train_loader, optimizer, epoch)

		# train evaluation
		avg_loss, accuracy = _evaluate_data_set(model, train_loader)
		train_loss.append(avg_loss)
		train_acc.append(accuracy)


		print '[EPOCH:{}]\t loss={:.4f}\t accuracy={:.4f}'.format(epoch, avg_loss, accuracy)

	return train_loss, train_acc


