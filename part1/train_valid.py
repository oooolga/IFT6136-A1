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

def _evaluate_data_set(model, data_loader, start=None, end=None):

	model.eval()
	total_loss, correct = 0, 0
	total_data, total_batch = 0, 0

	for batch_idx, (data, target) in enumerate(data_loader):

		if end and batch_idx >= end:
			break

		if start and batch_idx < start:
			continue

		data = data.view(-1, 784)

		data, target = Variable(data, volatile=True, requires_grad=False), \
					Variable(target, requires_grad=False)

		output = model(data)

		total_loss += F.nll_loss(output, target).data[0]

		_, predicted = torch.max(output.data, 1)

		correct += (predicted == target.data).sum()

		total_data += len(data)
		total_batch += 1

	avg_loss = total_loss / float(total_batch)
	accuracy = correct / float(total_data)
	return avg_loss, accuracy


def run(model, train_loader, test_loader, total_epoch, lr, momentum, result_path,
		num_valid_batch=None, verbose=True):

	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = [], [], [], [], [], []

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

		output = '[EPOCH {}]\ttrain_loss={:.3f}  train_acc={:.3f}  '.format(epoch,
																		   avg_loss,
																		   accuracy)

		# valid evaluation
		if test_loader and num_valid_batch:
			avg_loss, accuracy = _evaluate_data_set(model, test_loader, end=num_valid_batch)
			val_loss.append(avg_loss)
			val_acc.append(accuracy)
			output += 'valid_loss={:.3f}  valid_acc={:.3f}  '.format(avg_loss, accuracy)

		# test evaluation
		if test_loader:
			avg_loss, accuracy = _evaluate_data_set(model, test_loader, start=num_valid_batch)
			test_loss.append(avg_loss)
			test_acc.append(accuracy)
			output += 'test_loss={:.3f}  test_acc={:.3f}'.format(avg_loss, accuracy)

		print output


	return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc 


