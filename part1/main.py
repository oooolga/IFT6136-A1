__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

from util import *

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
						help='Learning rate')
	parser.add_argument('-m', '--momentum', default=0.5, type=float, help="Momentum")
	parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training')
	parser.add_argument('--test_batch_size', default=1000, type=int,
						help='Mini-batch size for testing')
	parser.add_argument('--plot_iter', default=200, type=int, help='Plot iteration')
	parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
	parser.add_argument('-w', '--weight_init_method', default='glorot', type=str,
						help='Weight initialization method')
	parser.add_argument('-r', '--result_path', default='./result', type=str,
						help='Result path')
	parser.add_argument('-a', '--alpha', default=1.0, type=float,
						help='Alpha')

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Weight initialization method: {}'.format(args.weight_init_method))
	print('Mini-batch size: {}'.format(args.batch_size))
	print('Nonlinearity: {}'.format('ReLU'))
	print('Alpha: {}\n'.format(args.alpha))

if __name__ == '__main__':

	args = parse()
	output_model_setting(args)

	if args.alpha > 1:
		print 'Alpha has to be in range (0,1].'
		exit()

	torch.manual_seed(args.seed)
	
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	train_loader, valid_loader, test_loader = load_data(batch_size=args.batch_size,
										  test_batch_size=args.test_batch_size,
										  alpha=args.alpha)

	print 'Number of training data: {}\n'.format(len(train_loader.dataset))

	model = Net(128, 128, args.weight_init_method)

	if use_cuda:
		model.cuda()


	train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = \
		run(model, train_loader, None, total_epoch=args.epoch,
			lr = args.learning_rate, momentum=args.momentum, result_path=args.result_path,
			valid_loader=valid_loader)

	plt.plot(range(1, 1+args.epoch), train_loss, 'ro-', label='train')
	plt.plot(range(1, 1+args.epoch), val_loss, 'bs-', label='valid')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.title('Epoch vs Loss')
	plt.legend(loc=4)

	plt.savefig('{}/Learning_Curves_4_loss.png'.format(args.result_path))
	plt.clf()

	plt.plot(range(1, 1+args.epoch), train_acc, 'ro-', label='train')
	plt.plot(range(1, 1+args.epoch), val_acc, 'bs-', label='valid')

	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')

	plt.title('Epoch vs Accuracy')
	plt.legend(loc=4)

	plt.savefig('{}/Learning_Curves_4_result.png'.format(args.result_path))
	plt.clf()