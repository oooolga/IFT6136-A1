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

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Weight initialization method: {}'.format(args.weight_init_method))
	print('Mini-batch size: {}'.format(args.batch_size))
	print('Nonlinearity: {}\n'.format('ReLU'))

if __name__ == '__main__':

	args = parse()

	torch.cuda.manual_seed_all(args.seed)

	train_loader, test_loader = load_data(batch_size=args.batch_size,
										  test_batch_size=args.test_batch_size)

	model = Net(500, 500, args.weight_init_method)

	run(model, train_loader, test_loader, total_epoch=args.epoch,
		lr = args.learning_rate, momentum=args.momentum, result_path=args.result_path)