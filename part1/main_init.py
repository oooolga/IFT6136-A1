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
	parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
	parser.add_argument('-r', '--result_path', default='./result', type=str,
						help='Result path')
	parser.add_argument('-a', '--alpha', default=1.0, type=float,
						help='Alpha')

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Mini-batch size: {}'.format(args.batch_size))
	print('Nonlinearity: {}\n'.format('ReLU'))

if __name__ == '__main__':

	args = parse()

	output_model_setting(args)

	if args.alpha > 1:
		print 'Alpha has to be in range (0,1].'
		exit()

	torch.manual_seed(args.seed)

	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	train_loader, _, _ = load_data(batch_size=args.batch_size,
								   test_batch_size=args.test_batch_size,
								   alpha=args.alpha)

	print 'Number of training data: {}\n'.format(len(train_loader.dataset))

	results = {}

	for w_m in weight_init_methods:
		model = Net(500, 500, w_m)

		if use_cuda:
			model.cuda()

		method_result, _, _, _, _, _ = run(model, train_loader, None,
										   total_epoch=args.epoch,
										   lr = args.learning_rate,
										   momentum=args.momentum,
										   result_path=args.result_path)

		results[w_m] = method_result


	plt.plot(range(args.epoch), results['zero'], 'ro-', label='zero')
	plt.plot(range(args.epoch), results['normal'], 'bs-', label='normal')
	plt.plot(range(args.epoch), results['glorot'], 'g^-', label='glorot')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.title('Epoch vs Loss for 3 Different Initialization Methods')
	plt.legend(loc=1)

	plt.savefig('{}/main_init_result.png'.format(args.result_path))