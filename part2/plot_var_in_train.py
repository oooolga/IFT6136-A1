import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

loss_100 = pickle.load(open("train_loss100.pkl", "rb"))
loss_1 = pickle.load(open("train_loss1.pkl", "rb"))
plt.figure()
steps = [ i+1 for i in range(5000)]
plt_100, = plt.plot(steps, loss_100, 'rx')
plt_1, = plt.plot(steps, loss_1, 'bx')
plt.legend([plt_100, plt_1], ["batch size 100","batch size 1"])
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("var_in_train.png")
