import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss_100 = np.array(pickle.load(open("train_loss100.pkl", "rb")))
loss_1 = np.array(pickle.load(open("train_loss1.pkl", "rb")))
plt.figure()
steps = np.arange(0,5000,20)
plt_100, = plt.plot(steps, loss_100[steps], 'rx')
plt_1, = plt.plot(steps, loss_1[steps], 'bx')
plt.legend([plt_100, plt_1], ["batch size 100","batch size 1"])
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("var_in_train.png")
