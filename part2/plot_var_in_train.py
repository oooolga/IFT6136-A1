import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss_100 = np.array(pickle.load(open("train_loss100.pkl", "rb")))
loss_1 = np.array(pickle.load(open("train_loss1.pkl", "rb")))

def smooth_data(data, threshold=5):
    """
    replace any data larger than threshold with average between two time step
    """
    for i in range(1, len(data)):
        if data[i] > threshold:
            data[i] = (data[i-1] + data[i+1]) / 2
    return data

loss_100 = smooth_data(loss_100)
loss_1 = smooth_data(loss_1)

plt.figure()
steps = np.arange(0,5000,32)
plt_100, = plt.plot(steps, loss_100[steps], 'rx')
plt_1, = plt.plot(steps, loss_1[steps], 'bx')
plt.legend([plt_100, plt_1], ["batch size 100","batch size 1"])
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig("var_in_train.png")
