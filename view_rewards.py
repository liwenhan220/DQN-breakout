import numpy as np
import matplotlib.pyplot as plt


data = np.load('train_rewards.npy', allow_pickle=True)

plt.xlim(0, len(data))
plt.plot(np.arange(len(data)), data)
plt.show()
