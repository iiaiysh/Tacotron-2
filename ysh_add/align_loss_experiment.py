import numpy as np

import pickle

import os

from PIL import Image

from matplotlib import pyplot as plt

# pickle_list = os.listdir('./')
# pickle_list = [item for item in pickle_list if item.endswith('.pickle')]
#
#
# for item in pickle_list:
#     with open(item, 'rb') as f:
#         align = pickle.load(f)
#
#     plt.imshow(align, origin='lower', aspect='auto')
#     plt.show()




a = np.ones([200, 300])
k = 0.001
N = a.shape[1]
T = a.shape[0]
for i in range(T):
    for j in range(N):
        a[i,j] = np.math.exp(-np.math.pow(i/T - j/N, 2)/k)



plt.imshow(a)
plt.show()
a = 1