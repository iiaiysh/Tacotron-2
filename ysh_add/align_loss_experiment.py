import numpy as np

import pickle

import os

from PIL import Image

from matplotlib import pyplot as plt

import tensorflow as tf
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
        a[i,j] = 1 - np.math.exp(-np.math.pow(i/T - j/N, 2)/k)

b = np.pad(a, ((0, 10), (0, 20)), 'constant', constant_values=1)

plt.imshow(b)
plt.show()


align = tf.convert_to_tensor(np.ones([200, 300], dtype=np.float64))

loss = tf.reduce_sum(align * a)

sess = tf.Session()
lo = sess.run(loss)

a = 1