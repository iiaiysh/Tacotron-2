import numpy as np

import pickle

import os

from PIL import Image

from matplotlib import pyplot as plt

pickle_list = os.listdir('./')
pickle_list = [item for item in pickle_list if item.endswith('.pickle')]


for item in pickle_list:
    with open(item, 'rb') as f:
        align = pickle.load(f)

    plt.imshow(align, origin='lower', aspect='auto')
    plt.show()

    a = 1

a = 1