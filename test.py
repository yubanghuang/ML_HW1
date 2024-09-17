from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# load mnist dataset
digits = datasets.load_digits()
print(digits)

fig ,ax = plt.subplots(nrows=8,ncols=8)
for i , axi in enumerate(ax.flat) :
    axi.imshow(digits.images[i],cmap='gray')
plt.show()
