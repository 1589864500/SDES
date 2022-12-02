import os
import numpy as np


X = np.load('X.npy')
F = np.load('F.npy')
gametable = np.load('gametable.npy')

print(X.shape, ' ', F.shape, ' ', gametable.shape)

print(F[-1])
print(X[-1], np.sum(X[-1]))
# print(gametable)
