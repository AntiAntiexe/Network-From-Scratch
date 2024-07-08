
#Raw python
'''import math

layer_output = [4.8, 1.21, 2, 2.385]

E = math.e

exp_values = []

for i in layer_output:
    exp_values.append(E**i)

norm_base = sum(exp_values)

norm_values = []

for val in exp_values:
    norm_values.append(val/norm_base)

print(norm_values)

print(sum(norm_values))
'''


#Numpy
import numpy as np
import nnfs

nnfs.init()

layer_output = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)

print(np.sum(layer_output, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

#print(sum(norm_values))