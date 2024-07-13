'''import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)'''

import numpy as np
'''
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

neg_log = -np.log(softmax_output[range(len(softmax_output)), class_targets])

average_loss = np.mean(neg_log)

print(average_loss)
'''

'''samples = 3
y_pred_clipped = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
y_true = [0, 1, 1]

if len(y_true.shape) == 1:
    correct_confidences = y_pred_clipped[range(samples), y_true]

print(correct_confidences)'''

'''
softmax_output = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = [0, 2, 1]

predictions = np.argmax(softmax_output, axis=1)

accuracy = np.mean(predictions == class_targets)

print(accuracy)
'''


import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=2)

plt.scatter(X  [:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()