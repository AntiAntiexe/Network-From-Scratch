import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probabilities = self.exp_val / np.sum(self.exp_val, axis=1, keepdims=True)
        self.output = self.probabilities


X, y = spiral_data(samples= 100, classes=3)

dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(n_inputs=3, n_neurons=3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output)









