import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
np.random.seed(0)

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.layer_output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


layer1 = Layer_Dense(n_inputs=2, n_neurons=3)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(n_inputs=3, n_neurons=3)
activation2 = Activation_Softmax()
layer1.forward(X)
activation1.forward(layer1.layer_output)
layer2.forward(activation1.output)
activation2.forward(layer2.layer_output)

print(activation2.output[:5])
