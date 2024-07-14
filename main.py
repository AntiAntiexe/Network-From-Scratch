import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() #Initialise the nnfs for ease of use

'''
Layer class defines the neural network architecture.
n_inputs is the user inputs
n_neurons is the number of neurons in the layer.

forward defines the forward pass of the layer by multiplying the inputs and the weights using dot product
then adding the biases.
'''
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)   )
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


'''
Activation function which is Rectified linear unit.

Is used inbetween the layers. After using forward from layer dense use the forward pass from ReLu.
'''
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

'''
Activation function which is Softmax activation function.

Softmax divides the value by the sum of the rest of the values
Is used in the final layer of the neural network.
The softmax forward self.output is the output of the network.
'''
class Activation_Softmax:
    def forward(self, inputs):
        self.exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probabilities = self.exp_val / np.sum(self.exp_val, axis=1, keepdims=True)
        self.output = self.probabilities

'''
Loss defines how "right" the neural network is
'''
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

'''
Loss categorical cross entropy loss defines how "right" the neural network is.
The Forward function fist find the length if the prediction list. The clips the prediction list so values dont
go past infinity.

Then check whether the y_true input is a one hot vector or a scalar list.

if one hot vector:
multiply the y_pred_clipped by the y_true using axis 1 to keep it the same shape.

elif scalar:
using the range of the samples (which is the length of y_pred) then select the corresponding scalar value from the 
y_pred_clipped.
'''
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


'''
Create the neural network architecture.
'''

X, y = spiral_data(samples= 100, classes=3) #Smaple Data of 300

dense1 = Layer_Dense(n_inputs=2, n_neurons=3) #use n_inputs of 2 as it is X, y data, you can use as many n_neurons
                                              # you want
activation1 = Activation_ReLU()

dense2 = Layer_Dense(n_inputs=3, n_neurons=3) #3 n_inputs and there were 3 n_neurons in the last layer
activation2 = Activation_Softmax()

dense1.forward(X) #First layer forward
activation1.forward(dense1.output) #First layer activation function

dense2.forward(activation1.output) #Outout layer forward
activation2.forward(dense2.output) #Outout layer activation function

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(output=activation2.output, y=y) #the output of the network is inputted into the loss
                                                               #function and the y_pred is just teh y values from the
                                                               # spiral data

print('Loss:', loss)









