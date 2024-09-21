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
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)



'''
Activation function which is Rectified linear unit.

Is used inbetween the layers. After using forward from layer dense use the forward pass from ReLu.
'''
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.dinputs <= 0] = 0

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

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and Gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate jacobian marric of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Outputs layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and retrun loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1 )

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


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
    def backward(self, dvalues, y_true):

        # Number of Samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # if the bales are sparse, trun them into on hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true * dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples



'''
Create the neural network architecture by calling the functions and classes.
'''
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
'''
# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense( 3 , 3 )

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# Let's see output of the first few samples:
print (loss_activation.output[:5])

# Print loss value
print ( 'loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis = 1 )
if len (y.shape) == 2:
    y = np.argmax(y, axis = 1 )

accuracy = np.mean(predictions == y)

# Print accuracy
print ('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print (dense1.dweights)
print (dense1.dbiases)
print (dense2.dweights)
print (dense2.dbiases)





