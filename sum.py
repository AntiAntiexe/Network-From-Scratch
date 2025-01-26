'''inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# for each neuron weight and neuron bias in weights and biases respectively, and the indexes will be the same
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        # sum the weights times the inputs
        neuron_output += weight * n_input

    # Add bias
    neuron_output += neuron_bias

    #We then put the neuron out puts to a list of this layers outputs, so if there was a neuron after this one it can use these as the input.
    layer_outputs.append(neuron_output)

print(layer_outputs)'''

import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)


print(output)