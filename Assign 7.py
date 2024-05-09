#assignnment 7

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed random numbers to make calculation deterministic
np.random.seed(1)

# Initialize weights randomly with mean 0
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Synapses (weights)
synapse_0 = 2 * np.random.random((input_neurons, hidden_neurons)) - 1
synapse_1 = 2 * np.random.random((hidden_neurons, output_neurons)) - 1

# Training
epochs = 60000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    # Error calculation
    layer_2_error = y - layer_2

    # Backpropagation
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    synapse_1 += layer_1.T.dot(layer_2_delta) * learning_rate
    synapse_0 += layer_0.T.dot(layer_1_delta) * learning_rate

    # Print error every 10000 epochs
    if epoch % 10000 == 0:
        print("Error:", np.mean(np.abs(layer_2_error)))

# Output after training
print("\nOutput after training:")
print(layer_2)
