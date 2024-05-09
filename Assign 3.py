#assignment 3

import numpy as np


# Define the perceptron class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum >= 0 else 0


# Function to convert ASCII to binary representation
def ascii_to_binary(ascii_value):
    binary_rep = format(ascii_value, '08b')
    return [int(bit) for bit in binary_rep]


# Training data: ASCII values for digits 0 - 9
training_data = {
    '0': 48,
    '1': 49,
    '2': 50,
    '3': 51,
    '4': 52,
    '5': 53,
    '6': 54,
    '7': 55,
    '8': 56,
    '9': 57
}


# Create a perceptron with input size of 8 (binary representation of ASCII values)
perceptron = Perceptron(input_size=8)


# Train the perceptron
epochs = 1000
learning_rate = 0.1
for epoch in range(epochs):
    for digit, ascii_value in training_data.items():
        binary_input = np.array(ascii_to_binary(ascii_value))
        target = 1 if int(digit) % 2 == 0 else 0  # 1 for even, 0 for odd
        prediction = perceptron.predict(binary_input)
        error = target - prediction
        perceptron.weights += learning_rate * error * binary_input
        perceptron.bias += learning_rate * error


# Test the perceptron
test_data = {
    'Odd': [49, 51, 53, 55, 57],  # ASCII for odd numbers
    'Even': [48, 50, 52, 54, 56]  # ASCII for even numbers
}

for label, ascii_values in test_data.items():
    for ascii_value in ascii_values:
        character = chr(ascii_value)  # Convert ASCII value back to character
        binary_input = np.array(ascii_to_binary(ascii_value))
        prediction = perceptron.predict(binary_input)
        even_odd = "Even" if prediction == 1 else "Odd"
        print(f"Predicted label for ASCII {ascii_value} ({character}): {even_odd} ({prediction})")
