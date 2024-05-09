#assignnment 9

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        # Forward pass through the network
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)
        return output
    
    def backward(self, inputs, targets, output):
        # Backward pass through the network
        output_error = targets - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta) * self.learning_rate
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for input_data, target_data in zip(inputs, targets):
                # Forward pass
                output = self.forward(input_data)
                # Backward pass
                self.backward(input_data.reshape(1, -1), target_data.reshape(1, -1), output)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {np.mean(np.square(targets - self.forward(inputs)))}')
    
    def predict(self, inputs):
        return self.forward(inputs)

# Example usage
if __name__ == "__main__":
    # Define input and target data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])  # XOR function
    
    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    nn.train(X_train, y_train, epochs=1000)
    
    # Test the trained network
    predictions = nn.predict(X_train)
    print("Predictions:", predictions)
