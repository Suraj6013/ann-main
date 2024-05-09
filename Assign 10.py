#assignnment 10

import numpy as np

class HopfieldNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.zeros((input_size, input_size))
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            # Update weights using Hebbian learning rule
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)  # Set diagonal to 0
    
    def recall(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern)
        output_pattern = np.copy(input_pattern)
        for _ in range(max_iterations):
            for i in range(self.input_size):
                # Update each neuron asynchronously
                activation = np.dot(self.weights[i], output_pattern)
                output_pattern[i] = 1 if activation > 0 else -1
        return output_pattern

# Example usage
if __name__ == "__main__":
    # Define input patterns
    patterns = [
        [1, 1, -1, -1],
        [-1, -1, 1, 1],
        [1, -1, -1, 1],
        [-1, 1, 1, -1]
    ]

    # Initialize and train the Hopfield network
    hopfield_net = HopfieldNetwork(input_size=len(patterns[0]))
    hopfield_net.train(patterns)

    # Test the network by recalling the stored patterns
    for pattern in patterns:
        recalled_pattern = hopfield_net.recall(pattern)
        print("Input Pattern:", pattern)
        print("Recalled Pattern:", recalled_pattern)
        print()
