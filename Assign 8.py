#assignnment 8

import numpy as np

class ART:
    def __init__(self, input_size, vigilance_parameter):
        self.input_size = input_size
        self.vigilance_parameter = vigilance_parameter
        self.w = np.random.rand(input_size)
        self.b = np.random.rand(input_size)
    
    def _normalize_input(self, x):
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        else:
            return x / norm
    
    def _match(self, x):
        epsilon = 1e-10  # Small epsilon value to avoid division by zero
        return np.sum(np.minimum(self.w, x)) / (np.sum(self.w) + epsilon)
    
    def _classify(self, x):
        return self._match(x) >= self.vigilance_parameter
    
    def _learn(self, x):
        self.w = np.minimum(self.w, x)
    
    def train(self, X):
        for x in X:
            x = self._normalize_input(x)
            print("Normalized input:", x)
            if not self._classify(x):
                print("Learning:", x)
                self._learn(x)
        print("Training complete. Learned weights:", self.w)
    
    def predict(self, X):
        predictions = []
        for x in X:
            x = self._normalize_input(x)
            print("Normalized input:", x)
            predictions.append(self._classify(x))
        return np.array(predictions, dtype=int)

# Example usage
if __name__ == "__main__":
    # Define input data
    X_train = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    # Initialize ART network
    art = ART(input_size=4, vigilance_parameter=0.1)

    # Train the network
    art.train(X_train)

    # Test the network
    X_test = np.array([
        [1, 1, 0, 1],
        [0, 0, 0, 0]
    ])
    predictions = art.predict(X_test)
    print("Predictions:", predictions)
