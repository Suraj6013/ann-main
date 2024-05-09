#assignnment 5

import numpy as np

class BAM:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.W = np.dot(self.X.T, self.Y)

    def recall_X(self, Y):
        return np.dot(Y, self.W.T)

    def recall_Y(self, X):
        return np.dot(X, self.W)

# Define two pairs of vectors
X = [[1, 0, 1],
    [0, 1, 0]]

Y = [[1, 1],
    [0, 0]]

# Create BAM object
bam = BAM(X, Y)

# Recall Y given X
X_input = np.array([1, 0, 1])
Y_output = bam.recall_Y(X_input)
print("Recalled Y:", Y_output)

# Recall X given Y
Y_input = np.array([1, 1])
X_output = bam.recall_X(Y_input)
print("Recalled X:", X_output)
