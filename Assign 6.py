#assignnment 6

import numpy as np
import tensorflow as tf

# Define training data for numbers 0, 1, 2, and 39
training_data = {
    0: [[1,1,1],
        [1,0,1],
        [1,0,1],
        [1,0,1],
        [1,1,1]],

    1: [[0,1,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,1,0]],

    2: [[1,1,1],
        [0,0,1],
        [1,1,1],
        [1,0,0],
        [1,1,1]],

    39: [[1,1,1],
        [0,0,1],
        [1,1,1],
        [0,0,1],
        [1,1,1]]
}

# Convert training data to feature vectors and labels
X_train = []
y_train = []

for label, matrix in training_data.items():
    X_train.append(np.array(matrix).flatten())
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(15,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(40, activation='softmax')  # Update output size to accommodate label 39
])

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Test the trained model
test_data = {
    "Test 1": [[1,1,1],
                [1,0,1],
                [1,0,1],
                [1,0,1],
                [1,1,1]],

    "Test 2": [[0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0],
                [0,1,0]],

    "Test 3": [[1,1,1],
                [0,0,1],
                [1,1,1],
                [1,0,0],
                [1,1,1]]
}

for test_name, test_matrix in test_data.items():
    test_vector = np.array(test_matrix).flatten()
    prediction = model.predict(np.array([test_vector]))
    predicted_number = np.argmax(prediction)
    print(f"{test_name} - Predicted Number: {predicted_number}")
