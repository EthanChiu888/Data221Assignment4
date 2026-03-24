# Q6 — Convolutional Neural Network with Built-in Dataset

# Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to include the channel dimension
# Original shape: (28, 28)
# New shape: (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build the CNN model
cnn_model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model for at least 15 epochs
cnn_model.fit(X_train, y_train, epochs=15, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)

# Report test accuracy
print("Test Accuracy:", test_accuracy)

# Discussion:
# CNNs are generally preferred over fully connected networks for image data
# because they can automatically detect spatial patterns such as edges,
# shapes, and textures. They also use fewer parameters by sharing weights,
# which makes them more efficient and better suited for images.

# In this task, the convolution layer is learning useful visual features
# from the clothing images, such as edges, outlines, textures, and simple
# shape patterns. These features help the network distinguish between the
# different Fashion MNIST clothing categories.