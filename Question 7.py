# Q7 — CNN Error Analysis and Misclassification Study

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to include the channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build the CNN model
cnn_model = Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
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

# Generate predictions on the test set
y_pred_probabilities = cnn_model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Compute and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print("Rows = Actual class, Columns = Predicted class")
print(conf_matrix)

# Identify misclassified images
misclassified_indices = np.where(y_pred != y_test)[0]

print("\nNumber of misclassified images:", len(misclassified_indices))

# Print details of first three misclassified images
print("\nDetails of first three misclassified images:")
for i in range(3):
    idx = misclassified_indices[i]
    print(f"Image {i+1}: True = {class_names[y_test[idx]]}, Predicted = {class_names[y_pred[idx]]}")

# Visualize at least three misclassified images
plt.figure(figsize=(12, 4))

for i in range(3):
    idx = misclassified_indices[i]
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPredicted: {class_names[y_pred[idx]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Discussion:
# One pattern observed in the misclassifications is that visually similar clothing
# categories are often confused. For example, items like shirts, pullovers, and coats
# may look very similar in grayscale images, leading to incorrect predictions.

# A realistic way to improve CNN performance is to increase model complexity by adding
# more convolutional layers or filters, allowing the network to learn more detailed
# features. Another effective approach is data augmentation, which can help the model
# generalize better by exposing it to more varied training examples.