# Q4 — Neural Network for Binary Classification

# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets using an 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a neural network with one hidden layer
# For binary classification, MLPClassifier uses a sigmoid output unit automatically
nn_model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

# Compute training and test accuracy
train_accuracy = nn_model.score(X_train_scaled, y_train)
test_accuracy = nn_model.score(X_test_scaled, y_test)

# Report results
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Discussion:
# Feature scaling is necessary for neural networks because the model learns
# by adjusting weights through optimization methods such as gradient descent.
# If features are on very different scales, the learning process can become
# slower, less stable, and less effective because some features may dominate others.
# Standardizing the features helps the neural network train more efficiently
# and often improves model performance.
# An epoch is one complete pass through the entire training dataset during
# neural network training. In each epoch, the network processes all training
# examples and updates its weights to reduce prediction error.
# Multiple epochs allow the model to gradually improve its learned weights
# and better fit the training data.
