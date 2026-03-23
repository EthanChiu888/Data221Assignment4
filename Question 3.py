# Q3 — Controlling Tree Complexity and Interpretability

# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split the dataset into training and testing sets using an 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train a constrained Decision Tree model
# Here, max_depth=3 is used to control tree complexity
constrained_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
constrained_model.fit(X_train, y_train)

# Compute training and test accuracy
train_accuracy = constrained_model.score(X_train, y_train)
test_accuracy = constrained_model.score(X_test, y_test)

# Report results
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Get feature importances
importances = constrained_model.feature_importances_

# Sort features by importance in descending order
indices = np.argsort(importances)[::-1]

# Display the top five most important features
print("\nTop 5 Most Important Features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")




