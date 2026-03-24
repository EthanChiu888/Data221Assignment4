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
# max_depth=3 is used to control tree complexity
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
sorted_indices = np.argsort(importances)[::-1]

# Display the top five most important features
print("\nTop 5 Most Important Features:")
for i in range(5):
    print(f"{i+1}. {feature_names[sorted_indices[i]]}: {importances[sorted_indices[i]]:.4f}")

# Discussion:
# Controlling model complexity helps reduce overfitting because it prevents
# the decision tree from becoming too deep and memorizing the training data.
# A simpler tree is often better able to generalize to unseen test data.
#
# In this case, setting max_depth limits how many levels the tree can grow.
# This usually lowers training accuracy slightly, but it can improve test
# accuracy or make it closer to the training accuracy.
#
# Feature importance contributes to interpretability because it shows which
# features the decision tree relied on most when making decisions.
# This makes it easier to understand which variables were most influential
# in the classification process.




