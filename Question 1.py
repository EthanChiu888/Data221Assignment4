# Q1 — Dataset Exploration and Understanding

# Import required libraries
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load dataset
data = load_breast_cancer()

# Construct feature matrix X and target vector y
X = data.data
y = data.target

# Report shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Count number of samples in each class
unique, counts = np.unique(y, return_counts=True)

print("\nClass distribution:")
for label, count in zip(unique, counts):
    class_name = "malignant" if label == 0 else "benign"
    print(f"Class {label} ({class_name}): {count}")

# Discussion:
# The dataset is slightly imbalanced.
# There are more benign cases than malignant cases.
#
# Class balance is important because if a dataset is heavily imbalanced,
# a model may become biased toward the majority class and perform poorly
# on the minority class.
#
# For example, if most samples are benign, the model might predict "benign"
# most of the time and still achieve high accuracy, but it would fail to
# correctly identify malignant (cancerous) cases, which are more critical.