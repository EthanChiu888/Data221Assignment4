# Q2 — Decision Tree Model Using Entropy

# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets using 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train a Decision Tree classifier using entropy as the splitting criterion
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_model.fit(X_train, y_train)

# Define accuracy
train_accuracy = dt_model.score(X_train, y_train)
test_accuracy = dt_model.score(X_test, y_test)

# Report accuracy
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Discussion:
# In decision trees, entropy measures the amount of uncertainty or impurity
# in a node. A node has low entropy when most samples belong to one class,
# and high entropy when the classes are more mixed.

# The decision tree selects splits that reduce entropy as much as possible.
# This reduction in entropy is called information gain.

# If the training accuracy is much higher than the test accuracy, that suggests
# overfitting because the model may be learning the training data too closely.
# If both accuracies are high and close together, that suggests good generalization.

# The results suggest slight overfitting.
# This is because the training accuracy is higher than the test accuracy, which means the model
# fits the training data very well but does not perform quite as well on unseen data.

