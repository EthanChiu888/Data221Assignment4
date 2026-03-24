# Q5 — Model Evaluation and Comparison

# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

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

# -------------------------------
# Constrained Decision Tree Model
# -------------------------------
tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
tree_model.fit(X_train, y_train)

# Predict on the test set
y_pred_tree = tree_model.predict(X_test)

# Compute confusion matrix for the constrained decision tree
cm_tree = confusion_matrix(y_test, y_pred_tree)

# Display confusion matrix for the constrained decision tree
print("Constrained Decision Tree Confusion Matrix:")
print("Rows = Actual class, Columns = Predicted class")
print(cm_tree)

# -------------------------------
# Neural Network Model
# -------------------------------

# Standardize input features for the neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train neural network
nn_model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_nn = nn_model.predict(X_test_scaled)

# Compute confusion matrix for the neural network
cm_nn = confusion_matrix(y_test, y_pred_nn)

# Display confusion matrix for the neural network
print("\nNeural Network Confusion Matrix:")
print("Rows = Actual class, Columns = Predicted class")
print(cm_nn)

# Class labels:
# 0 = malignant
# 1 = benign

# Discussion:
# I would prefer the constrained decision tree for this task if its performance
# is similar to the neural network, because it is easier to interpret and explain.
# In a medical classification problem, interpretability is important because it helps
# us understand which features influenced the prediction.
# Advantage of the constrained decision tree:
# - It is easy to interpret because the splits and feature importance can be examined directly.
# Limitation of the constrained decision tree:
# - It may be too simple to capture more complex patterns in the data, especially when constrained.
# Advantage of the neural network:
# - It can learn more complex relationships in the data and may achieve strong predictive performance.
# Limitation of the neural network:
# - It is less interpretable, since its decisions are harder to explain compared with a decision tree.
# From the confusion matrices, both models may perform well, but the most important
# issue is minimizing false negatives, where a malignant case is predicted as benign.
# If both models have similar error patterns, the constrained decision tree is preferred
# because it provides good performance while remaining easier to interpret.





