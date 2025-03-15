import numpy as np
import pandas as pd

# Calculate Entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

# Compute Information Gain
def information_gain(X, y, feature_index):
    total_entropy = entropy(y)
    unique_values, counts = np.unique(X[:, feature_index], return_counts=True)
    
    weighted_entropy = 0
    for val, count in zip(unique_values, counts):
        subset_y = y[X[:, feature_index] == val]
        weighted_entropy += (count / len(y)) * entropy(subset_y)
    
    return total_entropy - weighted_entropy

# Build the Decision Tree Recursively
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return np.argmax(np.bincount(y))  # Return most common class

        # Find the best feature to split
        best_feature = np.argmax([information_gain(X, y, i) for i in range(X.shape[1])])
        unique_values = np.unique(X[:, best_feature])
        
        # Create tree dictionary
        tree = {}
        for value in unique_values:
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]
            tree[value] = self.fit(subset_X, subset_y, depth + 1)

        return {best_feature: tree}

    def predict(self, X):
        predictions = []
        for x in X:
            subtree = self.tree
            while isinstance(subtree, dict):
                feature = list(subtree.keys())[0]
                subtree = subtree.get(x[feature], 1)  # Default to class 1
            predictions.append(subtree)
        return np.array(predictions)

# Example Dataset
X = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 1]])
y = np.array([0, 1, 1, 0, 1])

# Train the Decision Tree
dt = DecisionTree(max_depth=2)
dt.tree = dt.fit(X, y)
print("Decision Tree:", dt.tree)

# Predict
X_test = np.array([[1, 0], [0, 1]])
print("Predictions:", dt.predict(X_test))
