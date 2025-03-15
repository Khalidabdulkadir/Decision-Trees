# Decision Trees - Machine Learning Algorithm

## Introduction

A **Decision Tree** is a supervised machine learning algorithm used for **classification and regression tasks**. It works by recursively splitting the dataset into subsets based on the most significant feature until a clear decision is reached.

## How Decision Trees Work

1. **Entropy Calculation** - Measures impurity in the dataset.
2. **Information Gain** - Determines the best feature to split on.
3. **Tree Construction** - Recursively splits data into decision nodes and leaf nodes.
4. **Prediction** - Navigates through the tree to classify new data.

## Prerequisites

Before using this project, ensure you have:

- Basic knowledge of machine learning
- Understanding of entropy and information gain
- Familiarity with Python (depending on implementation)

## Installation & Usage

### Python Implementation (Using Scikit-Learn)

```bash
pip install numpy pandas scikit-learn matplotlib
```

Run the script:

```bash
python decision_tree.py
```
## Example Dataset (Cats vs. Dogs)

| Ear Shape  | Whiskers  | Face Shape | Weight (kg) | Class |
| ---------- | --------- | ---------- | ----------- | ----- |
| 0 (Pointy) | 1 (Long)  | 0 (Round)  | 4           | Cat   |
| 1 (Floppy) | 0 (Short) | 1 (Narrow) | 22          | Dog   |

## Features

- **Ear Shape:** Pointy (0) or Floppy (1)
- **Whiskers:** Short (0) or Long (1)
- **Face Shape:** Round (0) or Narrow (1)
- **Weight:** Approximate weight of the animal

## Implementation Details

- **Python Implementation:** Uses `DecisionTreeClassifier` from Scikit-Learn.

## Visualization

To visualize the tree in Python:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=["Ear Shape", "Whiskers", "Face Shape", "Weight"], class_names=["Cat", "Dog"], filled=True)
plt.show()
```



