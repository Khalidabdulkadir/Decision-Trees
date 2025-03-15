from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd

# 📌 Step 1: Create a Simple Dataset (Cats vs Dogs)
data = {
    "Ear_Shape": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 0 = Pointy, 1 = Floppy
    "Whiskers": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],   # 1 = Long, 0 = Short
    "Face_Shape": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1],  # 0 = Round, 1 = Narrow
    "Weight": [4, 3.5, 20, 4.2, 18, 3.8, 25, 22, 4, 19],  # Weight in kg
    "Class": ["Cat", "Cat", "Dog", "Cat", "Dog", "Cat", "Dog", "Dog", "Cat", "Dog"]
}

df = pd.DataFrame(data)

# 📌 Step 3: Prepare Features and Labels
X = df[["Ear_Shape", "Whiskers", "Face_Shape", "Weight"]]
y = df["Class"]

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 📌 Step 4: Make Predictions
X_test_sample = [[0, 1, 0, 4],  # A new cat-like example
                 [1, 0, 1, 22]]  # A new dog-like example
predictions = clf.predict(X_test_sample)

print("Predictions for new samples:", predictions)


# 📌 Step 5: Evaluate the Model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# 📌 Step 6: Visualize the Decision Tree

plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()