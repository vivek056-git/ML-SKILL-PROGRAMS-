"""Heart Disease: Decision Tree Classification."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "csv", "heart.csv"))

X = df.drop(columns=["target"])
y = df["target"]

X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
report = classification_report(y_test, pred)

with open(os.path.join(BASE, "output", "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n{report}")
print(f"accuracy: {acc:.4f}")

plt.figure(figsize=(16, 9))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=["no", "yes"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (max_depth=4) — Heart Disease")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "output", "tree.png"), dpi=120)
print("saved output/metrics.txt, output/tree.png")
