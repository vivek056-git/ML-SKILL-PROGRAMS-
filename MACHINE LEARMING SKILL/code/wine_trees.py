"""Wine Quality: Decision Tree vs Random Forest Classification."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "csv", "winequality.csv"))

# bucket quality: bad (<=4), ok (5-6), good (>=7)
df["label"] = pd.cut(df["quality"], bins=[0, 4, 6, 10],
                     labels=["bad", "ok", "good"])
X = df.drop(columns=["quality", "label"])
y = df["label"]
print("class counts:", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}
lines = []
f1s = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    acc = accuracy_score(y_test, p)
    f1 = f1_score(y_test, p, average=None)
    f1s[name] = f1
    lines.append(f"{name}: accuracy={acc:.4f}")
    for c, v in zip(["bad", "ok", "good"], f1):
        lines.append(f"  f1({c})={v:.4f}")
    print(lines[-4])

with open(os.path.join(BASE, "output", "metrics.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

# feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (name, m) in zip(axes, models.items()):
    order = np.argsort(m.feature_importances_)
    ax.barh(np.array(X.columns)[order], m.feature_importances_[order])
    ax.set_title(name)
fig.suptitle("Feature Importance Comparison")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "output", "feature_importance_compare.png"), dpi=120)

# confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, m) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, m.predict(X_test)),
                           display_labels=["bad", "ok", "good"]).plot(ax=ax)
    ax.set_title(name)
fig.suptitle("Confusion Matrices")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "output", "confusion_matrices.png"), dpi=120)
print("saved metrics.txt, feature_importance_compare.png, confusion_matrices.png")
