"""Heart Disease: Random Forest Ensemble."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "csv", "heart.csv"))

X = df.drop(columns=["target"])
y = df["target"]

X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))

with open(os.path.join(BASE, "output", "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nOOB score: {rf.oob_score_:.4f}\n")
print(f"accuracy: {acc:.4f}  oob: {rf.oob_score_:.4f}")

# feature importance
imp = rf.feature_importances_
order = np.argsort(imp)
plt.figure(figsize=(8, 6))
plt.barh(np.array(X.columns)[order], imp[order])
plt.title("Random Forest — Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "output", "feature_importance.png"), dpi=120)

# confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, rf.predict(X_test)),
                       display_labels=["no", "yes"]).plot(ax=ax)
ax.set_title("Confusion Matrix")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "output", "confusion_matrix.png"), dpi=120)
print("saved metrics.txt, feature_importance.png, confusion_matrix.png")
