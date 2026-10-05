"""Titanic Survival: Full Logistic Regression Pipeline (end-to-end).
Run: python3 code/logistic_regression.py  (from project root)
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "csv", "titanic.csv"))

def add_family(X):
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    return X

num_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
cat_features = ["Pclass", "Sex", "Embarked"]

model = Pipeline([
    ("engineer", FunctionTransformer(add_family, validate=False)),
    ("cols", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         cat_features),
    ])),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba),
}
with open(os.path.join(OUT, "metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")
print(metrics)

# confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["died", "survived"]).plot(ax=ax)
ax.set_title("Confusion matrix — logistic regression")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("ROC curve — logistic regression")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=120)
plt.close()

joblib.dump(model, os.path.join(OUT, "logistic_model.joblib"))
print("saved metrics.txt, confusion_matrix.png, roc_curve.png, logistic_model.joblib")
