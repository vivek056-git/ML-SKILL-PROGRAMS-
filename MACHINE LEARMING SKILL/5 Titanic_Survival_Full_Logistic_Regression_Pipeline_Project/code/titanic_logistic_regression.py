"""
Project 5 — Titanic Survival: Full Logistic Regression Pipeline

Uses the standard Kaggle Titanic training data (891 rows, 12 columns)
available in this project folder.

Pipeline:
EDA -> feature engineering -> train/test split -> imputation ->
scaling/encoding -> logistic regression -> evaluation -> CV ->
ROC-AUC -> confusion matrix -> coefficient/odds-ratio analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset" / "titanic_train.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    common = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
    return common.get(title, "Rare")

df["Title"] = df["Name"].apply(extract_title)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
df["HasCabin"] = df["Cabin"].notna().astype(int)

features = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "FamilySize", "IsAlone",
    "FarePerPerson", "HasCabin"
]
target = "Survived"

X = df[features]
y = df[target]

num = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "FarePerPerson", "HasCabin"
]
cat = ["Sex", "Embarked", "Title"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ]), cat)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.50).astype(int)

metrics = pd.DataFrame([{
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
    "ROC_AUC": roc_auc_score(y_test, prob),
    "Log_Loss": log_loss(y_test, prob)
}])

metrics.to_csv(OUT / "logistic_regression_metrics.csv", index=False)
print(metrics)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1
)

print("5-fold mean ROC-AUC:", cv_scores["test_roc_auc"].mean())
print("5-fold mean accuracy:", cv_scores["test_accuracy"].mean())
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
