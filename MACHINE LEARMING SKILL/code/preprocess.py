"""Titanic Survival: Preprocessing Pipeline and Cleaned Dataset.
Run: python3 code/preprocess.py  (from project root)
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "csv", "titanic.csv"))
report = []
report.append(f"raw shape: {df.shape}")
report.append("nulls before:\n" + df.isna().sum().to_string())

# ---- feature engineering (before column transforms) ----
def add_family(X):
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    return X

engineer = FunctionTransformer(add_family, validate=False)

num_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
cat_features = ["Pclass", "Sex", "Embarked"]

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = Pipeline([
    ("engineer", engineer),
    ("cols", ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ], remainder="drop")),
])

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess.fit(X_train)
X_train_t = preprocess.transform(X_train)
X_test_t = preprocess.transform(X_test)

report.append(f"\ntransformed train shape: {X_train_t.shape}")
report.append(f"transformed test shape: {X_test_t.shape}")
report.append(f"nulls after transform (train): {int(np.isnan(X_train_t).sum())}")

# ---- cleaned full dataset (engineered + imputed, human-readable) ----
clean = add_family(df.copy())
clean["Age"] = clean["Age"].fillna(clean["Age"].median())
clean["Fare"] = clean["Fare"].fillna(clean["Fare"].median())
clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])
clean = clean.drop(columns=["Name", "Ticket", "Cabin"])
clean.to_csv(os.path.join(OUT, "titanic_cleaned.csv"), index=False)
report.append(f"\ncleaned csv shape: {clean.shape}")
report.append("nulls in cleaned csv:\n" + clean.isna().sum().to_string())
report.append("\ncleaned columns: " + ", ".join(clean.columns))

joblib.dump(preprocess, os.path.join(OUT, "preprocess_pipeline.joblib"))
joblib.dump((X_train, X_test, y_train, y_test),
            os.path.join(OUT, "splits.joblib"))

with open(os.path.join(OUT, "preprocessing_report.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))
print("\nsaved: titanic_cleaned.csv, preprocess_pipeline.joblib, splits.joblib")
