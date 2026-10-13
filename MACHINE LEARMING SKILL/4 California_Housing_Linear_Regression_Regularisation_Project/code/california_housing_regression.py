"""
Project 4 — California Housing: Linear Regression and Regularisation

The included CSV is a reproducible PRACTICE dataset with the official
California Housing schema. The notebook contains an option to load the
official scikit-learn California Housing dataset when it is available
locally or can be downloaded.

Official dataset characteristics:
20,640 rows, 8 numeric predictive features + target MedHouseVal.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "dataset" / "california_housing.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "MedHouseVal"
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

def pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

models = {
    "Linear Regression": pipe(LinearRegression()),
    "Ridge": pipe(Ridge(alpha=1.0)),
    "Lasso": pipe(Lasso(alpha=0.001, max_iter=30000)),
    "ElasticNet": pipe(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000))
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "model_comparison.csv", index=False)
print(results.to_string(index=False))

alphas = np.logspace(-4, 3, 15)

ridge_search = GridSearchCV(
    pipe(Ridge()),
    {"model__alpha": alphas},
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1
)
ridge_search.fit(X_train, y_train)

lasso_search = GridSearchCV(
    pipe(Lasso(max_iter=50000)),
    {"model__alpha": alphas},
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1
)
lasso_search.fit(X_train, y_train)

print("\nBest Ridge:", ridge_search.best_params_)
print("Best Lasso:", lasso_search.best_params_)

print("\nDone. Results saved to:", OUT)
