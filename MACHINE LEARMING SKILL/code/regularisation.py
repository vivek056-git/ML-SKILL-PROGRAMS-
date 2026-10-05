"""California Housing — Linear Regression and Regularisation.

Compares LinearRegression vs Ridge / Lasso / ElasticNet (with quick CV
for alpha selection) on the California housing dataset.
Outputs metrics, coefficient comparison plot, predicted-vs-actual plot.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ---------- data ----------
housing = fetch_california_housing(as_frame=True)
df = housing.frame.rename(columns={"MedHouseVal": "MedHouseVal"})
df.to_csv(os.path.join(CSV, "housing.csv"), index=False)
print(f"rows: {len(df)}, features: {list(housing.feature_names)}")

X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

# ---------- models ----------
alphas = np.logspace(-3, 3, 13)
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": RidgeCV(alphas=alphas, cv=3),
    "Lasso": LassoCV(alphas=alphas, cv=3, max_iter=5000),
    "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8],
                               cv=3, max_iter=5000),
}

results, coefs = [], {}
for name, m in models.items():
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    cv = float(cross_val_score(m, Xtr, y_train, cv=3,
                               scoring="neg_root_mean_squared_error").mean()) * -1
    alpha = getattr(m, "alpha_", None)
    results.append({"model": name, "rmse": rmse, "mae": mae,
                    "r2": r2, "cv_rmse": cv,
                    "alpha": round(float(alpha), 4) if alpha is not None else "-"})
    coefs[name] = m.coef_
    print(f"{name:16s} rmse={rmse:.4f} mae={mae:.4f} r2={r2:.4f} alpha={alpha}")

pd.DataFrame(results).to_csv(os.path.join(OUT, "metrics.csv"), index=False)
with open(os.path.join(OUT, "metrics.txt"), "w") as f:
    f.write("CALIFORNIA HOUSING — REGRESSION METRICS (test set)\n")
    f.write("==================================================\n\n")
    f.write(pd.DataFrame(results).to_string(index=False))
    best = min(results, key=lambda r: r["rmse"])
    f.write(f"\n\nBest model by test RMSE: {best['model']} "
            f"(RMSE={best['rmse']:.4f}, R2={best['r2']:.4f})\n")
    f.write("Regularisation (Ridge/Lasso/ElasticNet) shrinks coefficients, "
            "reducing overfitting vs plain LinearRegression.\n")

# ---------- coefficient comparison ----------
feat = housing.feature_names
x = np.arange(len(feat))
width = 0.2
plt.figure(figsize=(12, 5))
for i, (name, c) in enumerate(coefs.items()):
    plt.bar(x + (i - 1.5) * width, c, width=width, label=name)
plt.xticks(x, feat, rotation=30, ha="right")
plt.axhline(0, color="k", lw=0.8)
plt.title("Coefficient comparison (standardised features)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "coefficients.png"), dpi=120); plt.close()

# ---------- predicted vs actual (best model) ----------
best_name = min(results, key=lambda r: r["rmse"])["model"]
pred = models[best_name].predict(Xte)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=8, alpha=0.3)
lo, hi = y_test.min(), y_test.max()
plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
plt.xlabel("actual MedHouseVal"); plt.ylabel("predicted MedHouseVal")
plt.title(f"Predicted vs actual — {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "predicted_vs_actual.png"), dpi=120); plt.close()

print("done ->", OUT)
