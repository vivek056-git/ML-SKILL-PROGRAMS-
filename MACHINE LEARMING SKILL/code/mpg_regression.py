"""Auto MPG: Linear Regression, Feature Scaling, and Encoding.

- median-imputes horsepower, drops `name`, one-hot encodes `origin`
- compares LinearRegression WITHOUT scaling vs WITH StandardScaler:
  coefficient magnitudes differ, predictions are identical
- reports RMSE / R2, residual plot, scaling-comparison plot
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CSV = os.path.join(BASE, "csv", "mpg.csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"


def build_dataset():
    df = pd.read_csv(DATA_URL)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df = df.drop(columns=["name"])
    df = pd.get_dummies(df, columns=["origin"], prefix="origin", drop_first=True)
    df.to_csv(CSV, index=False)
    print(f"saved {CSV}: {df.shape}")
    return df


def main():
    df = build_dataset()
    X = df.drop(columns=["mpg"])
    y = df["mpg"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # A) without scaling
    lr_raw = LinearRegression().fit(X_train, y_train)
    pred_raw = lr_raw.predict(X_test)

    # B) with scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr_scaled = LinearRegression().fit(X_train_s, y_train)
    pred_scaled = lr_scaled.predict(X_test_s)

    # predictions must match (scaling is an affine transform)
    max_diff = np.abs(pred_raw - pred_scaled).max()

    metrics = {
        "rmse_raw": float(np.sqrt(mean_squared_error(y_test, pred_raw))),
        "r2_raw": float(r2_score(y_test, pred_raw)),
        "rmse_scaled": float(np.sqrt(mean_squared_error(y_test, pred_scaled))),
        "r2_scaled": float(r2_score(y_test, pred_scaled)),
        "max_pred_diff": float(max_diff),
    }
    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write("Auto MPG — Linear Regression\n\n")
        f.write("A) WITHOUT scaling:\n")
        f.write(f"  RMSE: {metrics['rmse_raw']:.4f}\n")
        f.write(f"  R2:   {metrics['r2_raw']:.4f}\n\n")
        f.write("B) WITH StandardScaler:\n")
        f.write(f"  RMSE: {metrics['rmse_scaled']:.4f}\n")
        f.write(f"  R2:   {metrics['r2_scaled']:.4f}\n\n")
        f.write(f"max |pred_raw - pred_scaled|: {max_diff:.2e} "
                "(identical predictions)\n\n")
        f.write("Coefficient magnitudes (raw vs scaled):\n")
        for n, c_raw, c_s in zip(feature_names, lr_raw.coef_, lr_scaled.coef_):
            f.write(f"  {n:20s} raw={c_raw:12.4f}  scaled={c_s:12.4f}\n")
        f.write(f"\nintercept: raw={lr_raw.intercept_:.4f} "
                f"scaled={lr_scaled.intercept_:.4f}\n")
        f.write("\npreprocessing: horsepower median-imputed, `name` dropped, "
                "`origin` one-hot encoded (drop_first)\n")

    # scaling comparison: |coefficients| raw vs scaled
    idx = np.arange(len(feature_names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(idx - 0.2, np.abs(lr_raw.coef_), width=0.4, label="no scaling")
    ax.bar(idx + 0.2, np.abs(lr_scaled.coef_), width=0.4, label="standardized")
    ax.set_xticks(idx); ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("|coefficient|")
    ax.set_title("Coefficient magnitudes: raw vs standardized features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling_comparison.png"), dpi=120)
    plt.close(fig)

    # residual plot (scaled model)
    resid = y_test.values - pred_scaled
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pred_scaled, resid, alpha=0.5, s=18)
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("predicted mpg"); ax.set_ylabel("residual (true - pred)")
    ax.set_title(f"Residual plot (RMSE={metrics['rmse_scaled']:.2f}, "
                 f"R2={metrics['r2_scaled']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "residual_plot.png"), dpi=120)
    plt.close(fig)

    print(f"RMSE raw={metrics['rmse_raw']:.4f} scaled={metrics['rmse_scaled']:.4f} "
          f"max pred diff={max_diff:.2e}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
