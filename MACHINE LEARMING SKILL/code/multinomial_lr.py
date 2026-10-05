"""Diabetes Severity: Multinomial Logistic Regression.

Buckets the continuous diabetes target into Low/Medium/High severity classes
and fits a multinomial (softmax) logistic regression with scaled features.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
CSV = os.path.join(BASE, "csv", "diabetes.csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)


def build_dataset():
    ds = load_diabetes(as_frame=True)
    df = ds.frame.copy()
    df = df.rename(columns={"target": "progression"})
    # 3 severity buckets by tertiles
    q1, q2 = df["progression"].quantile([1 / 3, 2 / 3])
    df["severity"] = pd.cut(
        df["progression"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
    )
    df.to_csv(CSV, index=False)
    print(f"saved {CSV}: {df.shape}, severity counts:\n{df['severity'].value_counts()}")
    return df


def main():
    df = build_dataset()
    feature_cols = [c for c in df.columns if c not in ("progression", "severity")]
    X, y = df[feature_cols], df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # sklearn>=1.5: lbfgs is multinomial by default for multiclass (param removed)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    classes = list(clf.classes_)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)

    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write("Multinomial Logistic Regression — Diabetes Severity\n")
        f.write(f"test accuracy: {acc:.4f}\n\n")
        f.write("Per-class precision / recall / f1:\n")
        f.write(report + "\n")
        f.write(f"features: {feature_cols}\n")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix — diabetes severity")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # coefficient heatmap (classes x features)
    coef = clf.coef_
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(coef, cmap="coolwarm", aspect="auto",
                   vmin=-np.abs(coef).max(), vmax=np.abs(coef).max())
    ax.set_xticks(range(len(feature_cols))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title("Multinomial LR coefficients per severity class")
    fig.colorbar(im, ax=ax, label="coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coefficient_heatmap.png"), dpi=120)
    plt.close(fig)

    print(f"accuracy: {acc:.4f}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
