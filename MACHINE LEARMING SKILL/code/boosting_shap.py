"""Heart Disease: Gradient Boosting + Permutation Importance.

NOTE: xgboost / lightgbm / shap could not be installed in this environment
(no disk space), so sklearn's GradientBoostingClassifier is used as the
boosting model and permutation importance replaces SHAP. The comparison
below pits it against a RandomForest baseline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "csv", "heart.csv"))

X = df.drop(columns=["target"])
y = df["target"]

X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    pr = m.predict_proba(X_test)[:, 1]
    results[name] = (accuracy_score(y_test, p), roc_auc_score(y_test, pr))
    print(f"{name}: acc={results[name][0]:.4f} auc={results[name][1]:.4f}")

best_name = max(results, key=lambda k: results[k][1])
best = models[best_name]

with open(os.path.join(BASE, "output", "metrics.txt"), "w") as f:
    f.write("model comparison (accuracy, roc-auc):\n")
    for n, (a, u) in results.items():
        f.write(f"{n}: accuracy={a:.4f} roc_auc={u:.4f}\n")
    f.write(f"\nbest (by roc-auc): {best_name}\n")

# permutation importance as interpretability stand-in for SHAP
perm = permutation_importance(best, X_test, y_test, n_repeats=10,
                              random_state=42, n_jobs=-1)
order = np.argsort(perm.importances_mean)
plt.figure(figsize=(8, 6))
plt.barh(np.array(X.columns)[order], perm.importances_mean[order],
         xerr=perm.importances_std[order])
plt.title(f"Permutation Importance — {best_name}\n(SHAP unavailable, see README)")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "output", "permutation_importance.png"), dpi=120)
print("saved metrics.txt, permutation_importance.png")
