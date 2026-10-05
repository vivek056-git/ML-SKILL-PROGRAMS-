"""Titanic Survival: EDA and ML Lifecycle Mapping.
Run: python3 code/eda_lifecycle.py  (from project root)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "csv", "titanic.csv"))
print("shape:", df.shape)

# ---- 1. missing values ----
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print("\nmissing values:\n", miss)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(miss.index, miss.values)
ax.set_title("Missing values per column")
ax.set_ylabel("count")
for i, v in enumerate(miss.values):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "missing_values.png"), dpi=120)
plt.close()

# ---- 2. target distribution ----
fig, ax = plt.subplots(figsize=(6, 4))
df["Survived"].value_counts().plot(kind="bar", ax=ax, color=["#c44e52", "#4c72b0"])
ax.set_title("Survival distribution")
ax.set_xticklabels(["Did not survive (0)", "Survived (1)"], rotation=0)
ax.set_ylabel("passengers")
for i, v in enumerate(df["Survived"].value_counts().values):
    ax.text(i, v + 10, str(v), ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "survival_dist.png"), dpi=120)
plt.close()
print("\nsurvival rate:", round(df["Survived"].mean(), 4))

# ---- 3. survival by sex / class ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x="Sex", y="Survived", data=df, ax=axes[0], palette="pastel")
axes[0].set_title("Survival rate by sex")
sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[1], palette="pastel")
axes[1].set_title("Survival rate by passenger class")
for ax in axes:
    ax.set_ylim(0, 1)
    ax.set_ylabel("survival rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "survival_by_sex_class.png"), dpi=120)
plt.close()
print("\nsurvival by sex:\n", df.groupby("Sex")["Survived"].mean().round(3))
print("\nsurvival by class:\n", df.groupby("Pclass")["Survived"].mean().round(3))

# ---- 4. age & fare distributions ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Age"].dropna(), bins=30, color="#55a868", edgecolor="black")
axes[0].set_title("Age distribution")
axes[0].set_xlabel("age")
axes[1].hist(df["Fare"].dropna(), bins=30, color="#8172b2", edgecolor="black")
axes[1].set_title("Fare distribution")
axes[1].set_xlabel("fare")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "age_fare_dist.png"), dpi=120)
plt.close()

# age by survival
fig, ax = plt.subplots(figsize=(7, 4))
for s, lbl in [(0, "did not survive"), (1, "survived")]:
    ax.hist(df.loc[df["Survived"] == s, "Age"].dropna(), bins=25,
            alpha=0.6, label=lbl)
ax.set_title("Age distribution by survival")
ax.set_xlabel("age")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "age_by_survival.png"), dpi=120)
plt.close()

# ---- 5. fare by class boxplot ----
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(x="Pclass", y="Fare", data=df, ax=ax, palette="pastel")
ax.set_title("Fare by passenger class")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fare_by_class.png"), dpi=120)
plt.close()

# ---- 6. correlation heatmap (numeric) ----
num = df.select_dtypes(include=[np.number])
corr = num.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation heatmap (numeric features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "correlation_heatmap.png"), dpi=120)
plt.close()
print("\ncorrelation with Survived:\n", corr["Survived"].sort_values(ascending=False).round(3))

# ---- 7. embarked ----
print("\nsurvival by embarked:\n", df.groupby("Embarked")["Survived"].mean().round(3))

print("\nEDA done. plots saved to", OUT)
