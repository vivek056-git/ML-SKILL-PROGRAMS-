"""
Titanic Survival — EDA and ML Lifecycle Mapping Project

Run from the project root:
    python code/titanic_eda_lifecycle.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "titanic_train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("TITANIC SURVIVAL — EDA AND ML LIFECYCLE MAPPING")
print("=" * 70)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all").T)

# Missing values
missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percentage": (df.isna().mean().values * 100).round(2),
    "data_type": df.dtypes.astype(str).values
})
print("\nMISSING VALUES")
print(missing_summary)
missing_summary.to_csv(OUTPUT_DIR / "missing_summary.csv", index=False)

# Duplicates
duplicate_summary = pd.DataFrame({
    "metric": ["total_rows", "duplicate_rows", "unique_rows"],
    "value": [len(df), int(df.duplicated().sum()), int(df.drop_duplicates().shape[0])]
})
print("\nDUPLICATES")
print(duplicate_summary)
duplicate_summary.to_csv(OUTPUT_DIR / "duplicate_summary.csv", index=False)

# Target
target_distribution = (
    df["Survived"].value_counts().sort_index()
      .rename_axis("Survived").reset_index(name="count")
)
target_distribution["percentage"] = (
    target_distribution["count"] / len(df) * 100
).round(2)
print("\nTARGET DISTRIBUTION")
print(target_distribution)
target_distribution.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)

# Survival by sex
survival_by_sex = (
    df.groupby("Sex", dropna=False)["Survived"]
      .agg(passengers="count", survivors="sum", survival_rate="mean")
      .reset_index()
)
survival_by_sex["survival_rate_percent"] = (
    survival_by_sex["survival_rate"] * 100
).round(2)
print("\nSURVIVAL BY SEX")
print(survival_by_sex)
survival_by_sex.to_csv(OUTPUT_DIR / "survival_by_sex.csv", index=False)

# Survival by class
survival_by_pclass = (
    df.groupby("Pclass", dropna=False)["Survived"]
      .agg(passengers="count", survivors="sum", survival_rate="mean")
      .reset_index()
)
survival_by_pclass["survival_rate_percent"] = (
    survival_by_pclass["survival_rate"] * 100
).round(2)
print("\nSURVIVAL BY PASSENGER CLASS")
print(survival_by_pclass)
survival_by_pclass.to_csv(OUTPUT_DIR / "survival_by_pclass.csv", index=False)

# Survival by age group
age_bins = pd.cut(
    df["Age"], bins=[0, 12, 18, 30, 50, 80],
    labels=["Child (0-12)", "Teen (13-18)", "Young Adult (19-30)",
            "Adult (31-50)", "Older Adult (51-80)"],
    include_lowest=True
)
survival_by_age_group = (
    pd.DataFrame({"Age_Group": age_bins, "Survived": df["Survived"]})
      .dropna()
      .groupby("Age_Group", observed=False)["Survived"]
      .agg(passengers="count", survivors="sum", survival_rate="mean")
      .reset_index()
)
survival_by_age_group["survival_rate_percent"] = (
    survival_by_age_group["survival_rate"] * 100
).round(2)
print("\nSURVIVAL BY AGE GROUP")
print(survival_by_age_group)
survival_by_age_group.to_csv(OUTPUT_DIR / "survival_by_age_group.csv", index=False)

# Correlation
numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr().round(4)
print("\nCORRELATION MATRIX")
print(corr)
corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

# Summary
dataset_summary = pd.DataFrame({
    "metric": ["rows", "columns", "survivors", "non_survivors",
               "survival_rate_percent", "missing_cells", "duplicate_rows"],
    "value": [len(df), df.shape[1], int(df["Survived"].sum()),
              int((df["Survived"] == 0).sum()),
              round(df["Survived"].mean() * 100, 2),
              int(df.isna().sum().sum()), int(df.duplicated().sum())]
})
dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)

# Figures
plt.figure(figsize=(8, 5))
plt.imshow(df.isna().astype(int).to_numpy(), aspect="auto", interpolation="nearest")
plt.title("Titanic Missing-Value Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.colorbar(label="Missing (1 = Yes)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_missingness_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(survival_by_sex["Sex"], survival_by_sex["survival_rate_percent"])
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_survival_by_sex.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(survival_by_pclass["Pclass"].astype(str), survival_by_pclass["survival_rate_percent"])
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_survival_by_pclass.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(survival_by_age_group["Age_Group"].astype(str), survival_by_age_group["survival_rate_percent"])
plt.title("Survival Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_survival_by_age_group.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(corr.to_numpy(), aspect="auto", interpolation="nearest")
plt.title("Numeric Feature Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar(label="Correlation")
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(target_distribution["Survived"].astype(str), target_distribution["count"])
plt.title("Target Distribution: Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_target_distribution.png", dpi=160, bbox_inches="tight")
plt.close()

print("\nPROJECT COMPLETE")
print("Check the outputs/ and outputs/figures/ folders.")
