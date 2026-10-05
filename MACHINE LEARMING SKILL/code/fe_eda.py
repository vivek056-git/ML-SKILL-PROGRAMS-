"""Adult Income — Feature Engineering and EDA.

Steps:
  1. Load raw UCI Adult data (no header), strip whitespace, '?' -> NaN.
  2. Save cleaned raw to csv/adult.csv.
  3. EDA: income distribution, income by education / sex / hours-per-week.
  4. Feature engineering: capital_net, age bins, hours bins, education groups.
  5. Save engineered sample, plots, and a text report.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "csv")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# ---------- 1. load & clean ----------
df = pd.read_csv(os.path.join(CSV, "adult.data"), header=None, names=COLS)
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()
df.replace("?", np.nan, inplace=True)
df = df.dropna().reset_index(drop=True)          # drop rows with missing
df.to_csv(os.path.join(CSV, "adult.csv"), index=False)
print(f"raw rows: 32561 (file), clean rows: {len(df)}")

df["high_income"] = (df["income"] == ">50K").astype(int)

# ---------- 2. EDA ----------
report = []
report.append(f"rows after cleaning: {len(df)}")
report.append(f"income distribution:\n{df['income'].value_counts(normalize=True).round(3)}")

plt.figure(figsize=(7, 4))
df["income"].value_counts().plot(kind="bar", color=["#e15759", "#4e79a7"])
plt.title("Income distribution")
plt.ylabel("count"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "income_distribution.png"), dpi=120); plt.close()

# income rate by education
edu = df.groupby("education")["high_income"].mean().sort_values()
report.append(f"\nhigh-income rate by education:\n{edu.round(3)}")
plt.figure(figsize=(9, 4))
edu.plot(kind="bar", color="#59a14f")
plt.title("P(income >50K) by education"); plt.ylabel("rate"); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "income_by_education.png"), dpi=120); plt.close()

# income rate by sex
sex = df.groupby("sex")["high_income"].mean()
report.append(f"\nhigh-income rate by sex:\n{sex.round(3)}")
plt.figure(figsize=(6, 4))
sex.plot(kind="bar", color=["#f28e2b", "#4e79a7"])
plt.title("P(income >50K) by sex"); plt.ylabel("rate"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "income_by_sex.png"), dpi=120); plt.close()

# hours-per-week vs income
plt.figure(figsize=(7, 4))
for lab, g in df.groupby("income"):
    plt.hist(g["hours-per-week"], bins=30, alpha=0.6, label=lab)
plt.title("Hours-per-week by income"); plt.xlabel("hours-per-week"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "hours_by_income.png"), dpi=120); plt.close()
report.append(f"\nmean hours-per-week: <=50K={df[df.income=='<=50K']['hours-per-week'].mean():.1f}, "
              f">50K={df[df.income=='>50K']['hours-per-week'].mean():.1f}")

# ---------- 3. feature engineering ----------
df["capital_net"] = df["capital-gain"] - df["capital-loss"]
df["age_bin"] = pd.cut(df["age"], bins=[0, 25, 40, 60, 100],
                       labels=["young", "adult", "middle", "senior"])
df["hours_bin"] = pd.cut(df["hours-per-week"], bins=[0, 30, 45, 100],
                         labels=["part-time", "full-time", "overtime"])
edu_map = {"Preschool": "dropout", "1st-4th": "dropout", "5th-6th": "dropout",
           "7th-8th": "dropout", "9th": "dropout", "10th": "dropout",
           "11th": "dropout", "12th": "dropout", "HS-grad": "high-school",
           "Some-college": "college", "Assoc-voc": "college", "Assoc-acdm": "college",
           "Bachelors": "bachelors", "Masters": "postgrad",
           "Prof-school": "postgrad", "Doctorate": "postgrad"}
df["education_group"] = df["education"].map(edu_map)

eng_rate = df.groupby("education_group")["high_income"].mean().sort_values()
report.append(f"\nhigh-income rate by engineered education_group:\n{eng_rate.round(3)}")
report.append(f"\nhigh-income rate by age_bin:\n{df.groupby('age_bin', observed=True)['high_income'].mean().round(3)}")
report.append(f"\nhigh-income rate by hours_bin:\n{df.groupby('hours_bin', observed=True)['high_income'].mean().round(3)}")
report.append(f"\ncapital_net: mean={df['capital_net'].mean():.1f}, "
              f"median={df['capital_net'].median():.1f}, "
              f"corr with high_income={df['capital_net'].corr(df['high_income']):.3f}")

df.to_csv(os.path.join(OUT, "adult_engineered.csv"), index=False)

with open(os.path.join(OUT, "eda_report.txt"), "w") as f:
    f.write("ADULT INCOME — EDA REPORT\n=======================\n\n")
    f.write("\n".join(report))
    f.write("\n\nKEY FINDINGS\n"
            "- Income is imbalanced (~76% <=50K).\n"
            "- Higher education strongly raises the >50K rate (Doctorate/Prof-school highest).\n"
            "- Males have a much higher >50K rate than females in this 1994 census sample.\n"
            "- >50K earners work more hours on average; overtime bin has the highest rate.\n"
            "- capital_net (gain-loss) correlates positively with high income.\n"
            "- Engineered features: capital_net, age_bin, hours_bin, education_group.\n")

print("done ->", OUT)
