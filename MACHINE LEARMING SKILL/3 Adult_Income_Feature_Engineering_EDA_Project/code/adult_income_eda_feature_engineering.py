from pathlib import Path
import pandas as pd, numpy as np
BASE=Path(__file__).resolve().parent.parent
DATA=BASE/"dataset"/"adult_income.csv"; OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
categorical=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"]
for c in categorical: df[c]=df[c].replace("?",np.nan)
df["has_capital_gain"]=(df["capital.gain"]>0).astype(int)
df["has_capital_loss"]=(df["capital.loss"]>0).astype(int)
df["net_capital"]=df["capital.gain"]-df["capital.loss"]
df["age_hours"]=df["age"]*df["hours.per.week"]
df["capital_gain_log1p"]=np.log1p(df["capital.gain"])
df["capital_loss_log1p"]=np.log1p(df["capital.loss"])
df["age_group"]=pd.cut(df["age"],[17,25,35,45,55,65,100],labels=["18-25","26-35","36-45","46-55","56-65","66+"])
pd.DataFrame({"column":df.columns,"missing_count":df.isna().sum().values,"missing_percentage":(df.isna().mean()*100).round(2).values}).to_csv(OUT/"missing_values_after_conversion.csv",index=False)
df.to_csv(OUT/"adult_income_feature_engineered.csv",index=False)
df["income"].value_counts().rename_axis("income").reset_index(name="count").to_csv(OUT/"income_distribution.csv",index=False)
pd.crosstab(df["education"],df["income"],normalize="index").reset_index().to_csv(OUT/"education_income_rates.csv",index=False)
pd.crosstab(df["occupation"],df["income"],normalize="index").to_csv(OUT/"occupation_income_rates.csv")
df[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week","has_capital_gain","has_capital_loss","net_capital","age_hours","capital_gain_log1p","capital_loss_log1p"]].describe().T.to_csv(OUT/"numeric_summary.csv")
print("Completed Adult Income EDA and feature engineering. Outputs:",OUT)
