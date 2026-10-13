from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "titanic_train.csv"
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Feature engineering
work = df.copy()
work["FamilySize"] = work["SibSp"] + work["Parch"] + 1
work["IsAlone"] = (work["FamilySize"] == 1).astype(int)
work["FarePerPerson"] = work["Fare"] / work["FamilySize"]
work["Title"] = work["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
common_titles = ["Mr", "Miss", "Mrs", "Master"]
work["Title"] = work["Title"].where(work["Title"].isin(common_titles), "Rare")

# Readable cleaned dataset
clean = work.copy()
clean["Age"] = clean["Age"].fillna(clean["Age"].median())
clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])
clean["Fare"] = clean["Fare"].fillna(clean["Fare"].median())
clean["FarePerPerson"] = clean["FarePerPerson"].replace([np.inf, -np.inf], np.nan)
clean["FarePerPerson"] = clean["FarePerPerson"].fillna(clean["FarePerPerson"].median())
clean.to_csv(OUT / "titanic_cleaned_readable.csv", index=False)

# Modeling data
features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
            "FamilySize","IsAlone","FarePerPerson","Title"]
X = work[features]
y = df["Survived"]

numeric = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","FarePerPerson"]
categorical = ["Sex","Embarked","Title"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

# Split BEFORE fitting preprocessing: prevents test-set leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
names = preprocessor.get_feature_names_out()

train = pd.DataFrame(X_train_p, columns=names, index=X_train.index)
train["Survived"] = y_train
test = pd.DataFrame(X_test_p, columns=names, index=X_test.index)
test["Survived"] = y_test

train.to_csv(OUT / "train_preprocessed_with_target.csv", index=False)
test.to_csv(OUT / "test_preprocessed_with_target.csv", index=False)
train.drop(columns="Survived").to_csv(OUT / "X_train_preprocessed.csv", index=False)
test.drop(columns="Survived").to_csv(OUT / "X_test_preprocessed.csv", index=False)

pd.DataFrame({"processed_feature": names}).to_csv(
    OUT / "processed_feature_map.csv", index=False
)

print("Raw shape:", df.shape)
print("Raw missing cells:", int(df.isna().sum().sum()))
print("Cleaned missing cells:", int(clean.isna().sum().sum()))
print("Train shape:", X_train_p.shape)
print("Test shape:", X_test_p.shape)
print("Processed features:", len(names))
print("Done.")
