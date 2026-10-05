# ML Lifecycle Mapping — Titanic Survival EDA

Each EDA finding mapped to an ML lifecycle stage, with one concrete action for this dataset.

## 1. Problem framing
**Finding:** 38.4% survived — a binary classification problem with moderate class imbalance.
**Action:** Define success as F1/ROC-AUC, not raw accuracy, so the 62% majority class doesn't mask poor minority-class performance.

## 2. Data collection
**Finding:** 891 rows, 12 columns; `Cabin` 77% missing, `Age` 20% missing, `Embarked` 2 missing.
**Action:** Treat `Cabin` as unusable in its raw form (drop it or extract deck letter only); plan imputation for `Age`/`Embarked` instead of row deletion.

## 3. Exploratory data analysis
**Finding:** Females survived at ~74% vs males ~19%; 1st class ~63% vs 3rd class ~24%.
**Action:** Prioritise `Sex` and `Pclass` as top predictive features and one-hot encode them for the model.

## 4. Data preprocessing
**Finding:** `Age`/`Fare` are right-skewed; `Fare` correlates with `Pclass`.
**Action:** Median-impute `Age`/`Fare` (robust to skew), log-transform or scale `Fare`, and standard-scale numerics before regularised models.

## 5. Feature engineering
**Finding:** `SibSp` + `Parch` describe family size; solo travellers appear to die more often.
**Action:** Engineer `family_size = SibSp + Parch + 1` and `is_alone` flag; drop raw `Name`/`Ticket` identifiers.

## 6. Modeling
**Finding:** Strong linear-ish signals (sex, class) with a small dataset (891 rows).
**Action:** Start with logistic regression as a fast, interpretable baseline before trying tree ensembles.

## 7. Evaluation
**Finding:** Strongest numeric correlation with `Survived` is only −0.34 (`Pclass`) — no single numeric silver bullet.
**Action:** Use stratified 80/20 split + cross-validation and report precision/recall/F1/ROC-AUC, not accuracy alone.

## 8. Deployment & monitoring
**Finding:** Model depends on `Age` imputation and `Embarked` encoding learned from 1912 passenger data.
**Action:** Version the fitted preprocessing pipeline with the model and monitor input drift (e.g. missing-rate spikes) before serving predictions.
