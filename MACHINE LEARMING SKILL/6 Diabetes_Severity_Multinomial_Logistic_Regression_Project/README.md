# Project 6 — Diabetes Severity: Multinomial Logistic Regression

The included dataset is a **synthetic educational practice dataset**, not a clinical dataset.

- 2,000 rows
- 11 predictor features
- 3-class target `Severity`
- 0 = Low, 1 = Moderate, 2 = High
- Missing values included for preprocessing practice

## Workflow
EDA → missing-value handling → train/test split → median imputation → StandardScaler →
Multinomial Logistic Regression → multiclass metrics → confusion matrix → One-vs-Rest ROC-AUC →
5-fold Stratified Cross-Validation → coefficient analysis → predictions.

## Files
- `Diabetes_Severity_Multinomial_Logistic_Regression.ipynb`
- `code/diabetes_multinomial_logistic_regression.py`
- `dataset/diabetes_severity.csv`
- `outputs/`
- `requirements.txt`
- `run_project.bat`
