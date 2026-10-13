# Project 5 — Titanic Survival: Full Logistic Regression Pipeline

## Dataset
This project uses the standard Kaggle Titanic training dataset:
- 891 rows
- 12 original columns
- Target: `Survived`

Kaggle source:
https://www.kaggle.com/c/titanic/data

The dataset in `dataset/titanic_train.csv` is the Titanic training file already available
from the earlier Titanic project in this workspace.

## Pipeline

1. Data loading
2. Dataset information
3. Missing-value analysis
4. EDA
5. Feature engineering
6. Stratified train/test split
7. Numerical median imputation
8. Numerical standardisation
9. Categorical most-frequent imputation
10. One-hot encoding
11. Logistic Regression
12. Accuracy
13. Precision
14. Recall
15. F1-score
16. Log loss
17. ROC-AUC
18. Confusion matrix
19. 5-fold stratified cross-validation
20. Probability-threshold analysis
21. Coefficient and odds-ratio analysis
22. Test-set predictions

## Engineered Features

- `Title`
- `FamilySize`
- `IsAlone`
- `FarePerPerson`
- `HasCabin`

## Main Files

- `Titanic_Survival_Full_Logistic_Regression_Pipeline.ipynb`
- `code/titanic_logistic_regression.py`
- `dataset/titanic_train.csv`
- `outputs/logistic_regression_metrics.csv`
- `outputs/test_predictions.csv`
- `outputs/confusion_matrix.csv`
- `outputs/classification_report.csv`
- `outputs/cross_validation_results.csv`
- `outputs/threshold_analysis.csv`
- `outputs/logistic_coefficients_odds_ratios.csv`
- `outputs/missing_values_report.csv`
- `outputs/feature_engineered_dataset.csv`
- `outputs/figures/`

## Install

pip install -r requirements.txt

## Run

Open the notebook in Jupyter, or run:

run_project.bat
