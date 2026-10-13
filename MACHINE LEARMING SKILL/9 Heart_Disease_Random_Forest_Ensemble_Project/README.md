# Project 9 — Heart Disease: Random Forest Ensemble

## Dataset
The included 920-row CSV is a reproducible educational practice dataset modeled on the
common Heart Disease feature schema. It is NOT a clinical dataset and must not be used
for diagnosis or treatment.

Target: `target` — 0 = no disease, 1 = disease.

## Model
RandomForestClassifier:
- n_estimators=300
- max_depth=8
- min_samples_leaf=3
- class_weight=balanced
- random_state=42

## Workflow
EDA → missing-value analysis → imputation → OneHotEncoder → Random Forest →
Accuracy/Precision/Recall/F1/ROC-AUC/Log Loss → confusion matrix → ROC curve →
feature importance → 5-fold stratified cross-validation.

## Files
- `Heart_Disease_Random_Forest_Ensemble.ipynb`
- `code/heart_disease_random_forest.py`
- `dataset/heart_disease.csv`
- `outputs/`
- `requirements.txt`
- `run_project.bat`
