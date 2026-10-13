# Project 8 — Heart Disease: Decision Tree Classification

## Important dataset note
The included 920-row CSV is a reproducible educational practice dataset modeled on the
common Heart Disease feature schema. It is NOT a clinical dataset and must not be used
for diagnosis or treatment.

## Target
`target`: 0 = no disease, 1 = disease.

## Workflow
EDA → missing-value analysis → imputation → OneHotEncoder → Decision Tree →
Accuracy/Precision/Recall/F1/ROC-AUC/Log Loss → confusion matrix → ROC curve →
feature importance → tree visualization → 5-fold stratified cross-validation.

## Main files
- `Heart_Disease_Decision_Tree_Classification.ipynb`
- `code/heart_disease_decision_tree.py`
- `dataset/heart_disease.csv`
- `outputs/`
- `requirements.txt`
- `run_project.bat`
