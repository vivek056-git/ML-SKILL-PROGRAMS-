# Project 7 — Auto MPG: Linear Regression, Feature Scaling & Encoding

The included dataset is a 398-row reproducible practice dataset with an Auto MPG-style schema.

Target: `mpg`

Features: cylinders, displacement, horsepower, weight, acceleration, model_year, origin, make.

Missing horsepower values are included for imputation practice.

## Workflow
EDA → missing-value analysis → train/test split → numerical median imputation →
StandardScaler → categorical OneHotEncoder → Linear Regression → MAE/MSE/RMSE/R² →
residual analysis → coefficients → 5-fold cross-validation.

## Files
- Auto_MPG_Linear_Regression_Scaling_Encoding.ipynb
- code/auto_mpg_linear_regression.py
- dataset/auto_mpg.csv
- outputs/
- requirements.txt
- run_project.bat
