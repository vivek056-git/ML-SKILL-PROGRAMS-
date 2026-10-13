# Project 4 — California Housing: Linear Regression and Regularisation

## What this project covers

1. Dataset inspection
2. Exploratory Data Analysis
3. Correlation analysis
4. Train/test split
5. Preprocessing pipeline
6. StandardScaler
7. Linear Regression
8. Ridge Regression (L2)
9. Lasso Regression (L1)
10. Elastic Net
11. MAE, MSE, RMSE and R²
12. 5-fold cross-validation
13. Ridge/Lasso alpha tuning
14. Coefficient analysis
15. Actual vs predicted visualisation
16. CSV result exports

## Dataset

The standard California Housing dataset has 20,640 observations and 8 numeric predictive
attributes plus the `MedHouseVal` target. The data was derived from the 1990 U.S. Census.

Official scikit-learn documentation:
https://scikit-learn.org/stable/datasets/real_world.html

The standard dataset source is StatLib:
http://lib.stat.cmu.edu/datasets/houses.zip

A Kaggle California Housing regression competition also exists:
https://www.kaggle.com/competitions/playground-series-s3e1/data

## Important note about the included CSV

The current execution environment did not have the official California Housing file cached and
could not directly download external files. Therefore, the included
`dataset/california_housing.csv` is a **reproducible practice dataset with the official schema
and 20,640 rows**. It is explicitly not presented as the official dataset.

For an assignment that specifically requires the official California Housing data, replace the
included CSV with the official dataset. The notebook's workflow remains the same.

## Main files

- `California_Housing_Linear_Regression_Regularisation.ipynb`
- `code/california_housing_regression.py`
- `dataset/california_housing.csv`
- `outputs/model_comparison.csv`
- `outputs/tuned_model_results.csv`
- `outputs/regularisation_tuning.csv`
- `outputs/model_coefficients.csv`
- `outputs/test_predictions.csv`
- `outputs/descriptive_statistics.csv`
- `outputs/feature_correlation_matrix.csv`
- `outputs/dataset_summary.csv`
- `outputs/figures/`
