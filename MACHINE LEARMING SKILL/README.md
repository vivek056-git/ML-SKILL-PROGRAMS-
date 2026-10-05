# Wine Quality — Decision Tree and Random Forest Classification

Compares tree models on the UCI red wine quality dataset. Target `quality`
is bucketed into 3 classes: bad (≤4), ok (5–6), good (≥7).

## Pipeline
1. Load `csv/winequality.csv` (1599 rows, 11 features)
2. Bucket quality into bad/ok/good
3. 80/20 stratified split
4. DecisionTree (max_depth=6) vs RandomForest (200 trees) — accuracy + per-class F1

## Results
See `output/metrics.txt`.
Plots: `output/feature_importance_compare.png`, `output/confusion_matrices.png`.

## Run
```
python3 code/wine_trees.py
```
