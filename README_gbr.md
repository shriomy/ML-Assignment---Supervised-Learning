
# Gradient Boosting Regressor👾

## Overview
Gradient Boosting is an ensemble machine learning technique that builds multiple decision trees sequentially, where each new tree tries to correct the errors made by the previous trees.

## Why Gradient Boosting for Medical Costs?
♦️ Handles non-linear relationships (e.g., age vs cost spikes at certain ages)

♠️ Works with mixed data types (numerical + categorical)

♥️ Robust to outliers (medical cost outliers common)

♣️ Provides feature importance (tells you what drives costs)


## Methodology
1. Imported all necessary libraries
2. Scikit-learn is used for hyperparameter tuning
3. Loaded Dataset
4. Feature selection and defined columns to consider
5. Created virtual environment -> python -m venv venv
6. Installed libraries -> pip install pandas numpy scikit-learn matplotlib seaborn joblib
7. Configured feature selection, preprocessing and data splitting
8. Training model configuration with initial hyperparamters
9. Trained the model
10. Testing with validation data
11. Hyperparameter tuning
12. Repeated till a good model arrives
13. Saved the model
14. Saved model tested on testing set
15. Added test results and analyss

## Model Training steps
🕐 Run .ipnyb file from the kernel on VS code or google colabs after uploading or,
🕐 Run python gradient_boosting_regressor_model.py to run the python code on VS code

## 🃏Hyperparameters used:
- n_estimators:
- learning_rate:
- max_depth:
- min_samples_split:
- subsample:

## 🏁Test Results and analysis: 

## 🎲Try yourself: Run python gradient_boosting_regressor_model.py or Run .ipnyb file from kernel (VS code) or on google colabs


