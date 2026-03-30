# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# For hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("All libraries imported successfully!")

# Load your dataset (update filename to match yours)
df = pd.read_csv('medical_insurance.csv')

# Quick exploration
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nData types:")
print(df.dtypes.value_counts())
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Define columns to drop (leakage + irrelevant)
columns_to_drop = [
    'person_id', 
    'policy_term_years', 
    'policy_changes_last_2yrs',
    'provider_quality', 
    'risk_score', 
    'annual_premium', 
    'monthly_premium',
    'claims_count', 
    'avg_claim_amount', 
    'total_claims_paid'
]

# Remove target from features
target = 'annual_medical_cost'

# Create feature matrix X and target y
X = df.drop(columns=columns_to_drop + [target])
y = df[target]

print(f"Features after removal: {X.shape[1]}")
print(f"\nRemaining features:\n{X.columns.tolist()}")

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols[:10]}... (total: {len(numerical_cols)})")

# Apply Label Encoding for categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nAll categorical features encoded successfully!")

# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from remaining (10% of total = 12.5% of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 of 80% = 10% of total
)

print("="*50)
print("DATA SPLIT RESULTS")
print("="*50)
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTarget mean:")
print(f"Train: ${y_train.mean():,.2f}")
print(f"Val: ${y_val.mean():,.2f}")
print(f"Test: ${y_test.mean():,.2f}")

# Initialize model with reasonable defaults
gb_model = GradientBoostingRegressor(
    n_estimators=100,        # 100 trees
    learning_rate=0.1,       # contribution of each tree
    max_depth=4,             # tree depth
    min_samples_split=5,     # minimum samples to split a node
    min_samples_leaf=2,      # minimum samples in leaf node
    subsample=0.8,           # use 80% of data for each tree
    random_state=42
)

# Train the model
print("Training Gradient Boosting Regressor...")
gb_model.fit(X_train, y_train)
print("Training complete!")

# Predictions
y_train_pred = gb_model.predict(X_train)
y_val_pred = gb_model.predict(X_val)

# Evaluate
def evaluate_model(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{set_name} SET PERFORMANCE:")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    return mae, rmse, r2

print("="*50)
print("INITIAL MODEL PERFORMANCE")
print("="*50)
train_metrics = evaluate_model(y_train, y_train_pred, "TRAIN")
val_metrics = evaluate_model(y_val, y_val_pred, "VALIDATION")

# Check for overfitting
if train_metrics[2] - val_metrics[2] > 0.1:
    print("\n⚠️ Possible overfitting detected (R² difference > 0.1)")
else:
    print("\n✓ Good generalization (no severe overfitting)")

print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
    'min_samples_split': [5, 10],
    'subsample': [0.7, 0.8, 0.9]
}

# Randomized search (faster than grid search)
gb_tuned = GradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV(
    gb_tuned,
    param_distributions=param_grid,
    n_iter=20,  # Try 20 combinations
    cv=3,       # 3-fold cross validation
    scoring='r2',
    random_state=42,
    n_jobs=-1   # Use all CPU cores
)

print("Searching for best parameters (this may take 2-5 minutes)...")
random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation R²: {random_search.best_score_:.4f}")

# Use best model
best_gb = random_search.best_estimator_

# Predict with tuned model
y_val_pred_tuned = best_gb.predict(X_val)

print("="*50)
print("TUNED MODEL - VALIDATION SET")
print("="*50)
val_mae = mean_absolute_error(y_val, y_val_pred_tuned)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))
val_r2 = r2_score(y_val, y_val_pred_tuned)

print(f"MAE: ${val_mae:,.2f}")
print(f"RMSE: ${val_rmse:,.2f}")
print(f"R² Score: {val_r2:.4f}")

# Compare with initial model
print(f"\nImprovement vs initial model:")
print(f"MAE: Improved by ${val_metrics[0] - val_mae:,.2f}")
print(f"R²: Improved by {val_r2 - val_metrics[2]:.4f}")

# Final evaluation on test set
y_test_pred = best_gb.predict(X_test)

print("="*50)
print("FINAL MODEL - TEST SET (UNSEEN DATA)")
print("="*50)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"MAE: ${test_mae:,.2f}")
print(f"RMSE: ${test_rmse:,.2f}")
print(f"R² Score: {test_r2:.4f}")

# Interpret R²
if test_r2 > 0.8:
    print("\n✓ Excellent model (explains >80% of variance)")
elif test_r2 > 0.6:
    print("\n✓ Good model (explains 60-80% of variance)")
elif test_r2 > 0.4:
    print("\n⚠️ Moderate model (explains 40-60% of variance)")
else:
    print("\n⚠️ Poor model (<40% variance explained - consider more feature engineering)")

