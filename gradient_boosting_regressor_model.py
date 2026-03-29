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