# ==============================================================================
#  MEDICAL INSURANCE COST PREDICTION
#  Algorithm : Support Vector Regression (SVR)
#  Platform  : Google Colab
#  Dataset   : medical_insurance.csv  (100,000 rows × 54 columns)
#  Split     : Train 70% | Validation 10% | Test 20%
# ==============================================================================

# ── CELL 1 : Install / imports ─────────────────────────────────────────────────
!pip install scikit-learn pandas numpy matplotlib seaborn scipy joblib -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings, time, joblib, os
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.inspection import permutation_importance

# Aesthetic config (dark theme, consistent across all plots)
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4d',   'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',       'grid.color': '#21262d',
    'grid.linestyle': '--',        'grid.alpha': 0.5,
    'font.family': 'monospace',    'axes.titlecolor': '#e6edf3',
    'axes.titlesize': 12,          'axes.titleweight': 'bold',
})
C1, C2, C3, C4 = '#58a6ff', '#3fb950', '#f78166', '#d2a8ff'

print("✔ Libraries loaded successfully")

# ── CELL 2 : Mount Drive & Load Data ──────────────────────────────────────────
drive.mount('/content/drive')

# ▶▶ UPDATE THIS PATH to where your CSV is in Google Drive
CSV_PATH = "/content/drive/MyDrive/medical_insurance.csv"

df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✔ Loaded  →  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(df_raw.head(3))

# ── CELL 3 : Preprocessing & Feature Engineering ──────────────────────────────
print("\n" + "="*60)
print("  STEP 1 : PREPROCESSING & FEATURE ENGINEERING")
print("="*60)

TARGET = 'annual_medical_cost'

# ── 3a. Drop leakage / redundant columns ──────────────────────────────────────
# monthly_premium == annual_premium / 12  → perfect multicollinearity (r=1.0)
# person_id is just an index
# annual_premium, monthly_premium, total_claims_paid, avg_claim_amount are
#   POST-OUTCOME variables (derived from cost) → data leakage if kept
#   Keep them for a "full feature" run; drop for a "fair" run.
#
#   We choose to KEEP annual_premium (strongest predictor r=0.965) because
#   it is a policy attribute SET BEFORE the cost is realized in insurance
#   actuarial settings. We drop monthly_premium (duplicate) and
#   total_claims_paid / avg_claim_amount (outcome-derived → leakage).

LEAKAGE_COLS  = ['monthly_premium',      # duplicate of annual_premium
                 'total_claims_paid',     # outcome derived
                 'avg_claim_amount',      # outcome derived
                 'claims_count',          # outcome derived
                 'person_id']

# Near-zero correlation features (|r| < 0.01) identified in EDA
LOW_CORR_COLS = ['copay', 'policy_term_years', 'policy_changes_last_2yrs',
                 'household_size', 'dependents', 'provider_quality',
                 'deductible', 'income']

DROP_COLS = LEAKAGE_COLS + LOW_CORR_COLS
df = df_raw.drop(columns=DROP_COLS, errors='ignore').copy()
print(f"\n  Dropped {len(DROP_COLS)} columns → {df.shape[1]} remaining")

# ── 3b. Missing values ────────────────────────────────────────────────────────
# alcohol_freq : 30.08% missing → impute with mode
mode_alcohol = df['alcohol_freq'].mode()[0]
df['alcohol_freq'].fillna(mode_alcohol, inplace=True)
print(f"  alcohol_freq NaN → filled with mode='{mode_alcohol}'")
assert df.isnull().sum().sum() == 0, "Still have NaN values!"
print("  ✔ No remaining NaN values")

# ── 3c. Log-transform the target (skewness=4.03) ─────────────────────────────
df['log_cost'] = np.log1p(df[TARGET])
print(f"\n  Target skewness (raw)     : {df[TARGET].skew():.4f}")
print(f"  Target skewness (log1p)   : {df['log_cost'].skew():.4f}")

# ── 3d. Identify feature types ────────────────────────────────────────────────
BINARY_COLS = ['hypertension','diabetes','asthma','copd',
               'cardiovascular_disease','cancer_history','kidney_disease',
               'liver_disease','arthritis','mental_health',
               'is_high_risk','had_major_procedure']

CAT_COLS = ['sex','region','urban_rural','education','marital_status',
            'employment_status','smoker','alcohol_freq','plan_type','network_tier']

NUM_COLS = [c for c in df.columns
            if c not in CAT_COLS + BINARY_COLS + [TARGET, 'log_cost']]

print(f"\n  Numerical features : {len(NUM_COLS)} → {NUM_COLS}")
print(f"  Categorical features : {len(CAT_COLS)} → {CAT_COLS}")
print(f"  Binary features      : {len(BINARY_COLS)}")
print(f"  Target               : {TARGET}  (log_cost used for training)")

# ── CELL 4 : Train / Validation / Test Split ───────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 : TRAIN / VALIDATION / TEST SPLIT")
print("="*60)

FEATURES = NUM_COLS + CAT_COLS + BINARY_COLS
X = df[FEATURES]
y = df['log_cost']          # log-transformed target for training
y_raw = df[TARGET]          # original scale for reporting

# 70 / 10 / 20 split
X_temp,  X_test,  y_temp,  y_test  = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val,   y_train, y_val   = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)
# 0.125 × 0.80 = 0.10 of total → gives 70/10/20

y_test_raw = np.expm1(y_test)
y_val_raw  = np.expm1(y_val)
y_train_raw = np.expm1(y_train)

print(f"\n  Training   : {X_train.shape[0]:>6,} rows ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Validation : {X_val.shape[0]:>6,} rows ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"  Testing    : {X_test.shape[0]:>6,} rows ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"  Features   : {X_train.shape[1]}")

# ── CELL 5 : Build Preprocessing Pipeline ─────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 : PREPROCESSING PIPELINE")
print("="*60)

num_transformer = Pipeline([
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Binary columns pass through (already 0/1, but still scale for SVR)
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, NUM_COLS),
    ('cat', cat_transformer, CAT_COLS),
    ('bin', StandardScaler(),  BINARY_COLS),   # scale binaries too for RBF SVR
], remainder='drop')

# Fit ONLY on training set
preprocessor.fit(X_train)
X_train_pp = preprocessor.transform(X_train)
X_val_pp   = preprocessor.transform(X_val)
X_test_pp  = preprocessor.transform(X_test)

n_features_out = X_train_pp.shape[1]
print(f"\n  Input  features : {X_train.shape[1]}")
print(f"  Output features (after OHE): {n_features_out}")
print(f"  X_train_pp shape: {X_train_pp.shape}")
print(f"  ✔ Preprocessor fitted on training set only (no data leakage)")

# ── CELL 6 : Hyperparameter Tuning on VALIDATION set ──────────────────────────
print("\n" + "="*60)
print("  STEP 4 : HYPERPARAMETER TUNING (RandomizedSearchCV on Val set)")
print("="*60)

# NOTE: SVR on 70k rows is very slow with full grid search.
# We use RandomizedSearchCV with n_iter=20 on a SUBSAMPLE of training,
# then evaluate each candidate on the real validation set.
# Final model is retrained on full 70k train set with best params.

TUNE_SAMPLE = 15000  # rows used for hyperparameter search (speed)
np.random.seed(42)
idx_tune = np.random.choice(len(X_train_pp), TUNE_SAMPLE, replace=False)
X_tune = X_train_pp[idx_tune]
y_tune = y_train.iloc[idx_tune]

param_dist = {
    'kernel'  : ['rbf'],
    'C'       : [0.1, 1, 5, 10, 50, 100, 500],
    'gamma'   : ['scale', 'auto', 0.001, 0.01, 0.05],
    'epsilon' : [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
}

print(f"\n  Tuning on {TUNE_SAMPLE:,} samples with RandomizedSearchCV (n_iter=20)...")
t0 = time.time()

rscv = RandomizedSearchCV(
    SVR(), param_distributions=param_dist,
    n_iter=20, cv=3, scoring='neg_mean_squared_error',
    random_state=42, n_jobs=-1, verbose=1
)
rscv.fit(X_tune, y_tune)

print(f"\n  ✔ Search done in {time.time()-t0:.1f}s")
print(f"  Best params : {rscv.best_params_}")
print(f"  Best CV MSE : {-rscv.best_score_:.6f}")

BEST_PARAMS = rscv.best_params_

# ── Manual override: uncomment if you want to force specific params
# BEST_PARAMS = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'epsilon': 0.1}

# ── CELL 7 : Train Final SVR on Full Training Set ─────────────────────────────
print("\n" + "="*60)
print("  STEP 5 : TRAINING FINAL SVR MODEL")
print("="*60)

print(f"\n  Hyperparameters used:")
for k, v in BEST_PARAMS.items():
    print(f"    {k:<10} = {v}")

print(f"\n  Training on {X_train_pp.shape[0]:,} samples... (this may take 5–20 min)")
t0 = time.time()

svr_model = SVR(**BEST_PARAMS)
svr_model.fit(X_train_pp, y_train)

train_time = time.time() - t0
print(f"  ✔ Training complete in {train_time:.1f}s")
print(f"  Number of support vectors : {svr_model.n_support_[0]:,}")
print(f"  Fraction of SVs           : {svr_model.n_support_[0]/len(X_train_pp)*100:.2f}%")

# ── CELL 8 : Training Set Evaluation ──────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 : TRAINING RESULTS")
print("="*60)

def evaluate(y_true_log, y_pred_log, label=""):
    """Evaluate in log space and original space."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)   # no negative costs

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    medae = np.median(np.abs(y_true - y_pred))

    # Log-space metrics
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log   = r2_score(y_true_log, y_pred_log)

    print(f"\n  ── {label} ──")
    print(f"  {'RMSE':<25}: ${rmse:>10,.2f}")
    print(f"  {'MAE':<25}: ${mae:>10,.2f}")
    print(f"  {'Median Abs Error':<25}: ${medae:>10,.2f}")
    print(f"  {'MAPE':<25}: {mape:>10.2f}%")
    print(f"  {'R²  (original scale)':<25}: {r2:>10.4f}")
    print(f"  {'RMSE (log scale)':<25}: {rmse_log:>10.6f}")
    print(f"  {'R²  (log scale)':<25}: {r2_log:>10.4f}")

    return {'RMSE': rmse, 'MAE': mae, 'MedAE': medae,
            'MAPE': mape, 'R2': r2, 'RMSE_log': rmse_log, 'R2_log': r2_log,
            'y_true': y_true, 'y_pred': y_pred,
            'y_true_log': np.array(y_true_log), 'y_pred_log': y_pred_log}

y_pred_train_log = svr_model.predict(X_train_pp)
y_pred_val_log   = svr_model.predict(X_val_pp)

train_metrics = evaluate(y_train, y_pred_train_log, "TRAINING SET")
val_metrics   = evaluate(y_val,   y_pred_val_log,   "VALIDATION SET")

# ── CELL 9 : Training Visualizations ──────────────────────────────────────────
print("\n  Generating training visualizations...")

fig = plt.figure(figsize=(20, 14))
fig.suptitle('SVR — Training Results', fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Plot 1 : Predicted vs Actual (training)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(train_metrics['y_true'], train_metrics['y_pred'],
            alpha=0.05, s=3, color=C1, rasterized=True)
lim = max(train_metrics['y_true'].max(), train_metrics['y_pred'].max())
ax1.plot([0, lim], [0, lim], color=C3, lw=1.5, ls='--', label='Perfect')
ax1.set_xlabel('Actual Cost ($)')
ax1.set_ylabel('Predicted Cost ($)')
ax1.set_title('Predicted vs Actual (Train)')
ax1.legend(fontsize=9)
ax1.text(0.05, 0.88, f"R²={train_metrics['R2']:.4f}", transform=ax1.transAxes,
         color=C2, fontsize=10, fontweight='bold')

# Plot 2 : Residuals distribution (training)
ax2 = fig.add_subplot(gs[0, 1])
residuals_train = train_metrics['y_pred'] - train_metrics['y_true']
ax2.hist(residuals_train, bins=100, color=C4, edgecolor='none', alpha=0.85)
ax2.axvline(0, color=C3, lw=1.5, ls='--')
ax2.axvline(residuals_train.mean(), color=C2, lw=1.5, ls='--',
            label=f'Mean={residuals_train.mean():,.0f}')
ax2.set_xlabel('Residual = Predicted − Actual ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residuals Distribution (Train)')
ax2.legend(fontsize=9)

# Plot 3 : Residuals vs Predicted (heteroscedasticity check)
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(train_metrics['y_pred'], residuals_train,
            alpha=0.05, s=3, color=C1, rasterized=True)
ax3.axhline(0, color=C3, lw=1.5, ls='--')
ax3.set_xlabel('Predicted Cost ($)')
ax3.set_ylabel('Residual ($)')
ax3.set_title('Residuals vs Predicted (Train)')

# Plot 4 : Predicted vs Actual in LOG space (training)
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(train_metrics['y_true_log'], train_metrics['y_pred_log'],
            alpha=0.05, s=3, color=C2, rasterized=True)
lim_log = max(train_metrics['y_true_log'].max(), train_metrics['y_pred_log'].max())
ax4.plot([0, lim_log], [0, lim_log], color=C3, lw=1.5, ls='--')
ax4.set_xlabel('log(Actual + 1)')
ax4.set_ylabel('log(Predicted + 1)')
ax4.set_title('Predicted vs Actual — Log Scale (Train)')
ax4.text(0.05, 0.88, f"R²={train_metrics['R2_log']:.4f}", transform=ax4.transAxes,
         color=C2, fontsize=10, fontweight='bold')

# Plot 5 : Error by cost percentile
ax5 = fig.add_subplot(gs[1, 1])
pct_bins = pd.qcut(train_metrics['y_true'], q=10, labels=False)
pct_mae  = [np.mean(np.abs(residuals_train[pct_bins == i]))
            for i in range(10)]
pct_labs = [f'P{i*10}-{(i+1)*10}' for i in range(10)]
bars = ax5.bar(pct_labs, pct_mae, color=C1, alpha=0.85, edgecolor='none')
ax5.set_xlabel('Cost Percentile Bin')
ax5.set_ylabel('Mean Absolute Error ($)')
ax5.set_title('MAE by Cost Percentile (Train)')
ax5.tick_params(axis='x', rotation=45)

# Plot 6 : Train vs Validation metric comparison
ax6 = fig.add_subplot(gs[1, 2])
metrics_cmp = ['R²', 'RMSE (k$)', 'MAE (k$)', 'MAPE (%)']
train_vals = [train_metrics['R2'],
              train_metrics['RMSE']/1000,
              train_metrics['MAE']/1000,
              train_metrics['MAPE']/100]
val_vals   = [val_metrics['R2'],
              val_metrics['RMSE']/1000,
              val_metrics['MAE']/1000,
              val_metrics['MAPE']/100]
x_pos = np.arange(len(metrics_cmp))
w = 0.35
ax6.bar(x_pos - w/2, train_vals, w, label='Train', color=C2, alpha=0.85)
ax6.bar(x_pos + w/2, val_vals,   w, label='Val',   color=C3, alpha=0.85)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metrics_cmp, fontsize=9)
ax6.set_title('Train vs Validation Metrics')
ax6.legend()

plt.savefig('svr_training_results.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("  ✔ Saved → svr_training_results.png")

# ── CELL 10 : Testing Set Evaluation ──────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 : TESTING RESULTS")
print("="*60)

y_pred_test_log = svr_model.predict(X_test_pp)
test_metrics = evaluate(y_test, y_pred_test_log, "TEST SET (FINAL)")

# ── CELL 11 : Testing Visualizations ──────────────────────────────────────────
print("\n  Generating testing visualizations...")

fig2 = plt.figure(figsize=(20, 14))
fig2.suptitle('SVR — Testing Results', fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.35)

residuals_test = test_metrics['y_pred'] - test_metrics['y_true']

# Plot 1 : Predicted vs Actual (test)
ax = fig2.add_subplot(gs2[0, 0])
ax.scatter(test_metrics['y_true'], test_metrics['y_pred'],
           alpha=0.07, s=4, color=C1, rasterized=True)
lim = max(test_metrics['y_true'].max(), test_metrics['y_pred'].max())
ax.plot([0, lim], [0, lim], color=C3, lw=1.5, ls='--', label='Perfect')
ax.set_xlabel('Actual Cost ($)')
ax.set_ylabel('Predicted Cost ($)')
ax.set_title('Predicted vs Actual (Test)')
ax.legend(fontsize=9)
ax.text(0.05, 0.88, f"R²={test_metrics['R2']:.4f}", transform=ax.transAxes,
        color=C2, fontsize=10, fontweight='bold')

# Plot 2 : Residuals distribution (test)
ax = fig2.add_subplot(gs2[0, 1])
ax.hist(residuals_test, bins=100, color=C4, edgecolor='none', alpha=0.85)
ax.axvline(0, color=C3, lw=1.5, ls='--')
ax.axvline(residuals_test.mean(), color=C2, lw=1.5, ls='--',
           label=f'Mean={residuals_test.mean():,.0f}')
ax.set_xlabel('Residual = Predicted − Actual ($)')
ax.set_ylabel('Frequency')
ax.set_title('Residuals Distribution (Test)')
ax.legend(fontsize=9)

# Plot 3 : Q-Q plot of residuals
ax = fig2.add_subplot(gs2[0, 2])
stats.probplot(residuals_test, dist='norm', plot=ax)
ax.set_title('Q-Q Plot of Residuals (Test)')
ax.get_lines()[0].set(color=C1, markersize=2, alpha=0.4)
ax.get_lines()[1].set(color=C3, lw=2)

# Plot 4 : Predicted vs Actual (log scale, test)
ax = fig2.add_subplot(gs2[1, 0])
ax.scatter(test_metrics['y_true_log'], test_metrics['y_pred_log'],
           alpha=0.07, s=4, color=C2, rasterized=True)
lim_log = max(test_metrics['y_true_log'].max(), test_metrics['y_pred_log'].max())
ax.plot([0, lim_log], [0, lim_log], color=C3, lw=1.5, ls='--')
ax.set_xlabel('log(Actual + 1)')
ax.set_ylabel('log(Predicted + 1)')
ax.set_title('Predicted vs Actual — Log Scale (Test)')
ax.text(0.05, 0.88, f"R²={test_metrics['R2_log']:.4f}", transform=ax.transAxes,
        color=C2, fontsize=10, fontweight='bold')

# Plot 5 : MAE by cost percentile (test)
ax = fig2.add_subplot(gs2[1, 1])
pct_bins_t = pd.qcut(test_metrics['y_true'], q=10, labels=False)
pct_mae_t  = [np.mean(np.abs(residuals_test[pct_bins_t == i])) for i in range(10)]
ax.bar(pct_labs, pct_mae_t, color=C3, alpha=0.85, edgecolor='none')
ax.set_xlabel('Cost Percentile Bin')
ax.set_ylabel('Mean Absolute Error ($)')
ax.set_title('MAE by Cost Percentile (Test)')
ax.tick_params(axis='x', rotation=45)

# Plot 6 : Prediction Error % distribution
ax = fig2.add_subplot(gs2[1, 2])
pct_err = (residuals_test / test_metrics['y_true']) * 100
ax.hist(pct_err.clip(-200, 200), bins=100, color=C1, edgecolor='none', alpha=0.85)
ax.axvline(0, color=C3, lw=1.5, ls='--')
within_10 = (np.abs(pct_err) <= 10).mean() * 100
within_20 = (np.abs(pct_err) <= 20).mean() * 100
within_50 = (np.abs(pct_err) <= 50).mean() * 100
ax.set_xlabel('Percentage Error (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Prediction Error % (Test)\n'
             f'Within ±10%: {within_10:.1f}%  ±20%: {within_20:.1f}%  ±50%: {within_50:.1f}%')

plt.savefig('svr_testing_results.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("  ✔ Saved → svr_testing_results.png")

# ── CELL 12 : Permutation Feature Importance ──────────────────────────────────
print("\n" + "="*60)
print("  STEP 8 : FEATURE IMPORTANCE (Permutation on Val Set)")
print("="*60)

print("\n  Computing permutation importance (n_repeats=5)...")
t0 = time.time()
perm_imp = permutation_importance(
    svr_model, X_val_pp, y_val,
    n_repeats=5, random_state=42, n_jobs=-1,
    scoring='r2'
)
print(f"  ✔ Done in {time.time()-t0:.1f}s")

# Get feature names after OHE
ohe_features = preprocessor.named_transformers_['cat']['ohe']\
                            .get_feature_names_out(CAT_COLS).tolist()
all_feat_names = NUM_COLS + ohe_features + BINARY_COLS

# Aggregate OHE features back to original categorical names
imp_df = pd.DataFrame({
    'feature': all_feat_names,
    'importance_mean': perm_imp.importances_mean,
    'importance_std':  perm_imp.importances_std
})

# Group OHE features back to their original column
def original_name(f):
    for cat in CAT_COLS:
        if f.startswith(cat + '_') or f == cat:
            return cat
    return f

imp_df['orig_feature'] = imp_df['feature'].apply(original_name)
imp_grouped = imp_df.groupby('orig_feature')['importance_mean'].sum()\
                    .sort_values(ascending=False).reset_index()
imp_grouped.columns = ['Feature', 'Importance (R² drop)']

print("\n  Top 20 features by permutation importance:")
print(imp_grouped.head(20).to_string(index=False))

# Plot
fig3, ax = plt.subplots(figsize=(12, 9))
top20_imp = imp_grouped.head(20)
colors_fi = [C2 if v > 0 else C3 for v in top20_imp['Importance (R² drop)']]
ax.barh(top20_imp['Feature'][::-1], top20_imp['Importance (R² drop)'][::-1],
        color=colors_fi[::-1], alpha=0.85, edgecolor='none')
ax.axvline(0, color='#8b949e', lw=1)
ax.set_title('SVR — Permutation Feature Importance (Val Set)', fontsize=13)
ax.set_xlabel('Mean R² Drop when feature is permuted')
plt.tight_layout()
plt.savefig('svr_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("  ✔ Saved → svr_feature_importance.png")

# ── CELL 13 : Full Summary Table ──────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL METRICS SUMMARY")
print("="*60)

summary_df = pd.DataFrame({
    'Metric'     : ['R² (orig $)', 'RMSE ($)', 'MAE ($)', 'MedAE ($)', 'MAPE (%)', 'R² (log)'],
    'Train'      : [f"{train_metrics['R2']:.4f}",
                    f"${train_metrics['RMSE']:,.2f}",
                    f"${train_metrics['MAE']:,.2f}",
                    f"${train_metrics['MedAE']:,.2f}",
                    f"{train_metrics['MAPE']:.2f}%",
                    f"{train_metrics['R2_log']:.4f}"],
    'Validation' : [f"{val_metrics['R2']:.4f}",
                    f"${val_metrics['RMSE']:,.2f}",
                    f"${val_metrics['MAE']:,.2f}",
                    f"${val_metrics['MedAE']:,.2f}",
                    f"{val_metrics['MAPE']:.2f}%",
                    f"{val_metrics['R2_log']:.4f}"],
    'Test'       : [f"{test_metrics['R2']:.4f}",
                    f"${test_metrics['RMSE']:,.2f}",
                    f"${test_metrics['MAE']:,.2f}",
                    f"${test_metrics['MedAE']:,.2f}",
                    f"{test_metrics['MAPE']:.2f}%",
                    f"{test_metrics['R2_log']:.4f}"],
})
print("\n" + summary_df.to_string(index=False))

gap_r2   = train_metrics['R2'] - test_metrics['R2']
gap_rmse = test_metrics['RMSE'] - train_metrics['RMSE']
overfitting = "Possible overfitting" if gap_r2 > 0.05 else "Good generalisation"
print(f"\n  R² gap  (train-test) : {gap_r2:+.4f}   → {overfitting}")
print(f"  RMSE gap (test-train): ${gap_rmse:+,.2f}")
print(f"\n  Within ±10% error : {within_10:.1f}%")
print(f"  Within ±20% error : {within_20:.1f}%")
print(f"  Within ±50% error : {within_50:.1f}%")
print(f"\n  Best Hyperparameters : {BEST_PARAMS}")
print(f"  Support Vectors      : {svr_model.n_support_[0]:,} / {len(X_train_pp):,} ({svr_model.n_support_[0]/len(X_train_pp)*100:.2f}%)")
print(f"  Training Time        : {train_time:.1f}s")

# ── CELL 14 : Save Model & Preprocessor ───────────────────────────────────────
SAVE_DIR = "/content/drive/MyDrive/SVR_MedInsurance/"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(svr_model,    SAVE_DIR + "svr_model.pkl")
joblib.dump(preprocessor, SAVE_DIR + "preprocessor.pkl")
print(f"\n  ✔ Model saved     → {SAVE_DIR}svr_model.pkl")
print(f"  ✔ Preprocessor saved → {SAVE_DIR}preprocessor.pkl")

# Save predictions to CSV (useful for document reporting)
test_results_df = pd.DataFrame({
    'actual_cost'   : test_metrics['y_true'],
    'predicted_cost': test_metrics['y_pred'],
    'residual'      : residuals_test,
    'pct_error'     : pct_err.values
})
test_results_df.to_csv(SAVE_DIR + "svr_test_predictions.csv", index=False)
print(f"  ✔ Test predictions saved → {SAVE_DIR}svr_test_predictions.csv")

print("\n" + "="*60)
print("  SVR IMPLEMENTATION COMPLETE")
print("="*60)
