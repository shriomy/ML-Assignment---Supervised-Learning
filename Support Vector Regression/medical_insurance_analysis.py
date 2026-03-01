"""
=============================================================================
  MEDICAL INSURANCE COST PREDICTION — DATASET ANALYSIS SCRIPT
  Run this BEFORE model development to understand the data fully.
  Dataset: medical_insurance.csv (100,000 rows, 54+ columns)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Aesthetic config ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'axes.titlecolor':  '#e6edf3',
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
})
ACCENT   = '#58a6ff'
ACCENT2  = '#3fb950'
ACCENT3  = '#f78166'
ACCENT4  = '#d2a8ff'
PALETTE  = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#ffa657', '#79c0ff']

# =============================================================================
# 0. LOAD DATA
# =============================================================================
CSV_PATH = "medical_insurance.csv"   # ← change if your path differs

print("\n" + "="*70)
print("  MEDICAL INSURANCE COST PREDICTION — DATASET ANALYSIS")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"\n✔  Loaded  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

TARGET = 'annual_medical_cost'

# =============================================================================
# 1. BASIC OVERVIEW
# =============================================================================
print("\n" + "─"*70)
print("  1. BASIC OVERVIEW")
print("─"*70)

print(f"\n  Columns ({df.shape[1]}):")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:>2}. {col}")

print(f"\n  Data Types:")
print(df.dtypes.value_counts().to_string())

print(f"\n  Memory Usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# =============================================================================
# 2. MISSING VALUES
# =============================================================================
print("\n" + "─"*70)
print("  2. MISSING VALUES")
print("─"*70)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if missing_df.empty:
    print("\n  ✔  No missing values found — dataset is complete!")
else:
    print(f"\n  Columns with missing values ({len(missing_df)}):")
    print(missing_df.to_string())

# =============================================================================
# 3. TARGET VARIABLE ANALYSIS
# =============================================================================
print("\n" + "─"*70)
print(f"  3. TARGET VARIABLE: {TARGET}")
print("─"*70)

target = df[TARGET]
print(f"\n  Count   : {target.count():,}")
print(f"  Mean    : ${target.mean():>12,.2f}")
print(f"  Median  : ${target.median():>12,.2f}")
print(f"  Std Dev : ${target.std():>12,.2f}")
print(f"  Min     : ${target.min():>12,.2f}")
print(f"  Max     : ${target.max():>12,.2f}")
print(f"  Range   : ${target.max()-target.min():>12,.2f}")
print(f"  Skewness: {target.skew():>13.4f}")
print(f"  Kurtosis: {target.kurtosis():>13.4f}")

q1, q3  = target.quantile(0.25), target.quantile(0.75)
iqr     = q3 - q1
outliers_low  = target[target < (q1 - 1.5*iqr)]
outliers_high = target[target > (q3 + 1.5*iqr)]
print(f"\n  IQR     : ${iqr:>12,.2f}   (Q1=${q1:,.2f}  Q3=${q3:,.2f})")
print(f"  Outliers (IQR rule): {len(outliers_low)+len(outliers_high):,}  "
      f"({(len(outliers_low)+len(outliers_high))/len(df)*100:.2f}%)")

# PLOT 1 — Target distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'TARGET: {TARGET}  —  Distribution Analysis', fontsize=14, fontweight='bold', color='#e6edf3', y=1.02)

axes[0].hist(target, bins=80, color=ACCENT, edgecolor='none', alpha=0.85)
axes[0].axvline(target.mean(),   color=ACCENT3, lw=2, ls='--', label=f'Mean   ${target.mean():,.0f}')
axes[0].axvline(target.median(), color=ACCENT2, lw=2, ls='--', label=f'Median ${target.median():,.0f}')
axes[0].set_title('Raw Distribution')
axes[0].set_xlabel('Annual Medical Cost ($)')
axes[0].legend(fontsize=9)

axes[1].hist(np.log1p(target), bins=80, color=ACCENT4, edgecolor='none', alpha=0.85)
axes[1].set_title('Log-Transformed Distribution')
axes[1].set_xlabel('log(Annual Medical Cost + 1)')

stats.probplot(target, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot (Normality Check)')
axes[2].get_lines()[0].set(color=ACCENT, markersize=2, alpha=0.4)
axes[2].get_lines()[1].set(color=ACCENT3, lw=2)

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("\n  ✔  Saved → 01_target_distribution.png")

# =============================================================================
# 4. FEATURE CATEGORIZATION
# =============================================================================
print("\n" + "─"*70)
print("  4. FEATURE CATEGORIZATION")
print("─"*70)

drop_cols   = ['person_id', TARGET]
num_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols    = df.select_dtypes(include=['object']).columns.tolist()
bin_cols    = [c for c in num_cols if df[c].nunique() == 2 and c not in drop_cols]
num_cols_f  = [c for c in num_cols if c not in drop_cols and c not in bin_cols]

print(f"\n  Numerical (continuous) : {len(num_cols_f)} features")
print(f"    {num_cols_f}")
print(f"\n  Categorical (object)   : {len(cat_cols)} features")
print(f"    {cat_cols}")
print(f"\n  Binary (0/1)           : {len(bin_cols)} features")
print(f"    {bin_cols}")

# =============================================================================
# 5. NUMERICAL FEATURES — DESCRIPTIVE STATS
# =============================================================================
print("\n" + "─"*70)
print("  5. NUMERICAL FEATURES — DESCRIPTIVE STATISTICS")
print("─"*70)

num_stats = df[num_cols_f].describe().T
num_stats['skew']     = df[num_cols_f].skew()
num_stats['kurtosis'] = df[num_cols_f].kurtosis()
print(f"\n{num_stats[['mean','std','min','25%','50%','75%','max','skew','kurtosis']].round(3).to_string()}")

# =============================================================================
# 6. CATEGORICAL FEATURES — VALUE COUNTS
# =============================================================================
print("\n" + "─"*70)
print("  6. CATEGORICAL FEATURES — VALUE COUNTS & TARGET MEAN")
print("─"*70)

for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"\n  [{col}]  —  {n_unique} unique values")
    vc = df[col].value_counts()
    group_mean = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
    summary = pd.DataFrame({'Count': vc, 'Mean Cost ($)': group_mean.round(2)})
    print(summary.to_string())

# =============================================================================
# 7. CORRELATION WITH TARGET
# =============================================================================
print("\n" + "─"*70)
print(f"  7. CORRELATION WITH TARGET ({TARGET})")
print("─"*70)

corr_all     = df[num_cols_f + bin_cols + [TARGET]].corr()[TARGET].drop(TARGET)
corr_sorted  = corr_all.abs().sort_values(ascending=False)

print(f"\n  Top 20 most correlated features (by |r|):")
top20 = corr_sorted.head(20)
for feat, val in top20.items():
    direction = "▲" if corr_all[feat] > 0 else "▼"
    bar = "█" * int(abs(val) * 30)
    print(f"    {direction} {feat:<40} r={corr_all[feat]:>7.4f}  {bar}")

print(f"\n  Bottom 10 (weakest correlation):")
bot10 = corr_sorted.tail(10)
for feat, val in bot10.items():
    print(f"    {feat:<40} r={corr_all[feat]:>7.4f}")

# PLOT 2 — Correlation bar chart
fig, ax = plt.subplots(figsize=(14, 8))
top_feats = corr_all.reindex(corr_sorted.head(25).index)
colors    = [ACCENT2 if v > 0 else ACCENT3 for v in top_feats.values]
bars = ax.barh(top_feats.index[::-1], top_feats.values[::-1], color=colors[::-1], alpha=0.85, edgecolor='none')
ax.axvline(0, color='#8b949e', lw=1)
ax.set_title(f'Top 25 Feature Correlations with {TARGET}', fontsize=13)
ax.set_xlabel('Pearson Correlation Coefficient')
for bar, val in zip(bars, top_feats.values[::-1]):
    ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8, color='#c9d1d9')
plt.tight_layout()
plt.savefig('02_correlation_with_target.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("\n  ✔  Saved → 02_correlation_with_target.png")

# =============================================================================
# 8. FULL CORRELATION HEATMAP (top features)
# =============================================================================
top_feat_names = corr_sorted.head(15).index.tolist() + [TARGET]
corr_matrix    = df[top_feat_names].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5, linecolor='#21262d',
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — Top 15 Features + Target', fontsize=13)
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("\n  ✔  Saved → 03_correlation_heatmap.png")

# =============================================================================
# 9. KEY NUMERICAL FEATURES vs TARGET
# =============================================================================
key_num = corr_sorted.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Top 6 Numerical Features vs Annual Medical Cost', fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flatten(), key_num):
    ax.scatter(df[feat], df[TARGET], alpha=0.07, s=4, color=ACCENT, rasterized=True)
    m, b = np.polyfit(df[feat].fillna(df[feat].median()), df[TARGET], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
    ax.plot(x_line, m*x_line+b, color=ACCENT3, lw=1.5, label=f'r={corr_all[feat]:.3f}')
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('Annual Medical Cost ($)', fontsize=9)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('04_scatter_key_features.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("\n  ✔  Saved → 04_scatter_key_features.png")

# =============================================================================
# 10. CATEGORICAL FEATURES vs TARGET (Box plots)
# =============================================================================
plot_cat = [c for c in cat_cols if df[c].nunique() <= 8][:6]
n = len(plot_cat)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Categorical Features vs Annual Medical Cost', fontsize=14, fontweight='bold')

for ax, feat in zip(axes.flatten(), plot_cat):
    order = df.groupby(feat)[TARGET].median().sort_values(ascending=False).index
    data  = [df[df[feat]==cat][TARGET].values for cat in order]
    bp = ax.boxplot(data, patch_artist=True, labels=order, notch=False, vert=True)
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ['whiskers','caps','medians']:
        for line in bp[element]: line.set(color='#c9d1d9', linewidth=1.2)
    ax.set_title(feat, fontsize=11)
    ax.set_ylabel('Annual Medical Cost ($)', fontsize=9)
    ax.tick_params(axis='x', labelrotation=20, labelsize=8)

for ax in axes.flatten()[n:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('05_boxplots_categorical.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("\n  ✔  Saved → 05_boxplots_categorical.png")

# =============================================================================
# 11. BINARY FEATURES vs TARGET
# =============================================================================
print("\n" + "─"*70)
print("  11. BINARY FEATURES vs TARGET")
print("─"*70)

print(f"\n  {'Feature':<40} {'0-Mean':>12} {'1-Mean':>12} {'Diff':>10}  Sig?")
print(f"  {'─'*40} {'─'*12} {'─'*12} {'─'*10}  {'─'*4}")
sig_binary = []
for col in bin_cols:
    g0 = df[df[col]==0][TARGET]
    g1 = df[df[col]==1][TARGET]
    diff   = g1.mean() - g0.mean()
    t_stat, p_val = stats.ttest_ind(g0, g1)
    sig = "✔ YES" if p_val < 0.05 else "  NO"
    if p_val < 0.05: sig_binary.append((col, diff))
    print(f"  {col:<40} {g0.mean():>12,.2f} {g1.mean():>12,.2f} {diff:>+10,.2f}  {sig}")

# PLOT 6 — Binary features mean cost
if sig_binary:
    sig_binary.sort(key=lambda x: abs(x[1]), reverse=True)
    feats_b = [x[0] for x in sig_binary[:12]]
    diffs_b = [x[1] for x in sig_binary[:12]]
    colors_b = [ACCENT2 if d > 0 else ACCENT3 for d in diffs_b]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(feats_b[::-1], diffs_b[::-1], color=colors_b[::-1], alpha=0.85, edgecolor='none')
    ax.axvline(0, color='#8b949e', lw=1)
    ax.set_title('Binary Features: Mean Cost Difference (group=1 minus group=0)', fontsize=12)
    ax.set_xlabel('Δ Annual Medical Cost ($)')
    plt.tight_layout()
    plt.savefig('06_binary_feature_impact.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.show()
    print("\n  ✔  Saved → 06_binary_feature_impact.png")

# =============================================================================
# 12. OUTLIER SUMMARY
# =============================================================================
print("\n" + "─"*70)
print("  12. OUTLIER SUMMARY (IQR Method) — Numerical Features")
print("─"*70)

print(f"\n  {'Feature':<40} {'Outliers':>9} {'Pct':>8}  {'Min':>12}  {'Max':>12}")
print(f"  {'─'*40} {'─'*9} {'─'*8}  {'─'*12}  {'─'*12}")
for col in num_cols_f[:20]:
    q1c = df[col].quantile(0.25)
    q3c = df[col].quantile(0.75)
    iqrc = q3c - q1c
    n_out = ((df[col] < q1c - 1.5*iqrc) | (df[col] > q3c + 1.5*iqrc)).sum()
    pct   = n_out / len(df) * 100
    print(f"  {col:<40} {n_out:>9,} {pct:>7.2f}%  {df[col].min():>12.2f}  {df[col].max():>12.2f}")

# =============================================================================
# 13. DATA QUALITY & FEATURE ENGINEERING NOTES
# =============================================================================
print("\n" + "─"*70)
print("  13. DATA QUALITY REPORT & FEATURE ENGINEERING RECOMMENDATIONS")
print("─"*70)

print(f"""
  A. TARGET VARIABLE
     • {TARGET}: Mean=${target.mean():,.2f}, Skewness={target.skew():.3f}
     • {'⚠ Target is positively skewed — consider log1p transform for SVR' if target.skew() > 1 else '✔ Target skewness is acceptable'}
     • SVR is sensitive to feature scale → StandardScaler REQUIRED on all inputs

  B. CATEGORICAL ENCODING NEEDED
     • {cat_cols}
     • Strategy: OneHotEncoding for low-cardinality (≤8), OrdinalEncoding for ordered

  C. FEATURE SCALING (CRITICAL FOR SVR)
     • SVR with RBF kernel is distance-based → all numerical features must be scaled
     • Use StandardScaler (zero mean, unit variance) AFTER train/val/test split

  D. MISSING VALUES
     • {'None detected — no imputation needed' if missing_df.empty else f'Found in: {missing_df.index.tolist()} — impute with median/mode'}

  E. FEATURE SELECTION HINTS (for SVR efficiency)
     • SVR on 100k rows with 54+ features is computationally heavy
     • Consider removing near-zero variance features
     • Consider removing highly correlated pairs (|r| > 0.95 between features)
     • Top correlated features with target: {corr_sorted.head(10).index.tolist()}

  F. SPLIT STRATEGY
     • Training : 70% = {int(len(df)*0.70):,} rows
     • Validation: 10% = {int(len(df)*0.10):,} rows
     • Testing  : 20% = {int(len(df)*0.20):,} rows
     • Use random_state=42 for reproducibility

  G. SVR HYPERPARAMETER SEARCH SPACE (after EDA)
     • Kernel: RBF (recommended for nonlinear data)
     • C     : [0.1, 1, 10, 100]
     • Gamma : ['scale', 'auto', 0.001, 0.01]
     • Epsilon: [0.01, 0.1, 0.5, 1.0]
     • Use GridSearchCV or RandomizedSearchCV on VALIDATION set only
""")

# =============================================================================
# 14. NEAR-ZERO VARIANCE & HIGH CORRELATION FEATURES
# =============================================================================
print("\n" + "─"*70)
print("  14. NEAR-ZERO VARIANCE FEATURES")
print("─"*70)

variances = df[num_cols_f].var()
low_var = variances[variances < 0.01]
if low_var.empty:
    print("\n  ✔ No near-zero variance features found")
else:
    print(f"\n  Features with variance < 0.01:")
    print(low_var.to_string())

print("\n" + "─"*70)
print("  14b. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.90)")
print("─"*70)

corr_matrix_f = df[num_cols_f].corr().abs()
upper_tri = corr_matrix_f.where(np.triu(np.ones(corr_matrix_f.shape), k=1).astype(bool))
high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                   for col in upper_tri.columns
                   for row in upper_tri.index
                   if upper_tri.loc[row, col] > 0.90]
high_corr_pairs.sort(key=lambda x: -x[2])

if not high_corr_pairs:
    print("\n  ✔ No highly correlated feature pairs (> 0.90) found")
else:
    print(f"\n  {'Feature A':<35} {'Feature B':<35} {'|r|':>6}")
    print(f"  {'─'*35} {'─'*35} {'─'*6}")
    for a, b, r in high_corr_pairs:
        print(f"  {a:<35} {b:<35} {r:.4f}")

# =============================================================================
# 15. FINAL SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("  ANALYSIS COMPLETE — SUMMARY")
print("="*70)
print(f"""
  Dataset shape       : {df.shape[0]:,} rows × {df.shape[1]} columns
  Target              : {TARGET}
  Target mean         : ${target.mean():,.2f}
  Target range        : ${target.min():,.2f} – ${target.max():,.2f}
  Missing values      : {'None' if missing_df.empty else f"{missing_df.shape[0]} columns"}
  Numerical features  : {len(num_cols_f)}
  Categorical features: {len(cat_cols)}
  Binary features     : {len(bin_cols)}
  Top predictors      : {', '.join(corr_sorted.head(5).index.tolist())}

  Generated plots:
    01_target_distribution.png
    02_correlation_with_target.png
    03_correlation_heatmap.png
    04_scatter_key_features.png
    05_boxplots_categorical.png
    06_binary_feature_impact.png

  NEXT STEP → Run medical_insurance_svr.py to train the SVR model
""")
print("="*70 + "\n")