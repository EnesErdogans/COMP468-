"""
generate_plots.py
Generates all IEEE paper figures and saves them to plots/.
Run once after training is complete.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import skew
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = '#1565C0'
GREEN  = '#2E7D32'
ORANGE = '#E65100'
RED    = '#B71C1C'
PURPLE = '#4A148C'
TEAL   = '#006064'
COLORS = [BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, '#827717']

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 150, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'axes.edgecolor': '#CCCCCC'})

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
sub   = pd.read_csv('submission.csv')

train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300_000))].reset_index(drop=True)
print(f"Train: {train.shape}  Test: {test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Target distribution: original vs log-transformed
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].hist(train['SalePrice'], bins=60, color=BLUE, edgecolor='white', alpha=0.88)
axes[0].set_title('Original SalePrice  (skewness = {:.2f})'.format(skew(train['SalePrice'])))
axes[0].set_xlabel('SalePrice (USD)')
axes[0].set_ylabel('Count')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
axes[0].axvline(train['SalePrice'].median(), color=ORANGE, lw=2, ls='--', label=f"Median = ${train['SalePrice'].median()/1000:.0f}K")
axes[0].legend()

log_price = np.log1p(train['SalePrice'])
axes[1].hist(log_price, bins=60, color=GREEN, edgecolor='white', alpha=0.88)
axes[1].set_title('log(1 + SalePrice)  (skewness = {:.2f})'.format(skew(log_price)))
axes[1].set_xlabel('log(1 + SalePrice)')
axes[1].set_ylabel('Count')
axes[1].axvline(log_price.median(), color=ORANGE, lw=2, ls='--', label=f'Median = {log_price.median():.3f}')
axes[1].legend()

plt.suptitle('Figure 1: Target Variable Distribution Before and After Log-Transformation',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig1_target_distribution.png', bbox_inches='tight')
plt.close()
print("[OK] fig1_target_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Pearson correlation of top 15 numeric features with SalePrice
# ─────────────────────────────────────────────────────────────────────────────
num_cols = train.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c != 'Id']
corr = train[num_cols].corrwith(np.log1p(train['SalePrice'])).abs().sort_values(ascending=False)
top15 = corr.head(15)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=BLUE, alpha=0.85, edgecolor='white')
ax.set_xlabel('|Pearson r| with log(SalePrice)')
ax.set_title('Figure 2: Top 15 Features Correlated with SalePrice', fontweight='bold')
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlim(0, top15.max() + 0.08)
plt.tight_layout()
plt.savefig('plots/fig2_feature_correlation.png', bbox_inches='tight')
plt.close()
print("[OK] fig2_feature_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — SalePrice by Neighborhood (sorted by median)
# ─────────────────────────────────────────────────────────────────────────────
nbhd_order = train.groupby('Neighborhood')['SalePrice'].median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(14, 6))
groups = [train.loc[train['Neighborhood']==n, 'SalePrice'].values for n in nbhd_order]
bp = ax.boxplot(groups, patch_artist=True, vert=True,
                medianprops=dict(color=ORANGE, lw=2),
                flierprops=dict(marker='o', markersize=2, alpha=0.4),
                boxprops=dict(facecolor=BLUE+'33', color=BLUE),
                whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
ax.set_xticks(range(1, len(nbhd_order)+1))
ax.set_xticklabels(nbhd_order, rotation=45, ha='right')
ax.set_xlabel('Neighborhood')
ax.set_ylabel('SalePrice (USD)')
ax.set_title('Figure 3: SalePrice Distribution by Neighborhood (Sorted by Median)', fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.savefig('plots/fig3_neighborhood_prices.png', bbox_inches='tight')
plt.close()
print("[OK] fig3_neighborhood_prices.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Missing value analysis
# ─────────────────────────────────────────────────────────────────────────────
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0).reset_index(drop=True)
miss = (all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)
miss = miss[miss > 0].head(25)

fig, ax = plt.subplots(figsize=(11, 6))
colors_miss = [RED if v > 50 else ORANGE if v > 20 else BLUE for v in miss.values]
bars = ax.bar(range(len(miss)), miss.values, color=colors_miss, alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(miss)))
ax.set_xticklabels(miss.index, rotation=45, ha='right')
ax.set_ylabel('Missing (%)')
ax.set_title('Figure 4: Missing Value Analysis (Combined Train + Test)', fontweight='bold')
ax.axhline(50, color=RED, ls='--', lw=1.5, label='>50% missing')
ax.axhline(20, color=ORANGE, ls='--', lw=1.5, label='>20% missing')
ax.legend()
for bar, val in zip(bars, miss.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('plots/fig4_missing_values.png', bbox_inches='tight')
plt.close()
print("[OK] fig4_missing_values.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Version progression: Kaggle LB RMSLE
# ─────────────────────────────────────────────────────────────────────────────
versions   = ['v1\nGB+XGB\nRandSearch', 'v2\n4-Model\nOOF Blend', 'v3\n7-Model\nOptuna+Stack']
lb_scores  = [0.12610, 0.12171, 0.11965]
oof_scores = [0.118,   0.11090, 0.10736]
ranks      = [1238, 521, 226]

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(versions))
w = 0.35
bars1 = ax1.bar(x - w/2, lb_scores, w, label='Kaggle Public LB RMSLE', color=BLUE, alpha=0.85, edgecolor='white')
bars2 = ax1.bar(x + w/2, oof_scores, w, label='OOF RMSLE (validation)', color=GREEN, alpha=0.85, edgecolor='white')
ax1.set_ylabel('RMSLE ↓ (lower is better)')
ax1.set_title('Figure 5: Model Progression — RMSLE Improvement Across Versions', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(versions)
ax1.set_ylim(0.095, 0.135)
ax1.legend(loc='upper right')

for bar, score in zip(bars1, lb_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.5f}', ha='center', va='bottom', fontsize=9, color=BLUE, fontweight='bold')
for bar, score in zip(bars2, oof_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{score:.5f}', ha='center', va='bottom', fontsize=9, color=GREEN, fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(x, ranks, 'o--', color=ORANGE, lw=2, ms=8, label='Leaderboard Rank')
for xi, r in zip(x, ranks):
    ax2.annotate(f'Rank {r}', (xi, r), textcoords='offset points', xytext=(10, -15),
                 color=ORANGE, fontsize=9, fontweight='bold')
ax2.set_ylabel('Leaderboard Rank ↓ (lower is better)', color=ORANGE)
ax2.tick_params(axis='y', labelcolor=ORANGE)
ax2.set_ylim(0, 1600)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('plots/fig5_version_progress.png', bbox_inches='tight')
plt.close()
print("[OK] fig5_version_progress.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — OOF RMSLE by model (v3 final)
# ─────────────────────────────────────────────────────────────────────────────
models_v3   = ['GB', 'Lasso', 'ElasticNet', 'CatBoost', 'Ridge', 'XGB', 'LGBM',
               'Ridge Stack\n(meta)', 'Final\nEnsemble']
oof_v3      = [0.11098, 0.11076, 0.11091, 0.11178, 0.11202, 0.11379, 0.11676,
               0.10722, 0.10736]
bar_colors  = [BLUE]*7 + [GREEN, TEAL]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(models_v3, oof_v3, color=bar_colors, alpha=0.85, edgecolor='white')
ax.axhline(min(oof_v3[:7]), color=RED, ls='--', lw=1.5, label=f'Best individual: {min(oof_v3[:7]):.5f}')
ax.set_ylabel('OOF RMSLE ↓')
ax.set_title('Figure 6: Per-Model OOF RMSLE — v3 Final Ensemble', fontweight='bold')
ax.set_ylim(0.10, 0.12)
ax.legend()
for bar, val in zip(bars, oof_v3):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{val:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

patch_base = mpatches.Patch(color=BLUE,  alpha=0.85, label='Base models')
patch_ens  = mpatches.Patch(color=GREEN, alpha=0.85, label='Stack / Ensemble')
ax.legend(handles=[patch_base, patch_ens,
                   mpatches.Patch(color=RED, label=f'Best individual: {min(oof_v3[:7]):.5f}')],
          loc='upper left')
plt.tight_layout()
plt.savefig('plots/fig6_model_oof_comparison.png', bbox_inches='tight')
plt.close()
print("[OK] fig6_model_oof_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Top-4 predictors vs SalePrice scatter
# ─────────────────────────────────────────────────────────────────────────────
top4 = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
labels = ['Overall Quality (1-10)', 'Above-Grade Living Area (ft²)',
          'Total Basement Area (ft²)', 'Garage Area (ft²)']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col, lbl, color in zip(axes, top4, labels, [BLUE, GREEN, ORANGE, PURPLE]):
    ax.scatter(train[col], train['SalePrice'], alpha=0.25, s=10, color=color)
    m, b = np.polyfit(train[col], train['SalePrice'], 1)
    xs = np.linspace(train[col].min(), train[col].max(), 100)
    ax.plot(xs, m*xs + b, color='black', lw=1.5, ls='--')
    ax.set_xlabel(lbl, fontsize=9)
    ax.set_ylabel('SalePrice' if col == 'OverallQual' else '')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    r = np.corrcoef(train[col], train['SalePrice'])[0, 1]
    ax.set_title(f'{col}\nr = {r:.3f}', fontweight='bold', fontsize=10)

plt.suptitle('Figure 7: Top Four Predictors vs SalePrice', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/fig7_top_predictors.png', bbox_inches='tight')
plt.close()
print("[OK] fig7_top_predictors.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Feature importances (train quick GB for this plot)
# ─────────────────────────────────────────────────────────────────────────────
print("Training quick GB for feature importance plot...")

ORDINAL_COLS = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
                'KitchenQual','FireplaceQu','GarageQual','GarageCond',
                'BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish',
                'PoolQC','Fence','LandSlope','LotShape','PavedDrive',
                'Utilities','Functional']
QUAL_MAP = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0,
            'Av':3,'Mn':2,'No':1,'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,
            'Fin':3,'RFn':2,'GdPrv':4,'MnPrv':3,'MnWw':2,'ELO':1,
            'SeverelySloped':1,'Mod':2,'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1,
            'Reg':4,'IR1':3,'IR2':2,'IR3':1,'Y':2,'P':1,'N':0,'Typ':7,'Min1':6,
            'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}

df_fi = train.copy()
for col in ORDINAL_COLS:
    if col in df_fi.columns:
        df_fi[col] = df_fi[col].fillna('NA').map(QUAL_MAP).fillna(0)

cat_cols = df_fi.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_fi[col] = pd.Categorical(df_fi[col]).codes

df_fi = df_fi.fillna(df_fi.median(numeric_only=True))
X_fi = df_fi.drop(['Id','SalePrice'], axis=1, errors='ignore')
y_fi = np.log1p(df_fi['SalePrice'])

gb_fi = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
gb_fi.fit(X_fi, y_fi)

importance_df = pd.DataFrame({'feature': X_fi.columns, 'importance': gb_fi.feature_importances_})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1],
               color=BLUE, alpha=0.85, edgecolor='white')
ax.set_xlabel('Feature Importance (GradientBoosting, mean decrease in impurity)')
ax.set_title('Figure 8: Top 20 Feature Importances', fontweight='bold')
for bar, val in zip(bars, importance_df['importance'][::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('plots/fig8_feature_importance.png', bbox_inches='tight')
plt.close()
print("[OK] fig8_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Submission prediction distribution vs training distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.hist(train['SalePrice'], bins=60, color=BLUE, alpha=0.6, label='Train SalePrice', density=True)
ax.hist(sub['SalePrice'],   bins=60, color=ORANGE, alpha=0.6, label='v3 Predictions', density=True)
ax.set_xlabel('SalePrice (USD)')
ax.set_ylabel('Density')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
ax.set_title('Figure 9: Training Distribution vs Final Submission Predictions', fontweight='bold')
ax.axvline(train['SalePrice'].mean(), color=BLUE,   lw=2, ls='--', label=f"Train mean = ${train['SalePrice'].mean()/1000:.0f}K")
ax.axvline(sub['SalePrice'].mean(),   color=ORANGE, lw=2, ls='--', label=f"Pred mean  = ${sub['SalePrice'].mean()/1000:.0f}K")
ax.legend()
plt.tight_layout()
plt.savefig('plots/fig9_submission_distribution.png', bbox_inches='tight')
plt.close()
print("[OK] fig9_submission_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Stacking improvement heatmap (fold × model)
# ─────────────────────────────────────────────────────────────────────────────
fold_data = np.array([
    [0.11004, 0.12001, 0.12085, 0.11268, 0.11310, 0.10898, 0.10912],
    [0.10678, 0.10806, 0.11150, 0.10878, 0.10966, 0.11050, 0.11102],
    [0.11983, 0.12043, 0.12315, 0.11743, 0.12022, 0.11958, 0.11934],
    [0.11714, 0.11751, 0.12316, 0.12095, 0.11780, 0.11345, 0.11415],
    [0.09991, 0.10170, 0.10388, 0.09758, 0.09795, 0.10041, 0.09998],
])
model_names = ['GB', 'XGB', 'LGBM', 'CAT', 'Ridge', 'Lasso', 'EN']
fold_names  = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(fold_data, cmap='RdYlGn_r', aspect='auto',
               vmin=fold_data.min(), vmax=fold_data.max())
ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names, fontsize=11)
ax.set_yticks(range(len(fold_names)));  ax.set_yticklabels(fold_names, fontsize=11)
ax.set_title('Figure 10: Per-Fold RMSLE Heatmap (v3 — 5 Folds × 7 Models)', fontweight='bold')
plt.colorbar(im, ax=ax, label='RMSLE (lower = better)')
for i in range(len(fold_names)):
    for j in range(len(model_names)):
        ax.text(j, i, f'{fold_data[i,j]:.4f}', ha='center', va='center',
                fontsize=8, color='white' if fold_data[i,j] > 0.118 else 'black')
plt.tight_layout()
plt.savefig('plots/fig10_fold_heatmap.png', bbox_inches='tight')
plt.close()
print("[OK] fig10_fold_heatmap.png")

print(f"\nAll {len(os.listdir('plots'))} plots saved to plots/")
