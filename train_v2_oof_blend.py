"""
House Price Prediction - Improved Model v2
- 4 models: GradientBoosting, XGBoost, LightGBM, Ridge
- 5-fold Out-of-Fold (OOF) blending for robust ensemble
- Enhanced feature engineering (interaction features, extra flags)
- Weighted blend by OOF RMSLE
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.model_selection import KFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

RANDOM_STATE = 42
N_FOLDS      = 5
np.random.seed(RANDOM_STATE)

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}  Test: {test.shape}")

test_ids = test['Id']
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300_000))].reset_index(drop=True)
print(f"After outlier removal: {train.shape}")

y        = np.log1p(train['SalePrice'].values)
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0).reset_index(drop=True)

# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()

    none_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
                 'GarageType','GarageFinish','GarageQual','GarageCond',
                 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'MasVnrType']
    zero_cols = ['GarageYrBlt','GarageArea','GarageCars',
                 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                 'BsmtFullBath','BsmtHalfBath','MasVnrArea',
                 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
    df[[c for c in none_cols if c in df.columns]] = df[[c for c in none_cols if c in df.columns]].fillna('None')
    df[[c for c in zero_cols if c in df.columns]] = df[[c for c in zero_cols if c in df.columns]].fillna(0)

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

    # Fix GarageYrBlt typos
    mask = df['GarageYrBlt'] > 2100
    df.loc[mask, 'GarageYrBlt'] = df.loc[mask, 'YearBuilt']

    # ── Composite area features ───────────────────────────────────────────────
    df['TotalSF']         = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']       = (df['FullBath'] + 0.5*df['HalfBath']
                             + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
    df['TotalPorch']      = (df['OpenPorchSF'] + df['EnclosedPorch']
                             + df['3SsnPorch'] + df['ScreenPorch'])
    df['TotalOutdoorSF']  = (df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']
                             + df['3SsnPorch'] + df['ScreenPorch'] + df['PoolArea'])

    # ── Age features ─────────────────────────────────────────────────────────
    df['Age']          = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']   = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge']    = np.where(df['GarageYrBlt'] > 0,
                                  df['YrSold'] - df['GarageYrBlt'], 0)
    df['IsRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    # ── Binary flags ─────────────────────────────────────────────────────────
    df['HasPool']      = (df['PoolArea']     > 0).astype(int)
    df['HasGarage']    = (df['GarageArea']   > 0).astype(int)
    df['HasBsmt']      = (df['TotalBsmtSF']  > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces']   > 0).astype(int)
    df['HasMasVnr']    = (df['MasVnrArea']   > 0).astype(int)
    df['Is2ndFloor']   = (df['2ndFlrSF']     > 0).astype(int)
    df['IsNew']        = (df['YrSold'] == df['YearBuilt']).astype(int)

    # ── Quality × size interactions (most predictive) ─────────────────────────
    df['OverallQual_sq']  = df['OverallQual'] ** 2
    df['OverallScore']    = df['OverallQual'] * df['OverallCond']
    df['TotalSF_x_Qual']  = df['TotalSF']    * df['OverallQual']
    df['GrLiv_x_Qual']    = df['GrLivArea']  * df['OverallQual']
    df['Garage_x_Qual']   = df['GarageArea'] * df['OverallQual']
    df['Bsmt_x_Qual']     = df['TotalBsmtSF']* df['OverallQual']

    # ── Ratio features ────────────────────────────────────────────────────────
    df['LivLotRatio']  = df['GrLivArea'] / df['LotArea'].clip(lower=1)
    df['RoomSize']     = df['GrLivArea'] / (df['TotRmsAbvGrd'].clip(lower=1))
    df['BsmtRatio']    = df['TotalBsmtSF'] / df['TotalSF'].clip(lower=1)

    return df

all_data = feature_engineering(all_data)
n_train  = len(train)
print(f"After feature engineering: {all_data.shape}")

# ── Skewness correction (fit only on training slice) ─────────────────────────
train_slice = all_data.iloc[:n_train]
num_feats   = train_slice.select_dtypes(include=['int64','float64']).columns
skewed      = (train_slice[num_feats]
               .apply(lambda x: skew(x.dropna()))
               .pipe(lambda s: s[s.abs() > 0.75]).index)
print(f"Skewed features corrected: {len(skewed)}")
for feat in skewed:
    all_data[feat] = boxcox1p(all_data[feat].clip(lower=0), 0.15)

train_fe = all_data.iloc[:n_train]
test_fe  = all_data.iloc[n_train:]

# ── Preprocessing pipeline ────────────────────────────────────────────────────
ORDINAL_CATS = {
    'ExterQual':    ['Po','Fa','TA','Gd','Ex'],
    'ExterCond':    ['Po','Fa','TA','Gd','Ex'],
    'BsmtQual':     ['None','Po','Fa','TA','Gd','Ex'],
    'BsmtCond':     ['None','Po','Fa','TA','Gd','Ex'],
    'HeatingQC':    ['Po','Fa','TA','Gd','Ex'],
    'KitchenQual':  ['Po','Fa','TA','Gd','Ex'],
    'FireplaceQu':  ['None','Po','Fa','TA','Gd','Ex'],
    'GarageQual':   ['None','Po','Fa','TA','Gd','Ex'],
    'GarageCond':   ['None','Po','Fa','TA','Gd','Ex'],
    'BsmtExposure': ['None','No','Mn','Av','Gd'],
    'BsmtFinType1': ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'BsmtFinType2': ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'GarageFinish': ['None','Unf','RFn','Fin'],
    'Fence':        ['None','MnWw','GdWo','MnPrv','GdPrv'],
    'Utilities':    ['ELO','NoSeWa','NoSewr','AllPub'],
    'LandSlope':    ['Sev','Mod','Gtl'],
    'LotShape':     ['IR3','IR2','IR1','Reg'],
}

ORDINAL_COLS = [c for c in ORDINAL_CATS if c in train_fe.columns]
ord_categories = [ORDINAL_CATS[c] for c in ORDINAL_COLS]

X      = train_fe.drop(columns=['Id','SalePrice'], errors='ignore')
X_test = test_fe.drop(columns=['Id','SalePrice'], errors='ignore')

NUM_COLS = X.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_NOM  = [c for c in X.select_dtypes(include='object').columns if c not in ORDINAL_COLS]

print(f"Features: {len(NUM_COLS)} numeric, {len(ORDINAL_COLS)} ordinal, {len(CAT_NOM)} nominal")

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  RobustScaler())]),                      NUM_COLS),
    ('ord', OrdinalEncoder(categories=ord_categories,
                           handle_unknown='use_encoded_value',
                           unknown_value=-1),                         ORDINAL_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore',
                                            sparse_output=False))]),  CAT_NOM),
], remainder='drop')

# ── Model definitions ────────────────────────────────────────────────────────
PIPELINES = {
    'GB': Pipeline([('pre', preprocessor), ('model', GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, max_features=0.6, min_samples_split=4,
        loss='huber', random_state=RANDOM_STATE))]),

    'XGB': Pipeline([('pre', preprocessor), ('model', XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=1.0, min_child_weight=3,
        objective='reg:squarederror', random_state=RANDOM_STATE, verbosity=0))]),

    'LGBM': Pipeline([('pre', preprocessor), ('model', LGBMRegressor(
        n_estimators=3000, learning_rate=0.02, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.1, min_child_samples=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))]),

    'Ridge': Pipeline([('pre', preprocessor), ('model', Ridge(alpha=10.0))]),
}

# ── 5-Fold OOF training ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"5-Fold OOF Training  ({len(PIPELINES)} models)")
print(f"{'='*60}")

kfold     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = {name: np.zeros(n_train)        for name in PIPELINES}
tst_preds = {name: np.zeros(len(X_test))    for name in PIPELINES}

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X), 1):
    X_tr_f, X_val_f = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr_f, y_val_f = y[tr_idx], y[val_idx]
    print(f"\nFold {fold}/{N_FOLDS}  (train={len(tr_idx)}, val={len(val_idx)})")

    for name, pipe in PIPELINES.items():
        p = clone(pipe)
        p.fit(X_tr_f, y_tr_f)
        oof_preds[name][val_idx]  = p.predict(X_val_f)
        tst_preds[name]          += p.predict(X_test) / N_FOLDS
        fold_rmsle = np.sqrt(mean_squared_error(y_val_f, oof_preds[name][val_idx]))
        print(f"  {name:6s} fold RMSLE = {fold_rmsle:.5f}")

# ── OOF summary and blend weights ────────────────────────────────────────────
print(f"\n{'='*60}")
print("OOF Results:")
oof_scores = {}
for name in PIPELINES:
    rmsle = np.sqrt(mean_squared_error(y, oof_preds[name]))
    mae   = mean_absolute_error(np.expm1(y), np.expm1(oof_preds[name]))
    r2    = r2_score(y, oof_preds[name])
    oof_scores[name] = rmsle
    print(f"  {name:6s}  OOF RMSLE={rmsle:.5f}  MAE=${mae:,.0f}  R²={r2:.4f}")

# Inverse-RMSLE weights (lower error → higher weight)
total_inv = sum(1/s for s in oof_scores.values())
weights   = {n: (1/s)/total_inv for n, s in oof_scores.items()}
print(f"\nBlend weights:")
for n, w in weights.items():
    print(f"  {n:6s}: {w:.3f}")

# OOF blend RMSLE
blend_oof = sum(weights[n] * oof_preds[n] for n in PIPELINES)
blend_rmsle = np.sqrt(mean_squared_error(y, blend_oof))
blend_mae   = mean_absolute_error(np.expm1(y), np.expm1(blend_oof))
print(f"\nBlend OOF RMSLE = {blend_rmsle:.5f}  MAE=${blend_mae:,.0f}")

# ── Final predictions ─────────────────────────────────────────────────────────
blend_test = sum(weights[n] * tst_preds[n] for n in PIPELINES)
pred_final = np.expm1(blend_test)

# ── Save ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({'Id': test_ids, 'SalePrice': pred_final})
sub.to_csv('submission.csv', index=False)

print(f"\nSubmission saved: {len(sub)} rows")
print(f"  mean: ${pred_final.mean():,.0f}")
print(f"  min:  ${pred_final.min():,.0f}")
print(f"  max:  ${pred_final.max():,.0f}")
print(f"\nBest OOF RMSLE (blend): {blend_rmsle:.5f}")
print("DONE — submission.csv is ready.")
