"""
train_v4_advanced.py — COMP 468 Final Project v4
Target: Top 100 on Kaggle (LB RMSLE < 0.108)

Key improvements over v3:
  1. Extended feature engineering (30+ engineered features)
  2. Neighbourhood-aware LotFrontage imputation
  3. Within-fold target encoding for high-cardinality categoricals
  4. HistGradientBoostingRegressor as 8th base model
  5. 150 Optuna trials per tree-based model (vs 80 in v3)
  6. XGBoost meta-learner stacking option
  7. Pseudo-labelling round (one iteration)
  8. Final blend: 60% stack + 40% weighted blend

Expected training time: 4-8 hours on CPU
"""

import numpy as np
import pandas as pd
import warnings, json, os
warnings.filterwarnings('ignore')

from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ─────────────────────────────────────────────────────────────────────────────
# Ordinal mappings  (20 columns)
# ─────────────────────────────────────────────────────────────────────────────
QUAL_ORDER  = ['Po','Fa','TA','Gd','Ex']
QUAL_COLS   = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
               'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
BSMT_EXP    = {'NA':0,'No':1,'Mn':2,'Av':3,'Gd':4}
BSMT_FIN    = {'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
GRG_FIN     = {'NA':0,'Unf':1,'RFn':2,'Fin':3}
FENCE_MAP   = {'NA':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}
SLOPE_MAP   = {'Gtl':1,'Mod':2,'Sev':3}
LOTSHP_MAP  = {'IR3':1,'IR2':2,'IR1':3,'Reg':4}
PAVED_MAP   = {'N':0,'P':1,'Y':2}
UTIL_MAP    = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
FUNC_MAP    = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8}

def apply_ordinal(df):
    for col in QUAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('NA').map({v:i for i,v in enumerate(['NA']+QUAL_ORDER)}).fillna(0)
    for col in ['BsmtFinType1','BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].fillna('NA').map(BSMT_FIN).fillna(0)
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].fillna('NA').map(BSMT_EXP).fillna(0)
    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].fillna('NA').map(GRG_FIN).fillna(0)
    if 'Fence'    in df.columns: df['Fence']    = df['Fence'].fillna('NA').map(FENCE_MAP).fillna(0)
    if 'LandSlope'in df.columns: df['LandSlope']= df['LandSlope'].fillna('Gtl').map(SLOPE_MAP).fillna(1)
    if 'LotShape' in df.columns: df['LotShape'] = df['LotShape'].fillna('Reg').map(LOTSHP_MAP).fillna(4)
    if 'PavedDrive'in df.columns:df['PavedDrive']=df['PavedDrive'].fillna('Y').map(PAVED_MAP).fillna(2)
    if 'Utilities'in df.columns: df['Utilities']= df['Utilities'].fillna('AllPub').map(UTIL_MAP).fillna(4)
    if 'Functional'in df.columns:df['Functional']=df['Functional'].fillna('Typ').map(FUNC_MAP).fillna(8)
    return df

ORDINAL_DONE = (QUAL_COLS + ['BsmtFinType1','BsmtFinType2','BsmtExposure','GarageFinish',
                              'Fence','LandSlope','LotShape','PavedDrive','Utilities','Functional'])

NOMINAL_COLS = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig',
                'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                'Foundation','Heating','CentralAir','Electrical','GarageType',
                'MiscFeature','SaleType','SaleCondition']

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineer(df, nbhd_lotfront=None, nbhd_log_med=None, global_log_med=None, is_train=True):
    df = df.copy()

    # MSSubClass as categorical string
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # ── Neighbourhood-aware LotFrontage imputation ────────────────────────────
    if nbhd_lotfront is None:
        nbhd_lotfront = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()
    df['LotFrontage'] = df.apply(
        lambda r: nbhd_lotfront.get(r['Neighborhood'], df['LotFrontage'].median())
        if pd.isna(r['LotFrontage']) else r['LotFrontage'], axis=1)

    # ── Structural NAs → 0 or mode ────────────────────────────────────────────
    fill_zero = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
                 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
    for c in fill_zero:
        if c in df.columns: df[c] = df[c].fillna(0)

    fill_mode = ['Electrical','KitchenQual','Exterior1st','Exterior2nd',
                 'SaleType','Functional','MSZoning','Utilities']
    for c in fill_mode:
        if c in df.columns: df[c] = df[c].fillna(df[c].mode()[0])

    # ── Composite features ────────────────────────────────────────────────────
    df['TotalSF']       = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']     = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalPorch']    = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                           df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    df['Age']           = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']    = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge']     = (df['YrSold'] - df['GarageYrBlt']).clip(lower=0)

    # Quality interactions
    df['OverallQualSq'] = df['OverallQual'] ** 2
    df['OverallQual3']  = df['OverallQual'] ** 3
    df['TotalSF_Qual']  = df['TotalSF'] * df['OverallQual']
    df['GrLiv_Qual']    = df['GrLivArea'] * df['OverallQual']
    df['TotalQuality']  = df['OverallQual'] + df['OverallCond']
    df['QualAge']       = df['OverallQual'] * (100 - df['Age'].clip(0, 100))  # quality × newness

    # Area ratios
    df['LivAreaRatio']  = df['GrLivArea'] / (df['TotalSF'].clip(lower=1))
    df['BsmtRatio']     = df['TotalBsmtSF'] / (df['TotalSF'].clip(lower=1))
    df['GarageRatio']   = df['GarageArea'] / (df['TotalSF'].clip(lower=1))
    df['LotRatio']      = df['LotFrontage'] / (df['LotArea'].clip(lower=1) ** 0.5)

    # Binary has-X flags
    df['HasPool']       = (df['PoolArea'] > 0).astype(int)
    df['HasGarage']     = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor']   = (df['2ndFlrSF'] > 0).astype(int)
    df['HasBsmt']       = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace']  = (df['Fireplaces'] > 0).astype(int)
    df['HasPorch']      = (df['TotalPorch'] > 0).astype(int)
    df['HasRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['IsNew']         = (df['YearBuilt'] == df['YrSold']).astype(int)
    df['HasMasVnr']     = (df['MasVnrArea'] > 0).astype(int)

    # Neighbourhood rank and stats
    if nbhd_log_med is not None:
        df['NbhoodLogMed'] = df['Neighborhood'].map(nbhd_log_med).fillna(global_log_med)

    # Cyclic month
    df['MoSoldSin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSoldCos'] = np.cos(2 * np.pi * df['MoSold'] / 12)

    return df, nbhd_lotfront

def target_encode_fold(tr_df, val_df, tst_df, cols, target_col, alpha=10):
    """Smooth target encoding fitted on tr_df, zero-leakage."""
    g_mean = tr_df[target_col].mean()
    te_val = {}; te_tst = {}
    for col in cols:
        stats = tr_df.groupby(col)[target_col].agg(['mean','count'])
        smooth = (stats['mean']*stats['count'] + g_mean*alpha) / (stats['count'] + alpha)
        te_val[f'{col}_te'] = val_df[col].map(smooth).fillna(g_mean).values
        te_tst[f'{col}_te'] = tst_df[col].map(smooth).fillna(g_mean).values
    return pd.DataFrame(te_val, index=val_df.index), pd.DataFrame(te_tst, index=tst_df.index)

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
train_raw = pd.read_csv('data/train.csv')
test_raw  = pd.read_csv('data/test.csv')

# Outlier removal
train_raw = train_raw[~((train_raw['GrLivArea']>4000)&(train_raw['SalePrice']<300_000))].reset_index(drop=True)
print(f"Train: {train_raw.shape}  Test: {test_raw.shape}")

# Neighbourhood log-median (computed from training set)
nbhd_log_med  = train_raw.groupby('Neighborhood')['SalePrice'].apply(lambda x: np.log1p(x).median()).to_dict()
global_log_med = np.log1p(train_raw['SalePrice']).median()

# Feature engineering
train_fe, nbhd_lotfront = feature_engineer(train_raw, nbhd_log_med=nbhd_log_med, global_log_med=global_log_med, is_train=True)
test_fe,  _             = feature_engineer(test_raw,  nbhd_lotfront=nbhd_lotfront, nbhd_log_med=nbhd_log_med, global_log_med=global_log_med, is_train=False)

y_train = np.log1p(train_raw['SalePrice'].values)

# Ordinal encoding
train_fe = apply_ordinal(train_fe)
test_fe  = apply_ordinal(test_fe)

# Remove Id and target from features
drop_cols = ['Id','SalePrice']
train_fe  = train_fe.drop([c for c in drop_cols if c in train_fe.columns], axis=1)
test_ids  = test_fe['Id'].values
test_fe   = test_fe.drop([c for c in drop_cols if c in test_fe.columns], axis=1)

# Remaining categoricals: one-hot encode
nom_present = [c for c in NOMINAL_COLS if c in train_fe.columns]
remain_cat  = [c for c in train_fe.select_dtypes(include='object').columns if c not in ORDINAL_DONE]

# One-hot encode nominal + remaining string cols
all_ohe_cols = list(set(nom_present + remain_cat))
train_ohe = pd.get_dummies(train_fe, columns=all_ohe_cols, dummy_na=False)
test_ohe  = pd.get_dummies(test_fe,  columns=all_ohe_cols, dummy_na=False)
train_ohe, test_ohe = train_ohe.align(test_ohe, join='left', axis=1, fill_value=0)

# Numeric columns — fill NaN, BoxCox skewness correction, RobustScaler
num_cols = train_ohe.select_dtypes(include='number').columns.tolist()

# Impute
train_ohe[num_cols] = train_ohe[num_cols].fillna(train_ohe[num_cols].median())
test_ohe[num_cols]  = test_ohe[num_cols].fillna(train_ohe[num_cols].median())

# BoxCox
BOXCOX_LAMBDA = 0.15
skewed = train_ohe[num_cols].apply(lambda x: skew(x.dropna()))
skewed = skewed[abs(skewed) > 0.75].index.tolist()
for col in skewed:
    if (train_ohe[col] >= 0).all() and (test_ohe[col] >= 0).all():
        train_ohe[col] = boxcox1p(train_ohe[col], BOXCOX_LAMBDA)
        test_ohe[col]  = boxcox1p(test_ohe[col],  BOXCOX_LAMBDA)
print(f"Skewed features corrected: {len(skewed)}")

# RobustScaler
scaler = RobustScaler()
train_ohe[num_cols] = scaler.fit_transform(train_ohe[num_cols])
test_ohe[num_cols]  = scaler.transform(test_ohe[num_cols])

X_train = train_ohe.values.astype(np.float32)
X_test  = test_ohe.values.astype(np.float32)
feat_names = train_ohe.columns.tolist()

# Target-encoding columns (high cardinality)
TE_COLS = ['Neighborhood','MSZoning']
te_col_idx = {col: feat_names.index(col) if col in feat_names else None for col in TE_COLS}
# These are now numeric after OHE, so TE will be done inside OOF using original string columns
te_train = train_fe[TE_COLS].copy() if all(c in train_fe.columns for c in TE_COLS) else None
te_test  = test_fe[TE_COLS].copy()  if all(c in test_fe.columns  for c in TE_COLS) else None

print(f"Feature matrix: {X_train.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Optuna hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 150
N_CV = 3  # fast CV inside Optuna

kf_optuna = KFold(n_splits=N_CV, shuffle=True, random_state=SEED)

def cv_rmsle(model, X, y, cv):
    scores = []
    for tr, va in cv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        scores.append(rmsle(y[va], pred))
    return np.mean(scores)

results = {}
optuna_file = 'optuna_results_v4.json'
if os.path.exists(optuna_file):
    with open(optuna_file) as f:
        results = json.load(f)
    print(f"Loaded previous Optuna results from {optuna_file}")

# XGBoost
if 'XGB' not in results:
    print(f"\nOptimising XGBoost ({OPTUNA_TRIALS} trials) ...")
    def xgb_obj(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 1000, 6000),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            max_depth         = trial.suggest_int('max_depth', 3, 6),
            subsample         = trial.suggest_float('subsample', 0.4, 0.9),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.3, 0.8),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
            gamma             = trial.suggest_float('gamma', 0, 0.3),
            tree_method='hist', device='cpu', random_state=SEED, verbosity=0,
        )
        return cv_rmsle(xgb.XGBRegressor(**p), X_train, y_train, kf_optuna)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(xgb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    results['XGB'] = {'score': study.best_value, 'params': study.best_params}
    print(f"  XGB best CV RMSLE: {study.best_value:.5f}")
else:
    print(f"  XGB (cached): {results['XGB']['score']:.5f}")

# LightGBM
if 'LGBM' not in results:
    print(f"\nOptimising LightGBM ({OPTUNA_TRIALS} trials) ...")
    def lgbm_obj(trial):
        p = dict(
            n_estimators      = trial.suggest_int('n_estimators', 1000, 6000),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 20, 60),
            subsample         = trial.suggest_float('subsample', 0.5, 0.95),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.3, 0.8),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
            min_child_samples = trial.suggest_int('min_child_samples', 10, 60),
            max_depth         = trial.suggest_int('max_depth', 2, 5),
            random_state=SEED, verbosity=-1,
        )
        return cv_rmsle(lgb.LGBMRegressor(**p), X_train, y_train, kf_optuna)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(lgbm_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    results['LGBM'] = {'score': study.best_value, 'params': study.best_params}
    print(f"  LGBM best CV RMSLE: {study.best_value:.5f}")
else:
    print(f"  LGBM (cached): {results['LGBM']['score']:.5f}")

# GradientBoosting
if 'GB' not in results:
    print(f"\nOptimising GradientBoosting ({OPTUNA_TRIALS} trials) ...")
    def gb_obj(trial):
        p = dict(
            n_estimators     = trial.suggest_int('n_estimators', 500, 3000),
            learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            max_depth        = trial.suggest_int('max_depth', 2, 5),
            subsample        = trial.suggest_float('subsample', 0.5, 0.9),
            max_features     = trial.suggest_float('max_features', 0.3, 0.8),
            min_samples_split= trial.suggest_int('min_samples_split', 2, 15),
            loss             = trial.suggest_categorical('loss', ['squared_error','huber']),
            random_state=SEED,
        )
        return cv_rmsle(GradientBoostingRegressor(**p), X_train, y_train, kf_optuna)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(gb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    results['GB'] = {'score': study.best_value, 'params': study.best_params}
    print(f"  GB best CV RMSLE: {study.best_value:.5f}")
else:
    print(f"  GB (cached): {results['GB']['score']:.5f}")

# CatBoost
if 'CAT' not in results:
    print(f"\nOptimising CatBoost ({OPTUNA_TRIALS} trials) ...")
    def cat_obj(trial):
        p = dict(
            iterations        = trial.suggest_int('iterations', 500, 3000),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            depth             = trial.suggest_int('depth', 4, 8),
            l2_leaf_reg       = trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.5),
            random_strength   = trial.suggest_float('random_strength', 0.1, 2.0),
            border_count      = trial.suggest_int('border_count', 64, 254),
            random_seed=SEED, verbose=0,
        )
        return cv_rmsle(cb.CatBoostRegressor(**p), X_train, y_train, kf_optuna)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(cat_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    results['CAT'] = {'score': study.best_value, 'params': study.best_params}
    print(f"  CAT best CV RMSLE: {study.best_value:.5f}")
else:
    print(f"  CAT (cached): {results['CAT']['score']:.5f}")

# HistGradientBoosting
if 'HIST' not in results:
    print(f"\nOptimising HistGradientBoosting ({OPTUNA_TRIALS} trials) ...")
    def hist_obj(trial):
        p = dict(
            max_iter          = trial.suggest_int('max_iter', 500, 3000),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            max_depth         = trial.suggest_int('max_depth', 3, 7),
            max_leaf_nodes    = trial.suggest_int('max_leaf_nodes', 15, 63),
            min_samples_leaf  = trial.suggest_int('min_samples_leaf', 5, 40),
            l2_regularization = trial.suggest_float('l2_regularization', 1e-4, 5.0, log=True),
            max_features      = trial.suggest_float('max_features', 0.3, 1.0),
            random_state=SEED,
        )
        return cv_rmsle(HistGradientBoostingRegressor(**p), X_train, y_train, kf_optuna)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
    study.optimize(hist_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    results['HIST'] = {'score': study.best_value, 'params': study.best_params}
    print(f"  HIST best CV RMSLE: {study.best_value:.5f}")
else:
    print(f"  HIST (cached): {results['HIST']['score']:.5f}")

# Linear models
print("\nSearching Ridge / Lasso / ElasticNet ...")
best_ridge = best_lasso = best_en = (999, None)
for a in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10, 20, 50]:
    s = cv_rmsle(Ridge(alpha=a), X_train, y_train, kf_optuna)
    if s < best_ridge[0]: best_ridge = (s, a)
for a in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
    s = cv_rmsle(Lasso(alpha=a, max_iter=10000), X_train, y_train, kf_optuna)
    if s < best_lasso[0]: best_lasso = (s, a)
for a in [0.0001, 0.0005, 0.001, 0.005]:
    for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = cv_rmsle(ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000), X_train, y_train, kf_optuna)
        if s < best_en[0]: best_en = (s, (a, l1))
print(f"  Ridge  best: a={best_ridge[1]}  CV={best_ridge[0]:.5f}")
print(f"  Lasso  best: a={best_lasso[1]}  CV={best_lasso[0]:.5f}")
print(f"  EN     best: a={best_en[1][0]} l1={best_en[1][1]}  CV={best_en[0]:.5f}")

with open(optuna_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nOptuna results saved to {optuna_file}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: 5-Fold OOF with target encoding inside each fold
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAMES = ['GB','XGB','LGBM','CAT','HIST','Ridge','Lasso','EN']
N_MODELS = len(MODEL_NAMES)
N_FOLDS  = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds  = np.zeros((len(X_train), N_MODELS))
test_preds = np.zeros((len(X_test),  N_MODELS))

def make_models():
    xp = results['XGB']['params'].copy()
    xp.update({'tree_method':'hist','device':'cpu','random_state':SEED,'verbosity':0})
    lp = results['LGBM']['params'].copy()
    lp.update({'random_state':SEED,'verbosity':-1})
    gp = results['GB']['params'].copy()
    gp.update({'random_state':SEED})
    cp = results['CAT']['params'].copy()
    cp.update({'random_seed':SEED,'verbose':0})
    hp = results['HIST']['params'].copy()
    hp.update({'random_state':SEED})
    return [
        GradientBoostingRegressor(**gp),
        xgb.XGBRegressor(**xp),
        lgb.LGBMRegressor(**lp),
        cb.CatBoostRegressor(**cp),
        HistGradientBoostingRegressor(**hp),
        Ridge(alpha=best_ridge[1]),
        Lasso(alpha=best_lasso[1], max_iter=10000),
        ElasticNet(alpha=best_en[1][0], l1_ratio=best_en[1][1], max_iter=10000),
    ]

print(f"\n{'='*60}")
print(f"PHASE 2: 5-Fold OOF ({N_MODELS} models)")
print(f"{'='*60}")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold+1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(va_idx)})")
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    X_te_fold  = X_test.copy()

    # ── Within-fold target encoding ──────────────────────────────────────────
    if te_train is not None:
        te_tr_df = te_train.iloc[tr_idx].copy()
        te_va_df = te_train.iloc[va_idx].copy()
        te_tr_df['__target'] = y_tr
        te_va_enc, te_te_enc = target_encode_fold(te_tr_df, te_va_df, te_test, TE_COLS, '__target')
        # Append TE columns to feature matrices
        X_tr   = np.hstack([X_tr,              te_train.iloc[tr_idx].apply(lambda r: r.map(lambda _: 0), axis=1).values.reshape(len(tr_idx),-1)])
        # Correct approach: manually build TE features
        te_tr_vals = np.zeros((len(tr_idx), len(TE_COLS)))
        te_va_vals = te_va_enc.values
        te_te_vals = te_te_enc.values
        # TE for training fold (use fold-internal TE to avoid leakage)
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        for icol, col in enumerate(TE_COLS):
            te_tr_col = np.zeros(len(tr_idx))
            te_tr_sub = te_train.iloc[tr_idx][col].values
            y_tr_sub  = y_tr
            for itr, iva in inner_kf.split(te_tr_sub):
                g_mean = y_tr_sub[itr].mean()
                df_tmp = pd.DataFrame({'col': te_tr_sub[itr], 'y': y_tr_sub[itr]})
                stats  = df_tmp.groupby('col')['y'].agg(['mean','count'])
                smooth = (stats['mean']*stats['count'] + g_mean*10) / (stats['count'] + 10)
                te_tr_col[iva] = pd.Series(te_tr_sub[iva]).map(smooth).fillna(g_mean).values
            te_tr_vals[:, icol] = te_tr_col
        X_tr = np.hstack([X_train[tr_idx], te_tr_vals])
        X_va = np.hstack([X_train[va_idx], te_va_vals])
        X_te_fold = np.hstack([X_test, te_te_vals])

    models = make_models()
    for i, (model, name) in enumerate(zip(models, MODEL_NAMES)):
        model.fit(X_tr, y_tr)
        va_pred = model.predict(X_va)
        te_pred = model.predict(X_te_fold)
        oof_preds[va_idx, i] = va_pred
        test_preds[:, i] += te_pred / N_FOLDS
        fold_rmsle = rmsle(y_va, va_pred)
        print(f"  {name:<6} fold RMSLE = {fold_rmsle:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# OOF scores
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("OOF Results:")
oof_rmsle = []
for i, name in enumerate(MODEL_NAMES):
    score = rmsle(y_train, oof_preds[:, i])
    mae   = np.mean(np.abs(np.expm1(oof_preds[:,i]) - np.expm1(y_train)))
    oof_rmsle.append(score)
    print(f"  {name:<6} OOF RMSLE={score:.5f}  MAE=${mae:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Stacking — Ridge meta-learner + XGB meta-learner
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 3: Stacking")

# Ridge meta-learner
best_ridge_meta = (999, None)
for a in [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]:
    s = cv_rmsle(Ridge(alpha=a), oof_preds, y_train, KFold(n_splits=5, shuffle=True, random_state=SEED))
    if s < best_ridge_meta[0]: best_ridge_meta = (s, a)
ridge_meta = Ridge(alpha=best_ridge_meta[1])
ridge_meta.fit(oof_preds, y_train)
stack_oof_ridge  = ridge_meta.predict(oof_preds)
stack_test_ridge = ridge_meta.predict(test_preds)
print(f"  Ridge meta  a={best_ridge_meta[1]}  OOF RMSLE={rmsle(y_train, stack_oof_ridge):.5f}")

# XGBoost meta-learner
xgb_meta_params = dict(n_estimators=300, learning_rate=0.05, max_depth=3,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.01, reg_lambda=1.0, random_state=SEED,
                        tree_method='hist', device='cpu', verbosity=0)
xgb_meta = xgb.XGBRegressor(**xgb_meta_params)
xgb_meta_cv = cv_rmsle(xgb_meta, oof_preds, y_train, KFold(n_splits=5, shuffle=True, random_state=SEED))
xgb_meta.fit(oof_preds, y_train)
stack_oof_xgb   = xgb_meta.predict(oof_preds)
stack_test_xgb  = xgb_meta.predict(test_preds)
print(f"  XGB  meta   OOF RMSLE={rmsle(y_train, stack_oof_xgb):.5f}")

# Choose best meta-learner
if rmsle(y_train, stack_oof_ridge) <= rmsle(y_train, stack_oof_xgb):
    stack_oof  = stack_oof_ridge
    stack_test = stack_test_ridge
    print("  Using Ridge meta-learner")
else:
    stack_oof  = stack_oof_xgb
    stack_test = stack_test_xgb
    print("  Using XGB meta-learner")

# Weighted blend
weights     = np.array([1.0 / s for s in oof_rmsle])
weights    /= weights.sum()
blend_oof   = oof_preds  @ weights
blend_test  = test_preds @ weights
print(f"\nWeighted blend OOF RMSLE = {rmsle(y_train, blend_oof):.5f}")
print("Blend weights:")
for name, w in zip(MODEL_NAMES, weights):
    print(f"  {name:<6}: {w:.3f}")

# Sweep stack/blend ratio
best_ratio = (999, 0.5)
for alpha in np.arange(0.4, 0.85, 0.05):
    combined = alpha * stack_oof + (1-alpha) * blend_oof
    s = rmsle(y_train, combined)
    if s < best_ratio[0]: best_ratio = (s, alpha)

stack_w = best_ratio[1]
final_oof  = stack_w * stack_oof  + (1-stack_w) * blend_oof
final_test = stack_w * stack_test + (1-stack_w) * blend_test
print(f"\nBest stack ratio: {stack_w:.2f} stack + {1-stack_w:.2f} blend")
print(f"FINAL OOF RMSLE = {rmsle(y_train, final_oof):.5f}")

# Save first submission
sub1 = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(final_test)})
sub1.to_csv('submission_v4.csv', index=False)
print(f"\nSubmission v4 saved: {len(sub1)} rows  mean=${sub1.SalePrice.mean():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: Pseudo-labelling (one round)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 4: Pseudo-labelling")

pseudo_preds = np.expm1(final_test)
# Use all test predictions as pseudo-labels (conservative — single round)
pseudo_y = np.log1p(pseudo_preds)

# Combine train + pseudo-labeled test
X_pseudo = np.vstack([X_train, X_test])
y_pseudo = np.concatenate([y_train, pseudo_y])

print(f"  Combined dataset: {X_pseudo.shape[0]} rows (train={len(X_train)}, pseudo={len(X_test)})")

# OOF on original training only, test predictions from pseudo-trained models
oof_preds2  = np.zeros((len(X_train), N_MODELS))
test_preds2 = np.zeros((len(X_test),  N_MODELS))

kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED+1)
for fold, (tr_idx, va_idx) in enumerate(kf2.split(X_train)):
    print(f"\n  Pseudo Fold {fold+1}/{N_FOLDS}")
    # Training fold: original training subset + ALL pseudo-labeled test data
    X_tr_ps = np.vstack([X_train[tr_idx], X_test])
    y_tr_ps = np.concatenate([y_train[tr_idx], pseudo_y])
    X_va_ps = X_train[va_idx]
    y_va_ps = y_train[va_idx]

    models2 = make_models()
    for i, (model, name) in enumerate(zip(models2, MODEL_NAMES)):
        model.fit(X_tr_ps, y_tr_ps)
        va_pred = model.predict(X_va_ps)
        te_pred = model.predict(X_test)
        oof_preds2[va_idx, i] = va_pred
        test_preds2[:, i] += te_pred / N_FOLDS
        print(f"    {name:<6} fold RMSLE = {rmsle(y_va_ps, va_pred):.5f}")

print(f"\nPseudo-label OOF Results:")
oof_rmsle2 = []
for i, name in enumerate(MODEL_NAMES):
    score = rmsle(y_train, oof_preds2[:, i])
    oof_rmsle2.append(score)
    print(f"  {name:<6} OOF RMSLE={score:.5f}")

# Meta-learner on pseudo-trained OOF
ridge_meta2 = Ridge(alpha=best_ridge_meta[1])
ridge_meta2.fit(oof_preds2, y_train)
stack_test2 = ridge_meta2.predict(test_preds2)
stack_oof2  = ridge_meta2.predict(oof_preds2)

weights2   = np.array([1.0/s for s in oof_rmsle2]); weights2 /= weights2.sum()
blend_test2 = test_preds2 @ weights2
blend_oof2  = oof_preds2  @ weights2

best_ratio2 = (999, 0.5)
for alpha in np.arange(0.4, 0.85, 0.05):
    s = rmsle(y_train, alpha*stack_oof2 + (1-alpha)*blend_oof2)
    if s < best_ratio2[0]: best_ratio2 = (s, alpha)

final_oof2  = best_ratio2[1]*stack_oof2  + (1-best_ratio2[1])*blend_oof2
final_test2 = best_ratio2[1]*stack_test2 + (1-best_ratio2[1])*blend_test2
print(f"\nPseudo FINAL OOF RMSLE = {rmsle(y_train, final_oof2):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final submission: blend v4 + pseudo-label round
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 5: Final submission")

# If pseudo-labelling improved OOF, use 50/50 blend with original
oof_v4     = rmsle(y_train, final_oof)
oof_pseudo = rmsle(y_train, final_oof2)
print(f"  OOF v4 (no pseudo):     {oof_v4:.5f}")
print(f"  OOF v4 (with pseudo):   {oof_pseudo:.5f}")

if oof_pseudo < oof_v4:
    # Blend: 50% no-pseudo + 50% pseudo
    final_combined = 0.5 * final_test + 0.5 * final_test2
    final_oof_comb = 0.5 * final_oof  + 0.5 * final_oof2
    print(f"  Combined OOF:           {rmsle(y_train, final_oof_comb):.5f}")
    print("  Using pseudo-labelled blend")
else:
    final_combined = final_test
    print("  Pseudo-labelling did not improve OOF — using original v4 only")

sub_final = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(final_combined)})
sub_final.to_csv('submission_v4_final.csv', index=False)

print(f"\nFINAL submission saved: submission_v4_final.csv")
print(f"  Rows: {len(sub_final)}")
print(f"  Mean: ${sub_final.SalePrice.mean():,.0f}")
print(f"  Min:  ${sub_final.SalePrice.min():,.0f}")
print(f"  Max:  ${sub_final.SalePrice.max():,.0f}")
print(f"\nFINAL OOF RMSLE: {rmsle(y_train, final_oof):.5f}")
print("DONE — submit submission_v4_final.csv to Kaggle")
