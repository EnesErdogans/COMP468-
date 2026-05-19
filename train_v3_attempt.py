"""
House Price Prediction - BEST Model
=====================================
Models  : GradientBoosting, XGBoost (Optuna), LightGBM (Optuna),
          CatBoost, Ridge, Lasso, ElasticNet
Tuning  : Optuna Bayesian optimisation (80 trials per tree model)
CV      : 5-fold OOF for base models → Ridge meta-learner (stacking)
Features: 50+ engineered features including quality×size interactions,
          neighbourhood target rank, cyclic time encoding, and more
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_FOLDS      = 5
N_OPTUNA     = 80      # Optuna trials per tree model
np.random.seed(RANDOM_STATE)

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}  Test: {test.shape}", flush=True)

test_ids = test['Id']
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300_000))].reset_index(drop=True)
y = np.log1p(train['SalePrice'].values)

# ── Neighbourhood target rank (from training data only) ───────────────────────
nbhd_rank = (train.groupby('Neighborhood')['SalePrice']
             .median().rank().to_dict())

# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(df, nbhd_rank_map):
    df = df.copy()

    # Convert MSSubClass to categorical (it's a type code, not a number)
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Structural NaN → meaningful string/zero
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

    # ── Area composites ───────────────────────────────────────────────────────
    df['TotalSF']        = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']      = (df['FullBath'] + 0.5*df['HalfBath']
                            + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
    df['TotalPorch']     = (df['OpenPorchSF'] + df['EnclosedPorch']
                            + df['3SsnPorch'] + df['ScreenPorch'])
    df['TotalOutdoorSF'] = (df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']
                            + df['3SsnPorch'] + df['ScreenPorch'] + df['PoolArea'])

    # ── Age / time features ───────────────────────────────────────────────────
    df['Age']          = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']   = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge']    = np.where(df['GarageYrBlt'] > 0,
                                  df['YrSold'] - df['GarageYrBlt'], 0)
    df['IsRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    # Cyclic month encoding
    df['MoSold_sin']   = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSold_cos']   = np.cos(2 * np.pi * df['MoSold'] / 12)

    # ── Binary flags ─────────────────────────────────────────────────────────
    df['HasPool']      = (df['PoolArea']    > 0).astype(int)
    df['HasGarage']    = (df['GarageArea']  > 0).astype(int)
    df['HasBsmt']      = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces']  > 0).astype(int)
    df['HasMasVnr']    = (df['MasVnrArea']  > 0).astype(int)
    df['Is2ndFloor']   = (df['2ndFlrSF']    > 0).astype(int)
    df['IsNew']        = (df['YrSold'] == df['YearBuilt']).astype(int)

    # ── Quality × area interactions (most predictive) ─────────────────────────
    df['OverallQual_sq'] = df['OverallQual'] ** 2
    df['OverallQual_cb'] = df['OverallQual'] ** 3
    df['OverallScore']   = df['OverallQual'] * df['OverallCond']
    df['TotalSF_x_Qual'] = df['TotalSF']     * df['OverallQual']
    df['GrLiv_x_Qual']   = df['GrLivArea']   * df['OverallQual']
    df['Garage_x_Qual']  = df['GarageArea']  * df['OverallQual']
    df['Bsmt_x_Qual']    = df['TotalBsmtSF'] * df['OverallQual']
    df['Bath_x_Qual']    = df['TotalBath']   * df['OverallQual']

    # ── Ratio / density features ──────────────────────────────────────────────
    df['LivLotRatio']    = df['GrLivArea']  / df['LotArea'].clip(lower=1)
    df['RoomSize']       = df['GrLivArea']  / df['TotRmsAbvGrd'].clip(lower=1)
    df['BsmtRatio']      = df['TotalBsmtSF']/ df['TotalSF'].clip(lower=1)
    df['BsmtFinRatio']   = df['BsmtFinSF1'] / df['TotalBsmtSF'].clip(lower=1)
    df['GarageScore']    = df['GarageCars'] * df['GarageArea']

    # ── Neighbourhood quality rank (target-aware, computed from training) ──────
    df['Nbhd_Rank'] = df['Neighborhood'].map(nbhd_rank_map).fillna(
        np.median(list(nbhd_rank_map.values())))

    return df

all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0).reset_index(drop=True)
all_data = feature_engineering(all_data, nbhd_rank)
n_train  = len(train)
print(f"After feature engineering: {all_data.shape}", flush=True)

# ── Skewness correction ───────────────────────────────────────────────────────
train_slice = all_data.iloc[:n_train]
num_feats   = train_slice.select_dtypes(include=['int64','float64']).columns
skewed      = (train_slice[num_feats]
               .apply(lambda x: skew(x.dropna()))
               .pipe(lambda s: s[s.abs() > 0.75]).index)
print(f"Skewed features corrected: {len(skewed)}", flush=True)
for feat in skewed:
    all_data[feat] = boxcox1p(all_data[feat].clip(lower=0), 0.15)

train_fe = all_data.iloc[:n_train]
test_fe  = all_data.iloc[n_train:]

# ── Preprocessing ──────────────────────────────────────────────────────────────
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
    'Functional':   ['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
    'Utilities':    ['ELO','NoSeWa','NoSewr','AllPub'],
    'LandSlope':    ['Sev','Mod','Gtl'],
    'LotShape':     ['IR3','IR2','IR1','Reg'],
    'PavedDrive':   ['N','P','Y'],
    'CentralAir':   ['N','Y'],
}

ORDINAL_COLS = [c for c in ORDINAL_CATS if c in train_fe.columns]
ord_categories = [ORDINAL_CATS[c] for c in ORDINAL_COLS]

X      = train_fe.drop(columns=['Id','SalePrice'], errors='ignore')
X_test = test_fe.drop(columns=['Id','SalePrice'], errors='ignore')

NUM_COLS = X.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_NOM  = [c for c in X.select_dtypes(include='object').columns
            if c not in ORDINAL_COLS]

print(f"Features: {len(NUM_COLS)} numeric | {len(ORDINAL_COLS)} ordinal | {len(CAT_NOM)} nominal",
      flush=True)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  RobustScaler())]),                          NUM_COLS),
    ('ord', OrdinalEncoder(categories=ord_categories,
                           handle_unknown='use_encoded_value',
                           unknown_value=-1),                             ORDINAL_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore',
                                            sparse_output=False))]),      CAT_NOM),
], remainder='drop')

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ────────────────────────────────────────────────────────────────────────────
# PHASE 1 – Optuna hyperparameter search
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("PHASE 1: Optuna hyperparameter search", flush=True)
print(f"{'='*60}", flush=True)

def cv_rmsle(pipe):
    scores = cross_val_score(pipe, X, y, cv=kfold,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean()

# ── XGBoost Optuna ────────────────────────────────────────────────────────────
print(f"\nOptimising XGBoost ({N_OPTUNA} trials) ...", flush=True)

def obj_xgb(trial):
    p = {
        'n_estimators'     : trial.suggest_int  ('n_estimators',      500,  5000),
        'learning_rate'    : trial.suggest_float('learning_rate',    0.003, 0.08,  log=True),
        'max_depth'        : trial.suggest_int  ('max_depth',           3,    6),
        'subsample'        : trial.suggest_float('subsample',         0.55,  0.95),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',  0.45,  0.90),
        'reg_alpha'        : trial.suggest_float('reg_alpha',         1e-3,  5.0,  log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda',        0.2,   5.0,  log=True),
        'min_child_weight' : trial.suggest_int  ('min_child_weight',    1,    8),
        'gamma'            : trial.suggest_float('gamma',             0.0,   0.4),
    }
    pipe = Pipeline([('pre', clone(preprocessor)),
                     ('model', XGBRegressor(**p, objective='reg:squarederror',
                                            random_state=RANDOM_STATE, verbosity=0))])
    return cv_rmsle(pipe)

study_xgb = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(obj_xgb, n_trials=N_OPTUNA, show_progress_bar=False)
best_xgb = study_xgb.best_params
print(f"  XGB best CV RMSLE : {study_xgb.best_value:.5f}", flush=True)
print(f"  XGB best params   : {best_xgb}", flush=True)

# ── LightGBM Optuna ───────────────────────────────────────────────────────────
print(f"\nOptimising LightGBM ({N_OPTUNA} trials) ...", flush=True)

def obj_lgbm(trial):
    p = {
        'n_estimators'     : trial.suggest_int  ('n_estimators',      500,  5000),
        'learning_rate'    : trial.suggest_float('learning_rate',    0.003, 0.06,  log=True),
        'num_leaves'       : trial.suggest_int  ('num_leaves',         20,    80),
        'subsample'        : trial.suggest_float('subsample',         0.55,  0.95),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',  0.45,  0.90),
        'reg_alpha'        : trial.suggest_float('reg_alpha',         1e-3,  5.0,  log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda',        1e-3,  5.0,  log=True),
        'min_child_samples': trial.suggest_int  ('min_child_samples',   5,   40),
        'max_depth'        : trial.suggest_int  ('max_depth',          -1,    8),
    }
    pipe = Pipeline([('pre', clone(preprocessor)),
                     ('model', LGBMRegressor(**p, random_state=RANDOM_STATE,
                                             n_jobs=-1, verbose=-1))])
    return cv_rmsle(pipe)

study_lgbm = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgbm.optimize(obj_lgbm, n_trials=N_OPTUNA, show_progress_bar=False)
best_lgbm = study_lgbm.best_params
print(f"  LGBM best CV RMSLE: {study_lgbm.best_value:.5f}", flush=True)
print(f"  LGBM best params  : {best_lgbm}", flush=True)

# ── GradientBoosting Optuna ───────────────────────────────────────────────────
print(f"\nOptimising GradientBoosting ({N_OPTUNA} trials) ...", flush=True)

def obj_gb(trial):
    p = {
        'n_estimators'    : trial.suggest_int  ('n_estimators',    300,  2000),
        'learning_rate'   : trial.suggest_float('learning_rate',  0.005, 0.08, log=True),
        'max_depth'       : trial.suggest_int  ('max_depth',         3,    5),
        'subsample'       : trial.suggest_float('subsample',       0.55,  0.95),
        'max_features'    : trial.suggest_float('max_features',    0.3,   0.9),
        'min_samples_split': trial.suggest_int ('min_samples_split', 2,   15),
        'loss'            : trial.suggest_categorical('loss', ['huber','squared_error']),
    }
    pipe = Pipeline([('pre', clone(preprocessor)),
                     ('model', GradientBoostingRegressor(**p,
                                                         random_state=RANDOM_STATE))])
    return cv_rmsle(pipe)

study_gb = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_gb.optimize(obj_gb, n_trials=N_OPTUNA, show_progress_bar=False)
best_gb_params = study_gb.best_params
print(f"  GB  best CV RMSLE : {study_gb.best_value:.5f}", flush=True)
print(f"  GB  best params   : {best_gb_params}", flush=True)

# ── Linear model alpha search ─────────────────────────────────────────────────
print(f"\nSearching Ridge / Lasso / ElasticNet alphas ...", flush=True)
best_ridge_a, best_lasso_a, best_en_a, best_en_l1 = 10.0, 0.0005, 0.001, 0.5
best_ridge_s = best_lasso_s = best_en_s = 9999

for a in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100, 200]:
    s = cv_rmsle(Pipeline([('pre', clone(preprocessor)),
                            ('model', Ridge(alpha=a))]))
    if s < best_ridge_s: best_ridge_s, best_ridge_a = s, a

for a in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]:
    s = cv_rmsle(Pipeline([('pre', clone(preprocessor)),
                            ('model', Lasso(alpha=a, max_iter=10000))]))
    if s < best_lasso_s: best_lasso_s, best_lasso_a = s, a

for a in [0.0005, 0.001, 0.005, 0.01]:
    for l1 in [0.2, 0.4, 0.6, 0.8]:
        s = cv_rmsle(Pipeline([('pre', clone(preprocessor)),
                                ('model', ElasticNet(alpha=a, l1_ratio=l1,
                                                     max_iter=10000))]))
        if s < best_en_s: best_en_s, best_en_a, best_en_l1 = s, a, l1

print(f"  Ridge   best: α={best_ridge_a}  CV={best_ridge_s:.5f}", flush=True)
print(f"  Lasso   best: α={best_lasso_a}  CV={best_lasso_s:.5f}", flush=True)
print(f"  EN      best: α={best_en_a} l1={best_en_l1}  CV={best_en_s:.5f}", flush=True)

# Save Optuna results
optuna_results = {
    'xgb_best_value': study_xgb.best_value, 'xgb_best_params': best_xgb,
    'lgbm_best_value': study_lgbm.best_value, 'lgbm_best_params': best_lgbm,
    'gb_best_value': study_gb.best_value, 'gb_best_params': best_gb_params,
    'ridge_alpha': best_ridge_a, 'lasso_alpha': best_lasso_a,
    'en_alpha': best_en_a, 'en_l1': best_en_l1,
}
with open('optuna_results.json', 'w') as f:
    json.dump(optuna_results, f, indent=2)
# ── CatBoost Optuna ───────────────────────────────────────────────────────────
print(f"\nOptimising CatBoost ({N_OPTUNA} trials) ...", flush=True)

def obj_cat(trial):
    p = {
        'iterations'        : trial.suggest_int  ('iterations',       300,  4000),
        'learning_rate'     : trial.suggest_float('learning_rate',  0.003, 0.08,  log=True),
        'depth'             : trial.suggest_int  ('depth',             4,    8),
        'l2_leaf_reg'       : trial.suggest_float('l2_leaf_reg',     0.5,  10.0,  log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength'   : trial.suggest_float('random_strength',  0.0,  1.0),
        'border_count'      : trial.suggest_int  ('border_count',      32,  254),
    }
    pipe = Pipeline([('pre', clone(preprocessor)),
                     ('model', CatBoostRegressor(**p, random_seed=RANDOM_STATE,
                                                 verbose=0, allow_writing_files=False))])
    return cv_rmsle(pipe)

study_cat = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_cat.optimize(obj_cat, n_trials=N_OPTUNA, show_progress_bar=False)
best_cat = study_cat.best_params
print(f"  CAT best CV RMSLE : {study_cat.best_value:.5f}", flush=True)
print(f"  CAT best params   : {best_cat}", flush=True)

optuna_results = {
    'xgb_best_value': study_xgb.best_value, 'xgb_best_params': best_xgb,
    'lgbm_best_value': study_lgbm.best_value, 'lgbm_best_params': best_lgbm,
    'gb_best_value': study_gb.best_value, 'gb_best_params': best_gb_params,
    'cat_best_value': study_cat.best_value, 'cat_best_params': best_cat,
    'ridge_alpha': best_ridge_a, 'lasso_alpha': best_lasso_a,
    'en_alpha': best_en_a, 'en_l1': best_en_l1,
}
print("\nOptuna results saved to optuna_results.json", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# PHASE 2 – 5-Fold OOF with all models
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("PHASE 2: 5-Fold OOF Training", flush=True)
print(f"{'='*60}", flush=True)

PIPELINES = {
    'GB'  : Pipeline([('pre', preprocessor), ('model', GradientBoostingRegressor(
                **best_gb_params, random_state=RANDOM_STATE))]),
    'XGB' : Pipeline([('pre', preprocessor), ('model', XGBRegressor(
                **best_xgb, objective='reg:squarederror',
                random_state=RANDOM_STATE, verbosity=0))]),
    'LGBM': Pipeline([('pre', preprocessor), ('model', LGBMRegressor(
                **best_lgbm, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))]),
    'CAT' : Pipeline([('pre', preprocessor), ('model', CatBoostRegressor(
                **best_cat, random_seed=RANDOM_STATE,
                verbose=0, allow_writing_files=False))]),
    'Ridge': Pipeline([('pre', preprocessor), ('model', Ridge(alpha=best_ridge_a))]),
    'Lasso': Pipeline([('pre', preprocessor), ('model', Lasso(alpha=best_lasso_a,
                                                               max_iter=10000))]),
    'EN'  : Pipeline([('pre', preprocessor), ('model', ElasticNet(
                alpha=best_en_a, l1_ratio=best_en_l1, max_iter=10000))]),
}

oof_preds = {n: np.zeros(n_train)      for n in PIPELINES}
tst_preds = {n: np.zeros(len(X_test))  for n in PIPELINES}

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X), 1):
    X_tr_f, X_val_f = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr_f, y_val_f = y[tr_idx], y[val_idx]
    print(f"\nFold {fold}/{N_FOLDS}  (train={len(tr_idx)}, val={len(val_idx)})",
          flush=True)

    for name, pipe in PIPELINES.items():
        p = clone(pipe)
        p.fit(X_tr_f, y_tr_f)
        oof_preds[name][val_idx]  = p.predict(X_val_f)
        tst_preds[name]          += p.predict(X_test) / N_FOLDS
        fold_rmsle = np.sqrt(mean_squared_error(y_val_f, oof_preds[name][val_idx]))
        print(f"  {name:5s} fold RMSLE = {fold_rmsle:.5f}", flush=True)

# ── OOF summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("OOF Results (Level-1 base models):", flush=True)
oof_scores = {}
for name in PIPELINES:
    rmsle = np.sqrt(mean_squared_error(y, oof_preds[name]))
    mae   = mean_absolute_error(np.expm1(y), np.expm1(oof_preds[name]))
    oof_scores[name] = rmsle
    print(f"  {name:5s}  OOF RMSLE={rmsle:.5f}  MAE=${mae:,.0f}", flush=True)

# ────────────────────────────────────────────────────────────────────────────
# PHASE 3 – Level-2 stacking (Ridge meta-learner)
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("PHASE 3: Level-2 Ridge Stacking", flush=True)
print(f"{'='*60}", flush=True)

meta_X_train = np.column_stack([oof_preds[n] for n in PIPELINES])
meta_X_test  = np.column_stack([tst_preds[n] for n in PIPELINES])

best_meta_s, best_meta_a = 9999, 1.0
for a in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]:
    from sklearn.model_selection import cross_val_score as cvs2
    meta_scores = cvs2(Ridge(alpha=a), meta_X_train, y,
                       cv=kfold, scoring='neg_root_mean_squared_error')
    s = -meta_scores.mean()
    if s < best_meta_s:
        best_meta_s, best_meta_a = s, a
print(f"  Meta-Ridge best α={best_meta_a}  CV RMSLE={best_meta_s:.5f}", flush=True)

meta_model = Ridge(alpha=best_meta_a)
meta_model.fit(meta_X_train, y)
stacked_test = meta_model.predict(meta_X_test)
stacked_oof  = meta_model.predict(meta_X_train)
stack_oof_rmsle = np.sqrt(mean_squared_error(y, stacked_oof))
print(f"  Stacked OOF RMSLE = {stack_oof_rmsle:.5f}", flush=True)

# ── Simple inverse-RMSLE weighted blend ──────────────────────────────────────
total_inv  = sum(1/s for s in oof_scores.values())
weights    = {n: (1/s)/total_inv for n, s in oof_scores.items()}
blend_test = sum(weights[n] * tst_preds[n] for n in PIPELINES)
blend_oof  = sum(weights[n] * oof_preds[n] for n in PIPELINES)
blend_rmsle = np.sqrt(mean_squared_error(y, blend_oof))
print(f"\n  Weighted blend OOF RMSLE = {blend_rmsle:.5f}", flush=True)

# ── Final: average of stack + blend ──────────────────────────────────────────
# Weight: choose whichever is better, or take the mean
if stack_oof_rmsle < blend_rmsle:
    # 60% stacking + 40% blend
    final_test_log = 0.6*stacked_test + 0.4*blend_test
    final_oof_log  = 0.6*stacked_oof  + 0.4*blend_oof
else:
    final_test_log = blend_test
    final_oof_log  = blend_oof

final_oof_rmsle = np.sqrt(mean_squared_error(y, final_oof_log))
print(f"  Final OOF RMSLE           = {final_oof_rmsle:.5f}", flush=True)

pred_final = np.expm1(final_test_log)

# ── Save ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({'Id': test_ids, 'SalePrice': pred_final})
sub.to_csv('submission.csv', index=False)

joblib.dump(meta_model, 'meta_model.joblib')
for name, pipe in PIPELINES.items():
    full_pipe = clone(pipe)
    full_pipe.fit(X, y)
    joblib.dump(full_pipe, f'final_pipe_{name}.joblib')

print(f"\nSubmission saved: {len(sub)} rows", flush=True)
print(f"  mean: ${pred_final.mean():,.0f}", flush=True)
print(f"  min:  ${pred_final.min():,.0f}", flush=True)
print(f"  max:  ${pred_final.max():,.0f}", flush=True)
print(f"\nFinal OOF RMSLE: {final_oof_rmsle:.5f}", flush=True)
print("DONE — submission.csv is ready.", flush=True)
