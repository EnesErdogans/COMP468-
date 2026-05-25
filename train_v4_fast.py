"""
train_v4_fast.py — v4 improvement without re-running Optuna
Uses saved v3 hyperparameters + extended feature engineering + HistGB + pseudo-labelling
Expected runtime: 40-70 minutes on CPU
"""
import numpy as np
import pandas as pd
import json, warnings
warnings.filterwarnings('ignore')

from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

SEED = 42
np.random.seed(SEED)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ─────────────────────────────────────────────────────────────────────────────
# Ordinal mappings (20 columns)
# ─────────────────────────────────────────────────────────────────────────────
QUAL_COLS  = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
              'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
BSMT_EXP   = {'NA':0,'No':1,'Mn':2,'Av':3,'Gd':4}
BSMT_FIN   = {'NA':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
GRG_FIN    = {'NA':0,'Unf':1,'RFn':2,'Fin':3}
FENCE_MAP  = {'NA':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}
SLOPE_MAP  = {'Gtl':1,'Mod':2,'Sev':3}
LOTSHP_MAP = {'IR3':1,'IR2':2,'IR1':3,'Reg':4}
PAVED_MAP  = {'N':0,'P':1,'Y':2}
UTIL_MAP   = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}
FUNC_MAP   = {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8}

def apply_ordinal(df):
    q_map = {v:i for i,v in enumerate(['NA','Po','Fa','TA','Gd','Ex'])}
    for col in QUAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('NA').map(q_map).fillna(0)
    for col in ['BsmtFinType1','BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].fillna('NA').map(BSMT_FIN).fillna(0)
    for col_map, col_name in [
        (BSMT_EXP,'BsmtExposure'),(GRG_FIN,'GarageFinish'),(FENCE_MAP,'Fence'),
        (SLOPE_MAP,'LandSlope'),(LOTSHP_MAP,'LotShape'),(PAVED_MAP,'PavedDrive'),
        (UTIL_MAP,'Utilities'),(FUNC_MAP,'Functional')
    ]:
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(list(col_map.keys())[0]).map(col_map).fillna(0)
    return df

NOMINAL_COLS = ['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig',
                'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                'Foundation','Heating','CentralAir','Electrical','GarageType',
                'MiscFeature','SaleType','SaleCondition']

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (v4 — extended)
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineer(df, nbhd_lotfront=None, nbhd_log_med=None, global_log_med=None):
    df = df.copy()
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Neighbourhood LotFrontage imputation
    if nbhd_lotfront is None:
        nbhd_lotfront = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()
    df['LotFrontage'] = df.apply(
        lambda r: nbhd_lotfront.get(r['Neighborhood'], df['LotFrontage'].median())
        if pd.isna(r['LotFrontage']) else r['LotFrontage'], axis=1)

    # Structural NAs
    fill_zero = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
                 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
    for c in fill_zero:
        if c in df.columns: df[c] = df[c].fillna(0)
    fill_mode = ['Electrical','KitchenQual','Exterior1st','Exterior2nd',
                 'SaleType','Functional','MSZoning','Utilities']
    for c in fill_mode:
        if c in df.columns: df[c] = df[c].fillna(df[c].mode()[0])

    # ── Area composites ───────────────────────────────────────────────────────
    df['TotalSF']       = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']     = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalPorch']    = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                           df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])
    df['TotalArea']     = df['TotalSF'] + df['GarageArea'] + df['TotalPorch']

    # ── Age features ──────────────────────────────────────────────────────────
    df['Age']           = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']    = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge']     = (df['YrSold'] - df['GarageYrBlt']).clip(lower=0)

    # ── Quality interactions ──────────────────────────────────────────────────
    df['OverallQualSq'] = df['OverallQual'] ** 2
    df['OverallQual3']  = df['OverallQual'] ** 3
    df['TotalSF_Qual']  = df['TotalSF']    * df['OverallQual']
    df['GrLiv_Qual']    = df['GrLivArea']  * df['OverallQual']
    df['TotalQuality']  = df['OverallQual'] + df['OverallCond']
    df['QualAge']       = df['OverallQual'] * (100 - df['Age'].clip(0, 100))
    df['GrLivAreaSq']   = df['GrLivArea']  ** 2
    df['TotalSFSq']     = df['TotalSF']    ** 2

    # ── Ratios ────────────────────────────────────────────────────────────────
    df['LivAreaRatio']  = df['GrLivArea']   / (df['TotalSF'].clip(lower=1))
    df['BsmtRatio']     = df['TotalBsmtSF'] / (df['TotalSF'].clip(lower=1))
    df['GarageRatio']   = df['GarageArea']  / (df['TotalSF'].clip(lower=1))
    df['LotRatio']      = df['LotFrontage'] / (df['LotArea'].clip(lower=1) ** 0.5)

    # ── Binary flags ──────────────────────────────────────────────────────────
    df['HasPool']       = (df['PoolArea'] > 0).astype(int)
    df['HasGarage']     = (df['GarageArea'] > 0).astype(int)
    df['Has2ndFloor']   = (df['2ndFlrSF'] > 0).astype(int)
    df['HasBsmt']       = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace']  = (df['Fireplaces'] > 0).astype(int)
    df['HasPorch']      = (df['TotalPorch'] > 0).astype(int)
    df['HasRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['IsNew']         = (df['YearBuilt'] == df['YrSold']).astype(int)
    df['HasMasVnr']     = (df['MasVnrArea'] > 0).astype(int)

    # ── Neighbourhood stats ───────────────────────────────────────────────────
    if nbhd_log_med is not None:
        df['NbhoodLogMed'] = df['Neighborhood'].map(nbhd_log_med).fillna(global_log_med)

    # ── Cyclic month ──────────────────────────────────────────────────────────
    df['MoSoldSin'] = np.sin(2 * np.pi * df['MoSold'] / 12)
    df['MoSoldCos'] = np.cos(2 * np.pi * df['MoSold'] / 12)

    return df, nbhd_lotfront

def target_encode_fold(tr_df, val_ser, tst_ser, col, target_vals, alpha=10):
    g_mean = target_vals.mean()
    df_tmp = pd.DataFrame({'col': tr_df[col].values, 'y': target_vals})
    stats  = df_tmp.groupby('col')['y'].agg(['mean','count'])
    smooth = (stats['mean']*stats['count'] + g_mean*alpha) / (stats['count'] + alpha)
    val_enc = val_ser.map(smooth).fillna(g_mean).values
    tst_enc = tst_ser.map(smooth).fillna(g_mean).values
    return val_enc, tst_enc

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
train_raw = pd.read_csv('data/train.csv')
test_raw  = pd.read_csv('data/test.csv')
sub_v3    = pd.read_csv('submission.csv')   # v3 predictions for pseudo-labelling

train_raw = train_raw[~((train_raw['GrLivArea']>4000)&(train_raw['SalePrice']<300_000))].reset_index(drop=True)
print(f"Train: {train_raw.shape}  Test: {test_raw.shape}")

# Neighbourhood stats from training
nbhd_log_med  = train_raw.groupby('Neighborhood')['SalePrice'].apply(lambda x: np.log1p(x).median()).to_dict()
global_log_med = np.log1p(train_raw['SalePrice']).median()
y_train = np.log1p(train_raw['SalePrice'].values)

# Feature engineering
train_fe, nbhd_lotfront = feature_engineer(train_raw, nbhd_log_med=nbhd_log_med, global_log_med=global_log_med)
test_fe,  _             = feature_engineer(test_raw, nbhd_lotfront=nbhd_lotfront,
                                           nbhd_log_med=nbhd_log_med, global_log_med=global_log_med)

# Keep Neighborhood raw for target encoding inside OOF
te_train_nbhd = train_fe['Neighborhood'].copy()
te_test_nbhd  = test_fe['Neighborhood'].copy()
te_train_msz  = train_fe['MSZoning'].copy()
te_test_msz   = test_fe['MSZoning'].copy()

# Ordinal encoding
train_fe = apply_ordinal(train_fe)
test_fe  = apply_ordinal(test_fe)

# OHE nominal columns
all_ohe = [c for c in NOMINAL_COLS if c in train_fe.columns]
remain  = [c for c in train_fe.select_dtypes(include='object').columns if c not in all_ohe]
all_ohe_full = list(set(all_ohe + remain))

train_ohe = pd.get_dummies(train_fe.drop(['Id','SalePrice'], axis=1, errors='ignore'),
                           columns=all_ohe_full, dummy_na=False)
test_ids  = test_fe['Id'].values
test_ohe  = pd.get_dummies(test_fe.drop(['Id','SalePrice'], axis=1, errors='ignore'),
                           columns=all_ohe_full, dummy_na=False)
train_ohe, test_ohe = train_ohe.align(test_ohe, join='left', axis=1, fill_value=0)

# Fill NaN, BoxCox, RobustScaler
num_cols = train_ohe.select_dtypes(include='number').columns.tolist()
train_ohe[num_cols] = train_ohe[num_cols].fillna(train_ohe[num_cols].median())
test_ohe[num_cols]  = test_ohe[num_cols].fillna(train_ohe[num_cols].median())

BOXCOX_LAMBDA = 0.15
skewed = train_ohe[num_cols].apply(lambda x: skew(x.dropna()))
skewed = skewed[abs(skewed) > 0.75].index.tolist()
for col in skewed:
    if (train_ohe[col] >= 0).all() and (test_ohe[col] >= 0).all():
        train_ohe[col] = boxcox1p(train_ohe[col], BOXCOX_LAMBDA)
        test_ohe[col]  = boxcox1p(test_ohe[col],  BOXCOX_LAMBDA)
print(f"Skewed corrected: {len(skewed)}")

scaler = RobustScaler()
train_ohe[num_cols] = scaler.fit_transform(train_ohe[num_cols])
test_ohe[num_cols]  = scaler.transform(test_ohe[num_cols])

X_base  = train_ohe.values.astype(np.float32)
Xt_base = test_ohe.values.astype(np.float32)
print(f"Base feature matrix: {X_base.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Load saved v3 hyperparameters (no Optuna re-run needed)
# ─────────────────────────────────────────────────────────────────────────────
with open('optuna_results.json') as f:
    p = json.load(f)

xgb_p  = {**p['xgb'],  'tree_method':'hist','device':'cpu','random_state':SEED,'verbosity':0}
lgbm_p = {**p['lgbm'], 'random_state':SEED,'verbosity':-1}
gb_p   = {**p['gb'],   'random_state':SEED}
cat_p  = {**p['cat'],  'random_seed':SEED,'verbose':0}

# Good HistGB defaults (tuned by hand based on dataset size)
hist_p = dict(max_iter=2000, learning_rate=0.018, max_depth=4, max_leaf_nodes=31,
              min_samples_leaf=15, l2_regularization=0.1, max_features=0.7,
              random_state=SEED)

ridge_a = p['ridge_a']
lasso_a = p['lasso_a']
en_a, en_l1 = p['en_a'], p['en_l1']

MODEL_NAMES = ['GB','XGB','LGBM','CAT','HIST','Ridge','Lasso','EN']
N_MODELS    = len(MODEL_NAMES)
N_FOLDS     = 5

def make_models():
    return [
        GradientBoostingRegressor(**gb_p),
        xgb.XGBRegressor(**xgb_p),
        lgb.LGBMRegressor(**lgbm_p),
        cb.CatBoostRegressor(**cat_p),
        HistGradientBoostingRegressor(**hist_p),
        Ridge(alpha=ridge_a),
        Lasso(alpha=lasso_a, max_iter=10000),
        ElasticNet(alpha=en_a, l1_ratio=en_l1, max_iter=10000),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# OOF training (v4 features + within-fold target encoding)
# ─────────────────────────────────────────────────────────────────────────────
def run_oof(X_base, Xt_base, y, te_tr_nbhd, te_tst_nbhd, te_tr_msz, te_tst_msz,
            label="OOF"):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros((len(X_base), N_MODELS))
    test = np.zeros((len(Xt_base), N_MODELS))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_base)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}  (train={len(tr_idx)}, val={len(va_idx)})")
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Within-fold target encoding for Neighborhood and MSZoning
        va_nbhd, tst_nbhd = target_encode_fold(
            pd.DataFrame({'Neighborhood': te_tr_nbhd.iloc[tr_idx].values}),
            te_tr_nbhd.iloc[va_idx], te_tst_nbhd, 'Neighborhood', y_tr)
        va_msz, tst_msz = target_encode_fold(
            pd.DataFrame({'MSZoning': te_tr_msz.iloc[tr_idx].values}),
            te_tr_msz.iloc[va_idx], te_tst_msz, 'MSZoning', y_tr)

        # Build TE for training fold (inner 3-fold to prevent leakage)
        tr_nbhd_vals = np.zeros(len(tr_idx))
        tr_msz_vals  = np.zeros(len(tr_idx))
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        for itr, iva in inner_kf.split(tr_idx):
            for arr_out, ser_col, te_col in [
                (tr_nbhd_vals, te_tr_nbhd.iloc[tr_idx].values, 'Neighborhood'),
                (tr_msz_vals,  te_tr_msz.iloc[tr_idx].values,  'MSZoning'),
            ]:
                sub_tr = pd.DataFrame({'col': ser_col[itr], 'y': y_tr[itr]})
                stats  = sub_tr.groupby('col')['y'].agg(['mean','count'])
                g_mean = y_tr[itr].mean()
                smooth = (stats['mean']*stats['count'] + g_mean*10) / (stats['count'] + 10)
                arr_out[iva] = pd.Series(ser_col[iva]).map(smooth).fillna(g_mean).values

        # Append TE columns
        X_tr  = np.column_stack([X_base[tr_idx],  tr_nbhd_vals, tr_msz_vals])
        X_va  = np.column_stack([X_base[va_idx],  va_nbhd, va_msz])
        X_tst = np.column_stack([Xt_base,          tst_nbhd, tst_msz])

        models = make_models()
        for i, (model, name) in enumerate(zip(models, MODEL_NAMES)):
            model.fit(X_tr, y_tr)
            va_pred  = model.predict(X_va)
            tst_pred = model.predict(X_tst)
            oof[va_idx, i]  = va_pred
            test[:, i]     += tst_pred / N_FOLDS
            fold_s = rmsle(y_va, va_pred)
            print(f"    {name:<6} fold RMSLE = {fold_s:.5f}")

    print(f"\n{label} Results:")
    oof_scores = []
    for i, name in enumerate(MODEL_NAMES):
        s = rmsle(y, oof[:, i])
        mae = np.mean(np.abs(np.expm1(oof[:,i]) - np.expm1(y)))
        oof_scores.append(s)
        print(f"  {name:<6} OOF RMSLE={s:.5f}  MAE=${mae:,.0f}")
    return oof, test, oof_scores

def make_ensemble(oof, test_p, y, oof_scores):
    # Ridge meta-learner
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best = (999, None)
    for a in [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50]:
        scores = []
        for tr, va in kf5.split(oof):
            m = Ridge(alpha=a).fit(oof[tr], y[tr])
            scores.append(rmsle(y[va], m.predict(oof[va])))
        if np.mean(scores) < best[0]: best = (np.mean(scores), a)
    meta = Ridge(alpha=best[1]).fit(oof, y)
    stack_oof  = meta.predict(oof)
    stack_test = meta.predict(test_p)
    print(f"  Ridge meta a={best[1]}  stack OOF RMSLE={rmsle(y, stack_oof):.5f}")

    # Weighted blend
    weights    = np.array([1.0/s for s in oof_scores]); weights /= weights.sum()
    blend_oof  = oof    @ weights
    blend_test = test_p @ weights

    # Find best stack/blend ratio
    best_ratio = (999, 0.6)
    for alpha in np.arange(0.3, 0.85, 0.05):
        s = rmsle(y, alpha*stack_oof + (1-alpha)*blend_oof)
        if s < best_ratio[0]: best_ratio = (s, alpha)
    sw = best_ratio[1]
    final_oof  = sw*stack_oof  + (1-sw)*blend_oof
    final_test = sw*stack_test + (1-sw)*blend_test
    print(f"  Best ratio: {sw:.2f} stack + {1-sw:.2f} blend  FINAL OOF={rmsle(y, final_oof):.5f}")
    return final_oof, final_test

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 1: v4 OOF (v4 features, v3 hyperparams)")
print(f"{'='*60}")
oof1, test1, scores1 = run_oof(X_base, Xt_base, y_train,
                                te_train_nbhd, te_test_nbhd,
                                te_train_msz,  te_test_msz, "v4 OOF")
final_oof1, final_test1 = make_ensemble(oof1, test1, y_train, scores1)

sub1 = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(final_test1)})
sub1.to_csv('submission_v4.csv', index=False)
print(f"Saved submission_v4.csv  mean=${sub1.SalePrice.mean():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 2: Pseudo-labelling round")
print(f"{'='*60}")

# Use v4 predictions as pseudo-labels (more accurate than v3)
pseudo_y = np.log1p(np.expm1(final_test1))   # == final_test1

X_ps  = np.vstack([X_base,  Xt_base])
Xt_ps = Xt_base.copy()

te_nbhd_ps = pd.concat([te_train_nbhd, te_test_nbhd], ignore_index=True)
te_msz_ps  = pd.concat([te_train_msz,  te_test_msz],  ignore_index=True)
y_ps       = np.concatenate([y_train, pseudo_y])

print(f"Combined dataset: {X_ps.shape[0]} rows")

oof2, test2, scores2 = run_oof(X_ps, Xt_ps, y_ps,
                                te_nbhd_ps, te_test_nbhd,
                                te_msz_ps,  te_test_msz, "Pseudo OOF")
final_oof2_full, final_test2 = make_ensemble(oof2, test2, y_ps, scores2)

# OOF on training portion only for fair comparison
oof2_train = oof2[:len(y_train)]
oof2_score = rmsle(y_train, oof2_train @ np.array([1/s for s in scores2[:N_MODELS]]
                                           / np.sum([1/s for s in scores2[:N_MODELS]])))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PHASE 3: Final submission")
print(f"{'='*60}")

oof1_s = rmsle(y_train, final_oof1)
oof2_s = rmsle(y_train, oof2_train @ np.array([1/s for s in scores2]
                                     / np.sum([1/s for s in scores2])))

print(f"  v4 (no pseudo)  OOF RMSLE: {oof1_s:.5f}")
print(f"  v4 (w/ pseudo)  OOF RMSLE: {oof2_s:.5f}")

# Blend v4 + pseudo predictions
final_combined = 0.5 * final_test1 + 0.5 * final_test2
sub_final = pd.DataFrame({'Id': test_ids, 'SalePrice': np.expm1(final_combined)})
sub_final.to_csv('submission_v4_final.csv', index=False)

print(f"\nFINAL submission: submission_v4_final.csv")
print(f"  Rows: {len(sub_final)}")
print(f"  Mean: ${sub_final.SalePrice.mean():,.0f}")
print(f"  Min:  ${sub_final.SalePrice.min():,.0f}")
print(f"  Max:  ${sub_final.SalePrice.max():,.0f}")
print(f"\nDONE - submit submission_v4_final.csv to Kaggle")
