"""
Improved House Price Prediction Pipeline
- Advanced feature engineering (TotalSF, bathroom counts, age features, etc.)
- LotFrontage imputed by neighborhood median
- Skewness correction for numeric features (BoxCox on training stats only)
- Weighted blend of GradientBoosting + XGBoost predictions
- RandomizedSearchCV (n_iter=60, cv=5) for tuning both models
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

test_ids = test['Id']

# ── Remove outliers (De Cock recommendation) ─────────────────────────────────
outlier_mask = (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300_000)
print(f"Outliers removed: {outlier_mask.sum()}")
train = train[~outlier_mask].reset_index(drop=True)

y = np.log1p(train['SalePrice'].values)
all_data = pd.concat([train.drop('SalePrice', axis=1), test], axis=0).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()

    # Structural NaN means "feature absent", not missing
    none_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
                 'GarageType','GarageFinish','GarageQual','GarageCond',
                 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'MasVnrType']
    zero_cols = ['GarageYrBlt','GarageArea','GarageCars',
                 'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                 'BsmtFullBath','BsmtHalfBath','MasVnrArea']
    existing_none = [c for c in none_cols if c in df.columns]
    existing_zero = [c for c in zero_cols if c in df.columns]
    df[existing_none] = df[existing_none].fillna('None')
    df[existing_zero] = df[existing_zero].fillna(0)

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']    = (df['FullBath'] + 0.5*df['HalfBath']
                          + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
    df['TotalPorch']   = (df['OpenPorchSF'] + df['EnclosedPorch']
                          + df['3SsnPorch'] + df['ScreenPorch'])
    df['Age']          = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']   = df['YrSold'] - df['YearRemodAdd']
    df['HasPool']      = (df['PoolArea'] > 0).astype(int)
    df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt']      = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['IsNew']        = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    df['LivLotRatio']  = df['GrLivArea'] / df['LotArea'].clip(lower=1)

    # Typo correction: a handful of GarageYrBlt values exceed realistic range
    mask = df['GarageYrBlt'] > 2100
    df.loc[mask, 'GarageYrBlt'] = df.loc[mask, 'YearBuilt']

    return df

all_data = feature_engineering(all_data)
print(f"After feature engineering: {all_data.shape}")

# ── Skewness correction (fit on training slice only, apply to all) ────────────
n_train   = len(train)
train_slice = all_data.iloc[:n_train]
num_feats = train_slice.select_dtypes(include=['int64','float64']).columns
skewed_feats = (train_slice[num_feats]
                .apply(lambda x: skew(x.dropna()))
                .pipe(lambda s: s[s.abs() > 0.75])
                .index)
print(f"Skewed features corrected: {len(skewed_feats)}")
lam = 0.15
for feat in skewed_feats:
    all_data[feat] = boxcox1p(all_data[feat].clip(lower=0), lam)

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
ord_categories = [
    ORDINAL_CATS[c] if c in ORDINAL_CATS
    else sorted(train_fe[c].dropna().unique().tolist())
    for c in ORDINAL_COLS
]

X           = train_fe.drop(columns=['Id','SalePrice'], errors='ignore')
X_test_final = test_fe.drop(columns=['Id','SalePrice'], errors='ignore')

NUM_COLS = X.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_COLS = X.select_dtypes(include=['object']).columns.tolist()
CAT_NOM  = [c for c in CAT_COLS if c not in ORDINAL_COLS]

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

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmsle = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
    r2    = r2_score(y_true, y_pred)
    print(f"{name:20s}  RMSLE={rmsle:.4f}  MAE=${mae:,.0f}  R²={r2:.4f}")
    return rmsle

def run_search(pipe, grid, label):
    print(f"\n--- RandomizedSearchCV: {label} (n_iter=60) ---")
    rs = RandomizedSearchCV(pipe, grid, n_iter=60,
                            scoring='neg_root_mean_squared_error',
                            cv=5, random_state=RANDOM_STATE,
                            n_jobs=-1, verbose=1)
    rs.fit(X_tr, y_tr)
    print(f"Best CV RMSLE: {-rs.best_score_:.4f}")
    print(f"Best params:   {rs.best_params_}")
    return rs

# ── Models and search grids ───────────────────────────────────────────────────
pipe_gb = Pipeline([
    ('pre',   preprocessor),
    ('model', GradientBoostingRegressor(random_state=RANDOM_STATE)),
])
pipe_xgb = Pipeline([
    ('pre',   preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror',
                           random_state=RANDOM_STATE, verbosity=0)),
])

GB_GRID = {
    'model__n_estimators'    : [500, 800, 1000, 1500],
    'model__learning_rate'   : [0.01, 0.02, 0.05, 0.08],
    'model__max_depth'       : [3, 4, 5],
    'model__subsample'       : [0.7, 0.8, 0.85, 0.9],
    'model__min_samples_split': [2, 4, 6, 10],
    'model__max_features'    : ['sqrt', 0.5, 0.7],
    'model__loss'            : ['huber', 'squared_error'],
}
XGB_GRID = {
    'model__n_estimators'    : [500, 800, 1000, 1500],
    'model__learning_rate'   : [0.01, 0.02, 0.05, 0.08],
    'model__max_depth'       : [3, 4, 5],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'model__reg_alpha'       : [0, 0.05, 0.1, 0.5],
    'model__reg_lambda'      : [0.5, 1.0, 1.5, 2.0],
    'model__gamma'           : [0, 0.05, 0.1, 0.2],
    'model__subsample'       : [0.7, 0.8, 0.85, 0.9],
    'model__min_child_weight': [1, 3, 5, 7],
}

rs_gb  = run_search(pipe_gb,  GB_GRID,  'GradientBoosting')
rs_xgb = run_search(pipe_xgb, XGB_GRID, 'XGBoost')

# ── Validation ────────────────────────────────────────────────────────────────
p_gb  = rs_gb.predict(X_val)
p_xgb = rs_xgb.predict(X_val)

print("\n=== Validation Results ===")
gb_rmsle  = evaluate('GradientBoosting', y_val, p_gb)
xgb_rmsle = evaluate('XGBoost',          y_val, p_xgb)

w_gb  = xgb_rmsle / (gb_rmsle + xgb_rmsle)
w_xgb = gb_rmsle  / (gb_rmsle + xgb_rmsle)
p_blend = w_gb * p_gb + w_xgb * p_xgb
blend_rmsle = evaluate(f'Blend(GB={w_gb:.2f}/XGB={w_xgb:.2f})', y_val, p_blend)

# ── Refit on full training set ────────────────────────────────────────────────
print("\n--- Refitting on full training data ---")
best_gb  = rs_gb.best_estimator_
best_xgb = rs_xgb.best_estimator_
best_gb.fit(X, y)
best_xgb.fit(X, y)

# Weighted blend is the default; fall back to single best if blend doesn't win
candidates = [
    (blend_rmsle,  lambda: w_gb * best_gb.predict(X_test_final) + w_xgb * best_xgb.predict(X_test_final), 'Weighted blend'),
    (gb_rmsle,     lambda: best_gb.predict(X_test_final),  'GradientBoosting'),
    (xgb_rmsle,    lambda: best_xgb.predict(X_test_final), 'XGBoost'),
]
best_score, best_pred_fn, best_label = min(candidates, key=lambda t: t[0])
print(f"Using {best_label} (val RMSLE={best_score:.4f})")
pred_final = np.expm1(best_pred_fn())

# ── Save submission and pipelines ─────────────────────────────────────────────
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': pred_final})
submission.to_csv('submission.csv', index=False)
joblib.dump(best_gb,  'best_pipeline_gb.joblib')
joblib.dump(best_xgb, 'best_pipeline_xgb.joblib')

print(f"\nSubmission saved: {len(submission)} rows")
print(f"  mean: ${pred_final.mean():,.0f}")
print(f"  min:  ${pred_final.min():,.0f}")
print(f"  max:  ${pred_final.max():,.0f}")
print("\nDONE — submission.csv is ready for Kaggle upload.")
