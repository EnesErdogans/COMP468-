"""Quick training run — good feature engineering, fast search (n_iter=10, cv=3)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
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

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
test_ids = test['Id']

train = train[~((train['GrLivArea']>4000)&(train['SalePrice']<300_000))].reset_index(drop=True)
y = np.log1p(train['SalePrice'].values)
all_data = pd.concat([train.drop('SalePrice',axis=1), test], axis=0).reset_index(drop=True)

def feature_engineering(df):
    df = df.copy()
    none_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType',
                 'GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond',
                 'BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']
    zero_cols = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
                 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
    df[[c for c in none_cols if c in df.columns]] = df[[c for c in none_cols if c in df.columns]].fillna('None')
    df[[c for c in zero_cols if c in df.columns]] = df[[c for c in zero_cols if c in df.columns]].fillna(0)
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath']    = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['TotalPorch']   = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['Age']          = df['YrSold'] - df['YearBuilt']
    df['RemodelAge']   = df['YrSold'] - df['YearRemodAdd']
    df['HasPool']      = (df['PoolArea']>0).astype(int)
    df['HasGarage']    = (df['GarageArea']>0).astype(int)
    df['HasBsmt']      = (df['TotalBsmtSF']>0).astype(int)
    df['HasFireplace'] = (df['Fireplaces']>0).astype(int)
    df['IsNew']        = (df['YrSold']==df['YearBuilt']).astype(int)
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    df['LivLotRatio']  = df['GrLivArea'] / df['LotArea'].clip(lower=1)
    mask = df['GarageYrBlt'] > 2100
    df.loc[mask,'GarageYrBlt'] = df.loc[mask,'YearBuilt']
    return df

all_data = feature_engineering(all_data)
n_train = len(train)
train_slice = all_data.iloc[:n_train]
num_feats = train_slice.select_dtypes(include=['int64','float64']).columns
skewed_feats = train_slice[num_feats].apply(lambda x: skew(x.dropna())).pipe(lambda s: s[s.abs()>0.75]).index
for feat in skewed_feats:
    all_data[feat] = boxcox1p(all_data[feat].clip(lower=0), 0.15)

train_fe = all_data.iloc[:n_train]
test_fe  = all_data.iloc[n_train:]

ORDINAL_CATS = {
    'ExterQual':['Po','Fa','TA','Gd','Ex'],'ExterCond':['Po','Fa','TA','Gd','Ex'],
    'BsmtQual':['None','Po','Fa','TA','Gd','Ex'],'BsmtCond':['None','Po','Fa','TA','Gd','Ex'],
    'HeatingQC':['Po','Fa','TA','Gd','Ex'],'KitchenQual':['Po','Fa','TA','Gd','Ex'],
    'FireplaceQu':['None','Po','Fa','TA','Gd','Ex'],'GarageQual':['None','Po','Fa','TA','Gd','Ex'],
    'GarageCond':['None','Po','Fa','TA','Gd','Ex'],'BsmtExposure':['None','No','Mn','Av','Gd'],
    'BsmtFinType1':['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'BsmtFinType2':['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
    'GarageFinish':['None','Unf','RFn','Fin'],'Fence':['None','MnWw','GdWo','MnPrv','GdPrv'],
    'Utilities':['ELO','NoSeWa','NoSewr','AllPub'],'LandSlope':['Sev','Mod','Gtl'],
    'LotShape':['IR3','IR2','IR1','Reg'],
}
ORDINAL_COLS = [c for c in ORDINAL_CATS if c in train_fe.columns]
ord_categories = [ORDINAL_CATS[c] for c in ORDINAL_COLS]

X = train_fe.drop(columns=['Id','SalePrice'], errors='ignore')
X_test = test_fe.drop(columns=['Id','SalePrice'], errors='ignore')
NUM_COLS = X.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_NOM  = [c for c in X.select_dtypes(include='object').columns if c not in ORDINAL_COLS]

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp',SimpleImputer(strategy='median')),('sc',RobustScaler())]), NUM_COLS),
    ('ord', OrdinalEncoder(categories=ord_categories,handle_unknown='use_encoded_value',unknown_value=-1), ORDINAL_COLS),
    ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]), CAT_NOM),
], remainder='drop')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Features: {X_tr.shape[1]}, Train: {len(X_tr)}, Val: {len(X_val)}")

pipe_gb  = Pipeline([('pre',preprocessor),('model',GradientBoostingRegressor(random_state=RANDOM_STATE))])
pipe_xgb = Pipeline([('pre',preprocessor),('model',XGBRegressor(objective='reg:squarederror',random_state=RANDOM_STATE,verbosity=0))])

GB_GRID = {
    'model__n_estimators':[500,800,1000],'model__learning_rate':[0.02,0.05,0.08],
    'model__max_depth':[3,4,5],'model__subsample':[0.75,0.85,0.9],
    'model__min_samples_split':[2,5,10],'model__max_features':['sqrt',0.6],
    'model__loss':['huber','squared_error'],
}
XGB_GRID = {
    'model__n_estimators':[500,800,1000],'model__learning_rate':[0.02,0.05,0.08],
    'model__max_depth':[3,4,5],'model__colsample_bytree':[0.6,0.75,0.9],
    'model__reg_alpha':[0,0.05,0.1],'model__reg_lambda':[0.8,1.0,1.5],
    'model__gamma':[0,0.05,0.1],'model__subsample':[0.75,0.85,0.9],
    'model__min_child_weight':[1,3,5],
}

print("Training GradientBoosting (n_iter=20, cv=5)...")
rs_gb = RandomizedSearchCV(pipe_gb, GB_GRID, n_iter=20, scoring='neg_root_mean_squared_error',
                           cv=5, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
rs_gb.fit(X_tr, y_tr)
print(f"  GB best CV RMSLE: {-rs_gb.best_score_:.4f}  params: {rs_gb.best_params_}")

print("Training XGBoost (n_iter=20, cv=5)...")
rs_xgb = RandomizedSearchCV(pipe_xgb, XGB_GRID, n_iter=20, scoring='neg_root_mean_squared_error',
                             cv=5, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
rs_xgb.fit(X_tr, y_tr)
print(f"  XGB best CV RMSLE: {-rs_xgb.best_score_:.4f}  params: {rs_xgb.best_params_}")

# Validation
p_gb  = rs_gb.predict(X_val)
p_xgb = rs_xgb.predict(X_val)
gb_rmsle  = np.sqrt(mean_squared_error(y_val, p_gb))
xgb_rmsle = np.sqrt(mean_squared_error(y_val, p_xgb))
gb_mae  = mean_absolute_error(np.expm1(y_val), np.expm1(p_gb))
xgb_mae = mean_absolute_error(np.expm1(y_val), np.expm1(p_xgb))
print(f"\nGB   val => RMSLE={gb_rmsle:.4f}  MAE=${gb_mae:,.0f}  R2={r2_score(y_val,p_gb):.4f}")
print(f"XGB  val => RMSLE={xgb_rmsle:.4f}  MAE=${xgb_mae:,.0f}  R2={r2_score(y_val,p_xgb):.4f}")

w_gb  = xgb_rmsle / (gb_rmsle + xgb_rmsle)
w_xgb = gb_rmsle  / (gb_rmsle + xgb_rmsle)
p_blend = w_gb*p_gb + w_xgb*p_xgb
blend_rmsle = np.sqrt(mean_squared_error(y_val, p_blend))
print(f"Blend val => RMSLE={blend_rmsle:.4f}  (w_GB={w_gb:.2f}, w_XGB={w_xgb:.2f})")

# Refit on full data
print("\nRefitting on full training data...")
best_gb  = rs_gb.best_estimator_
best_xgb = rs_xgb.best_estimator_
best_gb.fit(X, y)
best_xgb.fit(X, y)

p_final_log = w_gb*best_gb.predict(X_test) + w_xgb*best_xgb.predict(X_test)
pred_final  = np.expm1(p_final_log)

sub = pd.DataFrame({'Id': test_ids, 'SalePrice': pred_final})
sub.to_csv('submission.csv', index=False)
joblib.dump(best_gb,  'best_pipeline_gb.joblib')
joblib.dump(best_xgb, 'best_pipeline_xgb.joblib')

print(f"\nSaved submission.csv ({len(sub)} rows)")
print(f"  mean=${pred_final.mean():,.0f}  min=${pred_final.min():,.0f}  max=${pred_final.max():,.0f}")
print(f"\nBest val RMSLE: {min(gb_rmsle, xgb_rmsle, blend_rmsle):.4f}")
print("DONE")
