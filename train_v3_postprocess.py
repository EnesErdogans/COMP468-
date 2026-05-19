"""
Post-processing: neighbourhood soft correction + power sweep
Modeli yeniden egitmeden submission'i hafifce duzelt.
"""
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error

train   = pd.read_csv('data/train.csv')
test    = pd.read_csv('data/test.csv')
sub_v3  = pd.read_csv('submission.csv')

# Outlier removal (egitimde yapildigi gibi)
train = train[~((train['GrLivArea']>4000)&(train['SalePrice']<300_000))].reset_index(drop=True)

# ── 1. Neighbourhood log-median from training ─────────────────────────────────
nbhd_log_med = train.groupby('Neighborhood')['SalePrice'].apply(
    lambda x: np.log1p(x).median()).to_dict()
global_log_med = np.log1p(train['SalePrice']).median()

test_log_nbhd = test['Neighborhood'].map(nbhd_log_med).fillna(global_log_med)
sub_log = np.log1p(sub_v3['SalePrice'].values)

# ── 2. Power sweep — tahminleri cok az kaydirarak en iyi OOF'u bul ─────────────
# OOF elimizde yok, sadece log-space distribution duzeltmesi yapabiliriz
# Neighbourhood blend: w * model_pred + (1-w) * nbhd_median
print("Neighbourhood blend test (w = model agirlik):")
for w in [0.97, 0.98, 0.99, 1.00]:
    blended_log = w * sub_log + (1-w) * test_log_nbhd.values
    blended     = np.expm1(blended_log)
    print(f"  w={w:.2f}  mean=${blended.mean():,.0f}  min=${blended.min():,.0f}  max=${blended.max():,.0f}")

# ── 3. Power adjustment sweep ─────────────────────────────────────────────────
# log(pred)^power etkisi — OOF yoksa guvenli test edemeyiz, ama aralik gosterelim
print("\nPower adjustment (pred^power):")
mean_log = sub_log.mean()
for power in [0.97, 0.98, 0.99, 1.00, 1.01, 1.02]:
    adj = np.expm1((sub_log - mean_log) * power + mean_log)
    print(f"  power={power:.2f}  mean=${adj.mean():,.0f}  min=${adj.min():,.0f}  max=${adj.max():,.0f}")

# ── 4. Sonuc: neighbourhood blend w=0.99 kaydet ──────────────────────────────
# Cok agresif degil — tahminlerin %99'u model, %1'i neighbourhood medyani
w = 0.99
blended_log = w * sub_log + (1-w) * test_log_nbhd.values
blended     = np.expm1(blended_log)

sub_out = pd.DataFrame({'Id': sub_v3['Id'], 'SalePrice': blended})
sub_out.to_csv('submission_pp.csv', index=False)
print(f"\nKaydedildi: submission_pp.csv")
print(f"  v3  mean=${sub_v3.SalePrice.mean():,.0f}")
print(f"  pp  mean=${blended.mean():,.0f}")
print(f"  fark: {(blended - sub_v3['SalePrice'].values).abs().mean():.0f} dolar ortalama")
