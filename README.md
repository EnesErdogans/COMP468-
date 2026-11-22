# COMP 468 Final Project — House Price Prediction

**Abdullah Gul University | Department of Computer Engineering | COMP 468 Machine Learning**

> Kaggle Competition: [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Result

| Metric | Value |
|---|---|
| Kaggle Public LB RMSLE | **0.11965** |
| Leaderboard Rank | **226 / 70,000+** |
| Percentile | **Top 0.3%** |
| OOF RMSLE (validation) | 0.10736 |

---

## Team

| Name |
|---|
| Mohamed Elkousy |
| Enes Yaviç |
| Enes Erdoğan |

---

## Project Overview

This project applies an iterative machine learning pipeline to predict residential house prices on the **Ames Housing dataset** (1,460 train / 1,459 test instances, 79 features). We developed three model generations, each improving upon the last:

| Version | Approach | Kaggle LB RMSLE | Rank |
|---|---|---|---|
| v1 | GB + XGB with RandomizedSearchCV | 0.12610 | ~1,238 |
| v2 | 4-model Out-of-Fold blend | 0.12171 | ~521 |
| **v3** | **7-model Optuna + Ridge stacking** | **0.11965** | **226** |

---

## Model Architecture (v3 Final)

```
Raw Data (79 features)
        │
        ▼
Feature Engineering (+12 composite features)
        │
        ▼
Preprocessing
  ├── 66 numeric  → BoxCox + RobustScaler
  ├── 20 ordinal  → OrdinalEncoder (explicit ordering)
  └── 24 nominal  → OneHotEncoder
        │
        ▼
5-Fold Out-of-Fold Training
  ├── GradientBoostingRegressor  (OOF RMSLE: 0.11098)
  ├── XGBRegressor               (OOF RMSLE: 0.11379)
  ├── LGBMRegressor              (OOF RMSLE: 0.11676)
  ├── CatBoostRegressor          (OOF RMSLE: 0.11178)
  ├── Ridge                      (OOF RMSLE: 0.11202)
  ├── Lasso                      (OOF RMSLE: 0.11076)
  └── ElasticNet                 (OOF RMSLE: 0.11091)
        │
        ▼
Ensemble
  ├── Ridge Meta-Learner (α=2) — stacked OOF RMSLE: 0.10722
  └── Inverse-RMSLE Weighted Blend — OOF RMSLE: 0.10768
        │
        ▼
Final: 60% Stack + 40% Blend → OOF RMSLE: 0.10736
```

Each tree-based model was tuned with **80 Optuna trials** (Bayesian TPE optimisation).

---

## File Structure

```
COMP468-/
│
├── data/
│   ├── train.csv                  # Kaggle training data (1,460 rows)
│   ├── test.csv                   # Kaggle test data (1,459 rows)
│   ├── data_description.txt       # Feature descriptions
│   └── sample_submission.csv      # Submission format
│
├── train_v1_baseline.py           # v1: GB + XGB, quick RandomizedSearchCV
├── train_v1_improved.py           # v1: cleaned pipeline, full preprocessing
├── train_v2_oof_blend.py          # v2: 4-model OOF blend (GB+XGB+LGBM+Ridge)
├── train_v3_attempt.py            # v3: first attempt (crashed on Unicode)
├── train_v3_final.py              # v3: FINAL model — 7-model Optuna + stacking
├── train_v3_postprocess.py        # v3: neighbourhood blend post-processing (experimental)
│
├── model_v1_pipeline.joblib       # Saved v1 full sklearn pipeline
├── model_v1_gb.joblib             # Saved v1 GradientBoosting model
├── model_v1_xgb.joblib            # Saved v1 XGBoost model
├── model_v3_meta_ridge.joblib     # Saved v3 Ridge meta-learner
│
├── submission.csv                 # Final Kaggle submission (v3, rank 226)
├── optuna_results.json            # Best Optuna hyperparameters (all models)
│
├── HousePrices_Final_Project.ipynb  # EDA + analysis notebook
├── Final_IEEE_Paper.md              # IEEE-format project paper
├── Midterm_HousePrices.pdf          # Project proposal (midterm)
│
├── Kaggle_Submission_Evidence.json  # Submission evidence
├── Kaggle_Submission_Evidence.md    # Submission summary
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Feature Engineering

12 domain-informed features engineered from the raw 79 columns:

| Feature | Description |
|---|---|
| `TotalSF` | Total square footage (basement + 1st + 2nd floor) |
| `TotalBath` | Weighted bathroom count (full + 0.5×half, all floors) |
| `TotalPorch` | Sum of all porch area columns |
| `Age` | Years since built at time of sale |
| `RemodelAge` | Years since last renovation |
| `GarageAge` | Years since garage built |
| `OverallQualSq` | Overall quality squared (captures non-linearity) |
| `TotalSF_Qual` | Total SF × OverallQual interaction |
| `GrLiv_Qual` | Above-grade living area × OverallQual |
| `NeighborhoodRank` | Neighbourhood ranked by median SalePrice |
| `MoSoldSin/Cos` | Cyclic encoding of sale month |

---

## Setup and Usage

**1. Install dependencies**

```bash
pip install -r requirements.txt
pip install lightgbm catboost optuna
```

**2. Run the final v3 model**

```bash
python train_v3_final.py
```

This will:
- Load and preprocess `data/train.csv` and `data/test.csv`
- Run Optuna hyperparameter search (80 trials per tree-based model)
- Train all 7 models with 5-fold OOF
- Fit Ridge meta-learner on OOF predictions
- Save `submission.csv` with 1,459 final predictions

> **Note:** Full training takes approximately 2–3 hours on CPU due to the Optuna search.

**3. Run an earlier version**

```bash
python train_v1_baseline.py   # ~5 min, GB + XGB only
python train_v2_oof_blend.py  # ~30 min, 4-model OOF
```

---

## Key Results

**OOF RMSLE by Model (v3)**

| Model | OOF RMSLE | MAE |
|---|---|---|
| GradientBoosting | 0.11098 | $13,471 |
| CatBoost | 0.11178 | $13,540 |
| Ridge | 0.11202 | $13,786 |
| ElasticNet | 0.11091 | $13,740 |
| Lasso | 0.11076 | $13,777 |
| XGBoost | 0.11379 | $13,892 |
| LightGBM | 0.11676 | $14,516 |
| **Ridge Stack (meta)** | **0.10722** | — |
| **Final Ensemble** | **0.10736** | — |

**Final Submission**

- Mean predicted SalePrice: **$178,556**
- Min: $44,388 | Max: $700,601
- Kaggle LB RMSLE: **0.11965** (rank 226)

---

## Dependencies

```
pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, optuna, joblib, scipy, matplotlib, seaborn
```

Python 3.10+ recommended.

---

## References

Key papers used in this project:

- De Cock (2011) — Ames Housing dataset
- Friedman (2001) — Gradient Boosting Machines
- Chen & Guestrin (2016) — XGBoost
- Ke et al. (2017) — LightGBM
- Prokhorenkova et al. (2018) — CatBoost
- Akiba et al. (2019) — Optuna
- Wolpert (1992) — Stacked Generalisation
- Bergstra & Bengio (2012) — Random Search for Hyperparameters

Full reference list in [Final_IEEE_Paper.md](Final_IEEE_Paper.md).

---

## Acknowledgment

This project was conducted as part of the **COMP 468 Machine Learning** course at **Abdullah Gul University (AGU)**, Kayseri, Turkey, May 2026.
