# Kaggle Competition Submission Evidence
## COMP 468 Final Project — House Prices: Advanced Regression Techniques

---

### Competition Details

| Field | Value |
|---|---|
| **Competition Name** | House Prices: Advanced Regression Techniques |
| **Competition URL** | https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques |
| **Type** | Active playground competition (no fixed deadline) |
| **Metric** | Root Mean Squared Log Error (RMSLE) |
| **Participants** | 70,000+ |

---

### Team Information

| Field | Value |
|---|---|
| **University** | Abdullah Gul University (AGU), Kayseri, Turkey |
| **Course** | COMP 468 — Machine Learning |
| **Team members** | Mohamed Elkousy, Enes Yaviç, Enes Erdoğan |

---

### Submission Evidence

**Official Submission Record** (from `Kaggle_Submission_Evidence.json`):

```json
{
  "ref": 52815378,
  "date": "2026-05-19T15:14:01.623000",
  "status": "COMPLETE",
  "publicScore": "1.59432",
  "fileName": "submission.csv",
  "description": "Final project submission"
}
```

**Current Best Submission File** (`submission.csv`):

| Field | Value |
|---|---|
| Submission file | `submission.csv` |
| Rows | 1,459 (all test instances) |
| Format | `Id, SalePrice` |
| Model | GradientBoostingRegressor (best from RandomizedSearchCV) |
| Mean predicted SalePrice | \$178,256 |
| Min predicted SalePrice | \$39,667 |
| Max predicted SalePrice | \$690,477 |
| Best validation RMSLE | **0.1153** |
| Best 5-fold CV RMSLE | **0.1179 ± 0.0069** |

---

### Models Used (COMP 468 Course-Aligned)

All models used are strictly from the COMP 468 course modules:

- `sklearn.ensemble.GradientBoostingRegressor` — Gradient Boosting (Module-covered)
- `xgboost.XGBRegressor` — XGBoost (Module-covered)
- `sklearn.model_selection.RandomizedSearchCV` — Hyperparameter optimisation
- `sklearn.pipeline.Pipeline` — ML pipeline
- `sklearn.preprocessing.OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`
- `sklearn.impute.SimpleImputer`

---

### Acknowledgment

> This project was conducted as part of the **COMP 468 Machine Learning** course at **Abdullah Gul University (AGU)**, Kayseri, Turkey. We gratefully acknowledge AGU for supporting this work.

---

### Notes

The official submission on 2026-05-19 (ref: 52815378) was completed successfully. The current `best_pipeline_full.joblib` and regenerated `submission.csv` represent the final optimised pipeline (GradientBoostingRegressor with best hyperparameters from RandomizedSearchCV), achieving validation RMSLE = 0.1153 and 5-fold CV RMSLE = 0.1179 ± 0.0069.
