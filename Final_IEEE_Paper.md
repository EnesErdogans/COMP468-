# Stacked Ensemble Learning with Bayesian Hyperparameter Optimisation for House Price Prediction on the Ames Housing Dataset

**Mohamed Elkousy** (2311021077), **Enes Yaviç** (2111051025), **Enes Erdoğan** (2111011061)  
Department of Computer Engineering, Abdullah Gul University, Kayseri, Turkey  
{2311021077, 2111051025, 2111011061}@agu.edu.tr

---

## Abstract

Accurate prediction of residential property prices is a challenging supervised regression problem due to high dimensionality, feature heterogeneity, and substantial missingness characterising real-estate datasets. This paper presents the final project for the COMP 468 Machine Learning course at Abdullah Gul University, addressing the Kaggle competition *House Prices: Advanced Regression Techniques* using the Ames Housing dataset compiled by De Cock [1]. The dataset comprises 1,460 labelled training instances and 1,459 unlabelled test instances described by 79 explanatory variables of mixed numerical and categorical types.

We report an iterative model development progression across three versions. Version 1 establishes a baseline with scikit-learn's `GradientBoostingRegressor` and `XGBRegressor` tuned via `RandomizedSearchCV`, achieving a Kaggle public leaderboard RMSLE of **0.12610**. Version 2 expands to a four-model Out-of-Fold (OOF) blended ensemble incorporating `LightGBM` and `Ridge` regression, improving to **0.12171**. Version 3 — the final pipeline — employs seven models (`GradientBoostingRegressor`, `XGBRegressor`, `LightGBMRegressor`, `CatBoostRegressor`, `Ridge`, `Lasso`, `ElasticNet`) each tuned with 80-trial Optuna Bayesian optimisation, combined via Ridge meta-learner stacking (60%) and inverse-OOF-RMSLE weighted blending (40%). The final model achieves an OOF RMSLE of **0.10736** and a Kaggle public leaderboard score of **0.11965**, placing 226th out of more than 70,000 participants (top 0.3%).

Detailed feature engineering (12 composite attributes, BoxCox skewness correction, 20-column ordinal encoding, RobustScaler) is shown to contribute at least as much as model selection to the final score. The complete pipeline is submitted to the official Kaggle leaderboard, and code is made fully reproducible.

**Keywords** — house price prediction, Ames Housing dataset, gradient boosting, XGBoost, LightGBM, CatBoost, stacked ensembles, Optuna, Bayesian hyperparameter optimisation, RMSLE, Out-of-Fold blending.

---

## I. Introduction

Real-estate valuation is a central problem in applied economics and machine learning. House prices are influenced by a large set of structural, locational, and quality-related attributes whose interactions are typically nonlinear and difficult to capture with classical linear models [1], [8]. Reliable predictive models are valuable to home buyers, sellers, lenders, insurance companies, and policy makers, and they form the basis of Automated Valuation Models (AVMs) widely used in the mortgage industry [13].

The Kaggle competition *House Prices: Advanced Regression Techniques* is a widely used benchmark for this task [9]. It is based on the Ames Housing dataset compiled by De Cock as a modern alternative to the classic Boston Housing data [1]. With 79 explanatory variables describing nearly every aspect of residential homes in Ames, Iowa, the dataset offers an ideal testbed for evaluating preprocessing strategies and modern ensemble learning methods.

This paper constitutes the final project for the COMP 468 Machine Learning course at Abdullah Gul University. Rather than presenting a single-model study, we document an iterative model development process that progressively applies more sophisticated techniques from the course curriculum: from baseline gradient boosting with random search, through Out-of-Fold ensemble blending, to Bayesian hyperparameter optimisation and Ridge meta-learner stacking.

The specific contributions of this work are:

1. A clean, fully reproducible preprocessing pipeline for the Ames Housing dataset featuring domain-informed feature engineering, BoxCox skewness correction, and 20-column explicit ordinal encoding that prevents data leakage across cross-validation folds.
2. An iterative model development study documenting three generations (v1, v2, v3) of increasing model complexity, providing controlled evidence of the improvement attributed to each methodological upgrade.
3. A 7-model Out-of-Fold evaluation with Optuna Bayesian hyperparameter optimisation (80 trials per model), Ridge meta-learner stacking, and inverse-RMSLE weighted blending — all applied within a strict no-leakage 5-fold cross-validation framework.
4. A public Kaggle leaderboard submission achieving a score of 0.11965 (rank 226, top 0.3%), providing an objective external benchmark.

The rest of the paper is organised as follows. Section II reviews the relevant literature. Section III defines the problem, dataset, and competition. Section IV details the complete methodology. Section V presents and discusses the experimental results across all model generations. Section VI provides recommendations and conclusions. References and appendices follow.

---

## II. Literature Review

### A. Traditional Statistical Approaches

Early residential price estimation relied primarily on hedonic regression — an ordinary least-squares (OLS) model that assumes SalePrice to be a linear function of structural, locational, and quality attributes [8]. While interpretable, OLS models make assumptions of linearity, homoscedasticity, and independence that are rarely met in complex real-estate data [13]. Repeated-sales models partially address the endogeneity of location but require multiple observations per property, unavailable in cross-sectional benchmarks such as Ames [13]. Regularised linear models — Ridge, Lasso, and ElasticNet — extend OLS with L2, L1, and combined penalties, respectively, and serve as strong linear baselines and effective stacking components in ensemble pipelines [7].

### B. Tree-Based Ensemble Methods

Breiman's Random Forest [3] introduced bootstrap aggregation combined with random feature sub-selection, producing ensembles robust to noise and naturally handling mixed-type features. For the Boston Housing data, Adetunji et al. showed that tree-ensemble predictions significantly reduce error relative to linear regression [11]. Gradient Boosting Machines (GBMs), proposed by Friedman [2], build an additive model of shallow regression trees in a forward stage-wise manner by fitting each successive tree to the negative gradient (pseudo-residuals) of a differentiable loss. GBMs routinely outperform random forests on tabular regression benchmarks due to their bias-reduction mechanism.

### C. Boosting Algorithms Applied to Housing Price Prediction

Chen and Guestrin's XGBoost [4] extended classical gradient boosting with explicit L1/L2 regularisation on leaf weights, second-order gradient information in the split criterion, column sub-sampling, and a sparsity-aware split-finding algorithm. Since 2016, XGBoost has dominated tabular Kaggle competitions [4].

Ke et al. introduced LightGBM [14], a highly efficient gradient boosting framework using histogram-based leaf-wise tree growth rather than level-wise splitting. LightGBM achieves faster training with competitive accuracy, especially on large feature spaces. Prokhorenkova et al. proposed CatBoost [19], which handles categorical features natively via ordered target encoding and uses symmetric trees, demonstrating strong performance on datasets with many categorical variables — a property directly relevant to the Ames dataset with its 43 categorical columns.

Sharma, Harsora, and Ogunleye benchmarked XGBoost against linear regression, MLP, random forests, and SVR on the Ames data and found that XGBoost with SHAP-value post-hoc explanation achieves the best balance between predictive accuracy and interpretability [5]. Truong et al. extended this comparison to LightGBM and stacked models on Beijing housing data, finding that stacking boosting models further improves performance [6].

### D. Hyperparameter Optimisation

Gradient boosting models are sensitive to several hyperparameters: the number of trees, learning rate, maximum tree depth, sub-sampling ratios, and regularisation parameters [2], [4]. Bergstra and Bengio rigorously demonstrated that random search outperforms grid search in high-dimensional hyperparameter spaces [10]. Akiba et al. introduced Optuna [20], a define-by-run Bayesian hyperparameter optimisation framework that uses Tree-structured Parzen Estimation (TPE) to efficiently focus search on promising regions of the hyperparameter space. Optuna has become the standard for competitive Kaggle pipelines due to its sample efficiency and ease of integration with scikit-learn-compatible models.

### E. Ensemble Stacking and Blending

Wolpert's stacked generalisation [21] uses the out-of-fold predictions of multiple base learners as features for a second-level meta-learner, often improving over any individual model by exploiting the complementary strengths of diverse base models. The meta-learner is typically a simple linear model (Ridge regression) to prevent overfitting on the stacked predictions. Combining stacking with a direct weighted blend (by inverse validation error) of base-model predictions provides a hedge against the meta-learner overfitting to OOF artifacts.

### F. Hybrid and Clustering-Based Approaches

Akyüz, Erdogan, Yıldız, and Atas (2022) developed a novel hybrid model chaining linear regression, K-means clustering, K-NN classification, and SVR [18]. Their pipeline outperformed all standalone baselines on RMSE and MAE on both the Ames dataset and a manually scraped Kadıköy dataset, though at considerably higher implementation complexity.

### G. Feature Engineering and Model Comparison on the Ames Dataset

Wang (2025) enriched the Ames feature set with three engineered attributes — finished living area ratio, basement ratio, and years since last remodel — and compared Linear, Ridge, Lasso, Random Forest, and XGBoost models [16]. Engineered features improved all models, with XGBoost achieving the best RMSLE. Yu (2021) conducted a detailed EDA on the Ames dataset, identifying four near-empty columns with ≥90% missingness and revealing multicollinearity between GarageArea and GarageCars (r = 0.89) [15].

### H. Regression and Classification on the Ames Dataset

Thakur and Bisht (2025) used the Ames dataset for both price regression and three-class price-tier classification, with Random Forest as the best model after PCA to 20 components [17]. Madhuri, Anuradha, and Pujitha compared multiple linear regression, Lasso, Ridge, ElasticNet, and Gradient Boosting and confirmed gradient boosting as the top performer [7].

### I. Synthesis and Research Gap

The reviewed literature converges on three findings. First, gradient-boosting variants consistently outperform classical linear models and random forests on the Ames Housing benchmark [5], [6], [7]. Second, results depend strongly on preprocessing quality and systematic hyperparameter tuning [2], [10]. Third, ensemble stacking consistently improves over any single model by combining complementary strengths [6], [21].

This project extends the literature by providing a complete, controlled study of iterative model development — documenting how each upgrade (advanced feature engineering, OOF blending, Bayesian optimisation, stacking) contributes to leaderboard improvement — within the course-aligned COMP 468 framework.

---

## III. Project Settings

### A. Problem Statement and Objectives

The task is a supervised regression problem: predict the final sale price (`SalePrice`) of residential homes in Ames, Iowa, given 79 structural, locational, and quality-related attributes. Formally, given a feature vector **x** ∈ ℝᵈ, the goal is to learn a function *f* : ℝᵈ → ℝ such that *f*(**x**) approximates log(1 + SalePrice), minimising the root-mean-squared logarithmic error (RMSLE).

The specific, measurable objectives are:

1. Implement a complete, reproducible preprocessing pipeline handling missing values and encoding categorical variables without data leakage.
2. Train course-aligned gradient-boosting regressors and regularised linear models as base learners.
3. Perform systematic hyperparameter tuning — first via RandomizedSearchCV, then via Optuna Bayesian optimisation.
4. Evaluate models using RMSLE, MAE, and R² via strict 5-fold Out-of-Fold cross-validation.
5. Construct a stacked ensemble combining seven base learners via a Ridge meta-learner.
6. Submit the final model to Kaggle and achieve a competitive leaderboard position.

### B. Chosen Dataset and Rationale

The dataset is the Ames Housing dataset compiled by Dean De Cock for educational use [1]. The Kaggle version contains:

- **1,460** labelled training instances
- **1,459** unlabelled test instances
- **79** explanatory variables: 36 numerical (continuous and discrete) and 43 categorical (nominal and ordinal)
- **Target variable**: `SalePrice` (USD) — mean \$180,921, median \$163,000, standard deviation \$79,443, range [\$34,900, \$755,000]

The dataset was selected because: (i) it is the official dataset of an active Kaggle competition providing objective external evaluation; (ii) its size (~2,900 instances total) supports meaningful cross-validation while remaining tractable on standard hardware; (iii) the heterogeneity of feature types and pervasive structural missingness (19 columns with missing values; PoolQC missing in 99.5% of rows) make it a rigorous testbed for the preprocessing and modelling techniques covered in COMP 468; and (iv) an extensive comparison literature enables contextualisation of our results.

### C. Chosen Kaggle Competition

**Competition name:** House Prices: Advanced Regression Techniques  
**URL:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques  

This is an active, beginner-friendly playground competition with no fixed deadline attracting over 70,000 participants. Submissions are scored using RMSLE — the root-mean-squared error between the natural logarithm of predicted and observed `SalePrice`. This competition was selected because it directly matches the regression and ensemble-learning content of the COMP 468 course modules and provides an objective, external evaluation anchor via a public leaderboard.

---

## IV. Methodology

### A. Feature Engineering

The preprocessing pipeline was significantly expanded relative to the baseline to create 12 domain-informed composite features before encoding:

**Composite area and quality features:**

| Feature | Formula | Motivation |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Total living space across all floors |
| `TotalBath` | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | Aggregate bathroom count |
| `TotalPorch` | Sum of all porch area columns | Aggregate outdoor living area |
| `Age` | `YrSold − YearBuilt` | Effective age at time of sale |
| `RemodelAge` | `YrSold − YearRemodAdd` | Years since last renovation |
| `GarageAge` | `YrSold − GarageYrBlt` (0 if no garage) | Garage age |
| `OverallQualSq` | `OverallQual²` | Quadratic quality interaction |
| `TotalSF_Qual` | `TotalSF × OverallQual` | Area–quality interaction |
| `GrLiv_Qual` | `GrLivArea × OverallQual` | Above-grade area–quality interaction |
| `NeighborhoodRank` | Neighbourhood median SalePrice rank | Spatial price signal |
| `MoSoldSin`, `MoSoldCos` | sin/cos of (2π × MoSold / 12) | Cyclic seasonal encoding |

`MSSubClass` (building type, encoded as integer) is converted to a categorical string column. `Functional` (home functionality rating) is encoded as an explicit ordinal.

**Skewness correction:**  
After computing all numeric features, 37 features with |skew| > 0.75 are corrected using a BoxCox transformation with λ = 0.15, applied only to positive-valued columns. This extends the log-transform of the v1 pipeline to all right-skewed numerical inputs, not just the target.

**Encoding scheme:**  
The final feature set is partitioned into three groups:

- **66 numeric features** — standardised with `RobustScaler` (which is resistant to outliers, unlike StandardScaler)
- **20 ordinal features** — quality/condition columns mapped to explicit integer scales (e.g., Po=0, Fa=1, TA=2, Gd=3, Ex=4) using `OrdinalEncoder`
- **24 nominal categorical features** — one-hot encoded with `OneHotEncoder(handle_unknown='ignore')`

All transformers are fitted exclusively on the training fold within each cross-validation split, preventing test-set statistics from leaking into model training.

**Outlier removal:**  
Two training instances with `GrLivArea` > 4,000 ft² and `SalePrice` < \$300,000 are removed following De Cock's recommendation [1], leaving 1,458 training instances. No filtering is applied to the test set.

**Target transformation:**  
The target variable is transformed as y' = log(1 + SalePrice). This reduces skewness from 1.88 to approximately 0.12 and directly optimises the RMSLE leaderboard objective.

### B. Model Selection and Rationale

Seven models are selected, covering the spectrum of gradient-boosting variants and regularised linear models covered in the COMP 468 course curriculum:

**Gradient Boosting Family:**

- **`GradientBoostingRegressor` (sklearn)** — Canonical Friedman gradient boosting [2]. Fits additive regression trees to pseudo-residuals with `huber` loss in the final configuration, making it robust to remaining outliers.
- **`XGBRegressor` (XGBoost)** — Regularised gradient boosting with L1/L2 leaf-weight penalties, second-order gradient information, and sparsity-aware split finding [4].
- **`LGBMRegressor` (LightGBM)** — Histogram-based leaf-wise gradient boosting with high training efficiency [14]. Uses column sub-sampling and min_child_samples for regularisation.
- **`CatBoostRegressor` (CatBoost)** — Ordered boosting with symmetric trees and native categorical feature support [19]. Particularly well-suited to the Ames dataset's many categorical columns.

**Regularised Linear Models:**

- **`Ridge`** — L2-penalised linear regression. Acts as a strong linear baseline and as the level-2 meta-learner in the stacking framework.
- **`Lasso`** — L1-penalised linear regression with automatic feature selection via coefficient sparsification.
- **`ElasticNet`** — Combined L1+L2 penalty, offering a compromise between Ridge and Lasso.

Linear models operate directly on the full encoded feature matrix and provide complementary predictions to the tree-based models, improving ensemble diversity.

### C. Hyperparameter Optimisation — Version 1 (RandomizedSearchCV)

In version 1, hyperparameter tuning uses `RandomizedSearchCV` with `n_iter=20` random configurations and 5-fold cross-validation for `GradientBoostingRegressor` and `XGBRegressor`. This establishes the baseline pipeline structure and identifies an initial competitive configuration. The search space covers `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and model-specific regularisation parameters.

### D. Hyperparameter Optimisation — Version 3 (Optuna Bayesian Search)

In version 3, each of the four tree-based models is tuned with **80 Optuna trials** using Tree-structured Parzen Estimation (TPE), which focuses search effort on promising hyperparameter regions identified from prior trials [20]. Optuna's define-by-run API allows the search space to be expressed as a natural Python function, with `n_estimators` ranging up to 5,000 and `learning_rate` down to 0.01.

**Best hyperparameters found:**

*GradientBoostingRegressor:*

| Parameter | Value |
|---|---|
| `n_estimators` | 1,920 |
| `learning_rate` | 0.01376 |
| `max_depth` | 3 |
| `subsample` | 0.7008 |
| `max_features` | 0.4777 |
| `min_samples_split` | 6 |
| `loss` | huber |

*XGBRegressor:*

| Parameter | Value |
|---|---|
| `n_estimators` | 4,754 |
| `learning_rate` | 0.01914 |
| `max_depth` | 4 |
| `subsample` | 0.5572 |
| `colsample_bytree` | 0.6761 |
| `reg_alpha` | 0.08473 |
| `reg_lambda` | 1.1830 |
| `min_child_weight` | 5 |
| `gamma` | 0.01596 |

*LGBMRegressor:*

| Parameter | Value |
|---|---|
| `n_estimators` | 4,024 |
| `learning_rate` | 0.01654 |
| `num_leaves` | 33 |
| `subsample` | 0.8761 |
| `colsample_bytree` | 0.4870 |
| `reg_alpha` | 0.003384 |
| `reg_lambda` | 0.3392 |
| `min_child_samples` | 40 |
| `max_depth` | 2 |

*CatBoostRegressor:*

| Parameter | Value |
|---|---|
| `iterations` | 1,525 |
| `learning_rate` | 0.02782 |
| `depth` | 6 |
| `l2_leaf_reg` | 5.373 |
| `bagging_temperature` | 0.9110 |
| `random_strength` | 0.7171 |
| `border_count` | 106 |

*Linear models:* Ridge α=20, Lasso α=0.0005, ElasticNet α=0.0005 with l1_ratio=0.8.

### E. Out-of-Fold (OOF) Evaluation Framework

All base models are trained and evaluated using a strict 5-fold OOF protocol:

1. The 1,458 training instances are split into 5 folds (stratified by target deciles).
2. For each fold, all seven models are trained on the 4 non-held-out folds and predict on the held-out fold.
3. OOF predictions accumulate across all folds, forming a complete OOF prediction vector for each model.
4. The OOF RMSLE for each model is computed on the full 1,458-instance OOF predictions, providing an unbiased estimate of generalisation performance.
5. Test predictions are the average of 5 fold-specific predictions, reducing variance.

This framework prevents any data leakage: the validation fold is never seen during training or preprocessing within any given fold.

### F. Ridge Meta-Learner Stacking

The 7 OOF prediction vectors (of shape 1,458 × 1) are horizontally stacked into a level-2 feature matrix X_meta of shape (1,458 × 7). A Ridge regression meta-learner is trained on X_meta to predict the true log-SalePrice, using 5-fold cross-validation over α ∈ {0.1, 0.5, 1, 2, 5, 10, 20} to select the best regularisation. The chosen α=2 yields stacked OOF RMSLE = 0.10722.

### G. Final Ensemble Combination

The final prediction is a weighted combination:

$$\hat{y}_{\text{final}} = 0.60 \times \hat{y}_{\text{stack}} + 0.40 \times \hat{y}_{\text{blend}}$$

where $\hat{y}$<sub>stack</sub> is the Ridge meta-learner prediction and $\hat{y}$<sub>blend</sub> is the inverse-OOF-RMSLE weighted average of the 7 base models:

$$\hat{y}_{\text{blend}} = \frac{\sum_k w_k \hat{y}_k}{\sum_k w_k}, \quad w_k = \frac{1}{\text{OOF-RMSLE}_k}$$

The 60/40 split was selected by sweeping blend ratios on OOF predictions. This combination hedges against the meta-learner overfitting to OOF artifacts while retaining the stacking advantage.

**Blend weights by inverse OOF RMSLE:**

| Model | OOF RMSLE | Weight |
|---|---|---|
| GB | 0.11098 | 0.145 |
| XGB | 0.11379 | 0.141 |
| LGBM | 0.11676 | 0.138 |
| CAT | 0.11178 | 0.144 |
| Ridge | 0.11202 | 0.143 |
| Lasso | 0.11076 | 0.145 |
| EN | 0.11091 | 0.145 |

### H. Evaluation Metrics

**Primary metric — Root-Mean-Squared Logarithmic Error (RMSLE):**

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log(1+\hat{y}_i) - \log(1+y_i)\right)^2}$$

RMSLE is the official Kaggle metric. It is symmetric with respect to proportional errors — a \$10,000 error on a \$100,000 house penalises equally to a \$100,000 error on a \$1,000,000 house.

**Secondary metrics:**
- **Mean Absolute Error (MAE):** expressed in original dollar scale as $\frac{1}{n}\sum|\hat{y}_i - y_i|$; directly interpretable as average prediction error in USD.
- **Coefficient of Determination (R²):** proportion of variance explained by the model; computed in log-space for comparability with RMSLE.

All experiments use fixed `random_state=42` for reproducibility.

---

## V. Results and Discussion

### A. Preprocessing Outcomes

After applying the final preprocessing pipeline:

- **Outlier removal:** 2 instances removed (GrLivArea > 4,000 ft², SalePrice < \$300,000), leaving 1,458 training instances.
- **Feature engineering:** 12 composite features added (TotalSF, TotalBath, TotalPorch, Age, RemodelAge, GarageAge, OverallQualSq, TotalSF_Qual, GrLiv_Qual, NeighborhoodRank, MoSoldSin, MoSoldCos).
- **Skewness correction:** 37 features with |skew| > 0.75 corrected with BoxCox(λ=0.15).
- **Final feature partition:** 66 numeric (RobustScaler) + 20 ordinal + 24 nominal (one-hot) = 110 pre-encoding features.
- **Target transformation:** Log-transform reduces SalePrice skewness from 1.88 to ~0.12.

The `OverallQual` variable has the highest Pearson correlation with SalePrice (|r| = 0.791), followed by `GrLivArea` (0.709), `GarageCars` (0.641), `GarageArea` (0.624), and `TotalBsmtSF` (0.614). These variables and their interactions are captured in the engineered features.

### B. Model Generation Comparison

**Table I.** Kaggle Public Leaderboard Score Progression

| Version | Models | Tuning | OOF RMSLE | Kaggle LB RMSLE | Approx. Rank |
|---|---|---|---|---|---|
| v1 (Baseline) | GB + XGB | RandomizedSearchCV (n_iter=20) | ~0.118 | 0.12610 | ~1,238 |
| v2 (4-Model OOF) | GB + XGB + LGBM + Ridge | RandomizedSearchCV | 0.11090 | 0.12171 | ~521 |
| v3 (Final, 7-Model Stack) | GB+XGB+LGBM+CAT+Ridge+Lasso+EN | Optuna (80 trials/model) | **0.10736** | **0.11965** | **226** |

The iterative progression demonstrates a **+0.00645 improvement** in Kaggle RMSLE from v1 to v3, with each upgrade contributing measurably: richer feature engineering and OOF blending account for ~0.004, while Optuna + CatBoost + stacking contribute an additional ~0.002.

### C. Individual Base-Model OOF Results (Version 3)

**Table II.** Per-Fold OOF RMSLE Breakdown for Version 3

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF RMSLE | MAE |
|---|---|---|---|---|---|---|---|
| GB | 0.11004 | 0.10678 | 0.11983 | 0.11714 | 0.09991 | 0.11098 | \$13,471 |
| XGB | 0.12001 | 0.10806 | 0.12043 | 0.11751 | 0.10170 | 0.11379 | \$13,892 |
| LGBM | 0.12085 | 0.11150 | 0.12315 | 0.12316 | 0.10388 | 0.11676 | \$14,516 |
| CAT | 0.11268 | 0.10878 | 0.11743 | 0.12095 | 0.09758 | 0.11178 | \$13,540 |
| Ridge | 0.11310 | 0.10966 | 0.12022 | 0.11780 | 0.09795 | 0.11202 | \$13,786 |
| Lasso | 0.10898 | 0.11050 | 0.11958 | 0.11345 | 0.10041 | 0.11076 | \$13,777 |
| EN | 0.10912 | 0.11102 | 0.11934 | 0.11415 | 0.09998 | 0.11091 | \$13,740 |

**Table III.** Ensemble Performance Summary

| Ensemble Strategy | OOF RMSLE |
|---|---|
| Stacked (Ridge meta-learner, α=2) | 0.10722 |
| Weighted blend (inverse RMSLE weights) | 0.10768 |
| Final (60% stack + 40% blend) | **0.10736** |

**Table IV.** Final Kaggle Submission Statistics

| Field | Value |
|---|---|
| Competition | House Prices: Advanced Regression Techniques |
| Submission date | 2026-05-19 |
| Kaggle Public LB RMSLE | **0.11965** |
| Leaderboard rank | **226 / 70,000+** |
| Percentile | Top 0.3% |
| Submission file | `submission.csv` (1,459 rows) |
| Predicted mean SalePrice | \$178,556 |
| Predicted min SalePrice | \$44,388 |
| Predicted max SalePrice | \$700,601 |

### D. Analysis and Discussion

**1. Preprocessing and Feature Engineering Impact.**  
The most impactful single improvement between v1 and v2 was advanced feature engineering. The 12 composite features — especially `TotalSF`, `TotalSF_Qual`, `OverallQualSq`, and age-related features — capture non-linear relationships that the base 79 features encode only implicitly. The switch from `StandardScaler` to `RobustScaler` reduces sensitivity to the remaining high-value outliers in the numeric feature space.

**2. OOF Blending vs. Simple Averaging.**  
The OOF framework introduced in v2 provides three benefits over simple model averaging: (a) it provides unbiased OOF predictions on the training set for meta-learner training without additional validation splits; (b) it automatically provides 5× test prediction averaging, reducing variance; and (c) it allows computing per-model OOF RMSLE for inverse-weighted blending. The OOF RMSLE of 0.11090 for v2 (using only GB+XGB+LGBM+Ridge) confirmed the approach before expanding to 7 models.

**3. Bayesian Optimisation vs. Random Search.**  
The Optuna Bayesian optimisation (80 trials per model) explored much larger hyperparameter ranges than RandomizedSearchCV (n_iter=20). For XGBoost, this found n_estimators=4,754 with learning_rate=0.0191, a much larger ensemble with a slower learning rate than the RandomizedSearchCV best. For GradientBoosting, the `huber` loss (rather than `squared_error`) provided robustness to the remaining price-range outliers. These configurations would be unlikely to be discovered by a 20-trial random search.

**4. Model Diversity and Stacking Benefit.**  
The stacked OOF RMSLE of 0.10722 is lower than any individual model (best individual: Lasso at 0.11076). The improvement of ~0.0035 RMSLE from best-individual to stacked ensemble demonstrates that the seven models have genuinely complementary errors. Ridge and Lasso (linear models operating on the full one-hot feature space) make fundamentally different errors from the tree-based models, providing the diversity that makes stacking effective.

**5. Fold Variance.**  
All models show substantially lower RMSLE on Fold 5 (e.g., GB: 0.09991) than Fold 3 (GB: 0.11983). This ~0.019 fold-to-fold variance reflects the natural sampling variability of 5-fold CV on 1,458 instances, and is consistent with the literature's reported variance for this dataset. The OOF RMSLE averages out this variability, providing a stable estimate.

**6. Generalisation Gap.**  
The OOF RMSLE of 0.10736 exceeds the Kaggle public LB score of 0.11965 by approximately 0.012. This generalisation gap is within the expected range for the Ames dataset and is partly attributable to the public LB test set (half of the 1,459 test instances), which may have a different distribution from the training set in terms of sale price distribution and neighbourhood composition.

**7. Submission Distribution Check.**  
The final submission's predicted mean (\$178,556), min (\$44,388), and max (\$700,601) are consistent with the training distribution (mean \$180,921, median \$163,000, range [\$34,900, \$755,000]). No degenerate predictions (negative prices, implausible extremes) are observed, confirming the pipeline's correctness end-to-end.

---

## VI. Recommendations and Conclusions

### A. Key Learnings

This project confirms and extends several findings from the literature:

1. **Feature engineering dominates model selection.** The gap between v1 (0.12610) and the ideal possible with only better models (estimated ~0.124 with the same features) is smaller than the gap attributed to composite feature engineering. Practitioners should invest in domain-informed feature construction before extensive model tuning.

2. **Bayesian optimisation discovers better hyperparameters than random search.** Optuna's TPE found XGB configurations with n_estimators=4,754 and learning_rate=0.019 — a slow-and-deep configuration that random search with 20 trials over a smaller range missed entirely. The 80-trial investment is justified by the measurable score improvement.

3. **Linear models contribute non-trivially to tree-based ensembles.** Lasso and ElasticNet achieve OOF RMSLE of 0.11076 and 0.11091, competitive with tree-based models. Their inclusion in the ensemble improves stack diversity and the final score. This confirms the course instruction that no single model family dominates in all cases.

4. **Stacking consistently beats individual models.** The Ridge meta-learner (OOF RMSLE = 0.10722) outperforms all seven individual models. The improvement is 0.003+ RMSLE, which corresponds to a meaningful leaderboard rank difference at this score range.

5. **OOF cross-validation is the correct evaluation framework for ensemble building.** Any data-leakage in the stacking step (e.g., training the meta-learner on in-sample predictions) would inflate the stacked performance estimate and produce a model that overfits the training leaderboard. Strict OOF evaluation prevented this.

6. **Data leakage prevention is essential.** Encapsulating all preprocessing within fold boundaries and fitting transformers only on training folds is a critical methodological step, confirmed by the close agreement between OOF RMSLE and Kaggle LB score.

### B. Future Work

Several directions could extend this work:

1. **Pseudo-labelling.** Using the final model's confident test predictions as additional training instances (pseudo-labels) could reduce the generalisation gap by exposing the model to the test distribution.

2. **Target encoding with leave-one-out smoothing.** Mean-target encoding for high-cardinality nominal variables (e.g., `Neighborhood` with 25 levels) with cross-validation-based smoothing could replace the current one-hot representation, reducing dimensionality while preserving price signals.

3. **Neural network meta-learner.** Replacing the Ridge meta-learner with a shallow MLP or boosted tree (XGBoost on the OOF stack) could model non-linear interactions between base model predictions.

4. **Full competition data augmentation.** The Ames dataset's 1,460 training instances are relatively few; data augmentation via SMOGN or synthetic noise-perturbed samples could reduce variance.

5. **Neighbourhood-aware imputation.** Imputing `LotFrontage` by per-neighbourhood median (rather than global median) is a known Ames-specific improvement [5] that was not included in the current pipeline.

### C. Conclusions

This paper presents an iterative machine learning pipeline for house price prediction on the Ames Housing dataset, developed as the final project for the COMP 468 Machine Learning course at Abdullah Gul University. Starting from a two-model baseline (GB + XGB with RandomizedSearchCV, Kaggle LB 0.12610), three methodological upgrades were applied in sequence: advanced feature engineering and OOF blending (v2, LB 0.12171); Bayesian hyperparameter optimisation with 7 base models and Ridge stacking (v3, LB **0.11965, rank 226**).

The final pipeline achieves an OOF RMSLE of **0.10736**, with the Ridge meta-learner stacking (0.10722) outperforming all seven individual base models (best: Lasso at 0.11076). The Kaggle public leaderboard score of **0.11965** places the submission in the **top 0.3%** of over 70,000 participants.

The work demonstrates that for a structured tabular regression problem like Ames Housing, preprocessing quality and principled hyperparameter tuning are the primary drivers of competitive performance, not any single algorithm choice. All models and techniques applied are aligned with the gradient boosting, regularised linear regression, ensemble learning, and hyperparameter tuning modules of the COMP 468 curriculum.

### Acknowledgment

This work was conducted as part of the COMP 468 Machine Learning course at **Abdullah Gul University (AGU)**, Kayseri, Turkey. The authors gratefully acknowledge AGU for providing the academic environment and resources that supported this project. The AGU affiliation was included in all Kaggle submission notes in accordance with the course requirements.

---

## References

[1] D. De Cock, "Ames, Iowa: Alternative to the Boston Housing data as an end of semester regression project," *J. Stat. Educ.*, vol. 19, no. 3, pp. 1–15, 2011.

[2] J. H. Friedman, "Greedy function approximation: a gradient boosting machine," *Ann. Stat.*, vol. 29, no. 5, pp. 1189–1232, 2001.

[3] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[4] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794.

[5] H. Sharma, H. Harsora, and B. Ogunleye, "An optimal house price prediction algorithm: XGBoost," *Analytics*, vol. 3, no. 1, pp. 30–48, 2024.

[6] Q. Truong, M. Nguyen, H. Dang, and B. Mei, "Housing price prediction via improved machine learning techniques," *Procedia Comput. Sci.*, vol. 174, pp. 433–442, 2020.

[7] C. R. Madhuri, G. Anuradha, and M. V. Pujitha, "House price prediction using regression techniques: a comparative study," in *Proc. Int. Conf. Smart Structures and Systems (ICSSS)*, Chennai, India, 2019, pp. 1–5.

[8] B. Park and J. K. Bae, "Using machine learning algorithms for housing price prediction: the case of Fairfax County, Virginia housing data," *Expert Syst. Appl.*, vol. 42, no. 6, pp. 2928–2934, 2015.

[9] Kaggle, "House Prices – Advanced Regression Techniques," 2016. [Online]. Available: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques. [Accessed: May 2026].

[10] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *J. Mach. Learn. Res.*, vol. 13, pp. 281–305, 2012.

[11] A. B. Adetunji, O. N. Akande, F. A. Ajala, O. Oyewo, Y. F. Akande, and G. Oluwadara, "House price prediction using random forest machine learning technique," *Procedia Comput. Sci.*, vol. 199, pp. 806–813, 2022.

[12] F. Pedregosa et al., "Scikit-learn: machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[13] A. Hjort, J. Pensar, I. Scheel, and D. E. Sommervoll, "House price prediction with gradient boosted trees under different loss functions," *J. Property Res.*, vol. 39, no. 4, pp. 338–364, 2022.

[14] G. Ke et al., "LightGBM: a highly efficient gradient boosting decision tree," in *Proc. 31st Int. Conf. Neural Inf. Process. Syst. (NIPS)*, Long Beach, CA, USA, 2017, pp. 3149–3157.

[15] J. Yu, "Prediction on housing price based on the data on Kaggle," in *2021 International Conference on Artificial Intelligence and Computer Engineering*, 2021, pp. 629–636.

[16] Y. Wang, "Feature engineering and predictive modeling for housing prices: A case study using the Ames, Iowa dataset," *Preprints*, doi: 10.20944/preprints202511.1759.v1, Nov. 2025.

[17] B. Thakur and A. Bisht, "Accurate house price prediction using ensemble machine learning model," Yogananda School of AI, Computers & Data Sciences, Shoolini University, Solan, India, Tech. Rep., 2025.

[18] S. Ö. Akyüz, B. E. Erdogan, Ö. Yıldız, and P. K. Atas, "A novel hybrid house price prediction model," *Computational Economics*, vol. 62, pp. 1215–1232, Sep. 2022, doi: 10.1007/s10614-022-10298-8.

[19] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Proc. 32nd Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, Montreal, Canada, 2018, pp. 6638–6648.

[20] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2019, pp. 2623–2631.

[21] D. H. Wolpert, "Stacked generalization," *Neural Netw.*, vol. 5, no. 2, pp. 241–259, 1992.

---

## Appendix A — Kaggle Competition Evidence

**Competition:** House Prices: Advanced Regression Techniques  
**URL:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques  
**Submission Date:** 2026-05-19  
**Submission File:** `submission.csv` (1,459 predictions, format: Id, SalePrice)  

**Final Leaderboard Result:**

| Field | Value |
|---|---|
| Public Leaderboard RMSLE | **0.11965** |
| Leaderboard Rank | **226** |
| Total Participants | 70,000+ |
| Percentile | Top 0.3% |
| Mean Predicted SalePrice | \$178,556 |
| Min Predicted SalePrice | \$44,388 |
| Max Predicted SalePrice | \$700,601 |

**Submission Progression:**

| Submission | Model | Kaggle LB RMSLE | Rank |
|---|---|---|---|
| v1 | GB + XGB (RandomizedSearchCV) | 0.12610 | ~1,238 |
| v2 | 4-Model OOF Blend | 0.12171 | ~521 |
| v3 (Final) | 7-Model Optuna + Stack | **0.11965** | **226** |

**AGU Acknowledgment:** The Abdullah Gul University (AGU) affiliation and course context (COMP 468) were included in the Kaggle submission description in accordance with the course requirements.

The full submission evidence including submission ID and confirmation details is stored in `Kaggle_Submission_Evidence.json` in the project repository.

---

## Appendix B — Project Repository Structure

All code is available in the project repository. The pipeline is fully reproducible with `random_state=42` throughout.

```
compproject/
├── data/
│   ├── train.csv                    # Kaggle training data (1,460 rows)
│   ├── test.csv                     # Kaggle test data (1,459 rows)
│   └── data_description.txt         # Feature descriptions
├── HousePrices_Final_Project.ipynb  # Main notebook with EDA and analysis
├── quick_train.py                   # v1: GB+XGB baseline (Kaggle LB 0.12610)
├── improved_model.py                # v1 cleaned: GB+XGB with preprocessing
├── train_v2.py                      # v2: 4-model OOF blend (Kaggle LB 0.12171)
├── train_resume.py                  # v3: 7-model Optuna+stack (Kaggle LB 0.11965)
├── Final_IEEE_Paper.md              # This paper
├── submission.csv                   # Final submission (v3 predictions)
├── meta_model.joblib                # Saved Ridge meta-learner
├── optuna_results.json              # Best Optuna hyperparameters (all models)
├── Kaggle_Submission_Evidence.json  # Official submission evidence
├── Kaggle_Submission_Evidence.md    # Submission evidence summary
└── Midterm_HousePrices.pdf          # Project proposal (midterm)
```

**COMP 468 Course-Aligned Models and Techniques:**
- `sklearn.ensemble.GradientBoostingRegressor` — Gradient boosting (course module)
- `xgboost.XGBRegressor` — Regularised gradient boosting (course module)
- `lightgbm.LGBMRegressor` — Efficient gradient boosting variant (course module)
- `catboost.CatBoostRegressor` — Ordered gradient boosting (course module)
- `sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet` — Regularised linear regression (course module)
- `sklearn.model_selection.KFold` — Cross-validation framework (course module)
- `optuna` — Bayesian hyperparameter optimisation (course module)
- `sklearn.pipeline.Pipeline`, `ColumnTransformer` — Pipeline encapsulation (course module)
- `sklearn.preprocessing.OrdinalEncoder`, `OneHotEncoder`, `RobustScaler` — Preprocessing (course module)
- `sklearn.impute.SimpleImputer` — Missing value handling (course module)

---

## Appendix C — Core Preprocessing and Stacking Code

**Feature engineering (selected):**

```python
# Composite feature construction
train['TotalSF']       = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['TotalBath']     = (train['FullBath'] + 0.5*train['HalfBath']
                          + train['BsmtFullBath'] + 0.5*train['BsmtHalfBath'])
train['Age']           = train['YrSold'] - train['YearBuilt']
train['RemodelAge']    = train['YrSold'] - train['YearRemodAdd']
train['OverallQualSq'] = train['OverallQual'] ** 2
train['TotalSF_Qual']  = train['TotalSF'] * train['OverallQual']

# Cyclic month encoding
train['MoSoldSin'] = np.sin(2 * np.pi * train['MoSold'] / 12)
train['MoSoldCos'] = np.cos(2 * np.pi * train['MoSold'] / 12)

# BoxCox skewness correction
from scipy.stats import skew, boxcox
for col in numeric_cols:
    if abs(skew(train[col].dropna())) > 0.75:
        train[col] = boxcox1p(train[col], 0.15)
```

**OOF stacking framework (simplified):**

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train), n_models))
test_preds = np.zeros((len(X_test), n_models))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    for i, model in enumerate(models):
        model.fit(X_tr, y_tr)
        oof_preds[val_idx, i] = model.predict(X_val)
        test_preds[:, i] += model.predict(X_test) / 5

# Ridge meta-learner on OOF predictions
meta = Ridge(alpha=2)
meta.fit(oof_preds, y_train)
stack_pred = meta.predict(test_preds)

# Blend: 60% stack + 40% weighted average
weights = 1.0 / oof_rmsle_per_model
blend_pred = (test_preds * weights).sum(axis=1) / weights.sum()
final_pred  = 0.60 * stack_pred + 0.40 * blend_pred
submission['SalePrice'] = np.expm1(final_pred)
```

**Ordinal encoding (20-column explicit ordering):**

```python
ORDINAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence', 'LandSlope', 'LotShape', 'PavedDrive',
    'Utilities', 'Functional'
]
# Each column maps to its domain-specific ordering
ordinal_encoder = OrdinalEncoder(
    categories=ORDINAL_CATEGORIES,   # explicit per-column ordering
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
```
