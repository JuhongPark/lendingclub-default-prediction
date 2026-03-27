# SNU Fintech AI - Lending Club Default Prediction

A two-stage default prediction model for Lending Club loans using SSAE (Semi-Supervised Stacked Autoencoder) for feature extraction, combined with traditional ML classifiers and portfolio-level objective function optimization.

## Pipeline

1. **Preprocessing** — Clean raw Lending Club data, handle missing values, encode categoricals
2. **SSAE Feature Extraction** — Denoise and compress high-dimensional features into latent representations
3. **1st Stage: Default Prediction** — Classify loan default using SSAE-encoded features + selected raw features
4. **2nd Stage: Moral Hazard Detection** — Identify late-stage defaults (>70% of repayment period elapsed)
5. **Evaluation** — Optimize by Portfolio Annualized Return, not just accuracy/F1

## Project Structure

```
├── notebooks/
│   ├── SSAE_Sample.ipynb                          # SSAE tutorial (breast cancer dataset)
│   ├── LendingClub_Cleaning_mod.ipynb             # Data preprocessing pipeline
│   ├── Lending_Club_SSAE.ipynb                    # Base SSAE + classifier pipeline
│   ├── Lending_Club_SSAE-BB-1000-JPark_0903.ipynb # Full experiment (5 classifiers)
│   ├── Lending_Club_SSAE-BB-1000_JS_0902.ipynb    # Final: 1st + 2nd stage with objective fn
│   └── objective_fn/
│       └── objective_fn_modified.ipynb             # LoanAnalysis class (annualized return)
├── data/                                          # CSV data (not tracked by git)
├── .gitignore
├── LICENSE
└── README.md
```

## Models

| Stage | Model | Description |
|-------|-------|-------------|
| 1st | Logistic Regression | Baseline classifier |
| 1st | LightGBM / XGBoost | Gradient boosting ensembles |
| 1st | Random Forest | Bagging ensemble |
| 1st | SVM | Support vector classifier |
| 2nd | Moral Hazard Classifier | Detects defaults after 70%+ repayment period |

All classifiers receive SSAE-encoded features (latent dim=8) concatenated with 14 hand-selected raw features.

## Setup

```bash
pip install torch scikit-learn shap xgboost lightgbm imbalanced-learn
```

Place `LC_Data_Cleaned_0829.csv` in the `data/` directory before running notebooks.

## Data

- **Source**: Lending Club loan data (2013–2014)
- **Target**: `loan_status_encoded` (0 = Fully Paid, 1 = Default)
- **Sampling**: Random undersampling applied to balance classes
