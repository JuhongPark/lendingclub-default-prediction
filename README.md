# Lending Club Default Prediction with SSAE

> **SNU Fintech AI Course Project**
>
> Traditional models optimize for accuracy — but a 90%-accurate model that approves the worst loans still loses money.
> This project optimizes for **portfolio return**, not just classification metrics.

---

## What This Does

| | |
|---|---|
| **Problem** | Predict loan defaults from 222K Lending Club loans (71 features) |
| **Core Idea** | Use SSAE to compress noisy features, then classify with ML models |
| **Key Difference** | Evaluate by **Portfolio Annualized Return**, not just F1/accuracy |

---

## Pipeline

```
  Raw Data (222K loans, 71 features)
       │
       ▼
  ┌──────────────┐
  │ Preprocessing │  Drop post-loan vars, one-hot encode, standardize
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ SSAE Encoder │  99 features → [64 → 32 → 8] → 8 latent dims
  └──────┬───────┘
         │
         ▼
  ┌────���─────────┐
  │    Concat    │  8 latent + 14 raw features = 22 final features
  └──────┬───────┘
         │
    ┌────┴─────┐
    ▼          ▼
 Stage 1    Stage 2
 Default    Moral Hazard
 Predict    Detection
```

---

## Models

### Stage 1 — Default Prediction

| Model | Role |
|-------|------|
| Logistic Regression | Baseline |
| LightGBM | Gradient boosting |
| XGBoost | Gradient boosting |
| Random Forest | Bagging ensemble |
| SVM | Support vector |

### Stage 2 — Moral Hazard Detection

Detects borrowers who repay 70%+ of term before defaulting — a costlier, harder-to-catch pattern.

---

## Technical Highlights

| Topic | Detail |
|-------|--------|
| **SSAE** | 3-layer denoising autoencoder, ReLU, 10K epochs, noise factor 0.2 |
| **Class Imbalance** | 83% non-default / 17% default — random undersampling applied |
| **Objective Function** | `LoanAnalysis` class: money-weighted annualized return per portfolio |
| **Explainability** | SHAP values for every model |
| **Feature Selection** | 14 domain features (FICO, DTI, income, etc.) kept alongside SSAE output |

---

## Project Structure

```
notebooks/
├── 00_ssae_tutorial.ipynb       # SSAE tutorial on toy dataset
├── 01_data_preprocessing.ipynb  # Data cleaning and feature engineering
├── 02_ssae_baseline.ipynb       # SSAE + classifier base pipeline
├── 03_model_experiment.ipynb    # Full experiment with 5 classifiers
├── 04_two_stage_model.ipynb     # Final: two-stage model + objective fn
└── objective_fn/
    └── portfolio_return.ipynb   # Portfolio annualized return calculator
data/                            # Lending Club CSVs (not tracked)
```

---

## Quick Start

```bash
pip install torch scikit-learn shap xgboost lightgbm imbalanced-learn
```

Place `LC_Data_Cleaned_0829.csv` in `data/`, then run notebooks in order.
