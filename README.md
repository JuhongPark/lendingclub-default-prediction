# Lending Club Default Prediction with SSAE

> **SNU Fintech AI Course Project**
>
> A 90%-accurate default model can still lose money if it approves the worst loans.
> This project goes beyond classification metrics — it optimizes for **portfolio-level annualized return**.

---

## Overview

| | |
|---|---|
| **Problem** | Predict loan defaults from 222K Lending Club loans with 71 features |
| **Approach** | Denoising SSAE for feature compression + ML classifiers |
| **Innovation** | Custom objective function based on portfolio annualized return |
| **Stage 2** | Moral hazard detection — late-stage defaults that are costlier to lenders |

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
  ┌──────────────┐
  │    Concat    │  8 SSAE latent + 14 domain features → 22 final features
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

| Model | Type |
|-------|------|
| Logistic Regression | Baseline |
| LightGBM / XGBoost | Gradient boosting |
| Random Forest | Bagging ensemble |
| SVM | Support vector |

All models are grid-searched and ranked by **portfolio annualized return**, not just F1 or accuracy.

### Stage 2 — Moral Hazard

Identifies borrowers who repay 70%+ of their term before defaulting — a pattern that is harder to detect and costlier for lenders than early defaults.

---

## Technical Highlights

| Topic | Detail |
|-------|--------|
| **SSAE** | 3-layer denoising autoencoder (ReLU, 10K epochs, noise factor 0.2) |
| **Class Imbalance** | 83/17 split — addressed with random undersampling |
| **Objective Function** | `LoanAnalysis` class computes money-weighted annualized return across TN/FP/FN/TP |
| **Explainability** | SHAP values computed for every model |
| **Feature Engineering** | 14 domain-selected features (FICO, DTI, income, etc.) preserved alongside SSAE latent output |

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 0 | `00_ssae_tutorial.ipynb` | SSAE walkthrough on a toy dataset |
| 1 | `01_data_preprocessing.ipynb` | Raw data cleaning, feature engineering, export |
| 2 | `02_ssae_baseline.ipynb` | End-to-end SSAE + classifier pipeline |
| 3 | `03_model_experiment.ipynb` | Grid search across 5 classifiers |
| 4 | `04_two_stage_model.ipynb` | **Final model** — two-stage prediction with portfolio objective |
| - | `objective_fn/portfolio_return.ipynb` | Portfolio annualized return calculator |

---

## Discussion

### ML

| Observation | Detail |
|-------------|--------|
| **Proxy ≠ true objective** | F1 ranking and portfolio return ranking diverge. `LoanAnalysis` assigns distinct revenue per confusion matrix cell (TN: total payments received, FN: net loss after recovery, TP/FP: treasury bond opportunity cost), so a classifier with strong F1 can still approve the costliest defaults |
| **Hybrid feature space** | 8 SSAE latent dims are concatenated with 14 domain features (FICO, DTI, income, etc.) — SHAP can then attribute predictions to both learned and human-readable features, keeping regulatory explainability viable |
| **Multi-explainer SHAP** | `LinearExplainer` for logistic regression, `TreeExplainer` for tree ensembles, `KernelExplainer` for SSAE-wrapped pipelines — matched per model family for consistent interpretability |
| **Undersampling cost** | 83/17 → 50/50 rebalancing improves default recall at the cost of majority-class information, risking miscalibrated precision in deployment |

### AI Safety

| Observation | Detail |
|-------------|--------|
| **Objective alignment** | Accuracy optimization approved bad loans; portfolio return aligns the model with lender economics by pricing each outcome differently (opportunity cost vs. net loss after recovery) — designing the right objective was harder than building the model |
| **Interpretability gap** | The best predictors (autoencoder latent dims) resist explanation; the explainable features (FICO, DTI) are weaker alone — "latent dim e3 was important" tells a regulator nothing |
| **Shifted incentives** | Stage 2 targets borrowers who complete ≥70% of their term before defaulting — costlier (less principal recovered, longer capital lock-up) and harder to detect, analogous to agents adapting behavior under changed incentive structures |
| **Unresolved fairness** | `addr_state` and `home_ownership` are not used directly but compressed into SSAE latent dims, where they can still proxy for protected attributes — a return-maximizing objective can amplify systemic disparities without explicit disparate impact auditing |
| **Dual-use of credit AI** | The same model that shields lenders from bad loans decides who gets funded — objective function and fairness constraint choices are societal design decisions, not just technical ones |

---

## Quick Start

```bash
pip install -r requirements.txt
```

Place the Lending Club dataset in `data/`, then run notebooks in order (`00` → `04`).
