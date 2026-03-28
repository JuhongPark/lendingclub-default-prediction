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

### Key Finding — No single metric replaces reality

> A technically strong model can still lose money, exclude creditworthy borrowers, or hide risk in opaque features.
> Technical metrics evaluate **how well** the model classifies — they do not evaluate **what happens** when those classifications reach real borrowers and real balance sheets.
> This project shows that evaluation must span three layers, and each reveals blind spots the others miss:

| Layer | What it misses | Example from this project |
|-------|---------------|--------------------------|
| **Technical (F1, AUC)** | Treats all errors equally — blind to business cost | A high-F1 model approved the costliest defaults because F1 does not distinguish a $5K loss from a $50K loss |
| **Business (portfolio return)** | Optimizes lender economics — blind to borrower welfare | `LoanAnalysis` prices each outcome differently (TN: payments received, FN: net loss, TP/FP: treasury bond opportunity cost), but says nothing about who gets denied credit or why |
| **Fairness (unaudited)** | Not measured — systemic disparities go undetected | `addr_state` and `home_ownership` are compressed into SSAE latent dims where they can still proxy for protected attributes, and no metric in the pipeline audits for this |

Optimizing any single layer without the others risks turning a technical success into a business failure or a societal one. Designing the right objective was harder than building the model — and even the best objective is still a proxy.

### ML

| Observation | Detail |
|-------------|--------|
| **Hybrid feature space** | 8 SSAE latent dims + 14 domain features (FICO, DTI, income, etc.) — SHAP attributes predictions to both learned and human-readable features, keeping regulatory explainability viable |
| **Multi-explainer SHAP** | `LinearExplainer`, `TreeExplainer`, `KernelExplainer` matched per model family for consistent interpretability |
| **Undersampling cost** | 83/17 → 50/50 rebalancing trades majority-class precision for default recall — invisible to recall alone |

### AI Safety

| Observation | Detail |
|-------------|--------|
| **Interpretability vs. predictive power** | The best predictors (latent dims) resist explanation; the explainable features (FICO, DTI) are weaker alone — technical accuracy and human accountability pull in opposite directions |
| **Shifted incentives** | Stage 2 targets borrowers who complete ≥70% of their term before defaulting — costlier and harder to detect; standard metrics miss these because they weight all defaults equally |
| **Dual-use of credit AI** | The same model that shields lenders decides who gets funded — objective function and fairness constraints are societal design decisions, not just technical ones |

---

## Quick Start

```bash
pip install -r requirements.txt
```

Place the Lending Club dataset in `data/`, then run notebooks in order (`00` → `04`).
