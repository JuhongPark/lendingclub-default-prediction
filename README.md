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

> **No single metric replaces reality.**
> A technically strong model can still lose money, exclude creditworthy borrowers, or satisfy regulators on paper while hiding risk in opaque features.
> This project demonstrates that evaluation must span technical performance, business impact, and societal fairness — and that each layer reveals blind spots the others miss.

### ML — Technical metrics alone are not enough

| Observation | Detail |
|-------------|--------|
| **Metric ≠ business outcome** | F1 ranking and portfolio return ranking diverge. `LoanAnalysis` prices each confusion matrix cell differently (TN: total payments received, FN: net loss after recovery, TP/FP: treasury bond opportunity cost) — a high-F1 model can still approve the costliest defaults because F1 treats all errors equally while the business does not |
| **Hybrid feature space** | 8 SSAE latent dims concatenated with 14 domain features (FICO, DTI, income, etc.) — SHAP can attribute predictions to both learned and human-readable features, keeping regulatory explainability viable alongside predictive power |
| **Multi-explainer SHAP** | `LinearExplainer` for logistic regression, `TreeExplainer` for tree ensembles, `KernelExplainer` for SSAE-wrapped pipelines — matched per model family for consistent interpretability |
| **Undersampling cost** | 83/17 → 50/50 rebalancing improves default recall but discards majority-class information — better sensitivity to defaults, worse precision on good loans, a trade-off invisible to recall alone |

### AI Safety — Beyond technical performance to real-world impact

| Observation | Detail |
|-------------|--------|
| **Objective alignment** | Accuracy optimization approved bad loans; portfolio return corrects this by pricing outcomes differently — but even portfolio return is a lender-side proxy that says nothing about borrower welfare. Designing the right objective was harder than building the model, and no single objective fully captures real-world impact |
| **Interpretability vs. predictive power** | The best predictors (autoencoder latent dims) resist explanation; the explainable features (FICO, DTI) are weaker alone — "latent dim e3 was important" tells a regulator nothing. Technical accuracy and human accountability pull in opposite directions |
| **Shifted incentives** | Stage 2 targets borrowers who complete ≥70% of their term before defaulting — costlier (less principal recovered, longer capital lock-up) and harder to detect. Standard metrics miss these late-stage patterns because they weight all defaults equally |
| **Fairness gap** | `addr_state` and `home_ownership` are compressed into SSAE latent dims, where they can still proxy for protected attributes — a return-maximizing objective can amplify systemic disparities, and no technical metric in the pipeline audits for this |
| **Dual-use of credit AI** | The same model that shields lenders from bad loans decides who gets funded — optimizing any single metric (accuracy, F1, or even portfolio return) without fairness constraints risks turning a technical success into a societal failure |

---

## Quick Start

```bash
pip install -r requirements.txt
```

Place the Lending Club dataset in `data/`, then run notebooks in order (`00` → `04`).
