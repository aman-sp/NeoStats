# Experimental Results & Model Benchmark — NeoStats Credit Risk Platform

## 1. Baseline Notebook Results vs Production Pipeline Results

### A. Original Notebook Experiments (`application_train_inner_merged.csv`, 11,043 rows)
*Note: Inner join caused a 96.4% row loss, and notebooks suffered from preprocessing/SMOTE leakage.*

| Notebook Model | Setup | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.90 | 0.91 | 0.00 | 0.00 | 0.00 | 0.63 |
| **Logistic Regression** | SMOTE + Scaler | 0.63 | 0.63 | 0.13 | 0.48 | 0.20 | 0.60 |
| **Logistic Regression** | Tuned | 0.67 | 0.67 | 0.17 | 0.62 | 0.27 | 0.71 |
| **Random Forest** | Baseline | 1.00 | 0.91 | 1.00 | 0.00 | 0.01 | 0.69 |
| **Random Forest** | Undersampling | 0.68 | 0.65 | 0.16 | 0.66 | 0.26 | 0.71 |
| **Random Forest** | SMOTE | 0.80 | 0.73 | 0.15 | 0.43 | 0.23 | 0.65 |
| **XGBoost** | Baseline | 0.90 | 0.90 | 0.39 | 0.06 | 0.11 | 0.68 |
| **XGBoost** | SMOTE + Tuned | 1.00 | 0.90 | 0.23 | 0.03 | 0.05 | 0.70 |
| **LightGBM** | Baseline | 0.97 | 0.91 | 0.58 | 0.03 | 0.07 | 0.71 |
| **LightGBM** | SMOTE + Tuned | 1.00 | 0.90 | 0.30 | 0.04 | 0.07 | 0.71 |

---

### B. Production Pipeline Results (Full 307,511 Applicants, Leak-Free Split)
*Evaluated on isolated held-out Test split (46,127 applicants, 3,724 defaults).*

| Model | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy | Brier Score |
|---|---|---|---|---|---|---|---|
| **LightGBM (Selected)** | **0.7716** | **0.2610** | **0.6898** | **0.1772** | **0.2841** | 0.6931 | **0.0670** (Calibrated) |
| **XGBoost** | 0.7695 | 0.2560 | 0.6847 | 0.1765 | 0.2831 | 0.6917 | 0.1862 (Raw) |
| **Logistic Regression** | 0.7544 | 0.2342 | 0.6829 | 0.1652 | 0.2663 | 0.6728 | 0.2037 (Raw) |
| **Random Forest** | 0.7507 | 0.2298 | 0.5473 | 0.1947 | 0.2872 | 0.7516 | 0.0712 (Raw) |

---

## 2. Confusion Matrix of Production LightGBM Model (Threshold = 0.50 on class-weighted probabilities)

- **True Negatives (TN):** 29,403 (Non-defaults correctly identified)
- **False Positives (FP):** 13,000 (Non-defaults flagged for manual underwriting review)
- **False Negatives (FN):** 1,155 (Defaults missed)
- **True Positives (TP):** 2,569 (Defaults correctly captured)
- **Default Detection Rate (Recall):** **68.98%** of all true defaults identified.

---

## 3. Probability Calibration Evidence

- **Raw LightGBM Probability Brier Score:** 0.1856
- **Calibrated (Sigmoid) Probability Brier Score:** **0.0670** (A 63.9% reduction in mean squared calibration error).
- **Adoption:** Calibrated model was adopted for production inference.

---

## 4. Empirical Validation Risk Bands

Based on 20-quantile analysis on the 46,127 validation records:

| Risk Band | Calibrated Probability Range | Empirical Validation Default Rate | Recommended Underwriting Action |
|---|---|---|---|
| **LOW RISK** | **< 2.93%** | **< 3.0%** (Actual: 2.34%) | Fast-track automated approval with preferential rates. |
| **MEDIUM RISK** | **2.93% to 15.31%** | **3.0% to 14.9%** (Actual: 7.91%) | Standard underwriting review and documentation verification. |
| **HIGH RISK** | **>= 15.31%** | **>= 15.0%** (Actual: 24.81%) | Senior credit committee escalation, collateral requirement, or limit restriction. |
