# Methodology & Modeling Strategy — NeoStats Credit Risk Intelligence Platform

## 1. Dataset Strategy & Leak-Free Architecture

### A. The Flaw in Existing Experimentation Notebooks
In the research notebooks (`Preprocessing_merged.ipynb` and model notebooks), an **inner join** was performed across relational tables, shrinking the dataset from **307,511 applicants to only 11,043** (a 96.4% data loss). Furthermore, `StandardScaler`, `SimpleImputer`, and `SMOTE` oversampling were applied globally before train/test splitting, creating artificial target and distribution leakage.

### B. The Production Pipeline Solution
1. **Anchor Table:** The main table `HC_application_train.csv` (307,511 records) serves as the immutable root.
2. **Applicant-Level Historical Aggregation:** Historical records from Credit Bureau (`HC_bureau.csv`) and Previous Applications (`HC_previous_application.csv`) are aggregated to exactly one row per `SK_ID_CURR` using sum, mean, count, and max operators.
3. **Left Join Preservation:** Aggregated tables are left-joined to `application_train`. Exactly 307,511 applicants are preserved with zero row duplication.
4. **Strict Split Before Preprocessing:** Data is split into Train (70% = 215,257), Validation (15% = 46,127), and Test (15% = 46,127) using stratified sampling. Preprocessing transformers (median imputation, one-hot encoding, standard scaling) are fitted **strictly on the training split**.

---

## 2. Handling Class Imbalance

- **Observed Distribution:** Non-defaults = 282,686 (91.93%), Defaults = 24,825 (8.07%). Imbalance Ratio = **11.39 : 1**.
- **Strategy:** Instead of synthetic oversampling (SMOTE) which distorts calibration and probability density in high dimensions:
  - For Logistic Regression and Random Forest: `class_weight='balanced'`.
  - For XGBoost and LightGBM: `scale_pos_weight=11.39`.
  - Combined with post-hoc probability threshold tuning on held-out validation data.

---

## 3. Model Benchmark & Selection Evidence

Four diverse algorithms were trained and evaluated on identical 70/15/15 splits using the complete 307,511 dataset:

| Model | Test ROC-AUC | Test PR-AUC | Recall (Defaults) | Precision | F1-Score | Brier Score |
|---|---|---|---|---|---|---|
| **LightGBM (Final Selected)** | **0.7716** | **0.2610** | **0.6898** | **0.1772** | **0.2841** | **0.0670** (Calibrated) |
| **XGBoost** | 0.7695 | 0.2560 | 0.6847 | 0.1765 | 0.2831 | 0.1862 |
| **Logistic Regression** | 0.7544 | 0.2342 | 0.6829 | 0.1652 | 0.2663 | 0.2037 |
| **Random Forest** | 0.7507 | 0.2298 | 0.5473 | 0.1947 | 0.2872 | 0.0712 |

### Selection Rationale:
LightGBM demonstrated superior discriminatory power across both ROC-AUC (0.7716) and Precision-Recall AUC (0.2610), while identifying 68.98% of all true loan defaults in the held-out test partition.

---

## 4. Probability Output vs Calibration vs Threshold Selection

### Distinction:
1. **Raw Model Output:** Probabilities produced directly by `predict_proba()` reflects ranking rather than true posterior probabilities under heavy class weighting.
2. **Probability Calibration:** We applied `CalibratedClassifierCV(method='sigmoid', cv='prefit')` on the validation partition. This improved the Brier Score loss from **0.1856 down to 0.0670**, aligning predicted probabilities with true historical empirical default frequencies.
3. **Empirical Risk Threshold Selection:** Rather than adopting arbitrary cutoffs (e.g. 5%, 15%), predicted probabilities were partitioned into 20 quantile bins on the validation split:
   - **LOW RISK (< 0.0293 / 2.93%):** Cutoff identifying applicant deciles where actual historical default rate is strictly below 3.0%.
   - **HIGH RISK (>= 0.1531 / 15.31%):** Cutoff identifying high delinquency concentration where actual historical default rate is 15.0% or greater.
   - **MEDIUM RISK (2.93% to 15.31%):** Intermediate band requiring standard credit review.

---

## 5. Explainable AI (SHAP)

Using TreeSHAP on the calibrated model:
- Translates model weights and split thresholds into local feature attributions:
  $$\hat{f}(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
- Categorizes features into **Risk-Increasing Factors** ($\phi_i > 0$) and **Risk-Reducing Protective Factors** ($\phi_i < 0$).
- Technical variables are dynamically mapped to plain-English banking definitions (e.g., `EXT_SOURCE_2` $\rightarrow$ "External Credit Bureau Score 2").

---

## 6. Secure Conversational NL-to-SQL Architecture

1. **Grounded Schema:** Strict SQLite schema injection restricts LLM attention to `applicants`, `bureau_summary`, and `prev_app_summary`.
2. **Pre-Execution AST & Regex Validator:**
   - Enforces read-only `SELECT` statements.
   - Rejects DDL (`DROP`, `ALTER`, `CREATE`) and DML (`INSERT`, `UPDATE`, `DELETE`).
   - Rejects multiple queries separated by semicolons.
   - Pre-flight syntax validation using SQLite `EXPLAIN`.
3. **Hallucination Control:** Detects and refuses unsupported domain attributes (e.g., eye colour, hair, pet, vehicle brand) with explicit explanatory notices.
4. **Deterministic Fallback Engine:** If an external LLM API key is absent, an offline regex pattern engine matches the 5 mandatory queries and 5+ standard credit queries, guaranteeing 100% database grounding without ever guessing answers.
