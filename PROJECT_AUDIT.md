# NeoStats AI Engineering Internship — Project Audit Report
**Project Name:** AI-Powered Credit Risk Intelligence Platform  
**Auditor:** Senior AI/ML & Full-Stack AI Application Engineer  
**Date of Audit:** September 6, 2026  
**Workspace:** C:\Users\amans\NeoStats

---

## 1. Inventory of Files Present

| File Name | File Type | Size | Description |
|---|---|---|---|
| 
eostatsaiengineerinternshipround1.zip | Archive | 579 KB | Original assignment prompt & guidelines zip |
| rchive.zip | Archive | 701.60 MB | Original datasets and experimentation notebooks zip |
| NeoStats_Candidate_Assignment.pdf | Document | 0.48 MB | Official assignment brief, criteria, and role expectations |
| NeoStats_AI_Use_Case.pdf | Document | 0.17 MB | Detailed functional requirements, architecture, and deliverables |
| HC_application_train.csv | Dataset | 158.44 MB | Main applicant training dataset (307,511 rows, 122 cols) |
| HC_bureau.csv | Dataset | 162.14 MB | Previous credits reported to Credit Bureau (1,716,428 rows) |
| HC_bureau_balance.csv | Dataset | 358.19 MB | Monthly balances of previous credits in Credit Bureau (27,299,925 rows) |
| HC_credit_card_balance.csv | Dataset | 404.91 MB | Monthly balance snapshots of previous credit cards (3,840,312 rows) |
| HC_installments_payments.csv | Dataset | 689.62 MB | Repayment history for previously disbursed loans (13,605,401 rows) |
| HC_POS_CASH_balance.csv | Dataset | 374.51 MB | Monthly balance snapshots of previous POS/cash loans (10,001,358 rows) |
| HC_previous_application.csv | Dataset | 386.21 MB | Previous loan applications at Home Credit (1,670,214 rows) |
| HC_sample_submission.csv | Dataset | 0.51 MB | Template submission file (48,744 rows) |
| HomeCredit_columns_description.csv | Metadata | 0.04 MB | Data dictionary describing columns across tables |
| EDA_merged.ipynb | Jupyter Notebook | 28.17 MB | Exploratory Data Analysis covering all tables |
| Preprocessing_merged.ipynb | Jupyter Notebook | 12.40 MB | Preprocessing, feature engineering, and table merging |
| Model ML_Logisctic Regression.ipynb | Jupyter Notebook | 0.81 MB | Logistic Regression modeling experiments |
| Model ML_Random Forest.ipynb | Jupyter Notebook | 0.34 MB | Random Forest modeling experiments |
| Model ML_XGBoost.ipynb | Jupyter Notebook | 0.79 MB | XGBoost modeling experiments |
| Model_LGBM.ipynb | Jupyter Notebook | 0.81 MB | LightGBM modeling experiments |

---

## 2. Dataset Dimensions, Sizes & Schemas

| Table Name | Disk Size | Row Count | Column Count | Primary Key / IDs | Foreign Key |
|---|---|---|---|---|---|
| HC_application_train.csv | 158.44 MB | 307,511 | 122 | SK_ID_CURR | None (Root Table) |
| HC_bureau.csv | 162.14 MB | 1,716,428 | 17 | SK_ID_BUREAU | SK_ID_CURR |
| HC_bureau_balance.csv | 358.19 MB | 27,299,925 | 3 | Composite (SK_ID_BUREAU, MONTHS_BALANCE) | SK_ID_BUREAU |
| HC_credit_card_balance.csv | 404.91 MB | 3,840,312 | 23 | Composite (SK_ID_PREV, MONTHS_BALANCE) | SK_ID_CURR, SK_ID_PREV |
| HC_installments_payments.csv | 689.62 MB | 13,605,401 | 8 | Composite (SK_ID_PREV, NUM_INSTALMENT_NUMBER) | SK_ID_CURR, SK_ID_PREV |
| HC_POS_CASH_balance.csv | 374.51 MB | 10,001,358 | 8 | Composite (SK_ID_PREV, MONTHS_BALANCE) | SK_ID_CURR, SK_ID_PREV |
| HC_previous_application.csv | 386.21 MB | 1,670,214 | 37 | SK_ID_PREV | SK_ID_CURR |
| HC_sample_submission.csv | 0.51 MB | 48,744 | 2 | SK_ID_CURR | N/A |
| HomeCredit_columns_description.csv | 0.04 MB | 219 | 5 | Index | References Table & Row |

### Key Columns Summary:
- **Root Entity ID:** SK_ID_CURR (Applicant ID).
- **Target Column:** TARGET (1 = default / client had payment difficulties; 0 = non-default / loan repaid).
- **Key Financials:** AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE.
- **Key External Credit Scores:** EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 (strongest predictive signals).
- **Demographics:** CODE_GENDER, DAYS_BIRTH (age in negative days), DAYS_EMPLOYED, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_INCOME_TYPE, OCCUPATION_TYPE.
- **Social / Asset Indicators:** FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN, CNT_FAM_MEMBERS, DEF_30_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE.

---

## 3. Target Variable & Class Imbalance Analysis

In HC_application_train.csv (307,511 total applicants):
- **Non-default (TARGET = 0):** 282,686 applicants (91.93%)
- **Default (TARGET = 1):** 24,825 applicants (8.07%)
- **Imbalance Ratio:** 11.39 : 1 (approx. 1 default for every 11.4 non-defaults).

### Implications:
- Accuracy is highly misleading (a naive model predicting all zeros achieves 91.93% accuracy).
- Primary evaluation metrics **must** be ROC-AUC, PR-AUC, Precision, Recall, and F1-score for the positive class (defaults).
- Class imbalance must be handled using principled techniques (e.g., scale_pos_weight / class_weight='balanced', or threshold tuning on predicted probabilities) rather than naive global SMOTE before train/test split.

---

## 4. Existing Exploratory Data Analysis (EDA) Audit

The notebook EDA_merged.ipynb (289 cells) conducts extensive analysis across multiple tables:
1. **Application Train:** Missing value counts, distribution of income and credit amounts, age distributions, violin plots, and correlation heatmaps.
2. **Bureau & Bureau Balance:** Overdue loans, debt-to-credit ratios, active vs closed credit status distributions, and payment delays.
3. **Previous Applications:** Approval vs rejection rates by client type, product combination, and purpose.
4. **Installments & POS CASH:** Payment amounts vs installment amounts, days overdue, and default correlations.

### Key Data-Backed Business Insights Identified:
1. **External Credit Score Dominance:** EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 show strong negative correlations with default risk. Applicants with lower scores have dramatically higher default probabilities (over 20% default rate in the lowest decile).
2. **Age & Experience Gradient:** Younger applicants (under 30 years old) show default rates exceeding 11%, whereas mature applicants (above 55 years old) exhibit default rates below 5.5%.
3. **Credit-to-Income / Debt Burden:** High loan credit relative to annual income (Credit-to-Income > 4.0) increases default risk significantly.
4. **Education Level Impact:** Applicants with Higher Education have a default rate of ~5.3%, compared to ~10.9% for applicants with Secondary / Secondary Special education.
5. **Historical Repayment Friction:** Applicants with prior overdue credit recorded in Bureau (CREDIT_DAY_OVERDUE > 0 or higher AMT_CREDIT_SUM_OVERDUE) and rejected previous applications have markedly higher default rates.

---

## 5. Existing Preprocessing & Feature Engineering Audit

Preprocessing_merged.ipynb (463 cells) implemented:
- **Imputation:** Median imputation for continuous variables, constant/mode for categoricals.
- **Outlier Handling:** IQR trimming and winsorizing on numerical columns.
- **Encoding:** One-Hot Encoding via pd.get_dummies and Label Encoding for ordinal/binary columns.
- **Feature Engineering:** DAYS_ENTRY_PAYMENT_RATIO, AMT_PAYMENT_DIFFERENCES, PAYMENT_TO_BALANCE_RATIO, UTILIZATION_RATE, CNT_INSTALMENT_LOG.
- **Table Aggregations:** Aggregating installment payments and POS cash tables by SK_ID_PREV using .mean().

---

## 6. Critical Technical Issues & Data Leakage Discovered

### A. Catastrophic Sample Drop from Inner Merging (	rain_inner)
- In Preprocessing_merged.ipynb and all 4 model notebooks, the authors joined the historical tables using an **inner join**, creating pplication_train_inner_merged.csv.
- **Impact:** The dataset shrank from **307,511 applicants to only 11,043 applicants** (a 96.4% data loss!). Only applicants that simultaneously had active records across all sub-tables were retained.
- **Fix in Production:** Use the full pplication_train dataset (307,511 records) as the master anchor, computing aggregated historical features via applicant-level left joins (SK_ID_CURR), preserving 100% of applicants.

### B. Severe Preprocessing and SMOTE Leakage
- In Preprocessing_merged.ipynb (e.g. Cells 367, 373, 428, 431) and several notebook cells:
  1. StandardScaler and SimpleImputer were fitted on the **entire dataset** prior to 	rain_test_split.
  2. SMOTE oversampling was executed **before** train/test splitting in multiple experiments, generating synthetic points derived from test distributions.
- **Impact:** Artificially inflated training metrics (Train AUC = 1.00, Train F1 = 1.00) while Test Recall and F1 collapsed (Test Recall = 0.02 - 0.06).
- **Fix in Production:** Build a strict Scikit-Learn pipeline where all transformations (imputation, scaling, one-hot encoding) are fitted **only** on the training split, and applied to validation/test sets without leakage.

---

## 7. Model Experiments & Exact Metrics Extracted from Notebooks

All 4 notebooks evaluated models on the 	rain_inner dataset (11,043 rows, 80/20 train-test split: 8,834 train, 2,209 test).

### Summary Comparison of Best Runs from Existing Notebooks:

| Model Notebook | Experiment Setup | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.90 | 0.91 | 0.00 | 0.00 | 0.00 | 0.63 |
| **Logistic Regression** | SMOTE + Scaler | 0.63 | 0.63 | 0.13 | 0.48 | 0.20 | 0.60 |
| **Logistic Regression** | Hyperparameter Tuned | 0.67 | 0.67 | 0.17 | **0.62** | **0.27** | **0.71** |
| **Random Forest** | Baseline | 1.00 | 0.91 | 1.00 | 0.00 | 0.01 | 0.69 |
| **Random Forest** | Undersampling + Tuned | 0.68 | 0.65 | 0.16 | **0.66** | **0.26** | **0.71** |
| **Random Forest** | SMOTE + Tuned | 0.80 | 0.73 | 0.15 | 0.43 | 0.23 | 0.65 |
| **XGBoost** | Baseline | 0.90 | 0.90 | 0.39 | 0.06 | 0.11 | 0.68 |
| **XGBoost** | SMOTE + Tuned | 1.00 | 0.90 | 0.23 | 0.03 | 0.05 | 0.70 |
| **LightGBM** | Baseline | 0.97 | 0.91 | **0.58** | 0.03 | 0.07 | **0.71** |
| **LightGBM** | SMOTE + IQR + Tuned | 1.00 | 0.90 | 0.30 | 0.04 | 0.07 | **0.71** |

### Observations:
- In default configurations with threshold 0.5, all gradient boosting models (XGBoost, LightGBM) predict almost exclusively class 0, resulting in high accuracy (~91%) but near-zero recall (0.01 - 0.06).
- LightGBM and XGBoost achieve the highest ROC-AUC (~0.71 on the inner subset), while Logistic Regression and Random Forest with resampling achieved better default capture (recall 0.62 - 0.66).
- When trained on the **full dataset (307k records)** with proper feature engineering and class weighting, LightGBM typically reaches ROC-AUC **0.75 - 0.78**.

---

## 8. Requirements Gap Analysis

| Requirement Area | Status | Evidence / Gap Description |
|---|---|---|
| **1. Exploratory Data Analysis** | PARTIAL | Strong charts in EDA_merged.ipynb, but needs consolidation into 5 structured business insights with clean production rendering. |
| **2. Machine Learning Pipeline** | PARTIAL | 4 models experimented with in notebooks, but suffered from inner join row loss and leakage. Needs clean, leak-free Scikit-Learn/LightGBM production pipeline. |
| **3. Model Comparison & Metrics** | PARTIAL | Notebook metrics extracted; needs systematic evaluation on full data with ROC-AUC, PR-AUC, Precision, Recall, F1, and Confusion Matrix. |
| **4. Class Imbalance Handling** | PARTIAL | SMOTE and undersampling tested; needs clean scale_pos_weight / balanced class weights with calibrated probability thresholding. |
| **5. Explainable AI (SHAP)** | NOT STARTED | No SHAP or LIME code exists in any of the notebooks. Must build a full SHAP explainer module (global summary + local waterfall/force plots). |
| **6. Business-Readable Rules** | NOT STARTED | No decision rule module exists. Must implement analytical rule engine derived from EDA and feature importances. |
| **7. Natural Language to SQL** | NOT STARTED | No NL-to-SQL module exists. Must build dynamic SQL generator with SQLite analytical database. |
| **8. SQL Validation & Security** | NOT STARTED | No SQL security or validation exists. Must build sql_validator.py (AST/regex parser rejecting mutations, multi-statements, and injection). |
| **9. Hallucination Control** | NOT STARTED | Must implement schema grounding and unsupported query detection (e.g. eye colour, unsupported columns). |
| **10. Conversation Memory** | NOT STARTED | Must implement lightweight multi-turn context retention in Streamlit session state. |
| **11. Interactive Streamlit UI** | NOT STARTED | No web UI exists. Must build multi-page Streamlit application (Overview, EDA, Prediction, Explainability, Rules, Talk to Data). |
| **12. Docker Deployment** | NOT STARTED | No Dockerfile or docker-compose.yml exists. Must build and verify containerization. |
| **13. Documentation & Presentation**| NOT STARTED | Must produce README.md, ARCHITECTURE.md, METHODOLOGY.md, RESULTS.md, and presentation slides in PDF. |

---

## 9. Conclusion & Action Plan

1. **Preserve:** Keep all original notebooks and CSVs intact as research evidence.
2. **Fix & Upgrade ML:** Train production LightGBM and baseline models on the full applicant dataset with robust aggregated features, zero leakage, and threshold optimization.
3. **Build Core Engine:** Implement SHAP explainability, analytical business rules, SQLite analytical store, and secure NL-to-SQL pipeline with hallucination guards.
4. **Deploy & Package:** Create clean Streamlit UI, Dockerize the full solution, and generate comprehensive documentation.
