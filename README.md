# AI-Powered Credit Risk Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20(Calibrated)-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20Warehouse-lightgrey.svg)](https://www.sqlite.org/)
[![Docker](https://img.shields.io/badge/Container-Docker%20%2B%20Compose-blue.svg)](https://www.docker.com/)

An end-to-end credit risk scoring, explainability, and conversational analytics platform built on the **Home Credit Default Risk** dataset.

> Note: the SQLite warehouse is generated locally at runtime via `python -m database.init_db` and is intentionally not stored in GitHub because of size limits.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Business Objective](#3-business-objective)
4. [Dataset Description & Structure](#4-dataset-description--structure)
5. [Exploratory Data Analysis & Business Insights](#5-exploratory-data-analysis--business-insights)
6. [Preprocessing & Leak-Free Pipeline](#6-preprocessing--leak-free-pipeline)
7. [Class Imbalance Management](#7-class-imbalance-management)
8. [Machine Learning Benchmark & Selection](#8-machine-learning-benchmark--selection)
9. [Final Model Performance & Metrics](#9-final-model-performance--metrics)
10. [Probability Output vs Calibration vs Risk Bands](#10-probability-output-vs-calibration-vs-risk-bands)
11. [Explainable AI (SHAP)](#11-explainable-ai-shap)
12. [Analytical Business Rules](#12-analytical-business-rules)
13. [Conversational "Talk to Data" System](#13-conversational-talk-to-data-system)
14. [SQL Security & Pre-Execution Validation](#14-sql-security--pre-execution-validation)
15. [Hallucination Control & Guardrails](#15-hallucination-control--guardrails)
16. [Conversation Memory](#16-conversation-memory)
17. [Interactive Streamlit UI](#17-interactive-streamlit-ui)
18. [Installation & Setup](#18-installation--setup)
19. [Environment Configuration](#19-environment-configuration)
20. [Local Execution Guide](#20-local-execution-guide)
21. [Docker Deployment Guide](#21-docker-deployment-guide)
22. [Tested Query Patterns](#22-tested-query-patterns)
23. [Automated Testing](#23-automated-testing)
24. [Known Limitations & Future Improvements](#24-known-limitations--future-improvements)

---

## 1. Project Overview
The **AI-Powered Credit Risk Intelligence Platform** bridges modern machine learning with institutional banking underwriting needs. It ingests complex applicant profiles and credit histories, predicts loan default probabilities, decomposes individual predictions using local SHAP explanations, evaluates analytical credit rules, and provides a secure natural-language SQL assistant for business analysts.

---

## 2. Problem Statement
Commercial retail lenders face three acute operational challenges:
1. **Severe Class Imbalance:** Loan defaults are rare events (~8% of applicants). Default model tuning often produces high accuracy by naively rejecting all positive default flags, leaving lenders vulnerable to catastrophic credit losses.
2. **Regulatory & Explainability Barriers:** Under regulatory frameworks such as Basel III and fair lending statutes, black-box credit decisions cannot be approved without transparent, audit-ready feature attribution.
3. **Information Friction:** Loan officers and risk committees need rapid, plain-English access to warehouse facts without writing manual SQL or risking data leaks.

---

## 3. Business Objective
- **Automate Risk Scoring:** Deliver calibrated default probability and segment applicants into actionable risk tiers (LOW, MEDIUM, HIGH).
- **Explain Every Decision:** Provide positive (risk-increasing) and negative (protective) factor attributions for every applicant.
- **Bridge Rules and ML:** Codify data-backed underwriting heuristics.
- **Democratize Data:** Allow natural language queries directly over the analytical data warehouse.
- **Deploy Safely:** Ensure complete separation between research notebooks and production-grade Python modules.

---

## 4. Dataset Description & Structure
The platform utilizes the **Home Credit Default Risk** relational dataset:
- **`HC_application_train.csv` (Root Entity):** 307,511 applicants, 122 initial attributes.
- **Target Variable (`TARGET`):**
  - `0`: Applicant repaid loan on schedule (282,686 records, 91.93%).
  - `1`: Applicant had severe payment difficulties / defaulted (24,825 records, 8.07%).
  - **Imbalance Ratio:** 11.39 : 1.
- **`HC_bureau.csv`:** 1,716,428 historical credits reported to the Credit Bureau.
- **`HC_previous_application.csv`:** 1,670,214 past loan applications submitted to Home Credit.

### Data Preservation Strategy:
Earlier research notebooks merged tables using an inner join, discarding 96.4% of applicants (shrinking 307k records down to 11k). This production system anchors on `application_train`, aggregates sub-tables to exactly one row per `SK_ID_CURR`, and applies left joins, **preserving 100% of the 307,511 applicants with zero duplication**.

---

## 5. Exploratory Data Analysis & Business Insights
Five empirical findings derived from data analysis across the portfolio:
1. **External Bureau Score Dominance:** The composite average of `EXT_SOURCE_1/2/3` exhibits the strongest monotonic negative correlation with default. Bottom decile applicants default at **21.4%**, compared to **2.1%** in the top decile (a 10.2x spread).
2. **Age & Employment Stability Gradient:** Default rates drop steadily from **12.3%** for borrowers under 25 years down to **5.4%** for borrowers aged 55+.
3. **Debt-to-Income Leverage:** Borrowers requesting loan credit exceeding 4.5x annual income default at **13.8%** vs **6.7%** for DTI < 2.0x.
4. **Educational Resilience:** Higher education holders default at **5.35%**, whereas lower secondary education holders default at **10.93%** (more than 2x higher).
5. **Historical Credit Delinquencies:** Applicants with past 30+ day bureau overdue records suffer a **28.6% default rate** vs **7.8%** for applicants with clean repayment records.

---

## 6. Preprocessing & Leak-Free Pipeline
All data transformations strictly adhere to leak-free engineering:
- **Stratified Partitioning:** 70% Train (215,257), 15% Validation (46,127), 15% Test (46,127).
- **Imputation:** Median imputation for continuous features, most_frequent imputation for categoricals.
- **Encoding:** Scikit-Learn `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Leakage Prevention:** Imputers, encoders, and scalers are fitted **strictly on the Train split** and applied downstream to Validation and Test sets.

---

## 7. Class Imbalance Management
- **No Global Pre-Split SMOTE:** Synthetic oversampling before train/test splitting contaminated prior notebooks.
- **Principled Weighting:**
  - `scale_pos_weight = 11.39` for LightGBM and XGBoost.
  - `class_weight = 'balanced'` for Logistic Regression and Random Forest.
  - Combined with post-hoc probability threshold tuning on held-out validation data.

---

## 8. Machine Learning Benchmark & Selection
All 4 candidate models were trained and benchmarked on identical held-out test sets (46,127 records) on the complete 307,511 dataset:

| Model | Test ROC-AUC | Test PR-AUC | Recall (Defaults) | Precision | F1-Score | Test Accuracy | Brier Score |
|---|---|---|---|---|---|---|---|
| **LightGBM (Selected)** | **0.7716** | **0.2610** | **68.98%** | **17.72%** | **0.2841** | 69.31% | **0.0670** (Calibrated) |
| **XGBoost** | 0.7695 | 0.2560 | 68.47% | 17.65% | 0.2831 | 69.17% | 0.1862 (Raw) |
| **Logistic Regression** | 0.7544 | 0.2342 | 68.29% | 16.52% | 0.2663 | 67.28% | 0.2037 (Raw) |
| **Random Forest** | 0.7507 | 0.2298 | 54.73% | 19.47% | 0.2872 | 75.16% | 0.0712 (Raw) |

**Selection Rationale:** LightGBM achieved the highest discriminatory power across both ROC-AUC (0.7716) and Precision-Recall AUC (0.2610), while identifying 68.98% of all true defaults.

---

## 9. Final Model Performance & Metrics
- **Test Confusion Matrix (Threshold 0.5 on weighted probabilities):**
  - True Negatives: 29,403 | False Positives: 13,000
  - False Negatives: 1,155 | True Positives: 2,569
  - **Default Capture (Recall):** 68.98% of true defaults captured.

---

## 10. Probability Output vs Calibration vs Risk Bands
The platform enforces a rigorous three-way conceptual separation:
1. **Raw Model Output:** Direct output from tree boosting reflects ranking rather than true posterior probability under class reweighting.
2. **Probability Calibration:** Sigmoid calibration fitted on validation data reduced Brier score error from **0.1856 to 0.0670** (a 63.9% improvement).
3. **Empirical Risk Threshold Selection:** Quantile analysis on 46,127 validation records established empirical cutoffs:
   - **LOW RISK (< 2.93%):** Validated historical default rate is strictly below 3.0%.
   - **MEDIUM RISK (2.93% to 15.31%):** Standard underwriting review band.
   - **HIGH RISK (>= 15.31%):** High-risk concentration band where empirical default rate exceeds 24.8%.

---

## 11. Explainable AI (SHAP)
- **Engine:** `shap.TreeExplainer` on the production model.
- **Decomposition:** Unpacks predictions into top risk-increasing factors (positive SHAP values) and risk-reducing mitigants (negative SHAP values).
- **Human Translation:** Automatically converts technical variable names (e.g. `EXT_SOURCE_2`, `CREDIT_INCOME_PERCENT`) into plain-English banking terms.
- **Visualization:** Renders interactive waterfall plots in the Streamlit UI.

---

## 12. Analytical Business Rules
The platform provides 6 evidence-backed analytical underwriting rules:
1. **RULE-ANL-01:** Severe External Bureau Score Depletion (`EXT_SOURCES_MEAN < 0.30`).
2. **RULE-ANL-02:** Excessive Debt-to-Income Leverage (`Credit / Income > 4.5x` or `Annuity / Income > 35%`).
3. **RULE-ANL-03:** Historical Bureau Delinquency (`Max Overdue Days > 30` or `Overdue Debt > $5,000`).
4. **RULE-ANL-04:** Institutional Rejection Precedent (`Past Loan Refusal Rate >= 50%`).
5. **RULE-ANL-05:** Youth & Short Employment Tenancy (`Age < 26` and `Employed < 1.5 yrs`).
6. **RULE-ANL-06 (Protective):** Prime Stability Profile (`EXT_SOURCES_MEAN > 0.65` and clean bureau history).

*Notice: These are purely analytical model-derived decision aids, not official statutory banking policy.*

---

## 13. Conversational "Talk to Data" System
A natural language query interface grounded exclusively in the SQLite database (`database/credit_risk_warehouse.db`):
- **Flow:** User Question $\rightarrow$ SQL Generator $\rightarrow$ SQL Validator $\rightarrow$ SQLite Execution $\rightarrow$ Plain-English Formatter.
- **Transparency:** The generated SQL is displayed alongside execution latency and row counts.
- **Grounding Guarantee:** All numerical figures in answers are extracted directly from database execution tables.

---

## 14. SQL Security & Pre-Execution Validation
The `src/chatbot/sql_validator.py` module runs **strictly before execution**:
- Enforces read-only `SELECT` statements.
- Rejects `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `ATTACH`, `PRAGMA`.
- Rejects semicolon chaining and multi-statement injection.
- Validates table names against whitelist (`applicants`, `bureau_summary`, `prev_app_summary`).
- Pre-flight dry-run syntax verification using SQLite `EXPLAIN`.

---

## 15. Hallucination Control & Guardrails
- Questions requesting attributes not present in the credit database (e.g., eye colour, hair, pet, vehicle brand) are intercepted and rejected with informative messages:
  > *"I cannot answer that question using the available dataset because 'blue eyes' is not recorded in the analytical credit risk warehouse."*

---

## 16. Conversation Memory
Maintains multi-turn context in `st.session_state`. Follow-up queries (e.g., *"What about females?"* following a gender question) resolve automatically against conversational history.

---

## 17. Interactive Streamlit UI
A 6-tab dashboard:
1. **Executive Overview:** High-level KPIs, benchmark comparison table, and threshold methodology.
2. **Exploratory Data Analysis:** 5 interactive Plotly visualizations with business interpretations.
3. **Risk Scoring & Assessment:** Interactive applicant input form with presets and real-time risk band badges.
4. **Explainable AI (SHAP):** Waterfall attribution plot and top positive/negative drivers.
5. **Analytical Business Rules:** Live policy evaluator for the current applicant.
6. **Talk to Data:** Conversational SQL chat interface with quick test prompt buttons.

---

## 18. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/aman-sp/NeoStats.git
cd NeoStats

# Create and activate virtual environment (Python 3.11+ recommended)
python -m venv .venv
.venv\Scripts\activate  # On macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 19. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure optional LLM API keys if desired:
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```
*Note: If no API key is provided, the platform operates completely using its verified deterministic SQL engine.*

---

## 20. Local Execution Guide

```bash
# 1. Initialize SQLite analytical warehouse
python -m database.init_db

# 2. Run automated validation
python -m pytest -q

# 3. Launch Streamlit UI
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 21. Docker Deployment Guide

The platform is fully containerized:
```bash
# Build and run container
docker compose up --build
```
Access the application at `http://localhost:8501`.

---

## 22. Tested Query Patterns
All 5 mandatory query patterns plus hallucination control are verified:
1. **Aggregation:** *"What is the average income of applicants?"*
2. **Filtering:** *"How many applicants with income above 200000 defaulted?"*
3. **Grouping:** *"What is the default rate by gender?"*
4. **Ranking:** *"Which occupation has the highest default rate?"*
5. **Comparison:** *"Compare the default rate of applicants with and without cars."*
6. **Hallucination Control Test:** *"What is the default rate of applicants with blue eyes?"*

---

## 23. Automated Testing
Run the automated regression suite covering prediction, SHAP, rules, SQL validation, database execution, and NL-to-SQL behavior:
```bash
python -m pytest -q
```
The project currently validates 7 core checks successfully in under 7 seconds.

---

## 24. Known Limitations & Future Improvements
1. **Loss Given Default (LGD) Modeling:** Current system models Probability of Default (PD). Future iterations will incorporate LGD and Exposure at Default (EAD) to estimate continuous Expected Loss ($EL = PD \times LGD \times EAD$).
2. **Social Graph Analytics:** Incorporate graph embeddings on social circle delinquency indicators (`DEF_30_CNT_SOCIAL_CIRCLE`, `DEF_60_CNT_SOCIAL_CIRCLE`) to flag organized default rings.
3. **Dynamic Drift Monitoring:** Integrate Population Stability Index (PSI) tracking to detect macro-economic concept drift in incoming applicant streams.
