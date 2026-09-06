# System Architecture — NeoStats Credit Risk Intelligence Platform

## 1. Architectural Overview

The **AI-Powered Credit Risk Intelligence Platform** is structured as an end-to-end, multi-tier decision-support system designed for credit risk assessment, explainable AI, regulatory auditability, and conversational data exploration.

```
+-----------------------------------------------------------------------------+
|                          PRESENTATION LAYER (STREAMLIT)                     |
|  [Executive Overview]  [EDA Insights]  [Risk Scoring]  [SHAP]  [Rules] [Chat]|
+---------------------------------------+-------------------------------------+
                                        |
+---------------------------------------v-------------------------------------+
|                              APPLICATION LAYER                              |
|   +-----------------------+   +-------------------+   +------------------+  |
|   |  CreditRiskPredictor  |   |CreditRiskExplainer|   |BusinessRulesEng. |  |
|   +-----------+-----------+   +---------+---------+   +--------+---------+  |
|               |                         |                      |            |
|   +-----------v-------------------------v----------------------v---------+  |
|   |                    Preprocessing & Feature Engineering               |  |
|   +-------------------------------------+--------------------------------+  |
+-----------------------------------------|-----------------------------------+
                                          |
+-----------------------------------------v-----------------------------------+
|                        CONVERSATIONAL TALK-TO-DATA LAYER                    |
|   User Query ---> NLToSQLGenerator ---> SQLValidator (Pre-execution AST)    |
|                          |                     | (Rejects DDL/DML/Attacks)  |
|                          v                     v                            |
|                 LLM API / Pattern Engine ---> QueryExecutor (Read-Only)     |
|                                                |                            |
|                                                v                            |
|                                       ResponseFormatter                     |
+-----------------------------------------+-----------------------------------+
                                          |
+-----------------------------------------v-----------------------------------+
|                       DATA & PERSISTENCE LAYER (SQLITE)                     |
|      [applicants]        [bureau_summary]        [prev_app_summary]         |
|     (307,511 rows)        (305,811 rows)           (338,857 rows)           |
+-----------------------------------------------------------------------------+
```

---

## 2. Component Design & Responsibilities

### A. Data & Preprocessing Layer (`src/data/`, `src/preprocessing/`)
- **`loader.py`**: Ingests `HC_application_train.csv` (307,511 records). Computes applicant-level aggregations from Credit Bureau (`HC_bureau.csv`) and Previous Applications (`HC_previous_application.csv`) and joins via `SK_ID_CURR` using left joins. Guarantees 1-to-1 applicant row preservation without duplication.
- **`feature_engineering.py`**: Computes financial risk ratios (Credit-to-Income, Annuity-to-Income, Payment Rate, Employment Stability, External Source composites).
- **`pipeline.py`**: Strict Scikit-Learn `ColumnTransformer` fitted **exclusively** on training splits, eliminating target, scaling, or imputation leakage.

### B. Machine Learning Layer (`src/models/`)
- **`train.py`**: Benchmarks Logistic Regression, Random Forest, XGBoost, and LightGBM with proper class weighting (`scale_pos_weight` / `class_weight='balanced'`).
- **`evaluate.py`**: Computes ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Confusion Matrix, Brier Score, and Calibration Curves.
- **`predict.py`**: Loads frozen preprocessor and calibrated model artifacts to generate default probabilities and assign risk bands.

### C. Explainability Layer (`src/explainability/`)
- **`shap_explainer.py`**: Utilizes `shap.TreeExplainer` on the production model. Maps technical variable names into human-readable banking terms and segments local explanations into top risk-increasing and risk-reducing drivers.

### D. Analytical Business Rules Engine (`src/rules/`)
- **`business_rules.py`**: Bridges machine learning feature importances and domain credit risk policies into human-interpretable analytical rules. Evaluates applicant profile against 6 evidence-backed rules.

### E. Conversational Talk-to-Data Layer (`src/chatbot/`)
- **`prompts.py`**: Injects live SQLite schema and strict hallucination boundaries into LLM system prompts.
- **`sql_validator.py`**: Multi-stage pre-execution security parser. Enforces read-only `SELECT` queries, rejects multi-statements, and validates table whitelist.
- **`sql_generator.py`**: Converts user natural language into SQL. Supports external LLM providers (Gemini / OpenAI) with seamless fallback to verified deterministic query templates.
- **`query_executor.py`**: Executes queries in read-only mode (`mode=ro`) on SQLite analytical database.
- **`response_formatter.py`**: Grounds conversational answers strictly in database execution results.

---

## 3. Security & Governance Architecture

1. **Database Immutability**: All analytical queries are executed via read-only URI connections (`file:db.sqlite?mode=ro`).
2. **Zero Destructive Queries**: `sql_validator.py` blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `PRAGMA`, and semicolon chaining before the query reaches the database engine.
3. **Hallucination Containment**: Questions referencing attributes not present in the data warehouse (e.g. eye colour, race, religion, pet) are intercepted and rejected with informative messages.
4. **Credential Isolation**: Zero API keys are hardcoded; credentials are read dynamically from `.env` or user runtime input.
