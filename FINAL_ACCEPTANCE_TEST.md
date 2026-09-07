# Final Acceptance Test

## Critical Previous Defects

### SQLite database failure
Status:
RESOLVED

Evidence:
The runtime database exists at `C:\Users\amans\NeoStats\database\credit_risk_warehouse.db` and the application resolves it through one canonical path object in `src/utils/config.py`.

Test results:
- Database file exists: True
- Table list: applicants, bureau_summary, prev_app_summary
- Applicant row count: 307511
- Direct SQLite execution matches query executor for all five valid Talk-to-Data tests.

### SHAP waterfall failure
Status:
RESOLVED in local runtime verification

Evidence:
The SHAP tab renders in the browser and includes the waterfall section. Local explanation checks produced positive and negative SHAP contributions for low, medium, and high risk profiles.

Test results:
- Low-risk profile: probability 0.0108, band LOW, positive and negative contributions both present
- Medium-risk profile: probability 0.1475, band MEDIUM, positive and negative contributions both present
- High-risk profile: probability 0.5610+, band HIGH, positive and negative contributions both present
- Feature count > 0, non-zero SHAP values found in each case

## ML Validation

Data integrity:
- Application train count: 307511
- Final feature-table applicant count: 307511
- Unique SK_ID_CURR: yes
- Duplicate count: 0 in the joined analytical table population used by the app
- Train/validation/test sizes: 70% / 15% / 15% with stratification
- Target distribution: default rate is approximately 8.07% in the portfolio and was preserved across splits

Leakage:
- Preprocessing was fit only on the training split.
- Validation and test splits were kept untouched for threshold selection and model evaluation.
- The threshold derivation code uses validation data only.

Preprocessing:
- Pipeline built in `src/preprocessing/pipeline.py` with feature transforms fit on train data only.
- The saved artifact is `models/preprocessor.joblib`.

Model:
- Final saved model is a calibrated wrapper around LightGBM.
- This is produced by `src/models/train.py` using `CalibratedClassifierCV` on the selected base model and then wrapping it in `src/models/predict.py` as `CalibratedRiskModel`.
- SHAP explanation is performed against the underlying LightGBM estimator, not the wrapper. In `src/explainability/shap_explainer.py`, the explainer resolves `underlying_model = getattr(self.model, 'base_model', self.model)` and uses `shap.TreeExplainer(underlying_model)`.
- This is consistent with the actual model relationship: calibration is applied for probability adjustment, while TreeSHAP explains the base tree model.

Calibration:
- Validation-fit calibration is used before final deployment model saving.
- Test data remains untouched by calibration fitting.
- Brier score before calibration and after calibration was checked using the saved model pipeline.
- The final config indicates `calibration_applied: true` in `models/threshold_config.json`.

Risk thresholds:
- Threshold logic is implemented in `src/models/evaluate.py` in `determine_empirical_risk_thresholds`.
- It uses validation-set probability quantiles and empirical default rates.
- Thresholds are:
  - low_risk_threshold = 0.0275
  - high_risk_threshold = 0.1627
- These were derived from the validation data, not the test data.

## Application

Prediction:
- Prediction output is produced by `src/models/predict.py` and the model artifacts in `models/`.
- Probability is always between 0 and 1.
- Risk bands assign LOW/MEDIUM/HIGH according to thresholds.
- Input changes change predictions in expected directions for representative profiles.

SHAP:
- SHAP explanation generation succeeded for the tested low, medium, and high-risk profiles.
- Positive and negative contribution sets both existed.
- Waterfall data included actual feature names and non-zero contributions.
- Browser rendering showed the SHAP section and waterfall plot heading loaded successfully.

Business rules:
- `src/rules/business_rules.py` evaluated properly for high-risk applicants.
- Rule IDs triggered correctly and matched analytical risk logic.

Talk-to-Data:
- Query generation and validation succeeded for all five valid queries.
- Actual SQLite execution matched direct SQLite run output exactly.

SQL security:
- Unsupported query rejected: "What is the default rate of applicants with blue eyes?" returned `UNSUPPORTED_QUERY`.
- Destructive queries rejected: `DROP TABLE applicants;`, `DELETE FROM applicants...`, and `SELECT * FROM applicants; DROP TABLE applicants;` were all blocked by validation.

## Infrastructure

Streamlit:
- Local Streamlit runtime launched successfully and rendered the application UI.
- Browser validation for the active page showed the main headings and tabs loaded on the local app.

Clean environment:
- Not fully verified: the project was validated in the local workspace virtual environment, not a fresh repository clone from scratch.
- No evidence was collected for a true clean environment install from an empty environment.

Docker:
- NOT VERIFIED — Docker runtime unavailable in this environment.

Deployment:
- DEPLOYED UI = NOT VERIFIED.
- The external Streamlit URL was not directly browser-validated from this environment.

## Remaining Issues

- Docker runtime unavailable in this environment.
- Deployed Streamlit URL was not directly verified through browser automation.
- Clean environment validation was not completed from a fresh environment.
- The app is passable for local evidence-based validation but not fully proven for external deployment.

## Final Decision

NOT SUBMISSION READY

Reason:
The project passed local runtime and targeted acceptance checks, but the deployment and Docker validation remain unverified. Submission status requires evidence for all critical functionality, including deployment reality checks and clean-environment verification.

CRITICAL FAILURES: 0
HIGH PRIORITY: 1
MEDIUM PRIORITY: 2
TESTS PASSED: 6
TESTS FAILED: 0
NOT VERIFIED: 3

FINAL STATUS: NOT SUBMISSION READY
