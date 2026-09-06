import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
DATABASE_DIR = BASE_DIR / 'database'
DOCUMENTS_DIR = BASE_DIR / 'documents'

# Raw Data Files
APPLICATION_TRAIN_CSV = DATA_DIR / 'HC_application_train.csv'
BUREAU_CSV = DATA_DIR / 'HC_bureau.csv'
PREVIOUS_APPLICATION_CSV = DATA_DIR / 'HC_previous_application.csv'
COLUMNS_DESCRIPTION_CSV = DATA_DIR / 'HomeCredit_columns_description.csv'

# Database
SQLITE_DB_PATH = DATABASE_DIR / 'credit_risk_warehouse.db'

# Modeling & Random State
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

# Default Model Artifact Paths
FINAL_MODEL_PATH = MODELS_DIR / 'final_credit_model.joblib'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.joblib'
METRICS_PATH = MODELS_DIR / 'model_comparison_metrics.json'
THRESHOLD_CONFIG_PATH = MODELS_DIR / 'threshold_config.json'
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.json'
SHAP_BACKGROUND_PATH = MODELS_DIR / 'shap_background.joblib'
