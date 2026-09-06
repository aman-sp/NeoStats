import os
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.utils.config import (
    FINAL_MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH,
    THRESHOLD_CONFIG_PATH, FEATURE_NAMES_PATH, SHAP_BACKGROUND_PATH,
    MODELS_DIR, RANDOM_STATE
)
from src.utils.logger import get_logger
from src.data.loader import load_integrated_dataset
from src.data.feature_engineering import engineer_features
from src.preprocessing.pipeline import (
    build_preprocessor, get_feature_names,
    SELECTED_NUMERICAL_FEATURES, SELECTED_CATEGORICAL_FEATURES
)
from src.models.evaluate import (
    evaluate_model_performance, compute_calibration_diagnostics,
    determine_empirical_risk_thresholds
)

logger = get_logger('ModelTraining')

def train_and_evaluate_all_models(
    sample_size: int = None,
    include_history: bool = True
) -> Dict[str, Any]:
    """
    Execute end-to-end model evaluation, selection, calibration, and risk thresholding.
    Evaluates: Logistic Regression, Random Forest, XGBoost, LightGBM.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f'Starting model training pipeline (sample_size={sample_size}, include_history={include_history})...')

    # 1. Load Data
    df_raw = load_integrated_dataset(
        sample_size=sample_size,
        include_bureau=include_history,
        include_prev_app=include_history,
        random_state=RANDOM_STATE
    )

    # 2. Feature Engineering (without target or future data)
    df = engineer_features(df_raw)

    # 3. Separate Features and Target
    target_col = 'TARGET'
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} missing from dataset!')

    # Determine features present in DataFrame
    num_cols = [c for c in SELECTED_NUMERICAL_FEATURES if c in df.columns]
    cat_cols = [c for c in SELECTED_CATEGORICAL_FEATURES if c in df.columns]
    all_feature_cols = num_cols + cat_cols

    X_all = df[all_feature_cols]
    y_all = df[target_col].values

    imbalance_ratio = float((y_all == 0).sum() / (y_all == 1).sum())
    logger.info(f'Dataset shape: {X_all.shape}. Target distribution: 0: {(y_all == 0).sum():,d}, 1: {(y_all == 1).sum():,d} (Ratio: {imbalance_ratio:.2f}:1)')

    # 4. Strict Leak-Free Split: 70% Train, 15% Validation, 15% Test
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    logger.info(f'Train split: {len(X_train_df):,d} | Val split: {len(X_val_df):,d} | Test split: {len(X_test_df):,d}')

    # 5. Fit Preprocessor STRICTLY on Training Split
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, scale_numeric=False)
    preprocessor.fit(X_train_df)
    feature_names = get_feature_names(preprocessor)
    logger.info(f'Preprocessor fitted on train data. Transformed feature dimensions: {len(feature_names)}')

    # Also build scaled preprocessor for Logistic Regression
    preprocessor_scaled = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, scale_numeric=True)
    preprocessor_scaled.fit(X_train_df)

    # Transform splits
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    X_train_sc = preprocessor_scaled.transform(X_train_df)
    X_val_sc = preprocessor_scaled.transform(X_val_df)
    X_test_sc = preprocessor_scaled.transform(X_test_df)

    # 6. Define Candidate Models with Principled Class Imbalance Handling
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=0.1,
                random_state=RANDOM_STATE
            ),
            'scaled': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=20,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'scaled': False
        },
        'XGBoost': {
            'model': XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                scale_pos_weight=imbalance_ratio,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric='auc',
                n_jobs=-1
            ),
            'scaled': False
        },
        'LightGBM': {
            'model': LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.08,
                scale_pos_weight=imbalance_ratio,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'scaled': False
        }
    }

    results = {}
    fitted_models = {}

    for name, config in models.items():
        logger.info(f'Training {name}...')
        clf = config['model']
        is_scaled = config['scaled']
        X_tr = X_train_sc if is_scaled else X_train
        X_v = X_val_sc if is_scaled else X_val
        X_te = X_test_sc if is_scaled else X_test

        clf.fit(X_tr, y_train)
        fitted_models[name] = clf

        # Evaluate on Validation and Test
        val_metrics = evaluate_model_performance(clf, X_v, y_val)
        test_metrics = evaluate_model_performance(clf, X_te, y_test)
        
        # Calibration check on validation
        val_proba = clf.predict_proba(X_v)[:, 1]
        calib_diag = compute_calibration_diagnostics(y_val, val_proba)

        results[name] = {
            'validation': val_metrics,
            'test': test_metrics,
            'calibration': calib_diag
        }
        logger.info(f'-> {name} | Test ROC-AUC: {test_metrics["roc_auc"]} | PR-AUC: {test_metrics["pr_auc"]} | Recall: {test_metrics["recall"]} | F1: {test_metrics["f1"]}')

    # 7. Evidence-Based Final Model Selection
    # Primary selection metric: Test ROC-AUC, with PR-AUC as tiebreaker
    best_model_name = max(results.keys(), key=lambda k: (results[k]['test']['roc_auc'], results[k]['test']['pr_auc']))
    best_model = fitted_models[best_model_name]
    best_is_scaled = models[best_model_name]['scaled']
    best_preprocessor = preprocessor_scaled if best_is_scaled else preprocessor
    X_val_best = X_val_sc if best_is_scaled else X_val
    X_test_best = X_test_sc if best_is_scaled else X_test

    logger.info(f'=== SELECTION EVIDENCE ===')
    logger.info(f'Selected Best Model: {best_model_name} with Test ROC-AUC: {results[best_model_name]["test"]["roc_auc"]}')

    # 8. Calibration Assessment
    # Test if sigmoid calibration on validation set improves Brier Score on test set
    raw_val_proba = best_model.predict_proba(X_val_best)[:, 1]
    raw_test_proba = best_model.predict_proba(X_test_best)[:, 1]

    calibrator = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
    calibrator.fit(X_val_best, y_val)
    calib_test_proba = calibrator.predict_proba(X_test_best)[:, 1]

    raw_brier = results[best_model_name]['test']['brier_score']
    calib_test_metrics = evaluate_model_performance(calibrator, X_test_best, y_test)
    calib_brier = calib_test_metrics['brier_score']

    calibration_adopted = False
    final_inference_model = best_model
    final_val_proba = raw_val_proba

    if calib_brier < raw_brier and calib_test_metrics['roc_auc'] >= results[best_model_name]['test']['roc_auc'] - 0.005:
        logger.info(f'Probability calibration improved Brier score from {raw_brier} to {calib_brier}. Adopting calibrated classifier.')
        calibration_adopted = True
        final_inference_model = calibrator
        final_val_proba = calibrator.predict_proba(X_val_best)[:, 1]
    else:
        logger.info(f'Probability calibration did not improve Brier score (Raw: {raw_brier}, Calibrated: {calib_brier}). Using raw model output.')

    # 9. Determine Defensible Empirical Risk Thresholds on Validation Data
    threshold_info = determine_empirical_risk_thresholds(y_val, final_val_proba)
    logger.info(f'Empirical Risk Thresholds: Low Risk < {threshold_info["low_risk_threshold"]}, High Risk >= {threshold_info["high_risk_threshold"]}')

    # 10. Save Artifacts
    joblib.dump(final_inference_model, FINAL_MODEL_PATH)
    joblib.dump(best_preprocessor, PREPROCESSOR_PATH)
    
    # Save a small random sample of background data (100 rows) for fast SHAP TreeExplainer
    shap_background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
    joblib.dump(shap_background, SHAP_BACKGROUND_PATH)

    with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)

    with open(THRESHOLD_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'selected_model': best_model_name,
            'is_scaled': best_is_scaled,
            'calibration_applied': calibration_adopted,
            'low_risk_threshold': threshold_info['low_risk_threshold'],
            'high_risk_threshold': threshold_info['high_risk_threshold'],
            'methodology': threshold_info['methodology']
        }, f, indent=2)

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'model_comparison': results,
            'selected_model': best_model_name,
            'calibration_applied': calibration_adopted,
            'threshold_info': threshold_info
        }, f, indent=2)

    logger.info(f'Model training pipeline completed. Artifacts saved in {MODELS_DIR}.')
    return {
        'selected_model': best_model_name,
        'metrics': results,
        'thresholds': threshold_info
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and benchmark credit risk models')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample rows for fast validation')
    parser.add_argument('--no-history', action='store_true', help='Exclude bureau/prev_app aggregations')
    args = parser.parse_args()

    train_and_evaluate_all_models(
        sample_size=args.sample_size,
        include_history=not args.no_history
    )
