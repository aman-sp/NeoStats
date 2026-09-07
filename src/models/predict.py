import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple
from src.utils.config import (
    FINAL_MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_CONFIG_PATH, FEATURE_NAMES_PATH
)
from src.data.feature_engineering import engineer_features

class CalibratedRiskModel:
    """Compatibility wrapper that supports predict_proba() and exposes the underlying base model for SHAP."""
    def __init__(self, base_model, calibrator=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibrated_classifiers_ = []

    def predict_proba(self, X):
        if self.calibrator is None:
            return self.base_model.predict_proba(X)
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(np.clip(raw.reshape(-1, 1), 1e-6, 1 - 1e-6))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

class CreditRiskPredictor:
    """Production Inference Engine for Applicant Default Risk."""

    def __init__(self):
        if not os.path.exists(FINAL_MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError('Trained model artifacts not found. Please train models first.')

        self.model = joblib.load(FINAL_MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        with open(THRESHOLD_CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)

        self.low_threshold = self.config['low_risk_threshold']
        self.high_threshold = self.config['high_risk_threshold']
        self.selected_model_name = self.config['selected_model']
        self.calibration_applied = self.config.get('calibration_applied', False)
        self.calibrated_brier_score = None

        try:
            with open('models/model_comparison_metrics.json', 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            selected = self.selected_model_name
            selected_test = metrics['model_comparison'][selected]['test']
            if self.calibration_applied and 'calibrated_brier_score' in selected_test:
                self.calibrated_brier_score = selected_test['calibrated_brier_score']
        except Exception:
            self.calibrated_brier_score = None

    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability and assign risk band for a single applicant.
        """
        df_input = pd.DataFrame([applicant_data])
        return self.predict_batch(df_input)[0]

    def predict_batch(self, df_input: pd.DataFrame) -> list:
        """
        Predict probability and assign risk band for a batch of applicants.
        """
        # Ensure feature engineering is applied
        df_feats = engineer_features(df_input)

        # Transform features
        X_trans = self.preprocessor.transform(df_feats)

        # Predict probability
        proba = self.model.predict_proba(X_trans)[:, 1]

        results = []
        for p in proba:
            p_val = float(p)
            if p_val < self.low_threshold:
                band = 'LOW'
                band_desc = 'Low default risk: Strong financial and credit profile.'
            elif p_val >= self.high_threshold:
                band = 'HIGH'
                band_desc = 'High default risk: Elevated vulnerability or historical repayment friction.'
            else:
                band = 'MEDIUM'
                band_desc = 'Moderate default risk: Standard underwriting review recommended.'

            results.append({
                'default_probability': round(p_val, 4),
                'default_probability_percentage': f'{p_val * 100:.1f}%',
                'risk_band': band,
                'risk_band_description': band_desc,
                'model_name': self.selected_model_name,
                'calibration_applied': self.calibration_applied,
                'thresholds': {
                    'low_threshold': self.low_threshold,
                    'high_threshold': self.high_threshold
                }
            })
        return results
