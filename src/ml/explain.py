import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from src.utils.config import (
    FINAL_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, SHAP_BACKGROUND_PATH
)
from src.data.feature_engineering import engineer_features

# Plain-English translation mapping for technical features
FEATURE_NAME_MAPPINGS = {
    'EXT_SOURCE_1': 'External Credit Bureau Score 1 (Normalized)',
    'EXT_SOURCE_2': 'External Credit Bureau Score 2 (Normalized)',
    'EXT_SOURCE_3': 'External Credit Bureau Score 3 (Normalized)',
    'EXT_SOURCES_MEAN': 'Average External Credit Score across rating agencies',
    'EXT_SOURCES_MIN': 'Lowest External Credit Score',
    'CREDIT_INCOME_PERCENT': 'Credit Burden (Total Credit to Annual Income Ratio)',
    'ANNUITY_INCOME_PERCENT': 'Annuity Burden (Annual Loan Payment to Income Ratio)',
    'PAYMENT_RATE': 'Payment Rate (Annuity to Total Credit Amount)',
    'GOODS_CREDIT_RATIO': 'Goods Price to Credit Ratio',
    'AGE_YEARS': 'Applicant Age (in years)',
    'EMPLOYED_YEARS': 'Employment Stability (Years in Current Employment)',
    'DAYS_BIRTH': 'Applicant Age (Days from Birth)',
    'DAYS_EMPLOYED': 'Employment Duration (Days)',
    'DAYS_EMPLOYED_PERCENT': 'Employment to Age Ratio',
    'INCOME_PER_PERSON': 'Household Per-Capita Income',
    'AMT_INCOME_TOTAL': 'Total Annual Income ($)',
    'AMT_CREDIT': 'Total Loan Credit Amount ($)',
    'AMT_ANNUITY': 'Annual Loan Annuity Payment ($)',
    'AMT_GOODS_PRICE': 'Goods Purchase Price ($)',
    'BUREAU_LOAN_COUNT': 'Total Past Credit Bureau Inquiries / Loans',
    'BUREAU_ACTIVE_LOANS': 'Count of Currently Active Bureau Credits',
    'BUREAU_MAX_DAYS_OVERDUE': 'Historical Maximum Overdue Days in Bureau',
    'BUREAU_MAX_OVERDUE': 'Historical Maximum Overdue Amount ($)',
    'BUREAU_TOTAL_CREDIT_SUM': 'Total Historical Credit Granted in Bureau ($)',
    'BUREAU_TOTAL_DEBT_SUM': 'Total Outstanding Credit Bureau Debt ($)',
    'PREV_APP_COUNT': 'Previous Loan Applications with Home Credit',
    'PREV_APP_REFUSED_COUNT': 'Previous Refused Loan Applications',
    'PREV_APP_APPROVED_COUNT': 'Previous Approved Loan Applications',
    'PREV_APP_REFUSAL_RATE': 'Previous Application Rejection Rate',
    'NAME_EDUCATION_TYPE_Higher education': 'Higher Education Degree',
    'NAME_EDUCATION_TYPE_Secondary / secondary special': 'Secondary Education Only',
    'NAME_INCOME_TYPE_Working': 'Income Source: Formal Employment',
    'NAME_INCOME_TYPE_Commercial associate': 'Income Source: Commercial Associate',
    'NAME_INCOME_TYPE_State servant': 'Income Source: Government / Civil Servant',
    'DEF_30_CNT_SOCIAL_CIRCLE': 'Default count in applicant social circle (30 days)',
    'DEF_60_CNT_SOCIAL_CIRCLE': 'Default count in applicant social circle (60 days)',
    'REGION_RATING_CLIENT_W_CITY': 'Regional Credit Risk Rating'
}

def get_readable_feature_name(raw_name: str) -> str:
    """Translate raw technical feature name into plain-English business name."""
    if raw_name in FEATURE_NAME_MAPPINGS:
        return FEATURE_NAME_MAPPINGS[raw_name]
    for key, val in FEATURE_NAME_MAPPINGS.items():
        if raw_name.startswith(key):
            return f'{val} ({raw_name})'
    return raw_name.replace('_', ' ').title()

class CreditRiskExplainer:
    """SHAP Explainer for individual applicant credit risk predictions."""

    def __init__(self):
        if not os.path.exists(FINAL_MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError('Model artifacts not found.')

        self.model = joblib.load(FINAL_MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)

        # Handle CalibratedClassifierCV wrapper if adopted
        underlying_model = self.model
        if hasattr(self.model, 'calibrated_classifiers_'):
            underlying_model = self.model.calibrated_classifiers_[0].estimator

        # Create TreeExplainer or LinearExplainer based on model type
        if hasattr(underlying_model, 'tree_') or hasattr(underlying_model, 'booster_') or hasattr(underlying_model, 'estimators_'):
            self.explainer = shap.TreeExplainer(underlying_model)
        else:
            shap_bg = joblib.load(SHAP_BACKGROUND_PATH)
            self.explainer = shap.LinearExplainer(underlying_model, shap_bg)

    def explain_instance(self, df_instance: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """
        Compute SHAP explanation for a single applicant instance.
        Returns top risk-increasing and top risk-reducing factors with readable descriptions.
        """
        df_feats = engineer_features(df_instance)
        X_trans = self.preprocessor.transform(df_feats)

        # Calculate SHAP values
        shap_values = self.explainer(X_trans)
        
        # Extract values for the positive default class (class 1)
        if len(shap_values.values.shape) == 3:
            vals = shap_values.values[0, :, 1]
            base_val = float(shap_values.base_values[0, 1])
        else:
            vals = shap_values.values[0, :]
            base_val = float(shap_values.base_values[0])

        feature_values = X_trans[0]
        factors = []
        for i, (name, val, fval) in enumerate(zip(self.feature_names, vals, feature_values)):
            factors.append({
                'feature_raw': name,
                'feature_readable': get_readable_feature_name(name),
                'shap_value': float(val),
                'feature_value': float(fval)
            })

        # Risk-increasing: highest positive SHAP values
        risk_increasing = sorted([f for f in factors if f['shap_value'] > 0], key=lambda x: x['shap_value'], reverse=True)[:top_n]
        # Risk-reducing: most negative SHAP values
        risk_reducing = sorted([f for f in factors if f['shap_value'] < 0], key=lambda x: x['shap_value'])[:top_n]

        return {
            'base_value': round(base_val, 4),
            'risk_increasing_factors': risk_increasing,
            'risk_reducing_factors': risk_reducing,
            'all_factors': factors
        }

    def generate_waterfall_plot(self, df_instance: pd.DataFrame, max_display: int = 10) -> plt.Figure:
        """
        Generate a matplotlib Figure with SHAP explanation for Streamlit.
        """
        df_feats = engineer_features(df_instance)
        X_trans = self.preprocessor.transform(df_feats)
        shap_values = self.explainer(X_trans)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Select positive class explanation if multi-output
        if len(shap_values.values.shape) == 3:
            single_explanation = shap.Explanation(
                values=shap_values.values[0, :, 1],
                base_values=shap_values.base_values[0, 1],
                data=X_trans[0],
                feature_names=[get_readable_feature_name(f) for f in self.feature_names]
            )
        else:
            single_explanation = shap.Explanation(
                values=shap_values.values[0, :],
                base_values=shap_values.base_values[0],
                data=X_trans[0],
                feature_names=[get_readable_feature_name(f) for f in self.feature_names]
            )

        shap.plots.waterfall(single_explanation, max_display=max_display, show=False)
        plt.tight_layout()
        return plt.gcf()
