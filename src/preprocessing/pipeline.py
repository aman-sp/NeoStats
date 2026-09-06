import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger('PreprocessingPipeline')

# Core candidate features with highest predictive signal
SELECTED_NUMERICAL_FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'PAYMENT_RATE',
    'GOODS_CREDIT_RATIO', 'DAYS_EMPLOYED_PERCENT', 'AGE_YEARS', 'EMPLOYED_YEARS',
    'INCOME_PER_PERSON', 'EXT_SOURCES_MEAN', 'EXT_SOURCES_MIN', 'EXT_SOURCES_MAX',
    'BUREAU_LOAN_COUNT', 'BUREAU_MAX_DAYS_OVERDUE', 'BUREAU_TOTAL_CREDIT_SUM',
    'BUREAU_TOTAL_DEBT_SUM', 'BUREAU_MAX_OVERDUE', 'BUREAU_ACTIVE_LOANS',
    'PREV_APP_COUNT', 'PREV_APP_TOTAL_CREDIT', 'PREV_APP_REFUSED_COUNT',
    'PREV_APP_APPROVED_COUNT', 'PREV_APP_REFUSAL_RATE'
]

SELECTED_CATEGORICAL_FEATURES = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
]

def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    scale_numeric: bool = False
) -> ColumnTransformer:
    """
    Build a leak-free Scikit-Learn ColumnTransformer.
    - Fits strictly on training split.
    - Imputes numerical features with median.
    - Imputes categorical features with most_frequent and OneHotEncodes.
    - Optionally applies StandardScaler (essential for Logistic Regression, optional for Trees).
    """
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    num_pipeline = Pipeline(num_steps)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor

def get_feature_names(column_transformer: ColumnTransformer) -> List[str]:
    """Extract output feature names from fitted ColumnTransformer."""
    feature_names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == 'remainder' or trans == 'drop':
            continue
        if hasattr(trans, 'named_steps') and 'encoder' in trans.named_steps:
            enc = trans.named_steps['encoder']
            cat_names = list(enc.get_feature_names_out(cols))
            feature_names.extend(cat_names)
        else:
            feature_names.extend(cols)
    return feature_names
