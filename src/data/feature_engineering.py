import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger('FeatureEngineering')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific credit risk features.
    Anchored on applicant attributes without data leakage.
    """
    logger.info('Engineering domain credit risk features...')
    df = df.copy()

    # 1. Handle anomalous DAYS_EMPLOYED
    # 365,243 is a known encoding for pensioner / not employed
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int32)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # 2. Credit and Income Ratios
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_PERCENT'] = (df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)).astype(np.float32)
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = (df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)).astype(np.float32)
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['PAYMENT_RATE'] = (df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)).astype(np.float32)
    if 'AMT_GOODS_PRICE' in df.columns and 'AMT_CREDIT' in df.columns:
        df['GOODS_CREDIT_RATIO'] = (df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1e-5)).astype(np.float32)

    # 3. Demographic & Employment Ratios
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = (df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)).astype(np.float32)
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(np.float32)
    if 'DAYS_EMPLOYED' in df.columns:
        df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED'] / 365.25).astype(np.float32)

    # 4. Household Per-Capita Income
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = (df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'].fillna(1).clip(lower=1))).astype(np.float32)

    # 5. External Source Composite Aggregations
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    if ext_cols:
        df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1).astype(np.float32)
        df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1).astype(np.float32)
        df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1).astype(np.float32)
        df['EXT_SOURCES_NAN_COUNT'] = df[ext_cols].isnull().sum(axis=1).astype(np.int32)

    logger.info(f'Feature engineering complete. Total features: {df.shape[1]}')
    return df
