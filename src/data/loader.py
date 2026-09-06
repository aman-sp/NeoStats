import pandas as pd
import numpy as np
import os
from typing import Optional
from src.utils.config import APPLICATION_TRAIN_CSV, BUREAU_CSV, PREVIOUS_APPLICATION_CSV
from src.utils.logger import get_logger

logger = get_logger('DataLoader')

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype(np.float32)
        elif col_type == 'int64' and col != 'SK_ID_CURR':
            df[col] = df[col].astype(np.int32)
    return df

def load_application_train(
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    logger.info(f'Loading application data from {APPLICATION_TRAIN_CSV}...')
    if sample_size:
        df = pd.read_csv(APPLICATION_TRAIN_CSV, nrows=sample_size * 2)
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f'Sampled {len(df)} rows for development verification.')
    else:
        df = pd.read_csv(APPLICATION_TRAIN_CSV)
        logger.info(f'Loaded full dataset: {df.shape[0]:,d} rows, {df.shape[1]} columns.')
    return optimize_dtypes(df)

def aggregate_bureau(bureau_path: Optional[str] = None) -> pd.DataFrame:
    path = bureau_path or BUREAU_CSV
    if not os.path.exists(path):
        logger.warning(f'Bureau file not found at {path}. Skipping.')
        return pd.DataFrame()

    logger.info('Aggregating Credit Bureau records by SK_ID_CURR...')
    cols = [
        'SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE',
        'CREDIT_DAY_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_MAX_OVERDUE'
    ]
    df_bureau = pd.read_csv(path, usecols=cols)
    
    agg_dict = {
        'SK_ID_BUREAU': 'count',
        'CREDIT_DAY_OVERDUE': 'max',
        'AMT_CREDIT_SUM': 'sum',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_MAX_OVERDUE': 'max'
    }
    
    bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(agg_dict).reset_index()
    bureau_agg.columns = [
        'SK_ID_CURR',
        'BUREAU_LOAN_COUNT',
        'BUREAU_MAX_DAYS_OVERDUE',
        'BUREAU_TOTAL_CREDIT_SUM',
        'BUREAU_TOTAL_DEBT_SUM',
        'BUREAU_MAX_OVERDUE'
    ]
    
    active_loans = df_bureau[df_bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_LOANS')
    bureau_agg = bureau_agg.merge(active_loans, on='SK_ID_CURR', how='left')
    bureau_agg['BUREAU_ACTIVE_LOANS'] = bureau_agg['BUREAU_ACTIVE_LOANS'].fillna(0).astype(np.int32)
    
    logger.info(f'Aggregated Bureau features for {len(bureau_agg):,d} unique applicants.')
    return optimize_dtypes(bureau_agg)

def aggregate_previous_applications(prev_path: Optional[str] = None) -> pd.DataFrame:
    path = prev_path or PREVIOUS_APPLICATION_CSV
    if not os.path.exists(path):
        logger.warning(f'Previous application file not found at {path}. Skipping.')
        return pd.DataFrame()

    logger.info('Aggregating Previous Application records by SK_ID_CURR...')
    cols = ['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_STATUS', 'AMT_CREDIT', 'AMT_APPLICATION']
    df_prev = pd.read_csv(path, usecols=cols)
    
    prev_agg = df_prev.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_APP_TOTAL_CREDIT=('AMT_CREDIT', 'sum'),
        PREV_APP_TOTAL_APPLICATION=('AMT_APPLICATION', 'sum')
    ).reset_index()
    
    status_counts = df_prev.pivot_table(
        index='SK_ID_CURR',
        columns='NAME_CONTRACT_STATUS',
        values='SK_ID_PREV',
        aggfunc='count',
        fill_value=0
    ).reset_index()
    
    refused_col = status_counts['Refused'] if 'Refused' in status_counts else 0
    approved_col = status_counts['Approved'] if 'Approved' in status_counts else 0
    
    status_df = pd.DataFrame({
        'SK_ID_CURR': status_counts['SK_ID_CURR'],
        'PREV_APP_REFUSED_COUNT': refused_col,
        'PREV_APP_APPROVED_COUNT': approved_col
    })
    
    prev_agg = prev_agg.merge(status_df, on='SK_ID_CURR', how='left')
    prev_agg['PREV_APP_REFUSAL_RATE'] = (prev_agg['PREV_APP_REFUSED_COUNT'] / prev_agg['PREV_APP_COUNT']).astype(np.float32)
    
    logger.info(f'Aggregated Previous Application features for {len(prev_agg):,d} unique applicants.')
    return optimize_dtypes(prev_agg)

def load_integrated_dataset(
    sample_size: Optional[int] = None,
    include_bureau: bool = True,
    include_prev_app: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    df_app = load_application_train(sample_size=sample_size, random_state=random_state)
    initial_rows = len(df_app)
    
    if include_bureau:
        df_bureau = aggregate_bureau()
        if not df_bureau.empty:
            df_app = df_app.merge(df_bureau, on='SK_ID_CURR', how='left')
            logger.info(f'Left joined Bureau features. Rows: {len(df_app):,d}')
            
    if include_prev_app:
        df_prev = aggregate_previous_applications()
        if not df_prev.empty:
            df_app = df_app.merge(df_prev, on='SK_ID_CURR', how='left')
            logger.info(f'Left joined Previous Application features. Rows: {len(df_app):,d}')
            
    assert len(df_app) == initial_rows, f'Error: Row count changed from {initial_rows} to {len(df_app)} after join!'
    return df_app
