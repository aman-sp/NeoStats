import os
import sys
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import SQLITE_DB_PATH, DATABASE_DIR
from src.data.loader import load_application_train, aggregate_bureau, aggregate_previous_applications
from src.data.feature_engineering import engineer_features
from src.utils.logger import get_logger

logger = get_logger('DatabaseInit')

def initialize_database(sample_size: int = None):
    """
    Build analytical SQLite database from raw datasets with optimized indexes.
    Populates: applicants, bureau_summary, prev_app_summary.
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    db_file = str(SQLITE_DB_PATH)
    logger.info(f'Initializing SQLite analytical warehouse at: {db_file}...')

    if os.path.exists(db_file):
        os.remove(db_file)
        logger.info('Removed existing database for clean reconstruction.')

    conn = sqlite3.connect(db_file)

    # 1. Populate applicants table
    logger.info('Loading and feature engineering applicant records...')
    df_app = load_application_train(sample_size=sample_size)
    df_app = engineer_features(df_app)

    # Filter columns to core analytical columns to keep DB lean and fast
    core_cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'AGE_YEARS', 'EMPLOYED_YEARS',
        'OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'EXT_SOURCES_MEAN', 'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT',
        'PAYMENT_RATE', 'GOODS_CREDIT_RATIO', 'INCOME_PER_PERSON'
    ]
    existing_cols = [c for c in core_cols if c in df_app.columns]
    df_app[existing_cols].to_sql('applicants', conn, index=False, if_exists='replace')
    logger.info(f'Table "applicants" populated with {len(df_app):,d} records.')

    # 2. Populate bureau_summary
    df_bureau = aggregate_bureau()
    if not df_bureau.empty:
        df_bureau.to_sql('bureau_summary', conn, index=False, if_exists='replace')
        logger.info(f'Table "bureau_summary" populated with {len(df_bureau):,d} records.')

    # 3. Populate prev_app_summary
    df_prev = aggregate_previous_applications()
    if not df_prev.empty:
        df_prev.to_sql('prev_app_summary', conn, index=False, if_exists='replace')
        logger.info(f'Table "prev_app_summary" populated with {len(df_prev):,d} records.')

    # 4. Create Performance Indexes
    logger.info('Creating performance indexes on key query fields...')
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_id ON applicants (SK_ID_CURR);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_target ON applicants (TARGET);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_gender ON applicants (CODE_GENDER);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_income_type ON applicants (NAME_INCOME_TYPE);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_education ON applicants (NAME_EDUCATION_TYPE);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_occupation ON applicants (OCCUPATION_TYPE);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_car ON applicants (FLAG_OWN_CAR);')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_applicants_income ON applicants (AMT_INCOME_TOTAL);')

    if not df_bureau.empty:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bureau_id ON bureau_summary (SK_ID_CURR);')
    if not df_prev.empty:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prev_id ON prev_app_summary (SK_ID_CURR);')

    conn.commit()
    conn.close()
    logger.info('Database initialization completed successfully.')

if __name__ == '__main__':
    initialize_database()
