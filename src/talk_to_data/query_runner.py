import os
import time
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional
from src.utils.config import SQLITE_DB_PATH
from src.utils.logger import get_logger

logger = get_logger('QueryExecutor')

def execute_query(
    sql_query: str,
    db_path: Optional[str] = None,
    timeout_seconds: int = 15
) -> Dict[str, Any]:
    """
    Execute a validated read-only SQL query against the analytical SQLite warehouse.
    Enforces read-only connection URI to guarantee database immutability.
    """
    path = db_path or str(SQLITE_DB_PATH)
    if not os.path.exists(path):
        return {
            'success': False,
            'error': f'Database file not found at {path}. Please run database initialization.',
            'data': None,
            'row_count': 0,
            'execution_time_ms': 0
        }

    start_time = time.time()
    try:
        # Connect in read-only mode via URI
        conn_uri = f'file:{os.path.abspath(path)}?mode=ro'
        conn = sqlite3.connect(conn_uri, uri=True, timeout=timeout_seconds)
        
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            'success': True,
            'error': None,
            'data': df,
            'row_count': len(df),
            'execution_time_ms': elapsed_ms
        }
    except sqlite3.OperationalError as e:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f'SQLite Operational Error: {str(e)}')
        return {
            'success': False,
            'error': f'Database error: {str(e)}',
            'data': None,
            'row_count': 0,
            'execution_time_ms': elapsed_ms
        }
    except Exception as e:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(f'Unexpected error executing query: {str(e)}')
        return {
            'success': False,
            'error': f'Unexpected execution error: {str(e)}',
            'data': None,
            'row_count': 0,
            'execution_time_ms': elapsed_ms
        }
