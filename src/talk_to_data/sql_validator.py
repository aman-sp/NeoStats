import re
import os
import sqlite3
from typing import Tuple, List, Optional
from src.utils.logger import get_logger

logger = get_logger('SQLValidator')

# Strict Whitelist of Allowed Tables
ALLOWED_TABLES = {'applicants', 'bureau_summary', 'prev_app_summary'}

# Blacklist of Forbidden Keywords & Dangerous Commands
FORBIDDEN_KEYWORDS = [
    'insert', 'update', 'delete', 'drop', 'alter', 'create',
    'attach', 'detach', 'pragma', 'vacuum', 'reindex', 'replace',
    'truncate', 'grant', 'revoke', 'commit', 'rollback',
    'savepoint', 'exec', 'execute', 'xp_', 'sp_', '--', '/*'
]

class SQLValidationError(Exception):
    """Raised when SQL fails security or syntax checks."""
    pass

def validate_sql(sql_query: str, db_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Perform rigorous multi-stage security validation on generated SQL.
    Validation happens STRICTLY BEFORE database execution.

    Stages:
    1. Blank / Null Check
    2. Multiple Statement Check (Prevent Semicolon Injection)
    3. Read-Only Command Enforcement (Must start with SELECT or WITH)
    4. Forbidden Keyword / DDL / DML Scan
    5. Table Whitelist Verification
    6. SQLite Engine Pre-flight (EXPLAIN parse without execution)
    """
    clean_sql = sql_query.strip()
    if not clean_sql:
        return False, 'Query is empty.'

    # Strip trailing semicolon if single statement
    if clean_sql.endswith(';'):
        clean_sql = clean_sql[:-1].strip()

    # Stage 2: Reject multiple statements
    if ';' in clean_sql:
        return False, 'Security Violation: Multiple statements separated by semicolons are strictly forbidden.'

    # Stage 3: Must begin with SELECT or WITH (for CTEs)
    first_word = clean_sql.split()[0].upper()
    if first_word not in ('SELECT', 'WITH'):
        return False, f'Security Violation: Only read-only SELECT queries are permitted. Found: {first_word}'

    # Stage 4: Scan for forbidden keywords
    lower_query = clean_sql.lower()
    for kw in FORBIDDEN_KEYWORDS:
        # Match whole word to avoid false positives (e.g. 'created_at' matching 'create')
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, lower_query):
            return False, f'Security Violation: Dangerous or mutating keyword "{kw.upper()}" is prohibited.'

    # Stage 5: Validate table references against whitelist
    # Extract words following FROM or JOIN
    table_pattern = r'\b(?:from|join)\s+([a-zA-Z0-9_]+)'
    found_tables = re.findall(table_pattern, lower_query)
    for tbl in found_tables:
        if tbl not in ALLOWED_TABLES:
            return False, f'Security Violation: Table "{tbl}" is not in the allowed database whitelist ({", ".join(ALLOWED_TABLES)}).'

    # Stage 6: SQLite Syntax Dry-Run (using EXPLAIN)
    try:
        from src.utils.config import SQLITE_DB_PATH
        actual_path = db_path or str(SQLITE_DB_PATH)
        if os.path.exists(actual_path):
            conn = sqlite3.connect(f'file:{os.path.abspath(actual_path)}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute(f'EXPLAIN {clean_sql}')
            conn.close()
    except sqlite3.OperationalError as e:
        return False, f'SQL Syntax Error: {str(e)}'
    except Exception as e:
        return False, f'SQL Validation Exception: {str(e)}'

    return True, clean_sql
