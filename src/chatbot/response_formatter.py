import pandas as pd
from typing import Dict, Any

def format_sql_response(
    question: str,
    sql: str,
    query_result: Dict[str, Any]
) -> str:
    """
    Format database execution output into clear, professional, human-readable response.
    All figures are derived directly from the SQL execution result.
    """
    if not query_result.get('success', False):
        return f"**Query Execution Error:** {query_result.get('error', 'Unknown database error')}"

    df = query_result.get('data')
    if df is None or df.empty:
        return "The query executed successfully, but returned no matching records in the analytical database."

    # Single numerical value output (e.g. Average income, total count)
    if df.shape == (1, 1):
        col_name = df.columns[0]
        val = df.iloc[0, 0]
        if 'pct' in col_name.lower() or 'rate' in col_name.lower():
            return f"Based on the analytical database, the **{col_name.replace('_', ' ')}** is **{val:.2f}%**."
        elif 'income' in col_name.lower() or 'credit' in col_name.lower() or 'annuity' in col_name.lower() or 'amount' in col_name.lower():
            return f"Based on the analytical database, the **{col_name.replace('_', ' ')}** is **${val:,.2f}**."
        elif isinstance(val, (int, float)):
            return f"Based on the analytical database, the count / metric is **{val:,.0f}**."
        else:
            return f"Result: **{val}**"

    # Aggregated table output (e.g. grouped by gender, education, occupation)
    lines = [f"**Query Results:** Found {len(df)} records."]
    
    # Render top records summary
    for idx, row in df.iterrows():
        row_parts = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if 'pct' in col.lower() or 'rate' in col.lower():
                    row_parts.append(f"{col}: **{val:.2f}%**")
                elif 'income' in col.lower() or 'credit' in col.lower() or 'amount' in col.lower():
                    row_parts.append(f"{col}: **${val:,.2f}**")
                else:
                    row_parts.append(f"{col}: **{val:.2f}**")
            elif isinstance(val, int):
                row_parts.append(f"{col}: **{val:,d}**")
            else:
                row_parts.append(f"{col}: **{val}**")
        lines.append(f"- " + ", ".join(row_parts))

    return "\n".join(lines)
