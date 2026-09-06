import os
import re
from typing import Dict, Any, Tuple, Optional, List
from src.chatbot.prompts import SYSTEM_PROMPT_TEMPLATE
from src.chatbot.sql_validator import validate_sql
from src.utils.logger import get_logger

logger = get_logger('SQLGenerator')

# Non-existent or unsupported domain terms to catch hallucinated inquiries
UNSUPPORTED_TERMS = {
    'eye', 'eyes', 'hair', 'race', 'religion', 'ethnicity',
    'pet', 'dog', 'cat', 'zodiac', 'horoscope', 'hobby',
    'favorite', 'credit card company', 'mastercard', 'visa',
    'car brand', 'toyota', 'bmw', 'mercedes', 'height', 'weight'
}

DETERMINISTIC_PATTERNS = [
    {
        'pattern': r'(?:average|mean)\s+income',
        'sql': 'SELECT ROUND(AVG(AMT_INCOME_TOTAL), 2) AS average_income_usd FROM applicants;',
        'description': 'Average annual income across all applicants'
    },
    {
        'pattern': r'income\s+(?:above|over|>)\s*200000.*default',
        'sql': 'SELECT COUNT(*) AS high_income_default_count FROM applicants WHERE AMT_INCOME_TOTAL > 200000 AND TARGET = 1;',
        'description': 'Count of applicants with income > $200,000 who defaulted'
    },
    {
        'pattern': r'default\s+rate.*gender',
        'sql': 'SELECT CODE_GENDER AS gender, COUNT(*) AS total_applicants, SUM(TARGET) AS default_count, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants GROUP BY CODE_GENDER;',
        'description': 'Default rate grouped by gender'
    },
    {
        'pattern': r'(?:occupation.*(?:highest|top|maximum|most)\s+default\s+rate|(?:highest|top|maximum|most)\s+default\s+rate.*occupation)',
        'sql': 'SELECT OCCUPATION_TYPE AS occupation, COUNT(*) AS total_applicants, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants WHERE OCCUPATION_TYPE IS NOT NULL GROUP BY OCCUPATION_TYPE HAVING COUNT(*) > 100 ORDER BY default_rate_pct DESC LIMIT 5;',
        'description': 'Top occupations ranked by default rate'
    },
    {
        'pattern': r'default\s+rate.*(?:car|cars)',
        'sql': 'SELECT FLAG_OWN_CAR AS owns_car, COUNT(*) AS total_applicants, SUM(TARGET) AS default_count, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants GROUP BY FLAG_OWN_CAR;',
        'description': 'Comparison of default rates between car owners and non-owners'
    },
    {
        'pattern': r'(?:overall|total)\s+default\s+rate',
        'sql': 'SELECT COUNT(*) AS total_applicants, SUM(TARGET) AS total_defaults, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants;',
        'description': 'Overall portfolio applicant count, default count, and default rate'
    },
    {
        'pattern': r'default\s+rate.*education',
        'sql': 'SELECT NAME_EDUCATION_TYPE AS education_level, COUNT(*) AS total_applicants, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC;',
        'description': 'Default rates segmented by educational attainment'
    },
    {
        'pattern': r'default\s+rate.*income\s+type',
        'sql': 'SELECT NAME_INCOME_TYPE AS income_category, COUNT(*) AS total_applicants, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants GROUP BY NAME_INCOME_TYPE ORDER BY default_rate_pct DESC;',
        'description': 'Default rates segmented by income / employment source'
    },
    {
        'pattern': r'default\s+rate.*(?:realty|house|housing|real estate)',
        'sql': 'SELECT FLAG_OWN_REALTY AS owns_real_estate, COUNT(*) AS total_applicants, ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct FROM applicants GROUP BY FLAG_OWN_REALTY;',
        'description': 'Default rate comparison by real estate ownership'
    },
    {
        'pattern': r'(?:average|mean)\s+credit.*education',
        'sql': 'SELECT NAME_EDUCATION_TYPE AS education, ROUND(AVG(AMT_CREDIT), 2) AS average_credit_usd FROM applicants GROUP BY NAME_EDUCATION_TYPE ORDER BY average_credit_usd DESC;',
        'description': 'Average loan credit amount grouped by education'
    }
]

class NLToSQLGenerator:
    """Generates secure SQL from Natural Language with Hallucination Guards."""

    def __init__(self):
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')

    def generate_sql(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Convert user question into validated SQL query.
        Guarantees:
        1. Explicit hallucination refusal for non-existent attributes.
        2. Strict pre-execution validation.
        3. Clear labelling of generation method (LLM vs Deterministic Pattern Engine).
        """
        user_query = question.strip()
        lower_q = user_query.lower()

        # Step 1: Hallucination Guardrail Check
        for term in UNSUPPORTED_TERMS:
            # Check for word boundary match
            if re.search(r'\b' + re.escape(term) + r'\b', lower_q):
                return {
                    'status': 'UNSUPPORTED_QUERY',
                    'sql': None,
                    'explanation': (
                        f'I cannot answer that question using the available dataset because "{term}" '
                        'is not recorded in the analytical credit risk warehouse. The database contains '
                        'financial ratios, loan parameters, age, education, housing, bureau debts, and repayment records.'
                    ),
                    'source': 'HallucinationGuard'
                }

        # Step 2: Check Conversation Context Resolution (e.g. "What about females?")
        resolved_query = lower_q
        if chat_history and len(chat_history) >= 2:
            last_user_msg = chat_history[-2]['content'].lower()
            if 'gender' in last_user_msg or 'male' in last_user_msg:
                if 'female' in lower_q or 'women' in lower_q:
                    resolved_query = 'default rate for females by gender'

        # Step 3: Check Deterministic Pattern Engine
        for pat in DETERMINISTIC_PATTERNS:
            if re.search(pat['pattern'], resolved_query):
                valid, clean_sql = validate_sql(pat['sql'])
                if valid:
                    return {
                        'status': 'SUCCESS',
                        'sql': clean_sql,
                        'explanation': f'Recognized query pattern: {pat["description"]}.',
                        'source': 'DeterministicPatternEngine'
                    }

        # Step 4: If LLM API key available, attempt Generative NL-to-SQL
        if self.gemini_key or self.openai_key:
            llm_result = self._call_llm_api(user_query, chat_history)
            return llm_result

        # Step 5: If no API key and no exact pattern matched, return clear status without guessing
        return {
            'status': 'NO_API_KEY_UNSUPPORTED',
            'sql': None,
            'explanation': (
                'Natural Language query did not match a predefined template and no external LLM API key '
                '(GEMINI_API_KEY or OPENAI_API_KEY) is configured in environment variables. '
                'To prevent fabricated answers, custom queries require an active API key. '
                'You can explore any of the supported query templates from the suggestions.'
            ),
            'source': 'SystemWarning'
        }

    def _call_llm_api(self, question: str, chat_history: Optional[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Invoke external LLM provider if configured."""
        try:
            if self.gemini_key:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=SYSTEM_PROMPT_TEMPLATE)
                prompt = question
                if chat_history:
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
                    prompt = f"Conversation History:\n{history_str}\n\nUser Question:\n{question}"
                resp = model.generate_content(prompt)
                raw_text = resp.text.strip()
            elif self.openai_key:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEMPLATE}]
                if chat_history:
                    messages.extend(chat_history[-4:])
                messages.append({'role': 'user', 'content': question})
                resp = client.chat.completions.create(model='gpt-4o-mini', messages=messages)
                raw_text = resp.choices[0].message.content.strip()
            else:
                return {'status': 'ERROR', 'sql': None, 'explanation': 'No API key configured.', 'source': 'API'}

            if 'UNSUPPORTED_QUERY:' in raw_text:
                return {
                    'status': 'UNSUPPORTED_QUERY',
                    'sql': None,
                    'explanation': raw_text.replace('UNSUPPORTED_QUERY:', '').strip(),
                    'source': 'LLM_HallucinationControl'
                }

            # Extract SQL code block
            sql_match = re.search(r'```(?:sql)?(.*?)```', raw_text, re.DOTALL | re.IGNORECASE)
            sql_candidate = sql_match.group(1).strip() if sql_match else raw_text

            is_valid, validation_msg = validate_sql(sql_candidate)
            if is_valid:
                return {
                    'status': 'SUCCESS',
                    'sql': validation_msg,
                    'explanation': 'SQL query generated and verified by LLM.',
                    'source': 'LLM_Generative'
                }
            else:
                return {
                    'status': 'VALIDATION_FAILED',
                    'sql': sql_candidate,
                    'explanation': f'Generated SQL rejected by security validator: {validation_msg}',
                    'source': 'SQLValidator'
                }
        except Exception as e:
            logger.error(f'LLM API generation failure: {str(e)}')
            return {
                'status': 'API_ERROR',
                'sql': None,
                'explanation': f'LLM API request encountered an error: {str(e)}',
                'source': 'LLM_Provider'
            }
