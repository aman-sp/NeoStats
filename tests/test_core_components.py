import os
import unittest
import pandas as pd
from src.models.predict import CreditRiskPredictor
from src.explainability.shap_explainer import CreditRiskExplainer
from src.rules.business_rules import evaluate_business_rules
from src.chatbot.sql_validator import validate_sql
from src.chatbot.query_executor import execute_query
from src.chatbot.sql_generator import NLToSQLGenerator
from src.chatbot.response_formatter import format_sql_response

class TestCreditRiskCoreComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = CreditRiskPredictor()
        cls.explainer = CreditRiskExplainer()
        cls.sql_gen = NLToSQLGenerator()

        # Sample test applicant (prime / lower risk profile)
        cls.low_risk_applicant = {
            'AMT_INCOME_TOTAL': 250000.0,
            'AMT_CREDIT': 300000.0,
            'AMT_ANNUITY': 18000.0,
            'AMT_GOODS_PRICE': 300000.0,
            'REGION_POPULATION_RELATIVE': 0.03,
            'DAYS_BIRTH': -18000, # ~49 years old
            'DAYS_EMPLOYED': -4000, # ~11 years
            'DAYS_REGISTRATION': -5000,
            'DAYS_ID_PUBLISH': -2000,
            'OWN_CAR_AGE': 3.0,
            'CNT_FAM_MEMBERS': 2.0,
            'REGION_RATING_CLIENT_W_CITY': 1,
            'HOUR_APPR_PROCESS_START': 12,
            'EXT_SOURCE_1': 0.75,
            'EXT_SOURCE_2': 0.82,
            'EXT_SOURCE_3': 0.79,
            'DEF_30_CNT_SOCIAL_CIRCLE': 0.0,
            'DEF_60_CNT_SOCIAL_CIRCLE': 0.0,
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CODE_GENDER': 'F',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'NAME_INCOME_TYPE': 'Working',
            'NAME_EDUCATION_TYPE': 'Higher education',
            'NAME_FAMILY_STATUS': 'Married',
            'NAME_HOUSING_TYPE': 'House / apartment',
            'OCCUPATION_TYPE': 'Managers',
            'BUREAU_LOAN_COUNT': 4,
            'BUREAU_ACTIVE_LOANS': 1,
            'BUREAU_MAX_DAYS_OVERDUE': 0,
            'BUREAU_TOTAL_CREDIT_SUM': 450000.0,
            'BUREAU_TOTAL_DEBT_SUM': 20000.0,
            'BUREAU_MAX_OVERDUE': 0.0,
            'PREV_APP_COUNT': 2,
            'PREV_APP_TOTAL_CREDIT': 150000.0,
            'PREV_APP_REFUSED_COUNT': 0,
            'PREV_APP_APPROVED_COUNT': 2,
            'PREV_APP_REFUSAL_RATE': 0.0
        }

        # Sample test applicant (high risk profile)
        cls.high_risk_applicant = {
            'AMT_INCOME_TOTAL': 65000.0,
            'AMT_CREDIT': 450000.0, # High burden > 6.9x
            'AMT_ANNUITY': 29000.0, # High annuity > 44%
            'AMT_GOODS_PRICE': 450000.0,
            'REGION_POPULATION_RELATIVE': 0.01,
            'DAYS_BIRTH': -8500, # ~23 years old
            'DAYS_EMPLOYED': -250, # < 1 year
            'DAYS_REGISTRATION': -1000,
            'DAYS_ID_PUBLISH': -500,
            'OWN_CAR_AGE': 15.0,
            'CNT_FAM_MEMBERS': 4.0,
            'REGION_RATING_CLIENT_W_CITY': 3,
            'HOUR_APPR_PROCESS_START': 9,
            'EXT_SOURCE_1': 0.15,
            'EXT_SOURCE_2': 0.20,
            'EXT_SOURCE_3': 0.18,
            'DEF_30_CNT_SOCIAL_CIRCLE': 2.0,
            'DEF_60_CNT_SOCIAL_CIRCLE': 2.0,
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CODE_GENDER': 'M',
            'FLAG_OWN_CAR': 'N',
            'FLAG_OWN_REALTY': 'N',
            'NAME_INCOME_TYPE': 'Working',
            'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
            'NAME_FAMILY_STATUS': 'Single / not married',
            'NAME_HOUSING_TYPE': 'With parents',
            'OCCUPATION_TYPE': 'Laborers',
            'BUREAU_LOAN_COUNT': 6,
            'BUREAU_ACTIVE_LOANS': 4,
            'BUREAU_MAX_DAYS_OVERDUE': 45,
            'BUREAU_TOTAL_CREDIT_SUM': 600000.0,
            'BUREAU_TOTAL_DEBT_SUM': 380000.0,
            'BUREAU_MAX_OVERDUE': 8500.0,
            'PREV_APP_COUNT': 4,
            'PREV_APP_TOTAL_CREDIT': 200000.0,
            'PREV_APP_REFUSED_COUNT': 3,
            'PREV_APP_APPROVED_COUNT': 1,
            'PREV_APP_REFUSAL_RATE': 0.75
        }

    def test_prediction_output(self):
        """Verify probability generation and calibrated risk banding."""
        res_low = self.predictor.predict_single(self.low_risk_applicant)
        self.assertIn(res_low['risk_band'], ['LOW', 'MEDIUM'])
        self.assertLess(res_low['default_probability'], 0.10)

        res_high = self.predictor.predict_single(self.high_risk_applicant)
        self.assertEqual(res_high['risk_band'], 'HIGH')
        self.assertGreater(res_high['default_probability'], 0.15)

    def test_shap_explanation(self):
        """Verify SHAP explanation decomposes actual feature contributions."""
        df_inst = pd.DataFrame([self.high_risk_applicant])
        explanation = self.explainer.explain_instance(df_inst, top_n=3)
        self.assertGreater(len(explanation['risk_increasing_factors']), 0)
        top_risk = explanation['risk_increasing_factors'][0]
        self.assertIn('feature_readable', top_risk)
        self.assertGreater(top_risk['shap_value'], 0)

    def test_business_rules(self):
        """Verify analytical decision rules fire appropriately."""
        triggered = evaluate_business_rules(self.high_risk_applicant)
        self.assertGreater(len(triggered), 0)
        rule_ids = [r['id'] for r in triggered]
        self.assertIn('RULE-ANL-03', rule_ids) # Overdue delinquency

    def test_sql_security_validator(self):
        """Verify rejection of destructive SQL and acceptance of read-only SELECT."""
        # Valid queries
        valid1, _ = validate_sql("SELECT COUNT(*) FROM applicants WHERE TARGET = 1;")
        self.assertTrue(valid1)

        valid2, _ = validate_sql("SELECT AVG(AMT_INCOME_TOTAL) FROM applicants;")
        self.assertTrue(valid2)

        # Destructive / injection attacks
        inv1, err1 = validate_sql("DROP TABLE applicants;")
        self.assertFalse(inv1)

        inv2, err2 = validate_sql("DELETE FROM applicants WHERE SK_ID_CURR = 100002;")
        self.assertFalse(inv2)

        inv3, err3 = validate_sql("SELECT * FROM applicants; DROP TABLE applicants;")
        self.assertFalse(inv3)

        inv4, err4 = validate_sql("INSERT INTO applicants (SK_ID_CURR) VALUES (1);")
        self.assertFalse(inv4)

        inv5, err5 = validate_sql("SELECT * FROM non_existent_table;")
        self.assertFalse(inv5)

    def test_database_execution(self):
        """Verify actual query execution returns real numbers from SQLite warehouse."""
        res = execute_query("SELECT COUNT(*) AS total FROM applicants;")
        self.assertTrue(res['success'])
        self.assertEqual(int(res['data'].iloc[0, 0]), 307511)

    def test_nl_to_sql_and_hallucination_control(self):
        """Test standard queries and unsupported question rejection."""
        # 1. Aggregation
        res_agg = self.sql_gen.generate_sql("What is the average income of applicants?")
        self.assertEqual(res_agg['status'], 'SUCCESS')
        exec_agg = execute_query(res_agg['sql'])
        self.assertTrue(exec_agg['success'])
        avg_inc = float(exec_agg['data'].iloc[0, 0])
        self.assertGreater(avg_inc, 100000)

        # 2. Filtering
        res_flt = self.sql_gen.generate_sql("How many applicants with income above 200000 defaulted?")
        self.assertEqual(res_flt['status'], 'SUCCESS')
        exec_flt = execute_query(res_flt['sql'])
        self.assertTrue(exec_flt['success'])

        # 3. Grouping
        res_grp = self.sql_gen.generate_sql("What is the default rate by gender?")
        self.assertEqual(res_grp['status'], 'SUCCESS')
        exec_grp = execute_query(res_grp['sql'])
        self.assertTrue(exec_grp['success'])

        # 4. Ranking
        res_rnk = self.sql_gen.generate_sql("Which occupation has the highest default rate?")
        self.assertEqual(res_rnk['status'], 'SUCCESS')
        exec_rnk = execute_query(res_rnk['sql'])
        self.assertTrue(exec_rnk['success'])

        # 5. Comparison
        res_cmp = self.sql_gen.generate_sql("Compare the default rate of applicants with and without cars.")
        self.assertEqual(res_cmp['status'], 'SUCCESS')
        exec_cmp = execute_query(res_cmp['sql'])
        self.assertTrue(exec_cmp['success'])

        # 6. Hallucination Control: Unsupported attribute (blue eyes)
        res_unsupp = self.sql_gen.generate_sql("What is the default rate of applicants with blue eyes?")
        self.assertEqual(res_unsupp['status'], 'UNSUPPORTED_QUERY')
        self.assertIsNone(res_unsupp['sql'])

if __name__ == '__main__':
    unittest.main()
