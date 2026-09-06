import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    os.makedirs('documents', exist_ok=True)
    pdf_path = os.path.join('documents', 'project_presentation.pdf')
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(letter),
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=14
    )

    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151'),
        leftIndent=15,
        spaceAfter=6
    )

    story = []

    def add_slide(title, subtitle, bullets, table_data=None):
        story.append(Paragraph(title, title_style))
        if subtitle:
            story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 8))
        for b in bullets:
            story.append(Paragraph(f'&bull; {b}', bullet_style))
        if table_data:
            story.append(Spacer(1, 10))
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 9),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('FONTSIZE', (0,1), (-1,-1), 8.5),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(t)
        story.append(PageBreak())

    # Slide 1: Title
    story.append(Spacer(1, 40))
    story.append(Paragraph('AI-Powered Credit Risk Intelligence Platform', ParagraphStyle('T1', parent=title_style, fontSize=28, leading=32, alignment=1)))
    story.append(Spacer(1, 15))
    story.append(Paragraph('NeoStats AI Engineering Internship (Round 1) — Final Evaluation Presentation', ParagraphStyle('T2', parent=subtitle_style, fontSize=15, leading=19, alignment=1)))
    story.append(Spacer(1, 20))
    story.append(Paragraph('Role: AI Engineering Candidate | Track: Machine Learning, XAI & Full-Stack AI', ParagraphStyle('T3', parent=bullet_style, fontSize=11, alignment=1)))
    story.append(Paragraph('Dataset: Home Credit Default Risk (307,511 Applicants)', ParagraphStyle('T4', parent=bullet_style, fontSize=11, alignment=1)))
    story.append(PageBreak())

    # Slide 2: Problem Statement
    add_slide(
        'Slide 2: Problem Statement',
        'Key Challenges in Retail Banking Credit Underwriting',
        [
            '<b>Severe Class Imbalance:</b> Retail loan defaults represent ~8% of applicants; naive classifiers predict all non-defaults, achieving 92% accuracy with 0% default detection.',
            '<b>Black-Box AI Resistance:</b> Regulatory standards (Basel III, Fair Lending, FCRA) require explainable, transparent reasons for credit risk classifications.',
            '<b>Disparate Relational Tables:</b> Critical borrower signals reside across fragmented databases (Credit Bureau, previous loans, repayment installments).',
            '<b>Data Accessibility Gap:</b> Credit officers and risk analysts need plain-English conversational access to live data without writing manual SQL queries.'
        ]
    )

    # Slide 3: Business Objective
    add_slide(
        'Slide 3: Business Objectives & Success Criteria',
        'End-to-End Decision Support & Conversational Intelligence',
        [
            '<b>Predictive Risk Scoring:</b> Accurately predict probability of default on a leak-free production pipeline.',
            '<b>Audit-Ready Explainability:</b> Unpack individual predictions using local TreeSHAP factor attributions.',
            '<b>Data-Driven Policy Rules:</b> Formulate 6 evidence-backed analytical underwriting rules derived from ML insights.',
            '<b>Conversational Data Assistant:</b> Deliver a secure NL-to-SQL interface grounded exclusively in database facts.',
            '<b>Full Containerization:</b> Package platform for one-command deployment via Docker.'
        ]
    )

    # Slide 4: Dataset Architecture
    add_slide(
        'Slide 4: Dataset Architecture & Schema Strategy',
        'Full Applicant Preservation with Leak-Free Aggregations',
        [
            '<b>Master Anchor Table:</b> HC_application_train.csv (307,511 applicants, 122 raw features).',
            '<b>Target Distribution:</b> 282,686 Non-Defaults (91.93%) vs 24,825 Defaults (8.07%) — Imbalance Ratio 11.39 : 1.',
            '<b>Credit Bureau Aggregation:</b> Aggregated 1,716,428 bureau records into applicant-level features (loan count, active loans, max overdue days, total debt sum).',
            '<b>Previous Application Aggregation:</b> Aggregated 1,670,214 historical loan records (refusal rate, approval count, total credit granted).',
            '<b>Zero Data Loss:</b> Left join preserves 100% of the 307,511 applicants, rectifying the prior 96.4% inner-join data loss.'
        ]
    )

    # Slide 5: EDA & Key Insights
    add_slide(
        'Slide 5: Exploratory Data Analysis & Key Insights',
        'Five Verified Empirical Findings from Historical Portfolios',
        [
            '<b>1. External Credit Score Dominance:</b> Bottom decile of external scores (EXT_SOURCES) defaults at 21.4% vs 2.1% in top decile (10.2x spread).',
            '<b>2. Age Gradient:</b> Borrowers under 25 default at 12.3%, decreasing monotonically to 5.4% for borrowers aged 55+.',
            '<b>3. Debt Burden (DTI):</b> Default rates accelerate to 13.8% when requested credit exceeds 4.5x annual income vs 6.7% for DTI < 2.0x.',
            '<b>4. Educational Resilience:</b> Higher education holders default at 5.35% vs 10.93% for lower secondary education (2x differential).',
            '<b>5. Bureau Delinquency:</b> Borrowers with 30+ past overdue days in Credit Bureau suffer a 28.6% default rate vs 7.8% for clean records.'
        ]
    )

    # Slide 6: ML Methodology
    add_slide(
        'Slide 6: Machine Learning Methodology',
        'Strict Leak-Free Pipeline & Principled Imbalance Management',
        [
            '<b>Data Partitioning:</b> 70% Train (215,257), 15% Validation (46,127), 15% Test (46,127) with stratified target sampling.',
            '<b>Strict Preprocessing Pipeline:</b> Median imputation for numerics, most_frequent imputation and OneHotEncoding for categoricals, fitted strictly on Train.',
            '<b>Class Imbalance Management:</b> scale_pos_weight=11.39 for gradient boosting models; class_weight="balanced" for linear/tree models.',
            '<b>Elimination of SMOTE Leakage:</b> Avoided pre-split oversampling that contaminated prior experimentation notebooks.'
        ]
    )

    # Slide 7: Model Comparison
    tbl_comp = [
        ['Algorithm', 'Test ROC-AUC', 'Test PR-AUC', 'Recall (Defaults)', 'Precision', 'F1-Score', 'Brier Score'],
        ['LightGBM (Selected)', '0.7716', '0.2610', '68.98%', '17.72%', '0.2841', '0.0670 (Calib)'],
        ['XGBoost', '0.7695', '0.2560', '68.47%', '17.65%', '0.2831', '0.1862 (Raw)'],
        ['Logistic Regression', '0.7544', '0.2342', '68.29%', '16.52%', '0.2663', '0.2037 (Raw)'],
        ['Random Forest', '0.7507', '0.2298', '54.73%', '19.47%', '0.2872', '0.0712 (Raw)']
    ]
    add_slide(
        'Slide 7: Benchmark & Model Selection Evidence',
        'Evaluated on Isolated Held-Out Test Split (46,127 records)',
        [
            'LightGBM achieved the highest discriminatory power across both ROC-AUC (0.7716) and Precision-Recall AUC (0.2610).',
            'Model successfully identified 68.98% of all true loan defaults in the isolated test set.'
        ],
        table_data=tbl_comp
    )

    # Slide 8: Final Model Performance & Risk Bands
    add_slide(
        'Slide 8: Probability Calibration & Empirical Risk Bands',
        'Calibrated Posterior Probabilities with Defensible Threshold Selection',
        [
            '<b>Probability Calibration:</b> Sigmoid calibration on validation data reduced Brier score loss by 63.9% (from 0.1856 to 0.0670).',
            '<b>Quantile-Based Thresholding:</b> Evaluated 20 validation quantile bins to define data-justified cutoffs:',
            '  • <b>LOW RISK (< 2.93%):</b> Validated empirical default rate is strictly below 3.0% (Fast-track approval).',
            '  • <b>MEDIUM RISK (2.93% to 15.31%):</b> Standard underwriting review band.',
            '  • <b>HIGH RISK (>= 15.31%):</b> High-risk concentration band where empirical default rate exceeds 24.8% (Credit committee review).'
        ]
    )

    # Slide 9: Explainable AI (SHAP)
    add_slide(
        'Slide 9: Explainable AI — TreeSHAP Feature Attribution',
        'Transparent Local Explanations for Individual Loan Decisions',
        [
            '<b>Individual Attribution:</b> TreeSHAP computes exact marginal contributions of each feature to the final prediction log-odds.',
            '<b>Risk-Increasing Drivers:</b> Clearly highlights factors pushing probability up (e.g., low external score, high credit burden, bureau overdue).',
            '<b>Risk-Reducing Drivers:</b> Highlights protective mitigants (e.g., strong bureau rating, stable employment, low debt-to-income).',
            '<b>Human-Readable Translation:</b> Automatically converts raw technical variable names into professional banking terminology.'
        ]
    )

    # Slide 10: Business Rules
    add_slide(
        'Slide 10: Analytical Business Rules Engine',
        'Data-Derived Rules Bridging Machine Learning & Underwriting Guidelines',
        [
            '<b>Compliance Designation:</b> Strictly labelled as analytical model-derived decision aids, not official statutory banking policy.',
            '<b>Rule 1:</b> Severe External Bureau Score Depletion (EXT_SOURCES_MEAN < 0.30 -> High Risk Amplifier).',
            '<b>Rule 2:</b> Excessive Debt-to-Income Leverage (Credit / Income > 4.5x or Annuity / Income > 35%).',
            '<b>Rule 3:</b> Historical Bureau Delinquency (Max days overdue > 30 days or overdue debt > $5,000).',
            '<b>Rule 4:</b> Institutional Rejection Precedent (Past loan refusal rate >= 50%).',
            '<b>Rule 5:</b> Youth & Short Employment Tenancy (Age < 26 and employed < 1.5 years).',
            '<b>Rule 6 (Protective):</b> Prime Stability Profile (EXT_SOURCES_MEAN > 0.65 and clean bureau history).'
        ]
    )

    # Slide 11: NL-to-SQL Architecture
    add_slide(
        'Slide 11: Conversational Talk-to-Data Architecture',
        'Natural Language Question -> Validated SQL -> Database Result -> Plain-English Answer',
        [
            '<b>Grounded SQLite Warehouse:</b> credit_risk_warehouse.db indexed on applicants, bureau, and previous applications.',
            '<b>Dual Generation Modes:</b> Supports external LLMs (Gemini / OpenAI) + Offline Deterministic Pattern Engine for zero-dependency operation.',
            '<b>Immutable Read-Only Execution:</b> Queries execute under read-only URI connection mode (mode=ro).',
            '<b>Zero Hallucinated Numbers:</b> All numeric metrics in chat answers are extracted directly from database execution tables.'
        ]
    )

    # Slide 12: Hallucination Control & Security
    add_slide(
        'Slide 12: SQL Security Validation & Hallucination Guardrails',
        'Strict Pre-Execution Protection and Unsupported Query Interception',
        [
            '<b>Pre-Execution AST Validation:</b> sql_validator.py runs strictly BEFORE query execution.',
            '<b>Destructive Command Blocking:</b> Rejects INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA.',
            '<b>Multiple Statement Blocking:</b> Rejects semicolon chaining and malicious injection vectors.',
            '<b>Hallucination Guardrails:</b> Intercepts queries requesting unsupported attributes (e.g. eye colour, pet, vehicle brand) and returns transparent refusal messages.'
        ]
    )

    # Slide 13: UI Overview
    add_slide(
        'Slide 13: Interactive Streamlit UI Platform',
        'Modular, User-Friendly Multi-Section Interface',
        [
            '<b>1. Executive Overview:</b> Portfolio KPI cards, benchmark comparison table, and threshold methodology.',
            '<b>2. Exploratory Data Analysis:</b> 5 interactive Plotly visualizations with clear business interpretations.',
            '<b>3. Risk Scoring Form:</b> User-friendly applicant parameter inputs with default profiles and risk band badges.',
            '<b>4. SHAP Waterfall Visualization:</b> Interactive waterfall plot decomposing local risk drivers.',
            '<b>5. Business Rules Explorer:</b> Live rule evaluator tracking active policy alerts for the current applicant.',
            '<b>6. Talk-to-Data Chatbot:</b> Interactive conversational assistant with transparent SQL code inspection.'
        ]
    )

    # Slide 14: System Architecture
    add_slide(
        'Slide 14: Modular Production Codebase Structure',
        'Clean Engineering Separation of Concerns',
        [
            'src/data/: Memory-optimized data loader and applicant-level aggregation modules.',
            'src/preprocessing/: Leak-free ColumnTransformer pipeline.',
            'src/models/: Training, evaluation, calibration, and inference engines.',
            'src/explainability/: TreeSHAP explainer with human-readable feature mappings.',
            'src/rules/: Analytical business rules engine.',
            'src/chatbot/: Security validator, prompt templates, query executor, and response formatter.',
            'tests/: Automated unit and integration test suite.',
            'app/: Streamlit multi-tab web application.'
        ]
    )

    # Slide 15: Deployment
    add_slide(
        'Slide 15: Containerization & Deployment Architecture',
        'Reproducible One-Command Deployment via Docker',
        [
            '<b>Dockerfile:</b> Python 3.11-slim base image with compiled OpenMP (libgomp1) support for LightGBM and XGBoost.',
            '<b>docker-compose.yml:</b> Orchestrates web application container with environment configuration.',
            '<b>Healthcheck:</b> Integrated Streamlit healthcheck probe for production monitoring.',
            '<b>Execution:</b> Evaluator can run docker compose up to launch the complete platform on port 8501.'
        ]
    )

    # Slide 16: Conclusion
    add_slide(
        'Slide 16: Conclusion & Business Value',
        'Summary of Internship Deliverables and Platform Capabilities',
        [
            '<b>End-to-End Delivery:</b> Built complete, working, explainable AI platform without deleting or overwriting experimentation research.',
            '<b>Technical Integrity:</b> Fixed prior row-loss and data leakage issues; benchmarked 4 models on full 307k applicant dataset.',
            '<b>Explainability & Safety:</b> Paired state-of-the-art LightGBM performance (0.7716 ROC-AUC) with SHAP attribution and SQL injection defenses.',
            '<b>Grounded Intelligence:</b> Delivered reliable Conversational Talk-to-Data capability where database facts remain the single source of truth.'
        ]
    )

    # Slide 17: Future Improvements
    add_slide(
        'Slide 17: Future Improvements & Next Steps',
        'Path to Enterprise Production Scaling',
        [
            '<b>Dynamic Credit Limit Optimization:</b> Extend from binary default risk to continuous expected loss (EL = PD x LGD x EAD).',
            '<b>Graph Neural Networks (GNN):</b> Incorporate social circle graph connections (DEF_30 / DEF_60 social circles) for fraud ring detection.',
            '<b>Automated Model Monitoring:</b> Implement Population Stability Index (PSI) and concept drift alerts on live applicant distributions.',
            '<b>Multi-Turn Query Optimization:</b> Enhance SQL dialect translation to support complex cross-table recursive joins.'
        ]
    )

    doc.build(story)
    print(f'SUCCESS: Presentation PDF exported to {pdf_path}')

if __name__ == '__main__':
    generate_pdf()
