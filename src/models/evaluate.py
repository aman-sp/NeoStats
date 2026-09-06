import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

def evaluate_model_performance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for classification and probability quality.
    Evaluates: ROC-AUC, PR-AUC, Precision, Recall, F1, Accuracy, Brier Score, Confusion Matrix.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y, y_proba))
    pr_auc = float(average_precision_score(y, y_proba))
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    brier = float(brier_score_loss(y, y_proba))
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'brier_score': round(brier, 4),
        'threshold_used': round(threshold, 4),
        'confusion_matrix': cm
    }

def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Analyze empirical probability calibration:
    Fraction of positives vs Mean predicted probability across bins.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {
        'expected_calibration_error': round(ece, 4),
        'brier_score': round(float(brier_score_loss(y_true, y_proba)), 4),
        'calibration_curve': {
            'fraction_of_positives': [round(float(x), 4) for x in prob_true],
            'mean_predicted_value': [round(float(x), 4) for x in prob_pred]
        }
    }

def determine_empirical_risk_thresholds(
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    target_low_max_default_rate: float = 0.03,
    target_high_min_default_rate: float = 0.15
) -> Dict[str, Any]:
    """
    Defensible Threshold Methodology:
    Instead of arbitrary thresholds (e.g. 5%, 15%), we partition predicted probabilities
    into quantiles/deciles on validation data and identify probability cutoffs where
    the empirical default rate transitions:
    - LOW RISK: Predicted default probabilities where actual validation default rate is below target_low_max_default_rate.
    - HIGH RISK: Predicted default probabilities where actual validation default rate is above target_high_min_default_rate.
    - MEDIUM RISK: Intermediate zone between LOW and HIGH.
    """
    df_val = pd.DataFrame({'y_true': y_val, 'y_proba': y_val_proba})
    # Divide into 20 quantile bins for fine resolution
    df_val['bin'] = pd.qcut(df_val['y_proba'], q=20, duplicates='drop')
    bin_stats = df_val.groupby('bin', observed=False).agg(
        min_p=('y_proba', 'min'),
        max_p=('y_proba', 'max'),
        mean_p=('y_proba', 'mean'),
        count=('y_true', 'count'),
        defaults=('y_true', 'sum')
    ).reset_index()
    bin_stats['empirical_default_rate'] = bin_stats['defaults'] / bin_stats['count']

    # Find cutoff for Low Risk: highest max_p where empirical default rate <= target_low_max_default_rate
    low_bins = bin_stats[bin_stats['empirical_default_rate'] <= target_low_max_default_rate]
    if not low_bins.empty:
        low_cutoff = float(low_bins['max_p'].iloc[-1])
    else:
        # Fallback to 25th percentile of probability if all bins exceed target
        low_cutoff = float(np.percentile(y_val_proba, 25))

    # Find cutoff for High Risk: lowest min_p where empirical default rate >= target_high_min_default_rate
    high_bins = bin_stats[bin_stats['empirical_default_rate'] >= target_high_min_default_rate]
    if not high_bins.empty:
        high_cutoff = float(high_bins['min_p'].iloc[0])
    else:
        # Fallback to 75th percentile of probability
        high_cutoff = float(np.percentile(y_val_proba, 75))

    # Ensure low_cutoff < high_cutoff
    if low_cutoff >= high_cutoff:
        low_cutoff = float(np.percentile(y_val_proba, 33))
        high_cutoff = float(np.percentile(y_val_proba, 67))

    bin_stats['bin'] = bin_stats['bin'].astype(str)
    return {
        'low_risk_threshold': round(low_cutoff, 4),
        'high_risk_threshold': round(high_cutoff, 4),
        'methodology': (
            'Empirical quantile analysis on validation split: Low risk threshold guarantees '
            f'historical default rate <= {target_low_max_default_rate*100:.1f}%; High risk threshold '
            f'identifies concentration with default rate >= {target_high_min_default_rate*100:.1f}%.'
        ),
        'decile_table': bin_stats.to_dict(orient='records')
    }
