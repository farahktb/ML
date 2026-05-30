"""
evaluation.py
-------------
Computes and returns all model evaluation metrics for the dashboard.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    roc_auc_score,
)


def get_all_metrics(model, X_test, y_test) -> dict:
    """
    Returns a comprehensive metrics dictionary for the dashboard.

    Key metrics for fraud detection
    ─────────────────────────────────
    • Accuracy  — overall correctness (misleading with imbalanced data)
    • Precision — of all flagged transactions, how many are truly fraud?
    • Recall    — of all actual frauds, how many did we catch? (MOST IMPORTANT)
    • F1-Score  — harmonic mean of precision & recall
    • Log-Loss  — measures confidence of probability predictions
    • AUC-ROC   — separability between classes
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "log_loss": round(log_loss(y_test, y_proba), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": cm,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "fraud_detection_rate": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
        "false_alarm_rate": round(fp / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0,
        "y_proba": y_proba,
    }


def explain_metrics() -> dict:
    """
    Returns academic explanations for each metric.
    Used on the Model Information page.
    """
    return {
        "accuracy": (
            "**Accuracy** = (TP + TN) / Total\n\n"
            "Proportion of all correctly classified transactions. "
            "⚠️ Misleading with imbalanced datasets — a model predicting "
            "all legit would score ~99.8% accuracy!"
        ),
        "precision": (
            "**Precision** = TP / (TP + FP)\n\n"
            "Of all transactions flagged as fraud, what fraction was truly fraudulent? "
            "High precision → fewer false alarms for the bank's fraud team."
        ),
        "recall": (
            "**Recall** = TP / (TP + FN)\n\n"
            "Of all actual fraudulent transactions, what fraction did the model catch? "
            "🚨 **This is the most important metric in fraud detection.** "
            "A missed fraud (FN) is far more costly than a false alarm (FP)."
        ),
        "f1": (
            "**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)\n\n"
            "Harmonic mean balancing precision and recall. "
            "Useful when the class distribution is uneven."
        ),
        "log_loss": (
            "**Log-Loss** (Binary Cross-Entropy)\n\n"
            "J = −[y·log(ŷ) + (1−y)·log(1−ŷ)]\n\n"
            "Measures the quality of probability estimates. Lower is better. "
            "This is the exact cost function minimised during training via gradient descent."
        ),
    }
