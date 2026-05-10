import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calc_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def calc_subset_report(df: pd.DataFrame, prob_col: str, label_col: str, threshold: float):
    report = {"all": calc_metrics(df[label_col], df[prob_col], threshold)}
    pos_df = df[df[label_col] == 1].copy()
    if len(pos_df) > 0:
        orig = pos_df[pos_df["source_type"] == "original"]
        syn = pos_df[pos_df["source_type"] == "synthetic"]
        if len(orig) > 0:
            report["original_positive_subset"] = calc_metrics(orig[label_col], orig[prob_col], threshold)
        if len(syn) > 0:
            report["synthetic_positive_subset"] = calc_metrics(syn[label_col], syn[prob_col], threshold)
    return report


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)