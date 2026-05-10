import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ========= 你只需要改这里 =========
OUTPUT_DIR = Path("./outputs_qwen05b_round1_focus")
VAL_FILE = OUTPUT_DIR / "oof_val_predictions.csv"
TEST_FILE = OUTPUT_DIR / "all_test_predictions.csv"

TH_START = 0.008
TH_END = 0.040
TH_STEP = 0.0001

OBJECTIVE = "f1"   # 可选: "f1", "f2"
MIN_PRECISION = None  # 例如 0.35；不设就填 None
# =================================


def get_prob_col(df: pd.DataFrame) -> str:
    candidates = ["prob", "score", "pred_prob", "positive_prob", "y_prob"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"未找到概率列。现有列: {list(df.columns)}")


def get_label_col(df: pd.DataFrame) -> str:
    candidates = ["label", "y_true", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"未找到标签列。现有列: {list(df.columns)}")


def eval_at_threshold(y_true, y_prob, th):
    y_pred = (y_prob >= th).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": th,
        "precision": p,
        "recall": r,
        "f1": f1,
        "f2": f2,
        "accuracy": acc,
        "confusion_matrix": cm,
    }


def pick_best(df_search: pd.DataFrame, objective: str, min_precision=None):
    cand = df_search.copy()
    if min_precision is not None:
        cand = cand[cand["precision"] >= min_precision].copy()

    if cand.empty:
        raise ValueError("没有任何阈值满足最小 precision 约束，请放宽 MIN_PRECISION。")

    sort_cols = [objective, "precision", "recall"]
    cand = cand.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)
    return cand.iloc[0].to_dict()


def main():
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    prob_col_val = get_prob_col(val_df)
    prob_col_test = get_prob_col(test_df)
    label_col_val = get_label_col(val_df)
    label_col_test = get_label_col(test_df)

    y_val = val_df[label_col_val].astype(int).values
    p_val = val_df[prob_col_val].astype(float).values

    y_test = test_df[label_col_test].astype(int).values
    p_test = test_df[prob_col_test].astype(float).values

    thresholds = np.arange(TH_START, TH_END + 1e-12, TH_STEP)
    rows = [eval_at_threshold(y_val, p_val, float(th)) for th in thresholds]
    search_df = pd.DataFrame(rows)

    best = pick_best(search_df, OBJECTIVE, MIN_PRECISION)
    best_th = float(best["threshold"])

    val_auc = roc_auc_score(y_val, p_val)
    val_ap = average_precision_score(y_val, p_val)

    test_metrics = eval_at_threshold(y_test, p_test, best_th)
    test_metrics["roc_auc"] = float(roc_auc_score(y_test, p_test))
    test_metrics["pr_auc"] = float(average_precision_score(y_test, p_test))

    out_csv = OUTPUT_DIR / "threshold_search_rescanned.csv"
    out_json = OUTPUT_DIR / "selected_threshold_rescanned.json"
    out_test_json = OUTPUT_DIR / "final_metrics_rescanned_test.json"

    search_df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": OBJECTIVE,
                "min_precision": MIN_PRECISION,
                "best_threshold_on_oof_val": best_th,
                "best_val_metrics": {
                    "precision": float(best["precision"]),
                    "recall": float(best["recall"]),
                    "f1": float(best["f1"]),
                    "f2": float(best["f2"]),
                    "accuracy": float(best["accuracy"]),
                    "roc_auc": float(val_auc),
                    "pr_auc": float(val_ap),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_test_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold_from_oof_val": best_th,
                "test_metrics": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("已完成细粒度阈值搜索")
    print(f"最佳阈值: {best_th:.4f}")
    print("验证集最佳结果:")
    print(
        f"  P={best['precision']:.4f}  R={best['recall']:.4f}  "
        f"F1={best['f1']:.4f}  F2={best['f2']:.4f}"
    )
    print("测试集应用该阈值后的结果:")
    print(
        f"  P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}  "
        f"F1={test_metrics['f1']:.4f}  F2={test_metrics['f2']:.4f}  "
        f"ACC={test_metrics['accuracy']:.4f}"
    )
    print(f"输出文件: {out_csv}")
    print(f"输出文件: {out_json}")
    print(f"输出文件: {out_test_json}")


if __name__ == "__main__":
    main()