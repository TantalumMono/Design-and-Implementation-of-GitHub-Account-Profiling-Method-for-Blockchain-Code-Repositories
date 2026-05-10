
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# =========================================================
# 0. CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 这里读取你上一份固定参数验证脚本输出的 OOF 预测
OOF_PRED_PATH = SCRIPT_DIR / "fixed_params_family_validation_outputs" / "fixed_params_family_oof_predictions.csv"
OUTPUT_DIR = SCRIPT_DIR / "fixed_params_threshold_refit_outputs"

BETA = 2.0


# =========================================================
# 1. UTILS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def calc_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def select_best_threshold_f1_priority(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0):
    thresholds = np.unique(np.round(y_proba, 10))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    best = None
    all_rows = []

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f_beta = calc_fbeta(p, r, beta=beta)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])

        row = {
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "f_beta": float(f_beta),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
        all_rows.append(row)

        if best is None:
            best = row
            continue

        better = (
            (row["f1"] > best["f1"])
            or (np.isclose(row["f1"], best["f1"]) and row["recall"] > best["recall"])
            or (np.isclose(row["f1"], best["f1"]) and np.isclose(row["recall"], best["recall"]) and row["precision"] > best["precision"])
            or (np.isclose(row["f1"], best["f1"]) and np.isclose(row["recall"], best["recall"]) and np.isclose(row["precision"], best["precision"]) and row["f_beta"] > best["f_beta"])
            or (np.isclose(row["f1"], best["f1"]) and np.isclose(row["recall"], best["recall"]) and np.isclose(row["precision"], best["precision"]) and np.isclose(row["f_beta"], best["f_beta"]) and row["threshold"] > best["threshold"])
        )
        if better:
            best = row

    return best, pd.DataFrame(all_rows)


# =========================================================
# 2. MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    if not OOF_PRED_PATH.exists():
        raise FileNotFoundError(f"未找到 OOF 预测文件: {OOF_PRED_PATH}")

    df = pd.read_csv(OOF_PRED_PATH)
    required = {"label", "y_proba"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OOF 文件缺少必要字段: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["y_proba"] = pd.to_numeric(df["y_proba"], errors="coerce")

    if df["y_proba"].isna().any():
        raise ValueError("y_proba 中存在无法解析的空值或非法值")

    y_true = df["label"].values
    y_proba = df["y_proba"].values

    best_row, threshold_df = select_best_threshold_f1_priority(y_true, y_proba, beta=BETA)

    # 用最佳阈值生成最终预测列
    best_threshold = best_row["threshold"]
    df["y_pred_refit_threshold"] = (df["y_proba"] >= best_threshold).astype(int)

    threshold_df.to_csv(
        OUTPUT_DIR / "threshold_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df.to_csv(
        OUTPUT_DIR / "fixed_params_family_oof_predictions_with_refit_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "source_oof_path": str(OOF_PRED_PATH),
        "selection_rule": "maximize_f1_then_recall_then_precision_then_fbeta_then_higher_threshold",
        "beta": BETA,
        "best_threshold_result": best_row,
    }

    with open(OUTPUT_DIR / "refit_threshold_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Best threshold refit done.")
    print("Best threshold:", best_threshold)
    print("Precision:", best_row["precision"])
    print("Recall:", best_row["recall"])
    print("F1:", best_row["f1"])
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
