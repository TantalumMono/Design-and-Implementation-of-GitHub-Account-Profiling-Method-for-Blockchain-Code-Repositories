from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DEFAULT_CSV_NAME = "fixed_params_family_oof_predictions_with_refit_threshold.csv"
DEFAULT_JSON_NAME = "refit_threshold_report.json"


def find_default_file(script_dir: Path, strict_name: str, prefix: str) -> Optional[Path]:
    p = script_dir / strict_name
    if p.exists():
        return p
    matches = sorted(script_dir.glob(prefix))
    return matches[0] if matches else None


def load_best_threshold(report_path: Path) -> Optional[float]:
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("best_threshold_result", {}).get("threshold")


def choose_prediction_column(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # 优先使用重定阈值后的预测列
    if "y_pred_refit_threshold" in df.columns:
        pred = pd.to_numeric(df["y_pred_refit_threshold"], errors="coerce").fillna(0).astype(int)
        return pred, "y_pred_refit_threshold"

    # 其次使用已有预测列
    if "y_pred" in df.columns:
        pred = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
        return pred, "y_pred"

    raise ValueError("未找到可用预测列。需要 y_pred_refit_threshold 或 y_pred。")


def calc_subset_metrics(sub: pd.DataFrame, pred_col: str) -> dict:
    y_true = pd.to_numeric(sub["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(sub[pred_col], errors="coerce").fillna(0).astype(int).to_numpy()

    if len(sub) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "recall": None,
            "precision": None,
            "f1": None,
            "tp": None,
            "fn": None,
            "tn": None,
            "fp": None,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分别统计原始正样本与生成正样本的识别准确率/召回率等指标")
    parser.add_argument("--pred_csv", type=str, default=None, help="预测结果 CSV 路径")
    parser.add_argument("--threshold_json", type=str, default=None, help="阈值报告 JSON 路径（可选，仅写入报告）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出文件夹路径")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    pred_csv = Path(args.pred_csv) if args.pred_csv else find_default_file(
        script_dir,
        DEFAULT_CSV_NAME,
        "fixed_params_family_oof_predictions_with_refit_threshold*.csv",
    )
    if pred_csv is None or not pred_csv.exists():
        raise FileNotFoundError(
            "未找到预测 CSV。请将 fixed_params_family_oof_predictions_with_refit_threshold.csv 放在脚本同目录，"
            "或使用 --pred_csv 指定路径。"
        )

    threshold_json = Path(args.threshold_json) if args.threshold_json else find_default_file(
        script_dir,
        DEFAULT_JSON_NAME,
        "refit_threshold_report*.json",
    )
    best_threshold = load_best_threshold(threshold_json) if threshold_json is not None else None

    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "positive_subset_accuracy_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    required_cols = {"label", "is_real_positive", "is_generated_positive"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"预测 CSV 缺少必要字段: {sorted(missing)}")

    pred_series, pred_col_name = choose_prediction_column(df)
    df["pred_used"] = pred_series

    real_pos = df[df["is_real_positive"] == 1].copy()
    gen_pos = df[df["is_generated_positive"] == 1].copy()
    all_pos = df[df["label"] == 1].copy()

    summary = {
        "input_csv": str(pred_csv),
        "threshold_report_json": str(threshold_json) if threshold_json is not None and threshold_json.exists() else None,
        "best_threshold_from_report": best_threshold,
        "prediction_column_used": pred_col_name,
        "all_positive": calc_subset_metrics(all_pos, "pred_used"),
        "original_positive": calc_subset_metrics(real_pos, "pred_used"),
        "synthetic_positive": calc_subset_metrics(gen_pos, "pred_used"),
    }

    rows = []
    for subset_name, subset_df in [
        ("all_positive", all_pos),
        ("original_positive", real_pos),
        ("synthetic_positive", gen_pos),
    ]:
        m = calc_subset_metrics(subset_df, "pred_used")
        rows.append({"subset": subset_name, **m})

    pd.DataFrame(rows).to_csv(output_dir / "positive_subset_metrics.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "positive_subset_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Prediction column used:", pred_col_name)
    if best_threshold is not None:
        print("Best threshold from report:", best_threshold)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
