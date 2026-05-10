# -*- coding: utf-8 -*-
"""
结构化特征合成偏置检查脚本
- 默认输入: 脚本同目录下的 augmented_300_with_meta.json
- 默认输出: 脚本同目录下的 structured_bias_probe_outputs/
- 标签: meta.is_synthetic (0=original, 1=synthetic)
- 特征: features 中所有可数值化字段
- 模型: LogisticRegression 线性探针
- 评估: 5折 StratifiedKFold ROC-AUC
- 额外输出: 全量拟合后的正/负向系数Top特征

说明：
1. 默认不需要传参，直接运行即可。
2. 若同目录下不存在严格文件名 augmented_300_with_meta.json，
   会自动尝试匹配 augmented_300_with_meta*.json 作为兜底。
3. 如需手动指定，也仍可通过 --input / --output_dir 覆盖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "augmented_300_with_meta.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "structured_bias_probe_outputs"


def is_numeric_like(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    return False


def resolve_input_path(user_input: str | None) -> Path:
    if user_input:
        path = Path(user_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"未找到输入文件: {path}")
        return path

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    candidates = sorted(SCRIPT_DIR.glob("augmented_300_with_meta*.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "未在脚本同目录下找到 augmented_300_with_meta.json，"
        "也未找到 augmented_300_with_meta*.json。"
    )


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是 list")
    return data


def infer_feature_names(samples: List[Dict[str, Any]]) -> List[str]:
    feat_names = set()
    for s in samples:
        feat = s.get("features", {})
        if not isinstance(feat, dict):
            continue
        for k, v in feat.items():
            if is_numeric_like(v):
                feat_names.add(k)
    return sorted(feat_names)


def build_dataframe(samples: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    feature_names = infer_feature_names(samples)
    rows = []
    for i, s in enumerate(samples):
        meta = s.get("meta", {}) if isinstance(s.get("meta", {}), dict) else {}
        feat = s.get("features", {}) if isinstance(s.get("features", {}), dict) else {}

        row: Dict[str, Any] = {
            "row_id": i,
            "is_synthetic": int(meta.get("is_synthetic", 0)),
            "family_label": meta.get("family_label", "UNKNOWN"),
            "augmentation_method": meta.get("augmentation_method", ""),
            "source_repo_full_name": meta.get("source_repo_full_name", ""),
        }

        for fn in feature_names:
            v = feat.get(fn, np.nan)
            if isinstance(v, bool):
                v = int(v)
            row[fn] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, feature_names


def build_probe(feature_names: List[str]) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipe, feature_names)],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def cross_validated_probe(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, float, float]:
    X = df[feature_names].copy()
    y = df["is_synthetic"].astype(int).values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = build_probe(feature_names)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob)

        rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "n_train_original": int((y_tr == 0).sum()),
                "n_train_synthetic": int((y_tr == 1).sum()),
                "n_test_original": int((y_te == 0).sum()),
                "n_test_synthetic": int((y_te == 1).sum()),
                "roc_auc": float(auc),
            }
        )

    fold_df = pd.DataFrame(rows)
    mean_auc = float(fold_df["roc_auc"].mean())
    std_auc = float(fold_df["roc_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0
    return fold_df, mean_auc, std_auc


def fit_full_and_export_coefs(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    X = df[feature_names].copy()
    y = df["is_synthetic"].astype(int).values

    clf = build_probe(feature_names)
    clf.fit(X, y)

    model: LogisticRegression = clf.named_steps["model"]
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return coef_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="可选：手动指定 augmented_300_with_meta.json 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="可选：手动指定输出目录")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_json_list(input_path)
    df, feature_names = build_dataframe(samples)

    if df["is_synthetic"].nunique() < 2:
        raise ValueError("is_synthetic 只有一个类别，无法做区分性探针")

    fold_df, mean_auc, std_auc = cross_validated_probe(df, feature_names)
    coef_df = fit_full_and_export_coefs(df, feature_names)

    fold_df.to_csv(output_dir / "structured_probe_fold_metrics.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(output_dir / "structured_probe_coefficients.csv", index=False, encoding="utf-8-sig")

    report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_samples": int(len(df)),
        "n_original": int((df["is_synthetic"] == 0).sum()),
        "n_synthetic": int((df["is_synthetic"] == 1).sum()),
        "n_structured_features": int(len(feature_names)),
        "cv": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "mean_roc_auc": mean_auc,
            "std_roc_auc": std_auc,
        },
        "interpretation": (
            "AUC 越接近 0.50，说明 original 与 synthetic 越不容易仅凭结构化特征被区分，"
            "结构化合成偏置越弱；若 AUC 明显高于 0.60，则提示存在可识别的结构化合成痕迹。"
        ),
        "top10_positive_coefficients": coef_df.head(10)[["feature", "coef"]].to_dict(orient="records"),
        "top10_negative_coefficients": coef_df.sort_values("coef", ascending=True).head(10)[["feature", "coef"]].to_dict(orient="records"),
    }

    with (output_dir / "structured_probe_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== Structured synthetic-bias probe finished ===")
    print(f"Input: {input_path}")
    print(f"Samples: {len(df)} | original={(df['is_synthetic'] == 0).sum()} | synthetic={(df['is_synthetic'] == 1).sum()}")
    print(f"Structured features: {len(feature_names)}")
    print(f"5-fold ROC-AUC: {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
