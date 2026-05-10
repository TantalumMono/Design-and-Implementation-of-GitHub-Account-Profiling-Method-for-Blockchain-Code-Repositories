#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined bias probe for augmented positive samples.

What it does:
1) Text bias check
   - synthetic vs original text distinguishability probe (TF-IDF + LogisticRegression, 5-fold ROC-AUC)
   - family-level original vs synthetic centroid cosine similarity
   - synthetic-to-source-original cosine similarity
   - unique-ratio checks for full text and README
   - top positive / negative text coefficients from a full-data probe

2) Family-level structured drift check
   - continuous features: absolute standardized mean difference (abs SMD) within each family
   - boolean features: absolute rate difference within each family
   - summary statistics across all family-feature comparisons

Default behavior:
- read augmented_300_with_meta.json in the same directory as this script
- if not found, fall back to glob: augmented_300_with_meta*.json
- write outputs to ./combined_bias_probe_outputs/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_STATE = 42
N_SPLITS = 5
TEXT_MAX_FEATURES = 6000
TEXT_MIN_DF = 2
TEXT_NGRAM_RANGE = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Input JSON path. Defaults to script-dir augmented_300_with_meta.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder. Defaults to script-dir/combined_bias_probe_outputs")
    return parser.parse_args()


def resolve_default_input(script_path: Path) -> Path:
    exact = script_path.parent / "augmented_300_with_meta.json"
    if exact.exists():
        return exact
    matches = sorted(script_path.parent.glob("augmented_300_with_meta*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {exact.name} in {script_path.parent}. "
        f"Also tried glob augmented_300_with_meta*.json"
    )


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of samples.")
    return data


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def build_dataframe(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    all_feature_names = set()
    for s in samples:
        feat = s.get("features", {}) or {}
        if isinstance(feat, dict):
            all_feature_names.update(feat.keys())
    all_feature_names = sorted(all_feature_names)

    for i, s in enumerate(samples):
        meta = s.get("meta", {}) or {}
        feat = s.get("features", {}) or {}

        row: Dict[str, Any] = {
            "row_id": i,
            "readme_text": safe_str(s.get("readme_text", "")),
            "description_text": safe_str(s.get("description_text", "")),
            "topics_text": safe_str(s.get("topics_text", "")),
            "combined_text": safe_str(s.get("combined_text", "")),
            "is_synthetic": int(meta.get("is_synthetic", 0)),
            "family_label": safe_str(meta.get("family_label", "UNKNOWN")),
            "source_raw_file_name": safe_str(meta.get("source_raw_file_name", "")),
            "source_repo_full_name": safe_str(meta.get("source_repo_full_name", "")),
            "augmentation_method": safe_str(meta.get("augmentation_method", "")),
            "repo_full_name": safe_str((feat or {}).get("repo_full_name", "")),
            "raw_file_name": safe_str((feat or {}).get("raw_file_name", "")),
        }
        row["full_text"] = "\n".join([
            row["readme_text"],
            row["description_text"],
            row["topics_text"],
            row["combined_text"],
        ]).strip()

        for fn in all_feature_names:
            row[f"feat__{fn}"] = feat.get(fn, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------- Text bias probe ----------

def run_text_probe(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    y = df["is_synthetic"].astype(int).values
    texts = df["full_text"].fillna("").astype(str).values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    all_oof = np.zeros(len(df), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, y), start=1):
        vectorizer = TfidfVectorizer(
            max_features=TEXT_MAX_FEATURES,
            min_df=TEXT_MIN_DF,
            ngram_range=TEXT_NGRAM_RANGE,
            sublinear_tf=True,
            lowercase=False,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b",
        )
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        pipe = make_pipeline(vectorizer, MaxAbsScaler(), clf)
        pipe.fit(texts[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(texts[te_idx])[:, 1]
        auc = roc_auc_score(y[te_idx], prob)
        all_oof[te_idx] = prob
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "roc_auc": float(auc),
        })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "text_probe_fold_metrics.csv", index=False, encoding="utf-8-sig")

    # Full-data fit for coefficients and centroid/similarity analysis
    vectorizer = TfidfVectorizer(
        max_features=TEXT_MAX_FEATURES,
        min_df=TEXT_MIN_DF,
        ngram_range=TEXT_NGRAM_RANGE,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b",
    )
    X = vectorizer.fit_transform(texts)
    scaler = MaxAbsScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(Xs, y)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    pos_df = coef_df.sort_values("coef", ascending=False).head(50).copy()
    neg_df = coef_df.sort_values("coef", ascending=True).head(50).copy()
    pos_df.to_csv(output_dir / "text_top_positive_coefficients.csv", index=False, encoding="utf-8-sig")
    neg_df.to_csv(output_dir / "text_top_negative_coefficients.csv", index=False, encoding="utf-8-sig")

    # Overall centroid cosine
    overall_orig_idx = df.index[df["is_synthetic"] == 0].tolist()
    overall_syn_idx = df.index[df["is_synthetic"] == 1].tolist()
    if len(overall_orig_idx) > 0 and len(overall_syn_idx) > 0:
        overall_orig_centroid = np.asarray(X[overall_orig_idx].mean(axis=0))
        overall_syn_centroid = np.asarray(X[overall_syn_idx].mean(axis=0))
        overall_centroid_cosine = float(cosine_similarity(overall_orig_centroid, overall_syn_centroid)[0, 0])
    else:
        overall_centroid_cosine = None

    # Family centroid cosine
    centroid_rows = []
    for family, sub in df.groupby("family_label"):
        orig_idx = sub.index[sub["is_synthetic"] == 0].tolist()
        syn_idx = sub.index[sub["is_synthetic"] == 1].tolist()
        if len(orig_idx) == 0 or len(syn_idx) == 0:
            continue
        orig_centroid = np.asarray(X[orig_idx].mean(axis=0))
        syn_centroid = np.asarray(X[syn_idx].mean(axis=0))
        cos = float(cosine_similarity(orig_centroid, syn_centroid)[0, 0])
        centroid_rows.append({
            "family_label": family,
            "n_original": int(len(orig_idx)),
            "n_synthetic": int(len(syn_idx)),
            "centroid_cosine": cos,
        })
    centroid_df = pd.DataFrame(centroid_rows).sort_values("family_label")
    centroid_df.to_csv(output_dir / "text_family_centroid_cosine.csv", index=False, encoding="utf-8-sig")

    # Synthetic-to-source similarity
    # Prefer source_raw_file_name -> original raw_file_name exact match; fall back to source_repo_full_name -> repo_full_name
    original_df = df[df["is_synthetic"] == 0].copy()
    raw_to_idx = {}
    repo_to_idx = {}
    for idx, row in original_df.iterrows():
        if row["raw_file_name"]:
            raw_to_idx.setdefault(row["raw_file_name"], idx)
        if row["repo_full_name"]:
            repo_to_idx.setdefault(row["repo_full_name"], idx)

    sim_rows = []
    for idx, row in df[df["is_synthetic"] == 1].iterrows():
        source_idx = None
        match_mode = None
        if row["source_raw_file_name"] and row["source_raw_file_name"] in raw_to_idx:
            source_idx = raw_to_idx[row["source_raw_file_name"]]
            match_mode = "source_raw_file_name"
        elif row["source_repo_full_name"] and row["source_repo_full_name"] in repo_to_idx:
            source_idx = repo_to_idx[row["source_repo_full_name"]]
            match_mode = "source_repo_full_name"
        if source_idx is None:
            continue
        sim = float(cosine_similarity(X[idx], X[source_idx])[0, 0])
        sim_rows.append({
            "synthetic_row_id": int(idx),
            "family_label": row["family_label"],
            "source_match_mode": match_mode,
            "source_row_id": int(source_idx),
            "source_raw_file_name": row["source_raw_file_name"],
            "source_repo_full_name": row["source_repo_full_name"],
            "cosine_similarity": sim,
        })
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(output_dir / "text_synthetic_to_source_similarity.csv", index=False, encoding="utf-8-sig")

    # Unique ratio checks
    def unique_ratio(series: pd.Series) -> float:
        vals = series.fillna("").astype(str)
        return float(vals.nunique() / max(len(vals), 1))

    full_unique_original = unique_ratio(df[df["is_synthetic"] == 0]["full_text"])
    full_unique_synthetic = unique_ratio(df[df["is_synthetic"] == 1]["full_text"])
    readme_unique_original = unique_ratio(df[df["is_synthetic"] == 0]["readme_text"])
    readme_unique_synthetic = unique_ratio(df[df["is_synthetic"] == 1]["readme_text"])

    report = {
        "n_samples": int(len(df)),
        "n_original": int((df["is_synthetic"] == 0).sum()),
        "n_synthetic": int((df["is_synthetic"] == 1).sum()),
        "cv": {
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "mean_roc_auc": float(fold_df["roc_auc"].mean()),
            "std_roc_auc": float(fold_df["roc_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        },
        "interpretation": (
            "AUC 越接近 0.50，说明 original 与 synthetic 越不容易仅凭文本特征被区分，文本合成偏置越弱；"
            "若 AUC 明显高于 0.60，则提示存在可识别的文本合成痕迹。"
        ),
        "overall_centroid_cosine": overall_centroid_cosine,
        "family_centroid_cosine_summary": {
            "mean": float(centroid_df["centroid_cosine"].mean()) if len(centroid_df) else None,
            "median": float(centroid_df["centroid_cosine"].median()) if len(centroid_df) else None,
            "min": float(centroid_df["centroid_cosine"].min()) if len(centroid_df) else None,
            "max": float(centroid_df["centroid_cosine"].max()) if len(centroid_df) else None,
        },
        "synthetic_to_source_similarity_summary": {
            "n_matched": int(len(sim_df)),
            "mean": float(sim_df["cosine_similarity"].mean()) if len(sim_df) else None,
            "median": float(sim_df["cosine_similarity"].median()) if len(sim_df) else None,
            "min": float(sim_df["cosine_similarity"].min()) if len(sim_df) else None,
            "q1": float(sim_df["cosine_similarity"].quantile(0.25)) if len(sim_df) else None,
            "q3": float(sim_df["cosine_similarity"].quantile(0.75)) if len(sim_df) else None,
            "max": float(sim_df["cosine_similarity"].max()) if len(sim_df) else None,
        },
        "unique_ratio": {
            "full_text_original": full_unique_original,
            "full_text_synthetic": full_unique_synthetic,
            "readme_original": readme_unique_original,
            "readme_synthetic": readme_unique_synthetic,
        },
        "top10_positive_coefficients": pos_df.head(10).to_dict(orient="records"),
        "top10_negative_coefficients": neg_df.head(10).to_dict(orient="records"),
    }
    return report


# ---------- Family-level structured drift ----------

def identify_structured_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    feat_cols = [c for c in df.columns if c.startswith("feat__")]
    continuous = []
    booleanish = []
    for c in feat_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        non_na = vals.dropna()
        if len(non_na) == 0:
            continue
        uniq = set(np.unique(non_na.values))
        if uniq.issubset({0, 1}):
            booleanish.append(c)
        else:
            continuous.append(c)
    return continuous, booleanish


def abs_smd(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    mx, my = float(x.mean()), float(y.mean())
    vx = float(x.var(ddof=1)) if len(x) > 1 else 0.0
    vy = float(y.var(ddof=1)) if len(y) > 1 else 0.0
    pooled = math.sqrt(max((vx + vy) / 2.0, 1e-12))
    return abs(mx - my) / pooled


def run_structured_family_drift(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    continuous_cols, boolean_cols = identify_structured_feature_types(df)

    cont_rows = []
    bool_rows = []

    for family, sub in df.groupby("family_label"):
        orig = sub[sub["is_synthetic"] == 0].copy()
        syn = sub[sub["is_synthetic"] == 1].copy()
        if len(orig) == 0 or len(syn) == 0:
            continue

        for c in continuous_cols:
            x = pd.to_numeric(orig[c], errors="coerce")
            y = pd.to_numeric(syn[c], errors="coerce")
            if x.notna().sum() == 0 or y.notna().sum() == 0:
                continue
            cont_rows.append({
                "family_label": family,
                "feature": c.replace("feat__", ""),
                "orig_n": int(x.notna().sum()),
                "syn_n": int(y.notna().sum()),
                "orig_mean": float(x.mean()),
                "syn_mean": float(y.mean()),
                "abs_smd": float(abs_smd(x, y)),
            })

        for c in boolean_cols:
            x = pd.to_numeric(orig[c], errors="coerce")
            y = pd.to_numeric(syn[c], errors="coerce")
            if x.notna().sum() == 0 or y.notna().sum() == 0:
                continue
            bool_rows.append({
                "family_label": family,
                "feature": c.replace("feat__", ""),
                "orig_n": int(x.notna().sum()),
                "syn_n": int(y.notna().sum()),
                "orig_rate": float(x.mean()),
                "syn_rate": float(y.mean()),
                "abs_diff": float(abs(float(x.mean()) - float(y.mean()))),
            })

    cont_df = pd.DataFrame(cont_rows).sort_values(["abs_smd", "family_label"], ascending=[False, True]) if cont_rows else pd.DataFrame()
    bool_df = pd.DataFrame(bool_rows).sort_values(["abs_diff", "family_label"], ascending=[False, True]) if bool_rows else pd.DataFrame()

    cont_df.to_csv(output_dir / "structured_family_continuous_smd.csv", index=False, encoding="utf-8-sig")
    bool_df.to_csv(output_dir / "structured_family_boolean_abs_diff.csv", index=False, encoding="utf-8-sig")

    report = {
        "n_structured_features_total": int(len(continuous_cols) + len(boolean_cols)),
        "n_continuous_features": int(len(continuous_cols)),
        "n_boolean_features": int(len(boolean_cols)),
        "continuous_summary": {
            "n_rows": int(len(cont_df)),
            "median_abs_smd": float(cont_df["abs_smd"].median()) if len(cont_df) else None,
            "mean_abs_smd": float(cont_df["abs_smd"].mean()) if len(cont_df) else None,
            "pct_abs_smd_gt_0_1": float((cont_df["abs_smd"] > 0.1).mean()) if len(cont_df) else None,
            "pct_abs_smd_gt_0_2": float((cont_df["abs_smd"] > 0.2).mean()) if len(cont_df) else None,
            "max_abs_smd": float(cont_df["abs_smd"].max()) if len(cont_df) else None,
            "top10": cont_df.head(10).to_dict(orient="records") if len(cont_df) else [],
        },
        "boolean_summary": {
            "n_rows": int(len(bool_df)),
            "median_abs_diff": float(bool_df["abs_diff"].median()) if len(bool_df) else None,
            "mean_abs_diff": float(bool_df["abs_diff"].mean()) if len(bool_df) else None,
            "pct_abs_diff_gt_0_02": float((bool_df["abs_diff"] > 0.02).mean()) if len(bool_df) else None,
            "pct_abs_diff_gt_0_05": float((bool_df["abs_diff"] > 0.05).mean()) if len(bool_df) else None,
            "pct_abs_diff_gt_0_1": float((bool_df["abs_diff"] > 0.1).mean()) if len(bool_df) else None,
            "max_abs_diff": float(bool_df["abs_diff"].max()) if len(bool_df) else None,
            "top10": bool_df.head(10).to_dict(orient="records") if len(bool_df) else [],
        },
    }
    return report


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    input_path = Path(args.input).resolve() if args.input else resolve_default_input(script_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (script_path.parent / "combined_bias_probe_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_path)
    df = build_dataframe(samples)

    # Basic counts
    n_original = int((df["is_synthetic"] == 0).sum())
    n_synthetic = int((df["is_synthetic"] == 1).sum())

    text_report = run_text_probe(df, output_dir)
    structured_report = run_structured_family_drift(df, output_dir)

    final_report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_samples": int(len(df)),
        "n_original": n_original,
        "n_synthetic": n_synthetic,
        "n_families": int(df["family_label"].nunique()),
        "family_counts": (
            df.groupby(["family_label", "is_synthetic"]).size().rename("n").reset_index().to_dict(orient="records")
        ),
        "text_bias": text_report,
        "structured_family_drift": structured_report,
        "notes": [
            "Text bias probe: 5-fold ROC-AUC for synthetic-vs-original distinguishability using TF-IDF + LogisticRegression.",
            "Structured family drift: continuous features use abs SMD within family; boolean features use absolute rate difference within family.",
            "AUC near 0.50 suggests weak global distinguishability; low SMD/abs_diff suggests weak family-level drift.",
        ],
    }

    with open(output_dir / "combined_bias_probe_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"Done. Input: {input_path}")
    print(f"Outputs written to: {output_dir}")
    print("Key files:")
    print("- combined_bias_probe_report.json")
    print("- text_probe_fold_metrics.csv")
    print("- text_family_centroid_cosine.csv")
    print("- text_synthetic_to_source_similarity.csv")
    print("- text_top_positive_coefficients.csv")
    print("- text_top_negative_coefficients.csv")
    print("- structured_family_continuous_smd.csv")
    print("- structured_family_boolean_abs_diff.csv")


if __name__ == "__main__":
    main()
