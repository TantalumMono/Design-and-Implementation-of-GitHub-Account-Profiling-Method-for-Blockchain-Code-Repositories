# -*- coding: utf-8 -*-
"""
Plot Top-10 SHAP beeswarm and bar figures for final XGBoost + TF-IDF model.

Output:
1. shap_beeswarm_top10.svg
2. shap_beeswarm_top10.pdf
3. shap_bar_top10.svg
4. shap_bar_top10.pdf

Notes:
- This script assumes your final model bundle contains:
  model
  structured_cols
  fill_values
  readme_vectorizer / vectorizer_readme
  aux_vectorizer / vectorizer_aux
  readme_text_col / aux_text_col, or default text columns

- If your bundle key names are slightly different, modify the key-reading part.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack
import shap


# =========================================================
# 0. Path config
# =========================================================
BASE_DIR = Path(".")

MODEL_BUNDLE_PATH = BASE_DIR / "final_model_artifacts" / "final_repo_binary_classifier_bundle.joblib"
DATA_PATH = BASE_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"

OUT_DIR = BASE_DIR / "shap_top10_vector_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10
RANDOM_SAMPLE_N = None
# 如果样本太多、SHAP 计算慢，可以改成 1000 或 1500。
# None 表示使用全部样本。


# =========================================================
# 1. Matplotlib config
# =========================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_current_figure(name: str):
    svg_path = OUT_DIR / f"{name}.svg"
    pdf_path = OUT_DIR / f"{name}.pdf"
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


# =========================================================
# 2. Helper functions
# =========================================================
def get_first_existing_key(d: dict, keys: list, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def prepare_structured_features(df: pd.DataFrame, structured_cols: list, fill_values: dict):
    X = df[structured_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    fill_series = pd.Series(fill_values)
    fill_series = fill_series.reindex(structured_cols)

    X = X.fillna(fill_series).fillna(0)
    return csr_matrix(X.to_numpy(dtype=np.float32))


def get_text_column(df: pd.DataFrame, preferred_cols: list):
    for c in preferred_cols:
        if c in df.columns:
            return c
    raise ValueError(f"None of these text columns exist in dataframe: {preferred_cols}")


def build_feature_matrix_and_names(df: pd.DataFrame, bundle: dict):
    model = bundle["model"]
    structured_cols = bundle["structured_cols"]
    fill_values = bundle.get("fill_values", {})

    # ---- structured features ----
    X_struct = prepare_structured_features(df, structured_cols, fill_values)
    feature_names = list(structured_cols)

    matrices = [X_struct]

    # ---- readme TF-IDF ----
    readme_vectorizer = get_first_existing_key(
        bundle,
        ["readme_vectorizer", "vectorizer_readme", "tfidf_readme_vectorizer"],
        default=None,
    )

    if readme_vectorizer is not None:
        readme_text_col = get_first_existing_key(
            bundle,
            ["readme_text_col", "text_col_readme"],
            default=None,
        )

        if readme_text_col is None:
            readme_text_col = get_text_column(
                df,
                [
                    "readme_text_clean",
                    "readme_clean",
                    "text_for_tfidf_clean",
                    "combined_text_clean",
                ],
            )

        readme_text = df[readme_text_col].fillna("").astype(str)
        X_readme = readme_vectorizer.transform(readme_text)
        readme_features = [
            f"tfidf_readme::{x}" for x in readme_vectorizer.get_feature_names_out()
        ]

        matrices.append(X_readme)
        feature_names.extend(readme_features)

    # ---- auxiliary TF-IDF ----
    aux_vectorizer = get_first_existing_key(
        bundle,
        ["aux_vectorizer", "vectorizer_aux", "tfidf_aux_vectorizer"],
        default=None,
    )

    if aux_vectorizer is not None:
        aux_text_col = get_first_existing_key(
            bundle,
            ["aux_text_col", "text_col_aux"],
            default=None,
        )

        if aux_text_col is None:
            aux_text_col = get_text_column(
                df,
                [
                    "aux_text_clean",
                    "description_topics_text_clean",
                    "combined_text_clean",
                    "text_for_tfidf_clean",
                ],
            )

        aux_text = df[aux_text_col].fillna("").astype(str)
        X_aux = aux_vectorizer.transform(aux_text)
        aux_features = [
            f"tfidf_aux::{x}" for x in aux_vectorizer.get_feature_names_out()
        ]

        matrices.append(X_aux)
        feature_names.extend(aux_features)

    # ---- fallback: single vectorizer ----
    # 如果你的 bundle 不是 readme + aux 两套向量器，而是单一 vectorizer，则走这里
    if readme_vectorizer is None and aux_vectorizer is None:
        vectorizer = get_first_existing_key(
            bundle,
            ["vectorizer", "tfidf_vectorizer"],
            default=None,
        )

        if vectorizer is not None:
            text_col = get_first_existing_key(
                bundle,
                ["text_col"],
                default=None,
            )

            if text_col is None:
                text_col = get_text_column(
                    df,
                    [
                        "combined_text_clean",
                        "text_for_tfidf_clean",
                        "readme_text_clean",
                    ],
                )

            text = df[text_col].fillna("").astype(str)
            X_text = vectorizer.transform(text)
            tfidf_features = [
                f"tfidf::{x}" for x in vectorizer.get_feature_names_out()
            ]

            matrices.append(X_text)
            feature_names.extend(tfidf_features)

    X_all = hstack(matrices, format="csr")

    if X_all.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name length mismatch: X has {X_all.shape[1]} columns, "
            f"but feature_names has {len(feature_names)} names."
        )

    return X_all, feature_names, model


def select_top_features(shap_values: np.ndarray, feature_names: list, top_n: int):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    return top_idx, top_names, mean_abs[top_idx]


# =========================================================
# 3. Main
# =========================================================
def main():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_BUNDLE_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print("Loading model bundle...")
    bundle = joblib.load(MODEL_BUNDLE_PATH)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    if RANDOM_SAMPLE_N is not None and len(df) > RANDOM_SAMPLE_N:
        df = df.sample(n=RANDOM_SAMPLE_N, random_state=42).reset_index(drop=True)
        print(f"Using random sample: n = {len(df)}")
    else:
        print(f"Using all samples: n = {len(df)}")

    print("Building feature matrix...")
    X_all, feature_names, model = build_feature_matrix_and_names(df, bundle)

    print(f"Feature matrix shape: {X_all.shape}")
    print(f"Feature name count: {len(feature_names)}")

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all)

    # Some SHAP versions may return list for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    print("Selecting top-10 features...")
    top_idx, top_names, top_importance = select_top_features(
        shap_values, feature_names, TOP_N
    )

    # Slice SHAP values and feature matrix to Top-N only
    shap_top = shap_values[:, top_idx]
    X_top = X_all[:, top_idx].toarray()

    # Save Top-N feature importance table
    top_df = pd.DataFrame({
        "rank": np.arange(1, TOP_N + 1),
        "feature": top_names,
        "mean_abs_shap": top_importance,
    })
    top_df.to_csv(
        OUT_DIR / "shap_top10_importance.csv",
        index=False,
        encoding="utf-8-sig",
        )
    print("Saved:", OUT_DIR / "shap_top10_importance.csv")

    # Build SHAP Explanation object
    explanation_top = shap.Explanation(
        values=shap_top,
        data=X_top,
        feature_names=top_names,
    )

    # =====================================================
    # Figure 1: Top-10 beeswarm
    # =====================================================
    print("Plotting Top-10 beeswarm...")
    plt.figure(figsize=(6.2, 4.8))
    shap.plots.beeswarm(
        explanation_top,
        max_display=TOP_N,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Beeswarm Plot (Top 10)", fontsize=12)
    save_current_figure("shap_beeswarm_top10")

    # =====================================================
    # Figure 2: Top-10 bar plot
    # =====================================================
    print("Plotting Top-10 bar...")
    plt.figure(figsize=(6.2, 4.2))
    shap.plots.bar(
        explanation_top,
        max_display=TOP_N,
        show=False,
    )
    plt.title("Global SHAP Importance (Top 10)", fontsize=12)
    save_current_figure("shap_bar_top10")

    print(f"\nAll Top-10 SHAP vector figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()