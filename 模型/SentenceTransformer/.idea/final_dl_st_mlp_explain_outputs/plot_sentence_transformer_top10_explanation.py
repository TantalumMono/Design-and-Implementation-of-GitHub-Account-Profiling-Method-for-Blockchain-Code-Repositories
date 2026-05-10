# -*- coding: utf-8 -*-
"""
Plot Top-10 explanation figures for SentenceTransformer + MLP/deep model.

Required input files:
1. dl_explanation_summary.json
2. global_branch_ablation.csv
3. global_structured_permutation_importance.csv
4. dl_predictions_for_explanation.csv

Output:
SVG and PDF vector figures in ./sentence_transformer_top10_explanation_figures/

Explanation types:
1. Branch ablation importance
2. Structured-feature permutation importance
3. Prediction score distribution
4. Train-set performance summary

Note:
These explanation results are based on the training set, not an independent holdout set.
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# =========================
# 0. Path config
# =========================
BASE_DIR = Path(".")

SUMMARY_PATH = BASE_DIR / "dl_explanation_summary.json"
BRANCH_ABLATION_PATH = BASE_DIR / "global_branch_ablation.csv"
STRUCTURED_IMPORTANCE_PATH = BASE_DIR / "global_structured_permutation_importance.csv"
PRED_PATH = BASE_DIR / "dl_predictions_for_explanation.csv"

OUT_DIR = BASE_DIR / "sentence_transformer_top10_explanation_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10


# =========================
# 1. Matplotlib config
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_vector(fig, name: str):
    """Save figure as both SVG and PDF."""
    svg_path = OUT_DIR / f"{name}.svg"
    pdf_path = OUT_DIR / f"{name}.pdf"

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def wrap_labels(labels, width=34):
    """Wrap long feature names for better layout."""
    return [textwrap.fill(str(x), width=width, break_long_words=False) for x in labels]


def add_hbar_labels(ax, values, fmt="{:.4f}", offset_ratio=0.02):
    """Add value labels to horizontal bars."""
    max_value = max(abs(float(v)) for v in values) if len(values) > 0 else 1.0
    offset = max_value * offset_ratio

    for patch, v in zip(ax.patches, values):
        x = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2

        ax.text(
            x + offset,
            y,
            fmt.format(v),
            va="center",
            ha="left",
            fontsize=8,
            )


def rename_ablation_name(x: str) -> str:
    mapping = {
        "remove_readme_branch": "Remove README branch",
        "remove_aux_branch": "Remove auxiliary text branch",
        "remove_structured_branch": "Remove structured branch",
    }
    return mapping.get(str(x), str(x))


# =========================
# 2. Load files
# =========================
for p in [SUMMARY_PATH, BRANCH_ABLATION_PATH, STRUCTURED_IMPORTANCE_PATH, PRED_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

branch_df = pd.read_csv(BRANCH_ABLATION_PATH)
struct_df = pd.read_csv(STRUCTURED_IMPORTANCE_PATH)
pred_df = pd.read_csv(PRED_PATH)

# Validate branch ablation CSV
required_branch_cols = {
    "ablation",
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "delta_pr_auc",
    "delta_roc_auc",
    "delta_f1",
}
missing_branch = required_branch_cols - set(branch_df.columns)
if missing_branch:
    raise ValueError(f"Branch ablation CSV missing columns: {sorted(missing_branch)}")

# Validate structured importance CSV
required_struct_cols = {"feature", "pr_auc_drop", "f1_drop"}
missing_struct = required_struct_cols - set(struct_df.columns)
if missing_struct:
    raise ValueError(f"Structured importance CSV missing columns: {sorted(missing_struct)}")

# Validate prediction CSV
required_pred_cols = {"label", "y_proba", "y_pred"}
missing_pred = required_pred_cols - set(pred_df.columns)
if missing_pred:
    raise ValueError(f"Prediction CSV missing columns: {sorted(missing_pred)}")

# Numeric coercion
for c in ["pr_auc", "roc_auc", "precision", "recall", "f1", "delta_pr_auc", "delta_roc_auc", "delta_f1"]:
    branch_df[c] = pd.to_numeric(branch_df[c], errors="coerce")

struct_df["pr_auc_drop"] = pd.to_numeric(struct_df["pr_auc_drop"], errors="coerce").fillna(0)
struct_df["f1_drop"] = pd.to_numeric(struct_df["f1_drop"], errors="coerce").fillna(0)

pred_df["label"] = pd.to_numeric(pred_df["label"], errors="coerce").fillna(0).astype(int)
pred_df["y_proba"] = pd.to_numeric(pred_df["y_proba"], errors="coerce")
pred_df["y_pred"] = pd.to_numeric(pred_df["y_pred"], errors="coerce").fillna(0).astype(int)

if pred_df["y_proba"].isna().any():
    raise ValueError("y_proba contains invalid or missing values.")

threshold = float(summary.get("threshold", 0.5))


# =========================
# 3. Compute train-set metrics
# =========================
y_true = pred_df["label"].values
y_score = pred_df["y_proba"].values
y_pred = pred_df["y_pred"].values

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
pr_auc = average_precision_score(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

computed_summary = {
    "threshold": threshold,
    "precision_train": float(precision),
    "recall_train": float(recall),
    "f1_train": float(f1),
    "pr_auc_train": float(pr_auc),
    "roc_auc_train": float(roc_auc),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}
with open(OUT_DIR / "computed_sentence_transformer_train_metrics.json", "w", encoding="utf-8") as f:
    json.dump(computed_summary, f, ensure_ascii=False, indent=2)

print("Computed train-set metrics:")
print(json.dumps(computed_summary, ensure_ascii=False, indent=2))


# =========================
# 4. Save Top-N tables
# =========================
top_f1 = (
    struct_df.sort_values(["f1_drop", "pr_auc_drop"], ascending=False)
    .head(TOP_N)
    .copy()
)

top_pr = (
    struct_df.sort_values(["pr_auc_drop", "f1_drop"], ascending=False)
    .head(TOP_N)
    .copy()
)

top_f1.to_csv(
    OUT_DIR / "sentence_transformer_top10_structured_by_f1_drop.csv",
    index=False,
    encoding="utf-8-sig",
    )
top_pr.to_csv(
    OUT_DIR / "sentence_transformer_top10_structured_by_pr_auc_drop.csv",
    index=False,
    encoding="utf-8-sig",
    )


# =========================
# 5. Figure 1: Top-10 structured features by F1 drop
# =========================
def plot_top10_structured_by_f1():
    df = top_f1.sort_values("f1_drop", ascending=True).copy()

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["f1_drop"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["feature"].values, width=38))
    ax.set_xlabel("F1 drop after feature permutation")
    ax.set_title("Top 10 structured features by F1 drop")

    add_hbar_labels(ax, df["f1_drop"].values, fmt="{:.4f}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig01_top10_structured_features_by_f1_drop")


# =========================
# 6. Figure 2: Top-10 structured features by PR-AUC drop
# =========================
def plot_top10_structured_by_pr_auc():
    df = top_pr.sort_values("pr_auc_drop", ascending=True).copy()

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["pr_auc_drop"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["feature"].values, width=38))
    ax.set_xlabel("PR-AUC drop after feature permutation")
    ax.set_title("Top 10 structured features by PR-AUC drop")

    add_hbar_labels(ax, df["pr_auc_drop"].values, fmt="{:.5f}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig02_top10_structured_features_by_pr_auc_drop")


# =========================
# 7. Figure 3: Branch ablation by F1 drop
# =========================
def plot_branch_ablation_f1():
    df = branch_df.copy()
    df["branch"] = df["ablation"].apply(rename_ablation_name)
    df["f1_drop_abs"] = -df["delta_f1"]
    df = df.sort_values("f1_drop_abs", ascending=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    y = np.arange(len(df))
    ax.barh(y, df["f1_drop_abs"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["branch"].values, width=36))
    ax.set_xlabel("F1 drop after branch removal")
    ax.set_title("Branch ablation importance by F1 drop")

    add_hbar_labels(ax, df["f1_drop_abs"].values, fmt="{:.4f}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig03_branch_ablation_by_f1_drop")


# =========================
# 8. Figure 4: Branch ablation by PR-AUC drop
# =========================
def plot_branch_ablation_pr_auc():
    df = branch_df.copy()
    df["branch"] = df["ablation"].apply(rename_ablation_name)
    df["pr_auc_drop_abs"] = -df["delta_pr_auc"]
    df = df.sort_values("pr_auc_drop_abs", ascending=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    y = np.arange(len(df))
    ax.barh(y, df["pr_auc_drop_abs"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["branch"].values, width=36))
    ax.set_xlabel("PR-AUC drop after branch removal")
    ax.set_title("Branch ablation importance by PR-AUC drop")

    add_hbar_labels(ax, df["pr_auc_drop_abs"].values, fmt="{:.5f}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig04_branch_ablation_by_pr_auc_drop")


# =========================
# 9. Figure 5: Train-set prediction score distribution
# =========================
def plot_score_distribution():
    pos = pred_df.loc[pred_df["label"] == 1, "y_proba"].dropna()
    neg = pred_df.loc[pred_df["label"] == 0, "y_proba"].dropna()

    max_score = max(float(pred_df["y_proba"].max()), threshold)
    bins = np.linspace(0, max_score * 1.02, 50)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(neg, bins=bins, density=True, alpha=0.65, label="Negative")
    ax.hist(pos, bins=bins, density=True, alpha=0.65, label="Positive")

    ax.axvline(threshold, linestyle="--", linewidth=1.0)
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.85,
        f"Threshold = {threshold:.4f}",
        rotation=90,
        ha="right",
        va="center",
        fontsize=9,
        )

    ax.set_xlabel("Predicted probability / risk score")
    ax.set_ylabel("Density")
    ax.set_title("Prediction score distribution on the training set")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig05_trainset_score_distribution")


# =========================
# 10. Figure 6: Train-set metrics
# =========================
def plot_trainset_metrics():
    labels = ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
    values = [precision, recall, f1, pr_auc, roc_auc]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.bar(labels, values, width=0.65)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SentenceTransformer model performance on the training set")

    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            )

    ax.text(
        0.5,
        -0.22,
        "Note: metrics are computed on train-set predictions.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig06_trainset_metrics")


# =========================
# 11. Main
# =========================
def main():
    plot_top10_structured_by_f1()
    plot_top10_structured_by_pr_auc()
    plot_branch_ablation_f1()
    plot_branch_ablation_pr_auc()
    plot_score_distribution()
    plot_trainset_metrics()

    print(f"\nAll SentenceTransformer explanation figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()