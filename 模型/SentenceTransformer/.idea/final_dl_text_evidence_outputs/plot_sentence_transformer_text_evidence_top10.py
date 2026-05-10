# -*- coding: utf-8 -*-
"""
Plot Top-10 text evidence figures for SentenceTransformer text-branch explanation.

Required input files:
1. text_evidence_overview.json
2. text_evidence_summary.csv
3. global_text_branch_ablation.csv
4. final_text_evidence_predictions.csv

Output:
SVG and PDF vector figures in ./sentence_transformer_text_evidence_top10_figures/

Figures:
1. Top text evidence units by probability drop
2. Top text evidence units by absolute probability drop
3. Top text sections by absolute probability drop
4. Global text branch ablation
5. Train/full-data prediction score distribution
6. Prediction metrics summary

Note:
These are qualitative explanation figures from the final full-data model.
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

OVERVIEW_PATH = BASE_DIR / "text_evidence_overview.json"
TEXT_SUMMARY_PATH = BASE_DIR / "text_evidence_summary.csv"
BRANCH_ABLATION_PATH = BASE_DIR / "global_text_branch_ablation.csv"
PRED_PATH = BASE_DIR / "final_text_evidence_predictions.csv"

OUT_DIR = BASE_DIR / "sentence_transformer_text_evidence_top10_figures"
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
    svg_path = OUT_DIR / f"{name}.svg"
    pdf_path = OUT_DIR / f"{name}.pdf"

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def wrap_label(x, width=56):
    x = str(x)
    return textwrap.fill(x, width=width, break_long_words=False)


def short_text(x, max_len=120):
    x = str(x).replace("\n", " ").strip()
    if len(x) <= max_len:
        return x
    return x[:max_len - 3] + "..."


def add_hbar_labels(ax, values, fmt="{:.4g}", offset_ratio=0.03):
    max_abs = max(abs(float(v)) for v in values) if len(values) else 1.0
    offset = max_abs * offset_ratio

    for patch, v in zip(ax.patches, values):
        y = patch.get_y() + patch.get_height() / 2
        x = patch.get_width()

        if x >= 0:
            ax.text(
                x + offset,
                y,
                fmt.format(v),
                va="center",
                ha="left",
                fontsize=8,
                )
        else:
            ax.text(
                x - offset,
                y,
                fmt.format(v),
                va="center",
                ha="right",
                fontsize=8,
                )


def make_unit_label(row):
    repo = row.get("repo_full_name", "")
    case_type = row.get("case_type", "")
    branch = row.get("top_text_branch", "")
    unit = short_text(row.get("top_text_unit", ""), max_len=105)
    return f"{case_type} | {repo} | {branch}: {unit}"


def make_section_label(row):
    repo = row.get("repo_full_name", "")
    case_type = row.get("case_type", "")
    section = row.get("top_section_title", "")
    return f"{case_type} | {repo} | Section: {section}"


def rename_branch(x):
    mapping = {
        "remove_readme_branch": "Remove README branch",
        "remove_aux_branch": "Remove auxiliary text branch",
    }
    return mapping.get(str(x), str(x))


# =========================
# 2. Load files
# =========================
for p in [OVERVIEW_PATH, TEXT_SUMMARY_PATH, BRANCH_ABLATION_PATH, PRED_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

with open(OVERVIEW_PATH, "r", encoding="utf-8") as f:
    overview = json.load(f)

text_df = pd.read_csv(TEXT_SUMMARY_PATH)
branch_df = pd.read_csv(BRANCH_ABLATION_PATH)
pred_df = pd.read_csv(PRED_PATH)

# Validate text evidence summary
required_text_cols = {
    "repo_full_name",
    "case_type",
    "label",
    "y_proba",
    "y_pred",
    "top_text_branch",
    "top_text_unit",
    "top_text_prob_drop",
    "top_section_title",
    "top_section_prob_drop",
}
missing_text = required_text_cols - set(text_df.columns)
if missing_text:
    raise ValueError(f"text_evidence_summary.csv missing columns: {sorted(missing_text)}")

# Validate branch ablation
required_branch_cols = {
    "ablation",
    "mean_prob_drop",
    "positive_mean_prob_drop",
    "negative_mean_prob_drop",
}
missing_branch = required_branch_cols - set(branch_df.columns)
if missing_branch:
    raise ValueError(f"global_text_branch_ablation.csv missing columns: {sorted(missing_branch)}")

# Validate predictions
required_pred_cols = {"label", "y_proba", "y_pred"}
missing_pred = required_pred_cols - set(pred_df.columns)
if missing_pred:
    raise ValueError(f"final_text_evidence_predictions.csv missing columns: {sorted(missing_pred)}")

# Numeric coercion
text_df["label"] = pd.to_numeric(text_df["label"], errors="coerce").fillna(0).astype(int)
text_df["y_proba"] = pd.to_numeric(text_df["y_proba"], errors="coerce")
text_df["y_pred"] = pd.to_numeric(text_df["y_pred"], errors="coerce").fillna(0).astype(int)
text_df["top_text_prob_drop"] = pd.to_numeric(text_df["top_text_prob_drop"], errors="coerce")
text_df["top_section_prob_drop"] = pd.to_numeric(text_df["top_section_prob_drop"], errors="coerce")

for c in ["mean_prob_drop", "positive_mean_prob_drop", "negative_mean_prob_drop"]:
    branch_df[c] = pd.to_numeric(branch_df[c], errors="coerce")

pred_df["label"] = pd.to_numeric(pred_df["label"], errors="coerce").fillna(0).astype(int)
pred_df["y_proba"] = pd.to_numeric(pred_df["y_proba"], errors="coerce")
pred_df["y_pred"] = pd.to_numeric(pred_df["y_pred"], errors="coerce").fillna(0).astype(int)

if pred_df["y_proba"].isna().any():
    raise ValueError("final_text_evidence_predictions.csv has invalid y_proba values.")

threshold = float(overview.get("threshold", 0.5))


# =========================
# 3. Compute prediction metrics
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
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}
with open(OUT_DIR / "computed_text_evidence_prediction_metrics.json", "w", encoding="utf-8") as f:
    json.dump(computed_summary, f, ensure_ascii=False, indent=2)

print("Computed prediction metrics:")
print(json.dumps(computed_summary, ensure_ascii=False, indent=2))


# =========================
# 4. Prepare Top-N tables
# =========================
text_df["abs_top_text_prob_drop"] = text_df["top_text_prob_drop"].abs()
text_df["abs_top_section_prob_drop"] = text_df["top_section_prob_drop"].abs()

top_text_positive = (
    text_df.sort_values("top_text_prob_drop", ascending=False)
    .head(TOP_N)
    .copy()
)

top_text_abs = (
    text_df.sort_values("abs_top_text_prob_drop", ascending=False)
    .head(TOP_N)
    .copy()
)

top_section_abs = (
    text_df.sort_values("abs_top_section_prob_drop", ascending=False)
    .head(TOP_N)
    .copy()
)

top_text_positive.to_csv(
    OUT_DIR / "top10_text_units_by_positive_prob_drop.csv",
    index=False,
    encoding="utf-8-sig",
    )
top_text_abs.to_csv(
    OUT_DIR / "top10_text_units_by_abs_prob_drop.csv",
    index=False,
    encoding="utf-8-sig",
    )
top_section_abs.to_csv(
    OUT_DIR / "top10_sections_by_abs_prob_drop.csv",
    index=False,
    encoding="utf-8-sig",
    )


# =========================
# 5. Figure 1: Top text units by positive probability drop
# =========================
def plot_top_text_units_positive():
    df = top_text_positive.sort_values("top_text_prob_drop", ascending=True).copy()
    labels = [wrap_label(make_unit_label(row), width=60) for _, row in df.iterrows()]
    values = df["top_text_prob_drop"].values

    fig_height = max(4.0, 0.52 * len(df))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Probability drop after removing top text unit")
    ax.set_title("Top text evidence units supporting suspicious prediction")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.01,
        -0.13,
        "Positive value: removing this text unit decreases predicted risk.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig01_top10_text_units_positive_prob_drop")


# =========================
# 6. Figure 2: Top text units by absolute probability drop
# =========================
def plot_top_text_units_abs():
    df = top_text_abs.sort_values("abs_top_text_prob_drop", ascending=True).copy()
    labels = [wrap_label(make_unit_label(row), width=60) for _, row in df.iterrows()]
    values = df["top_text_prob_drop"].values

    fig_height = max(4.0, 0.52 * len(df))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Signed probability drop after removing top text unit")
    ax.set_title("Top text evidence units by absolute effect")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.01,
        -0.13,
        "Positive: text unit supports suspicious prediction; negative: text unit suppresses suspicious prediction.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig02_top10_text_units_abs_prob_drop")


# =========================
# 7. Figure 3: Top sections by absolute probability drop
# =========================
def plot_top_sections_abs():
    df = top_section_abs.sort_values("abs_top_section_prob_drop", ascending=True).copy()
    labels = [wrap_label(make_section_label(row), width=58) for _, row in df.iterrows()]
    values = df["top_section_prob_drop"].values

    fig_height = max(4.0, 0.52 * len(df))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Signed probability drop after removing top section")
    ax.set_title("Top text sections by absolute effect")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.01,
        -0.13,
        "Positive: section supports suspicious prediction; negative: section suppresses suspicious prediction.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig03_top10_sections_abs_prob_drop")


# =========================
# 8. Figure 4: global text branch ablation
# =========================
def plot_global_branch_ablation():
    df = branch_df.copy()
    df["branch"] = df["ablation"].apply(rename_branch)

    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    ax.bar(x - width, df["mean_prob_drop"], width=width, label="All samples")
    ax.bar(x, df["positive_mean_prob_drop"], width=width, label="Positive samples")
    ax.bar(x + width, df["negative_mean_prob_drop"], width=width, label="Negative samples")

    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["branch"], rotation=0)
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Global text branch ablation")
    ax.legend(frameon=False, ncol=3)

    ax.text(
        0.01,
        -0.20,
        "Positive value: removing the branch decreases predicted risk.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig04_global_text_branch_ablation")


# =========================
# 9. Figure 5: prediction score distribution
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
    ax.set_title("Prediction score distribution")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig05_prediction_score_distribution")


# =========================
# 10. Figure 6: metrics bar
# =========================
def plot_prediction_metrics():
    labels = ["Precision", "Recall", "F1-score", "PR-AUC", "ROC-AUC"]
    values = [precision, recall, f1, pr_auc, roc_auc]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar(labels, values, width=0.65)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Prediction metrics for text evidence dataset")

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
        "Note: qualitative evidence from final full-data model.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig06_prediction_metrics")


# =========================
# 11. Main
# =========================
def main():
    plot_top_text_units_positive()
    plot_top_text_units_abs()
    plot_top_sections_abs()
    plot_global_branch_ablation()
    plot_score_distribution()
    plot_prediction_metrics()

    print(f"\nAll text-evidence figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()