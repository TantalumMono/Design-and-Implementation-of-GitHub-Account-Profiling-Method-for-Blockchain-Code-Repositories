# -*- coding: utf-8 -*-
"""
Plot compact Top-10 text evidence figures for SentenceTransformer text-branch explanation.

Required input files:
1. text_evidence_overview.json
2. text_evidence_summary.csv
3. global_text_branch_ablation.csv
4. final_text_evidence_predictions.csv

Output:
SVG and PDF vector figures in ./sentence_transformer_text_evidence_top10_figures_compact/

Main improvements:
- Avoid long text labels on y-axis.
- Use evidence IDs such as E1, E2, ...
- Save full text evidence mapping to CSV.
- Better figure layout for thesis/PPT.
"""

import json
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
BASE_DIR = Path("sentence_transformer_text_evidence_top10_figures")

OVERVIEW_PATH = BASE_DIR / "text_evidence_overview.json"
TEXT_SUMMARY_PATH = BASE_DIR / "text_evidence_summary.csv"
BRANCH_ABLATION_PATH = BASE_DIR / "global_text_branch_ablation.csv"
PRED_PATH = BASE_DIR / "final_text_evidence_predictions.csv"

OUT_DIR = BASE_DIR / "sentence_transformer_text_evidence_top10_figures_compact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 10


# =========================
# 1. Matplotlib config
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def save_vector(fig, name: str):
    svg_path = OUT_DIR / f"{name}.svg"
    pdf_path = OUT_DIR / f"{name}.pdf"

    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


def short_repo_name(x: str, max_len=32):
    x = str(x)
    if "/" in x:
        x = x.split("/")[-1]
    if len(x) > max_len:
        return x[: max_len - 3] + "..."
    return x


def short_text(x: str, max_len=180):
    x = str(x).replace("\n", " ").replace("\r", " ").strip()
    x = " ".join(x.split())
    if len(x) > max_len:
        return x[: max_len - 3] + "..."
    return x


def add_hbar_labels(ax, values, fmt="{:.4g}", offset_ratio=0.04):
    if len(values) == 0:
        return

    max_abs = max(abs(float(v)) for v in values)
    if max_abs == 0:
        max_abs = 1.0

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


def rename_branch(x):
    mapping = {
        "remove_readme_branch": "Remove README",
        "remove_aux_branch": "Remove AUX text",
    }
    return mapping.get(str(x), str(x))


def make_compact_label(row):
    evidence_id = row["evidence_id"]
    case_type = str(row.get("case_type", "case"))
    branch = str(row.get("top_text_branch", "text"))
    repo = short_repo_name(row.get("repo_full_name", ""), max_len=24)

    case_type = case_type.replace("high_score_", "high_")
    case_type = case_type.replace("hard_real_positive", "hard_pos")
    case_type = case_type.replace("hard_negative", "hard_neg")
    case_type = case_type.replace("false_positive", "FP")
    case_type = case_type.replace("false_negative", "FN")

    return f"{evidence_id} | {case_type} | {branch} | {repo}"


def make_section_label(row):
    evidence_id = row["evidence_id"]
    case_type = str(row.get("case_type", "case"))
    repo = short_repo_name(row.get("repo_full_name", ""), max_len=24)
    section = str(row.get("top_section_title", "section"))

    case_type = case_type.replace("high_score_", "high_")
    case_type = case_type.replace("hard_real_positive", "hard_pos")
    case_type = case_type.replace("hard_negative", "hard_neg")

    if len(section) > 24:
        section = section[:21] + "..."

    return f"{evidence_id} | {case_type} | {repo} | {section}"


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

required_branch_cols = {
    "ablation",
    "mean_prob_drop",
    "positive_mean_prob_drop",
    "negative_mean_prob_drop",
}
missing_branch = required_branch_cols - set(branch_df.columns)
if missing_branch:
    raise ValueError(f"global_text_branch_ablation.csv missing columns: {sorted(missing_branch)}")

required_pred_cols = {"label", "y_proba", "y_pred"}
missing_pred = required_pred_cols - set(pred_df.columns)
if missing_pred:
    raise ValueError(f"final_text_evidence_predictions.csv missing columns: {sorted(missing_pred)}")


# Numeric conversion
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
# 4. Prepare Top-N tables with compact IDs
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


def assign_ids(df, prefix="E"):
    df = df.copy().reset_index(drop=True)
    df["evidence_id"] = [f"{prefix}{i+1}" for i in range(len(df))]
    return df


top_text_positive = assign_ids(top_text_positive, prefix="E")
top_text_abs = assign_ids(top_text_abs, prefix="E")
top_section_abs = assign_ids(top_section_abs, prefix="S")


def save_evidence_mapping(df, filename):
    out = df.copy()
    out["top_text_unit_short"] = out["top_text_unit"].apply(lambda x: short_text(x, max_len=300))
    out["repo_short"] = out["repo_full_name"].apply(short_repo_name)
    keep_cols = [
        "evidence_id",
        "case_type",
        "repo_full_name",
        "label",
        "y_proba",
        "y_pred",
        "top_text_branch",
        "top_text_prob_drop",
        "top_section_title",
        "top_section_prob_drop",
        "top_text_unit_short",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    out[keep_cols].to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")


save_evidence_mapping(top_text_positive, "mapping_top_text_positive.csv")
save_evidence_mapping(top_text_abs, "mapping_top_text_abs.csv")
save_evidence_mapping(top_section_abs, "mapping_top_sections_abs.csv")


# =========================
# 5. Figure 1: Top text units by positive prob drop
# =========================
def plot_top_text_units_positive():
    df = top_text_positive.sort_values("top_text_prob_drop", ascending=True).copy()
    values = df["top_text_prob_drop"].values
    labels = [make_compact_label(row) for _, row in df.iterrows()]

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Probability drop after removing text unit")
    ax.set_title("Top text evidence units supporting suspicious prediction")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.0,
        -0.18,
        "Positive value: removing this text decreases predicted risk. Full text is saved in mapping_top_text_positive.csv.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.40, bottom=0.22)
    save_vector(fig, "fig01_top10_text_units_positive_prob_drop_compact")


# =========================
# 6. Figure 2: Top text units by absolute prob drop
# =========================
def plot_top_text_units_abs():
    df = top_text_abs.sort_values("abs_top_text_prob_drop", ascending=True).copy()
    values = df["top_text_prob_drop"].values
    labels = [make_compact_label(row) for _, row in df.iterrows()]

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signed probability drop after removing text unit")
    ax.set_title("Top text evidence units by absolute effect")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.0,
        -0.18,
        "Positive: supports suspicious prediction; negative: suppresses suspicious prediction. Full text is saved in mapping_top_text_abs.csv.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.40, bottom=0.22)
    save_vector(fig, "fig02_top10_text_units_abs_prob_drop_compact")


# =========================
# 7. Figure 3: Top sections by absolute prob drop
# =========================
def plot_top_sections_abs():
    df = top_section_abs.sort_values("abs_top_section_prob_drop", ascending=True).copy()
    values = df["top_section_prob_drop"].values
    labels = [make_section_label(row) for _, row in df.iterrows()]

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    y = np.arange(len(df))
    ax.barh(y, values, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signed probability drop after removing section")
    ax.set_title("Top text sections by absolute effect")

    ax.axvline(0, linewidth=1.0)
    add_hbar_labels(ax, values, fmt="{:.4g}")

    ax.text(
        0.0,
        -0.18,
        "Positive: section supports suspicious prediction; negative: section suppresses suspicious prediction.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.42, bottom=0.22)
    save_vector(fig, "fig03_top10_sections_abs_prob_drop_compact")


# =========================
# 8. Figure 4: global text branch ablation
# =========================
def plot_global_branch_ablation():
    df = branch_df.copy()
    df["branch"] = df["ablation"].apply(rename_branch)

    x = np.arange(len(df))
    width = 0.23

    fig, ax = plt.subplots(figsize=(6.2, 3.3))

    ax.bar(x - width, df["mean_prob_drop"], width=width, label="All")
    ax.bar(x, df["positive_mean_prob_drop"], width=width, label="Positive")
    ax.bar(x + width, df["negative_mean_prob_drop"], width=width, label="Negative")

    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["branch"], fontsize=9)
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Global text branch ablation")
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax.text(
        0.0,
        -0.22,
        "Positive value: removing the branch decreases predicted risk.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.24)
    save_vector(fig, "fig04_global_text_branch_ablation_compact")


# =========================
# 9. Figure 5: prediction score distribution
# =========================
def plot_score_distribution():
    pos = pred_df.loc[pred_df["label"] == 1, "y_proba"].dropna()
    neg = pred_df.loc[pred_df["label"] == 0, "y_proba"].dropna()

    max_score = max(float(pred_df["y_proba"].max()), threshold)
    bins = np.linspace(0, max_score * 1.02, 50)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    ax.hist(neg, bins=bins, density=True, alpha=0.65, label="Negative")
    ax.hist(pos, bins=bins, density=True, alpha=0.65, label="Positive")

    ax.axvline(threshold, linestyle="--", linewidth=1.0)
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.82,
        f"Threshold = {threshold:.4f}",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
        )

    ax.set_xlabel("Predicted probability / risk score")
    ax.set_ylabel("Density")
    ax.set_title("Prediction score distribution")
    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig05_prediction_score_distribution_compact")


# =========================
# 10. Figure 6: metrics bar
# =========================
def plot_prediction_metrics():
    labels = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    values = [precision, recall, f1, pr_auc, roc_auc]

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.bar(labels, values, width=0.62)

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
            fontsize=8,
            )

    ax.text(
        0.5,
        -0.20,
        "Qualitative evidence from final full-data model.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.22)
    save_vector(fig, "fig06_prediction_metrics_compact")


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

    print(f"\nAll compact text-evidence figures saved to: {OUT_DIR.resolve()}")
    print("Full text mappings are saved as CSV files in the same output directory.")


if __name__ == "__main__":
    main()