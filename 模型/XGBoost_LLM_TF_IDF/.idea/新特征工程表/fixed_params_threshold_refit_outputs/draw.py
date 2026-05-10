# -*- coding: utf-8 -*-
"""
Plot vector figures for fixed-parameter family-aware OOF validation results.

Required input files:
1. fixed_params_family_oof_predictions_with_refit_threshold.csv
2. threshold_search_results.csv
3. refit_threshold_report.json

Outputs:
SVG and PDF vector figures in ./oof_validation_vector_figures/

Figures:
1. Core metric bar chart
2. Confusion matrix
3. Threshold search curves
4. ROC curve
5. PR curve
6. OOF score distribution
7. Per-fold metrics (if outer_fold exists)
8. Positive recall by family (if family_id exists)
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
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
)


# =========================
# 0. Path config
# =========================
BASE_DIR = Path(".")

OOF_PATH = BASE_DIR / "fixed_params_family_oof_predictions_with_refit_threshold.csv"
THRESHOLD_CSV_PATH = BASE_DIR / "threshold_search_results.csv"
REPORT_PATH = BASE_DIR / "refit_threshold_report.json"

OUT_DIR = BASE_DIR / "oof_validation_vector_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Matplotlib global config
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["svg.fonttype"] = "none"   # keep text editable in SVG
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


def add_bar_labels(ax, fmt="{:.3f}", offset=0.01):
    """Add value labels above bars."""
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            )


def calc_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


# =========================
# 2. Load files
# =========================
if not OOF_PATH.exists():
    raise FileNotFoundError(f"Missing file: {OOF_PATH}")
if not THRESHOLD_CSV_PATH.exists():
    raise FileNotFoundError(f"Missing file: {THRESHOLD_CSV_PATH}")
if not REPORT_PATH.exists():
    raise FileNotFoundError(f"Missing file: {REPORT_PATH}")

oof_df = pd.read_csv(OOF_PATH)
thr_df = pd.read_csv(THRESHOLD_CSV_PATH)

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

# Required columns in OOF
required_oof_cols = {"label", "y_proba"}
missing_oof = required_oof_cols - set(oof_df.columns)
if missing_oof:
    raise ValueError(f"OOF CSV missing columns: {sorted(missing_oof)}")

# Required columns in threshold search
required_thr_cols = {"threshold", "precision", "recall", "f1"}
missing_thr = required_thr_cols - set(thr_df.columns)
if missing_thr:
    raise ValueError(f"Threshold CSV missing columns: {sorted(missing_thr)}")

# Numeric coercion
oof_df["label"] = pd.to_numeric(oof_df["label"], errors="coerce").fillna(0).astype(int)
oof_df["y_proba"] = pd.to_numeric(oof_df["y_proba"], errors="coerce")

if oof_df["y_proba"].isna().any():
    raise ValueError("OOF CSV has invalid y_proba values.")

thr_df["threshold"] = pd.to_numeric(thr_df["threshold"], errors="coerce")
thr_df["precision"] = pd.to_numeric(thr_df["precision"], errors="coerce")
thr_df["recall"] = pd.to_numeric(thr_df["recall"], errors="coerce")
thr_df["f1"] = pd.to_numeric(thr_df["f1"], errors="coerce")
if "f_beta" in thr_df.columns:
    thr_df["f_beta"] = pd.to_numeric(thr_df["f_beta"], errors="coerce")

# Read best result from json
best = report["best_threshold_result"]
best_threshold = float(best["threshold"])
best_precision = float(best["precision"])
best_recall = float(best["recall"])
best_f1 = float(best["f1"])
best_fbeta = float(best["f_beta"])
tn = int(best["tn"])
fp = int(best["fp"])
fn = int(best["fn"])
tp = int(best["tp"])

# Decide prediction column
if "y_pred_refit_threshold" in oof_df.columns:
    oof_df["y_pred_final"] = pd.to_numeric(
        oof_df["y_pred_refit_threshold"], errors="coerce"
    ).fillna(0).astype(int)
else:
    oof_df["y_pred_final"] = (oof_df["y_proba"] >= best_threshold).astype(int)

# Recompute summary metrics from OOF CSV
y_true = oof_df["label"].values
y_score = oof_df["y_proba"].values
y_pred = oof_df["y_pred_final"].values

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
fbeta = calc_fbeta(precision, recall, beta=2.0)
roc_auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn2, fp2, fn2, tp2 = cm.ravel()

# Save a local computed summary
computed_summary = {
    "threshold": best_threshold,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "f_beta": float(fbeta),
    "roc_auc": float(roc_auc),
    "pr_auc_ap": float(ap),
    "tn": int(tn2),
    "fp": int(fp2),
    "fn": int(fn2),
    "tp": int(tp2),
}
with open(OUT_DIR / "computed_oof_metrics.json", "w", encoding="utf-8") as f:
    json.dump(computed_summary, f, ensure_ascii=False, indent=2)

print("Computed OOF metrics:")
print(json.dumps(computed_summary, ensure_ascii=False, indent=2))


# =========================
# 3. Figure 1: core metrics bar chart
# =========================
def plot_core_metrics():
    labels = ["Precision", "Recall", "F1-score", r"$F_{\beta=2}$"]
    values = [precision, recall, f1, fbeta]

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.bar(labels, values, width=0.65)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("OOF validation performance at the refitted threshold")
    ax.axhline(0.80, linestyle="--", linewidth=1.0)
    ax.text(
        0.02,
        0.805,
        "Target = 0.80",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        fontsize=9,
    )

    add_bar_labels(ax, fmt="{:.3f}", offset=0.015)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig01_oof_core_metrics_bar")


# =========================
# 4. Figure 2: confusion matrix
# =========================
def plot_confusion_matrix():
    mat = np.array([[tn2, fp2], [fn2, tp2]])

    fig, ax = plt.subplots(figsize=(4.3, 3.8))
    im = ax.imshow(mat)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nnormal", "Predicted\nsuspicious"])
    ax.set_yticklabels(["Actual\nnormal", "Actual\nsuspicious"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{mat[i, j]}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_title("OOF confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_vector(fig, "fig02_oof_confusion_matrix")


# =========================
# 5. Figure 3: threshold search curves
# =========================
def plot_threshold_curves():
    df = thr_df.sort_values("threshold").copy()

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(df["threshold"], df["precision"], linewidth=1.5, label="Precision")
    ax.plot(df["threshold"], df["recall"], linewidth=1.5, label="Recall")
    ax.plot(df["threshold"], df["f1"], linewidth=1.5, label="F1-score")

    if "f_beta" in df.columns:
        ax.plot(df["threshold"], df["f_beta"], linewidth=1.5, label=r"$F_{\beta=2}$")

    ax.axvline(best_threshold, linestyle="--", linewidth=1.0)
    ax.text(
        best_threshold,
        0.04,
        f"Best threshold = {best_threshold:.4f}",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=9,
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold search curves based on OOF predictions")
    ax.legend(frameon=False, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig03_threshold_search_curves")


# =========================
# 6. Figure 4: ROC curve
# =========================
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(fpr, tpr, linewidth=1.6, label=f"ROC-AUC = {roc_auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve based on OOF predictions")
    ax.legend(frameon=False, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig04_oof_roc_curve")


# =========================
# 7. Figure 5: PR curve
# =========================
def plot_pr_curve():
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap_value = average_precision_score(y_true, y_score)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.plot(r, p, linewidth=1.6, label=f"AP = {ap_value:.3f}")
    ax.axhline(baseline, linestyle="--", linewidth=1.0, label=f"Baseline = {baseline:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall curve based on OOF predictions")
    ax.legend(frameon=False, loc="lower left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig05_oof_pr_curve")


# =========================
# 8. Figure 6: OOF score distribution
# =========================
def plot_score_distribution():
    pos = oof_df.loc[oof_df["label"] == 1, "y_proba"].dropna()
    neg = oof_df.loc[oof_df["label"] == 0, "y_proba"].dropna()

    max_score = max(float(oof_df["y_proba"].max()), best_threshold)
    bins = np.linspace(0, max_score * 1.02, 50)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.hist(neg, bins=bins, density=True, alpha=0.65, label="Negative")
    ax.hist(pos, bins=bins, density=True, alpha=0.65, label="Positive")

    ax.axvline(best_threshold, linestyle="--", linewidth=1.0)
    ax.text(
        best_threshold,
        ax.get_ylim()[1] * 0.85,
        f"Threshold = {best_threshold:.4f}",
        rotation=90,
        ha="right",
        va="center",
        fontsize=9,
        )

    ax.set_xlabel("Predicted probability / risk score")
    ax.set_ylabel("Density")
    ax.set_title("OOF score distribution")
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig06_oof_score_distribution")


# =========================
# 9. Figure 7: per-fold metrics
# =========================
def plot_fold_metrics():
    if "outer_fold" not in oof_df.columns:
        print("Skip per-fold metrics: 'outer_fold' column not found.")
        return

    fold_rows = []
    for fold_id, g in oof_df.groupby("outer_fold"):
        y_t = g["label"].values
        y_p = g["y_pred_final"].values
        if len(np.unique(y_t)) < 2:
            continue

        fold_rows.append({
            "outer_fold": fold_id,
            "precision": precision_score(y_t, y_p, zero_division=0),
            "recall": recall_score(y_t, y_p, zero_division=0),
            "f1": f1_score(y_t, y_p, zero_division=0),
        })

    if not fold_rows:
        print("Skip per-fold metrics: no valid folds.")
        return

    fold_df = pd.DataFrame(fold_rows).sort_values("outer_fold").reset_index(drop=True)
    fold_df.to_csv(OUT_DIR / "computed_fold_metrics.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(fold_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - width, fold_df["precision"], width=width, label="Precision")
    ax.bar(x, fold_df["recall"], width=width, label="Recall")
    ax.bar(x + width, fold_df["f1"], width=width, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(v)}" for v in fold_df["outer_fold"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance across OOF outer folds")
    ax.legend(frameon=False, ncol=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig07_oof_fold_metrics")


# =========================
# 10. Figure 8: positive recall by family
# =========================
def plot_family_positive_recall():
    if "family_id" not in oof_df.columns:
        print("Skip family recall: 'family_id' column not found.")
        return

    pos_df = oof_df[oof_df["label"] == 1].copy()
    if pos_df.empty:
        print("Skip family recall: no positive samples.")
        return

    fam = (
        pos_df.groupby("family_id")
        .agg(
            n_positive=("label", "size"),
            tp=("y_pred_final", "sum"),
        )
        .reset_index()
    )
    fam["recall"] = fam["tp"] / fam["n_positive"]
    fam = fam.sort_values(["recall", "n_positive"], ascending=[True, False]).reset_index(drop=True)

    fam.to_csv(OUT_DIR / "computed_family_positive_recall.csv", index=False, encoding="utf-8-sig")

    fig_height = max(3.5, 0.38 * len(fam))
    fig, ax = plt.subplots(figsize=(6.6, fig_height))

    y = np.arange(len(fam))
    ax.barh(y, fam["recall"])

    ax.set_yticks(y)
    ax.set_yticklabels(fam["family_id"].astype(str))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_title("Positive recall by family (OOF validation)")

    for idx, row in fam.iterrows():
        ax.text(
            row["recall"] + 0.02,
            idx,
            f"{int(row['tp'])}/{int(row['n_positive'])}",
            va="center",
            fontsize=8,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig08_oof_family_positive_recall")


# =========================
# 11. Main
# =========================
def main():
    plot_core_metrics()
    plot_confusion_matrix()
    plot_threshold_curves()
    plot_roc_curve()
    plot_pr_curve()
    plot_score_distribution()
    plot_fold_metrics()
    plot_family_positive_recall()

    print(f"\nAll vector figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()