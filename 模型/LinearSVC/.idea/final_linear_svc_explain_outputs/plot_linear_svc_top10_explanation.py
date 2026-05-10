# -*- coding: utf-8 -*-
"""
Plot Top-10 explanation figures for LinearSVC.

Required input files:
1. linear_svc_explanation_summary.json
2. linear_svc_global_coefficients.csv

Expected columns in CSV:
feature, coef, abs_coef

Output:
SVG and PDF vector figures in ./linear_svc_top10_explanation_figures/

Figures:
1. Top-10 positive coefficients
2. Top-10 negative coefficients
3. Top-10 coefficients by absolute value
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. Path config
# =========================
BASE_DIR = Path(".")

SUMMARY_PATH = BASE_DIR / "linear_svc_explanation_summary.json"
COEF_PATH = BASE_DIR / "linear_svc_global_coefficients.csv"

OUT_DIR = BASE_DIR / "linear_svc_top10_explanation_figures"
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


def add_hbar_labels(ax, values, offset_ratio=0.02, fmt="{:.4f}"):
    """Add labels to horizontal bars."""
    max_abs = max(abs(float(v)) for v in values) if len(values) > 0 else 1.0
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


# =========================
# 2. Load data
# =========================
if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"Missing file: {SUMMARY_PATH}")

if not COEF_PATH.exists():
    raise FileNotFoundError(f"Missing file: {COEF_PATH}")

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

coef_df = pd.read_csv(COEF_PATH)

required_cols = {"feature", "coef"}
missing = required_cols - set(coef_df.columns)
if missing:
    raise ValueError(f"Coefficient CSV missing columns: {sorted(missing)}")

coef_df["coef"] = pd.to_numeric(coef_df["coef"], errors="coerce")
coef_df = coef_df.dropna(subset=["coef"]).copy()

if "abs_coef" not in coef_df.columns:
    coef_df["abs_coef"] = coef_df["coef"].abs()
else:
    coef_df["abs_coef"] = pd.to_numeric(coef_df["abs_coef"], errors="coerce")
    coef_df["abs_coef"] = coef_df["abs_coef"].fillna(coef_df["coef"].abs())

threshold = summary.get("threshold", None)
n_total = summary.get("n_total", None)
n_positive = summary.get("n_positive", None)
n_negative = summary.get("n_negative", None)
predicted_positive_rate = summary.get("predicted_positive_rate", None)

print("LinearSVC explanation summary:")
print(f"  threshold = {threshold}")
print(f"  n_total = {n_total}")
print(f"  n_positive = {n_positive}")
print(f"  n_negative = {n_negative}")
print(f"  predicted_positive_rate = {predicted_positive_rate}")


# =========================
# 3. Prepare Top-N tables
# =========================
top_pos = (
    coef_df[coef_df["coef"] > 0]
    .sort_values("coef", ascending=False)
    .head(TOP_N)
    .copy()
)

top_neg = (
    coef_df[coef_df["coef"] < 0]
    .sort_values("coef", ascending=True)
    .head(TOP_N)
    .copy()
)

top_abs = (
    coef_df.sort_values("abs_coef", ascending=False)
    .head(TOP_N)
    .copy()
)

# Save tables
top_pos.to_csv(OUT_DIR / "linear_svc_top10_positive_coefficients.csv", index=False, encoding="utf-8-sig")
top_neg.to_csv(OUT_DIR / "linear_svc_top10_negative_coefficients.csv", index=False, encoding="utf-8-sig")
top_abs.to_csv(OUT_DIR / "linear_svc_top10_abs_coefficients.csv", index=False, encoding="utf-8-sig")


# =========================
# 4. Figure 1: Top-10 positive coefficients
# =========================
def plot_top_positive():
    df = top_pos.sort_values("coef", ascending=True).copy()

    fig_height = max(3.6, 0.38 * len(df))
    fig, ax = plt.subplots(figsize=(6.6, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["coef"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["feature"].values, width=36))
    ax.set_xlabel("LinearSVC coefficient")
    ax.set_title("Top 10 positive coefficients\n(push toward suspicious class)")

    add_hbar_labels(ax, df["coef"].values, fmt="{:.4f}")

    ax.axvline(0, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig01_linear_svc_top10_positive_coefficients")


# =========================
# 5. Figure 2: Top-10 negative coefficients
# =========================
def plot_top_negative():
    # For display: most negative at top
    df = top_neg.sort_values("coef", ascending=False).copy()

    fig_height = max(3.6, 0.38 * len(df))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["coef"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["feature"].values, width=36))
    ax.set_xlabel("LinearSVC coefficient")
    ax.set_title("Top 10 negative coefficients\n(push toward normal class)")

    add_hbar_labels(ax, df["coef"].values, fmt="{:.4f}")

    ax.axvline(0, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig02_linear_svc_top10_negative_coefficients")


# =========================
# 6. Figure 3: Top-10 absolute coefficients, signed display
# =========================
def plot_top_abs_signed():
    df = top_abs.sort_values("abs_coef", ascending=True).copy()

    fig_height = max(3.8, 0.42 * len(df))
    fig, ax = plt.subplots(figsize=(7.0, fig_height))

    y = np.arange(len(df))
    ax.barh(y, df["coef"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(df["feature"].values, width=38))
    ax.set_xlabel("LinearSVC coefficient")
    ax.set_title("Top 10 global coefficients by absolute value")

    add_hbar_labels(ax, df["coef"].values, fmt="{:.4f}")

    ax.axvline(0, linewidth=1.0)
    ax.text(
        0.02,
        0.02,
        "Positive: suspicious class   |   Negative: normal class",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_vector(fig, "fig03_linear_svc_top10_abs_signed_coefficients")


# =========================
# 7. Main
# =========================
def main():
    plot_top_positive()
    plot_top_negative()
    plot_top_abs_signed()

    print(f"\nAll LinearSVC Top-10 explanation figures saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()