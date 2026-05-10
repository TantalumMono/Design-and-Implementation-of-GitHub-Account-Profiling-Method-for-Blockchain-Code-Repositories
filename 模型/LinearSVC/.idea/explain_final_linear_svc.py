
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
MODEL_BUNDLE_PATH = SCRIPT_DIR / "final_linear_svc_calibrated_artifacts" / "final_linear_svc_calibrated_bundle.joblib"
OUTPUT_DIR = SCRIPT_DIR / "final_linear_svc_explain_outputs"

README_TEXT_COL = "readme_text_clean"
DESCRIPTION_TEXT_COL = "description_text_clean"
TOPICS_TEXT_COL = "topics_text_clean"

TOP_GLOBAL = 30
TOP_LOCAL = 20
N_TOP_SCORE = 3
N_HARD_POS = 3
N_HARD_NEG = 3

FINAL_LINEAR_SVC_PARAMS = {
    "C": 0.0015,
    "penalty": "l2",
    "loss": "squared_hinge",
    "dual": "auto",
    "fit_intercept": True,
    "max_iter": 10000,
    "random_state": 42,
}
FINAL_WEIGHTS = {
    "real_positive": 4.5,
    "generated_positive": 1.0,
    "negative": 1.25,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_aux_text_series(df: pd.DataFrame) -> pd.Series:
    desc = df.get(DESCRIPTION_TEXT_COL, "").fillna("").astype(str)
    topics = df.get(TOPICS_TEXT_COL, "").fillna("").astype(str)
    return (desc + " " + topics).str.strip()


def build_sample_weights(df: pd.DataFrame):
    w = np.full(len(df), FINAL_WEIGHTS["negative"], dtype=float)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = FINAL_WEIGHTS["real_positive"]
    w[gen_mask] = FINAL_WEIGHTS["generated_positive"]
    return w


def plot_bar(df_bar: pd.DataFrame, title: str, xlabel: str, save_path: Path):
    plt.figure(figsize=(10, 8))
    y = np.arange(len(df_bar))
    plt.barh(y, df_bar["value"].values)
    plt.yticks(y, df_bar["feature"].values, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到训练数据: {DATA_PATH}")
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"未找到最终 LinearSVC bundle: {MODEL_BUNDLE_PATH}")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    calibrated_model = bundle["model"]
    threshold = float(bundle["threshold"])
    readme_vectorizer = bundle["readme_vectorizer"]
    aux_vectorizer = bundle["aux_vectorizer"]
    fill_values = pd.Series(bundle["fill_values"])
    numeric_scaler = bundle["numeric_scaler"]
    structured_cols = bundle["structured_cols"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)

    # rebuild exact feature matrix
    X_num = df[structured_cols].copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(fill_values).fillna(0)
    X_num_scaled = numeric_scaler.transform(X_num.values.astype(np.float32))
    X_num_sp = csr_matrix(X_num_scaled)

    X_readme = readme_vectorizer.transform(df[README_TEXT_COL])
    X_aux = aux_vectorizer.transform(get_aux_text_series(df))
    X_all = hstack([X_num_sp, X_readme, X_aux], format="csr")

    y_true = df["label"].values
    y_proba = calibrated_model.predict_proba(X_all)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "label": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred,
    })
    if "repo_full_name" in df.columns:
        pred_df["repo_full_name"] = df["repo_full_name"].values
    if "family_id" in df.columns:
        pred_df["family_id"] = df["family_id"].astype(str).values
    if "is_real_positive" in df.columns:
        pred_df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int).values
    if "is_generated_positive" in df.columns:
        pred_df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int).values
    pred_df.to_csv(OUTPUT_DIR / "linear_svc_predictions_for_explanation.csv", index=False, encoding="utf-8-sig")

    # train matched plain LinearSVC for coefficient-based evidence
    explainer_linear = LinearSVC(**FINAL_LINEAR_SVC_PARAMS)
    sample_weight = build_sample_weights(df)
    try:
        explainer_linear.fit(X_all, y_true, sample_weight=sample_weight)
    except TypeError:
        explainer_linear.fit(X_all, y_true)

    coef = explainer_linear.coef_.ravel()
    intercept = float(explainer_linear.intercept_[0])

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)
    coef_df.to_csv(OUTPUT_DIR / "linear_svc_global_coefficients.csv", index=False, encoding="utf-8-sig")

    top_positive = coef_df.sort_values("coef", ascending=False).head(TOP_GLOBAL).copy()
    plot_bar(
        top_positive.rename(columns={"coef": "value"})[["feature", "value"]],
        "Top Positive Coefficients (push toward suspicious)",
        "coefficient",
        OUTPUT_DIR / "linear_svc_top_positive_coefficients.png",
    )

    top_negative = coef_df.sort_values("coef", ascending=True).head(TOP_GLOBAL).copy()
    top_negative_plot = top_negative.assign(value=-top_negative["coef"].values)[["feature", "value"]]
    plot_bar(
        top_negative_plot,
        "Top Negative Coefficients (push toward normal)",
        "absolute negative coefficient",
        OUTPUT_DIR / "linear_svc_top_negative_coefficients.png",
    )

    top_abs = coef_df.head(TOP_GLOBAL).copy()
    plot_bar(
        top_abs.rename(columns={"abs_coef": "value"})[["feature", "value"]],
        "Top Absolute Coefficients",
        "|coefficient|",
        OUTPUT_DIR / "linear_svc_top_absolute_coefficients.png",
    )

    # representative cases
    real_pos_mask = np.ones(len(df), dtype=bool)
    if "is_real_positive" in df.columns:
        real_pos_mask = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int).values == 1
    neg_mask = y_true == 0

    top_score_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=False).head(N_TOP_SCORE)
    hard_real_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=True).head(N_HARD_POS)
    hard_neg = pred_df.loc[neg_mask].sort_values("y_proba", ascending=False).head(N_HARD_NEG)

    selected_cases = pd.concat([
        top_score_pos.assign(case_type="high_score_real_positive"),
        hard_real_pos.assign(case_type="hard_real_positive"),
        hard_neg.assign(case_type="hard_negative"),
    ], axis=0)
    selected_cases.to_csv(OUTPUT_DIR / "selected_local_cases.csv", index=False, encoding="utf-8-sig")

    X_dense = X_all.toarray()
    local_summaries = []

    for row_idx in selected_cases.index.tolist():
        x = X_dense[row_idx]
        contrib = x * coef
        decision_score = float(np.dot(x, coef) + intercept)

        local_df = pd.DataFrame({
            "feature": feature_names,
            "feature_value": x,
            "coefficient": coef,
            "contribution": contrib,
            "abs_contribution": np.abs(contrib),
        }).sort_values("abs_contribution", ascending=False).head(TOP_LOCAL)

        repo_name = str(selected_cases.loc[row_idx].get("repo_full_name", f"sample_{row_idx}"))
        safe_name = repo_name.replace("/", "__").replace("\\", "__")[:80]

        local_df.to_csv(OUTPUT_DIR / f"local_explanation_{safe_name}.csv", index=False, encoding="utf-8-sig")

        plot_df = local_df.sort_values("contribution", ascending=False)
        plt.figure(figsize=(10, 7))
        y = np.arange(len(plot_df))
        plt.barh(y, plot_df["contribution"].values)
        plt.yticks(y, plot_df["feature"].values, fontsize=9)
        plt.gca().invert_yaxis()
        plt.title(f"Local Linear Contributions | {safe_name}")
        plt.xlabel("feature_value × coefficient")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"local_explanation_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        local_summaries.append({
            "row_index": int(row_idx),
            "repo_full_name": repo_name,
            "case_type": str(selected_cases.loc[row_idx]["case_type"]),
            "label": int(selected_cases.loc[row_idx]["label"]),
            "y_proba": float(selected_cases.loc[row_idx]["y_proba"]),
            "y_pred": int(selected_cases.loc[row_idx]["y_pred"]),
            "decision_score_linear": decision_score,
            "top_feature_1": str(local_df.iloc[0]["feature"]) if len(local_df) > 0 else None,
            "top_contribution_1": float(local_df.iloc[0]["contribution"]) if len(local_df) > 0 else None,
        })

    pd.DataFrame(local_summaries).to_csv(OUTPUT_DIR / "local_explanation_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "threshold": threshold,
        "n_total": int(len(df)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "global_top10_positive_coefficients": top_positive.head(10)[["feature", "coef"]].to_dict(orient="records"),
        "global_top10_negative_coefficients": top_negative.head(10)[["feature", "coef"]].to_dict(orient="records"),
        "note": "Explanation evidence is based on a matched plain LinearSVC decision function (feature_value × coefficient), while probabilities come from the calibrated model.",
    }
    with open(OUTPUT_DIR / "linear_svc_explanation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved LinearSVC explanation outputs to:", OUTPUT_DIR)
    print("- linear_svc_predictions_for_explanation.csv")
    print("- linear_svc_global_coefficients.csv")
    print("- linear_svc_top_positive_coefficients.png")
    print("- linear_svc_top_negative_coefficients.png")
    print("- linear_svc_top_absolute_coefficients.png")
    print("- selected_local_cases.csv")
    print("- local_explanation_*.csv")
    print("- local_explanation_*.png")
    print("- local_explanation_summary.csv")
    print("- linear_svc_explanation_summary.json")


if __name__ == "__main__":
    main()
