
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from xgboost import DMatrix

warnings.filterwarnings("ignore")

# =========================================================
# 0. CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
MODEL_BUNDLE_PATH = SCRIPT_DIR / "final_model_artifacts" / "final_repo_binary_classifier_bundle.joblib"
OUTPUT_DIR = SCRIPT_DIR / "final_model_shap_outputs"

MAX_GLOBAL_ROWS = 800
TOP_GLOBAL = 30
TOP_LOCAL = 15
RANDOM_STATE = 42

# 选择想解释的样本
# 1) 真正的高分正样本
# 2) 较难的正样本（低分正样本）
# 3) 高分负样本（容易误报的正常仓库）
N_TOP_REAL_POS = 3
N_HARD_REAL_POS = 3
N_HARD_NEG = 3


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def prepare_numeric(df: pd.DataFrame, structured_cols: list[str], fill_values: dict) -> csr_matrix:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    fill_series = pd.Series(fill_values)
    X = X.fillna(fill_series).fillna(0)
    return csr_matrix(X.to_numpy(dtype=np.float32))


def get_aux_text_series(df: pd.DataFrame, desc_col: str, topics_col: str) -> pd.Series:
    desc = df.get(desc_col, "").fillna("").astype(str)
    topics = df.get(topics_col, "").fillna("").astype(str)
    return (desc + " " + topics).str.strip()


def plot_bar(df_top: pd.DataFrame, title: str, save_path: Path):
    plt.figure(figsize=(10, 8))
    y = np.arange(len(df_top))
    plt.barh(y, df_top["importance"].values)
    plt.yticks(y, df_top["feature"].values, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_local_bar(local_df: pd.DataFrame, title: str, save_path: Path):
    plt.figure(figsize=(9, 6))
    y = np.arange(len(local_df))
    plt.barh(y, local_df["shap_value"].values)
    plt.yticks(y, local_df["feature"].values, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("SHAP contribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def try_make_shap_beeswarm(model, X_sample_dense, feature_names, save_path: Path):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_dense)
        plt.figure()
        shap.summary_plot(shap_values, X_sample_dense, feature_names=feature_names, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"[WARN] shap beeswarm skipped: {e}")
        return False


def main():
    ensure_dir(OUTPUT_DIR)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到训练数据: {DATA_PATH}")
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"未找到最终模型 bundle: {MODEL_BUNDLE_PATH}")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    readme_vectorizer = bundle["readme_vectorizer"]
    aux_vectorizer = bundle["aux_vectorizer"]
    fill_values = bundle["fill_values"]
    structured_cols = bundle["structured_cols"]
    feature_names = bundle["feature_names"]
    threshold = float(bundle["threshold"])
    readme_col = bundle["README_TEXT_COL"]
    desc_col = bundle["DESCRIPTION_TEXT_COL"]
    topics_col = bundle["TOPICS_TEXT_COL"]

    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df[readme_col] = df[readme_col].fillna("").astype(str)
    df[desc_col] = df[desc_col].fillna("").astype(str)
    df[topics_col] = df[topics_col].fillna("").astype(str)

    X_num = prepare_numeric(df, structured_cols, fill_values)
    X_readme = readme_vectorizer.transform(df[readme_col])
    X_aux = aux_vectorizer.transform(get_aux_text_series(df, desc_col, topics_col))
    X_all = hstack([X_num, X_readme, X_aux], format="csr")

    y_true = df["label"].values
    y_proba = model.predict_proba(X_all)[:, 1]
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

    pred_df.to_csv(OUTPUT_DIR / "final_model_predictions_for_shap.csv", index=False, encoding="utf-8-sig")

    # -------- global shap --------
    rng = np.random.default_rng(RANDOM_STATE)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    keep_pos = pos_idx
    remaining = max(0, MAX_GLOBAL_ROWS - len(keep_pos))
    keep_neg = rng.choice(neg_idx, size=remaining, replace=False) if len(neg_idx) > remaining else neg_idx

    sample_idx = np.concatenate([keep_pos, keep_neg])
    sample_idx = np.sort(sample_idx)

    X_sample = X_all[sample_idx]
    dm = DMatrix(X_sample, feature_names=feature_names)
    contribs = model.get_booster().predict(dm, pred_contribs=True)
    shap_values = contribs[:, :-1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_global_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False)

    shap_global_df.to_csv(OUTPUT_DIR / "shap_global_importance.csv", index=False, encoding="utf-8-sig")
    plot_bar(shap_global_df.head(TOP_GLOBAL), "Global SHAP Importance (Top 30)", OUTPUT_DIR / "shap_top30_bar.png")

    X_sample_dense = X_sample.toarray()
    try_make_shap_beeswarm(model, X_sample_dense, feature_names, OUTPUT_DIR / "shap_beeswarm_top25.png")

    # -------- local cases --------
    real_pos_mask = np.ones(len(df), dtype=bool)
    if "is_real_positive" in df.columns:
        real_pos_mask = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int).values == 1

    neg_mask = y_true == 0

    top_real_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=False).head(N_TOP_REAL_POS)
    hard_real_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=True).head(N_HARD_REAL_POS)
    hard_neg = pred_df.loc[neg_mask].sort_values("y_proba", ascending=False).head(N_HARD_NEG)

    selected_local = pd.concat([
        top_real_pos.assign(case_type="high_score_real_positive"),
        hard_real_pos.assign(case_type="hard_real_positive"),
        hard_neg.assign(case_type="hard_negative"),
    ], axis=0)

    selected_local.to_csv(OUTPUT_DIR / "selected_local_cases.csv", index=False, encoding="utf-8-sig")

    local_indices = selected_local.index.to_list()
    X_local = X_all[local_indices]
    dm_local = DMatrix(X_local, feature_names=feature_names)
    local_contribs = model.get_booster().predict(dm_local, pred_contribs=True)
    local_shap = local_contribs[:, :-1]

    for i, row_idx in enumerate(local_indices):
        row_meta = selected_local.loc[row_idx]
        one_shap = local_shap[i]
        local_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": one_shap,
            "abs_shap": np.abs(one_shap),
        }).sort_values("abs_shap", ascending=False).head(TOP_LOCAL)

        repo_name = row_meta.get("repo_full_name", f"sample_{row_idx}")
        safe_name = str(repo_name).replace("/", "__").replace("\\", "__")[:80]

        local_df.to_csv(OUTPUT_DIR / f"local_shap_{safe_name}.csv", index=False, encoding="utf-8-sig")
        plot_local_bar(
            local_df.sort_values("shap_value", ascending=False),
            f"{row_meta['case_type']} | {safe_name}",
            OUTPUT_DIR / f"local_shap_{safe_name}.png",
        )

    summary = {
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
        "threshold": threshold,
        "global_top10_features": shap_global_df.head(10).to_dict(orient="records"),
        "n_total": int(len(df)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }
    with open(OUTPUT_DIR / "shap_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved SHAP outputs to:", OUTPUT_DIR)
    print("- final_model_predictions_for_shap.csv")
    print("- shap_global_importance.csv")
    print("- shap_top30_bar.png")
    print("- shap_beeswarm_top25.png (if shap installed)")
    print("- selected_local_cases.csv")
    print("- local_shap_*.csv")
    print("- local_shap_*.png")
    print("- shap_analysis_summary.json")


if __name__ == "__main__":
    main()
