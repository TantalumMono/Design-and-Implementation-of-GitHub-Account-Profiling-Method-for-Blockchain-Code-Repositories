# -*- coding: utf-8 -*-
import os

# Must be set before importing sentence_transformers / transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home"
os.environ["HF_HUB_CACHE"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home\hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\st_cache"

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
BUNDLE_PATH = SCRIPT_DIR / "final_dl_st_mlp_artifacts" / "final_dl_st_mlp_bundle.pt"
OUTPUT_DIR = SCRIPT_DIR / "final_dl_st_mlp_explain_outputs"
CACHE_DIR = SCRIPT_DIR / "dl_st_embedding_cache_offline"

TOP_STRUCT_GLOBAL = 30
TOP_LOCAL_STRUCT = 20
N_TOP_SCORE = 3
N_HARD_POS = 3
N_HARD_NEG = 3

NON_STRUCTURED_COLS = {
    "label","family_id","repo_full_name","readme_text","readme_text_raw","readme_text_clean",
    "description_text","topics_text","combined_text","combined_text_clean","description_text_clean",
    "topics_text_clean","text_for_tfidf_clean","sample_id","collected_at","provenance_type",
    "is_real_positive","is_generated_positive","is_real_negative","sample_source","group_id",
    "source_repo_full_name","source_family","generated_from",
}
LEAKY_PREFIXES = ("genmeta__", "rewrite_meta__", "ablation__")
META_LEAKY_PREFIXES = ("meta_", "meta__", "source_", "generated_", "augmentation_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_leaky_feature_col(col: str) -> bool:
    if col in NON_STRUCTURED_COLS:
        return True
    if any(col.startswith(p) for p in LEAKY_PREFIXES):
        return True
    if any(col.startswith(p) for p in META_LEAKY_PREFIXES):
        return True
    return False


def get_aux_text_series(df: pd.DataFrame, description_col: str, topics_col: str) -> pd.Series:
    desc = df.get(description_col, "").fillna("").astype(str)
    topics = df.get(topics_col, "").fillna("").astype(str)
    return (desc + " " + topics).str.strip()


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def encode_texts_cached(texts, cache_path: Path, st_model_path: str):
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        return np.load(cache_path)

    model = SentenceTransformer(
        st_model_path,
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME", None),
        local_files_only=True,
        device="cpu",
    )
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    np.save(cache_path, emb)
    return emb


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
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"未找到最终模型 bundle: {BUNDLE_PATH}")

    bundle = torch.load(BUNDLE_PATH, map_location="cpu")
    fixed_params = bundle["fixed_params"]
    threshold = float(bundle["threshold"])
    st_model_path = bundle["st_model_path"]
    readme_col = bundle["README_TEXT_COL"]
    desc_col = bundle["DESCRIPTION_TEXT_COL"]
    topics_col = bundle["TOPICS_TEXT_COL"]
    structured_cols = bundle["structured_cols"]
    fill_values = pd.Series(bundle["fill_values"])
    input_dim = int(bundle["input_dim"])
    readme_dim = int(bundle["readme_dim"])
    aux_dim = int(bundle["aux_dim"])
    struct_dim = int(bundle["struct_dim"])

    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df[readme_col] = df[readme_col].fillna("").astype(str)
    df[desc_col] = df[desc_col].fillna("").astype(str)
    df[topics_col] = df[topics_col].fillna("").astype(str)

    readme_cache = CACHE_DIR / "readme_embeddings.npy"
    aux_cache = CACHE_DIR / "aux_embeddings.npy"
    readme_emb = encode_texts_cached(df[readme_col].tolist(), readme_cache, st_model_path)
    aux_emb = encode_texts_cached(get_aux_text_series(df, desc_col, topics_col).tolist(), aux_cache, st_model_path)

    X_num = df[structured_cols].copy()
    for col in structured_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(fill_values).fillna(0)

    scaler = StandardScaler()
    # reconstruct from current full data using saved fill_values; scale directly on current data is not ideal,
    # but for this final explainer we use the exact same full-data setup as training.
    X_num_scaled = scaler.fit_transform(X_num.values.astype(np.float32)).astype(np.float32)

    X_all = np.concatenate([readme_emb, aux_emb, X_num_scaled], axis=1).astype(np.float32)
    y_true = df["label"].values.astype(int)

    model = FusionMLP(
        input_dim=input_dim,
        hidden_dim=int(fixed_params["hidden_dim"]),
        num_layers=int(fixed_params["num_layers"]),
        dropout=float(fixed_params["dropout"]),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_all, dtype=torch.float32)).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "label": y_true,
        "y_logit": logits,
        "y_proba": probs,
        "y_pred": pred,
    })
    if "repo_full_name" in df.columns:
        pred_df["repo_full_name"] = df["repo_full_name"].values
    if "family_id" in df.columns:
        pred_df["family_id"] = df["family_id"].astype(str).values
    if "is_real_positive" in df.columns:
        pred_df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int).values
    if "is_generated_positive" in df.columns:
        pred_df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int).values
    pred_df.to_csv(OUTPUT_DIR / "dl_predictions_for_explanation.csv", index=False, encoding="utf-8-sig")

    baseline = {
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

    # ---------------------------
    # 1) Global branch ablation
    # ---------------------------
    readme_slice = slice(0, readme_dim)
    aux_slice = slice(readme_dim, readme_dim + aux_dim)
    struct_slice = slice(readme_dim + aux_dim, readme_dim + aux_dim + struct_dim)

    branch_results = []
    for branch_name, branch_slice in [
        ("remove_readme_branch", readme_slice),
        ("remove_aux_branch", aux_slice),
        ("remove_structured_branch", struct_slice),
    ]:
        X_ab = X_all.copy()
        X_ab[:, branch_slice] = 0.0
        with torch.no_grad():
            logits_ab = model(torch.tensor(X_ab, dtype=torch.float32)).numpy()
        probs_ab = 1.0 / (1.0 + np.exp(-logits_ab))
        pred_ab = (probs_ab >= threshold).astype(int)

        row = {
            "ablation": branch_name,
            "pr_auc": float(average_precision_score(y_true, probs_ab)),
            "roc_auc": float(roc_auc_score(y_true, probs_ab)),
            "precision": float(precision_score(y_true, pred_ab, zero_division=0)),
            "recall": float(recall_score(y_true, pred_ab, zero_division=0)),
            "f1": float(f1_score(y_true, pred_ab, zero_division=0)),
        }
        row["delta_pr_auc"] = row["pr_auc"] - baseline["pr_auc"]
        row["delta_roc_auc"] = row["roc_auc"] - baseline["roc_auc"]
        row["delta_f1"] = row["f1"] - baseline["f1"]
        branch_results.append(row)

    branch_df = pd.DataFrame(branch_results)
    branch_df.to_csv(OUTPUT_DIR / "global_branch_ablation.csv", index=False, encoding="utf-8-sig")

    # ---------------------------
    # 2) Global structured permutation importance
    # ---------------------------
    rng = np.random.default_rng(42)
    struct_importance = []
    X_base = X_all.copy()

    for j, feat in enumerate(structured_cols):
        X_perm = X_base.copy()
        col_idx = struct_slice.start + j
        X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])

        with torch.no_grad():
            logits_perm = model(torch.tensor(X_perm, dtype=torch.float32)).numpy()
        probs_perm = 1.0 / (1.0 + np.exp(-logits_perm))
        pred_perm = (probs_perm >= threshold).astype(int)

        pr_auc_perm = average_precision_score(y_true, probs_perm)
        f1_perm = f1_score(y_true, pred_perm, zero_division=0)

        struct_importance.append({
            "feature": feat,
            "pr_auc_drop": float(baseline["pr_auc"] - pr_auc_perm),
            "f1_drop": float(baseline["f1"] - f1_perm),
        })

    struct_imp_df = pd.DataFrame(struct_importance).sort_values(["pr_auc_drop", "f1_drop"], ascending=False)
    struct_imp_df.to_csv(OUTPUT_DIR / "global_structured_permutation_importance.csv", index=False, encoding="utf-8-sig")

    top_struct_plot = struct_imp_df.head(TOP_STRUCT_GLOBAL).copy()
    top_struct_plot = top_struct_plot.rename(columns={"pr_auc_drop": "value"})[["feature", "value"]]
    plot_bar(
        top_struct_plot,
        "Top Structured Features by PR-AUC Drop",
        "PR-AUC drop after permutation",
        OUTPUT_DIR / "global_structured_permutation_importance.png",
    )

    # ---------------------------
    # 3) Local evidence
    #    - branch ablation per case
    #    - gradient x input on structured features
    # ---------------------------
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

    local_summary_rows = []

    for row_idx in selected_cases.index.tolist():
        x = X_all[row_idx:row_idx + 1].copy()
        base_prob = float(probs[row_idx])
        base_logit = float(logits[row_idx])

        # branch ablation
        local_branch_rows = []
        for branch_name, branch_slice in [
            ("remove_readme_branch", readme_slice),
            ("remove_aux_branch", aux_slice),
            ("remove_structured_branch", struct_slice),
        ]:
            x_ab = x.copy()
            x_ab[:, branch_slice] = 0.0
            with torch.no_grad():
                logit_ab = float(model(torch.tensor(x_ab, dtype=torch.float32)).item())
            prob_ab = float(1.0 / (1.0 + np.exp(-logit_ab)))
            local_branch_rows.append({
                "ablation": branch_name,
                "base_prob": base_prob,
                "ablation_prob": prob_ab,
                "prob_drop": base_prob - prob_ab,
            })

        repo_name = str(selected_cases.loc[row_idx].get("repo_full_name", f"sample_{row_idx}"))
        safe_name = repo_name.replace("/", "__").replace("\\", "__")[:80]
        pd.DataFrame(local_branch_rows).to_csv(
            OUTPUT_DIR / f"local_branch_ablation_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # gradient x input on structured part
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        model.zero_grad()
        out_logit = model(x_tensor)
        out_logit.backward()

        grads = x_tensor.grad.detach().numpy()[0]
        x_np = x_tensor.detach().numpy()[0]
        gx = grads * x_np

        struct_vals = x_np[struct_slice]
        struct_grads = grads[struct_slice]
        struct_gx = gx[struct_slice]

        local_struct_df = pd.DataFrame({
            "feature": structured_cols,
            "feature_value_scaled": struct_vals,
            "gradient": struct_grads,
            "grad_x_input": struct_gx,
            "abs_grad_x_input": np.abs(struct_gx),
        }).sort_values("abs_grad_x_input", ascending=False).head(TOP_LOCAL_STRUCT)

        local_struct_df.to_csv(
            OUTPUT_DIR / f"local_structured_gradxinput_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plot_df = local_struct_df.copy().sort_values("grad_x_input", ascending=False)
        plt.figure(figsize=(10, 7))
        y = np.arange(len(plot_df))
        plt.barh(y, plot_df["grad_x_input"].values)
        plt.yticks(y, plot_df["feature"].values, fontsize=9)
        plt.gca().invert_yaxis()
        plt.title(f"Local Structured grad×input | {safe_name}")
        plt.xlabel("grad × input")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"local_structured_gradxinput_{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

        local_summary_rows.append({
            "row_index": int(row_idx),
            "repo_full_name": repo_name,
            "case_type": str(selected_cases.loc[row_idx]["case_type"]),
            "label": int(selected_cases.loc[row_idx]["label"]),
            "y_proba": float(selected_cases.loc[row_idx]["y_proba"]),
            "y_pred": int(selected_cases.loc[row_idx]["y_pred"]),
            "top_struct_feature": str(local_struct_df.iloc[0]["feature"]) if len(local_struct_df) > 0 else None,
            "top_struct_gradxinput": float(local_struct_df.iloc[0]["grad_x_input"]) if len(local_struct_df) > 0 else None,
            "max_branch_prob_drop": float(max(r["prob_drop"] for r in local_branch_rows)) if local_branch_rows else None,
        })

    pd.DataFrame(local_summary_rows).to_csv(OUTPUT_DIR / "local_explanation_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "threshold": threshold,
        "baseline_metrics_on_trainset": baseline,
        "note": (
            "For the final deep model with frozen SentenceTransformer embeddings, "
            "global explanation is provided via branch ablation and structured-feature permutation importance; "
            "local explanation is provided via branch ablation and grad×input on structured features. "
            "This is explanation evidence on the training set, not a separate holdout."
        ),
    }
    with open(OUTPUT_DIR / "dl_explanation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved deep model explanation outputs to:", OUTPUT_DIR)
    print("- dl_predictions_for_explanation.csv")
    print("- global_branch_ablation.csv")
    print("- global_structured_permutation_importance.csv")
    print("- global_structured_permutation_importance.png")
    print("- selected_local_cases.csv")
    print("- local_branch_ablation_*.csv")
    print("- local_structured_gradxinput_*.csv")
    print("- local_structured_gradxinput_*.png")
    print("- local_explanation_summary.csv")
    print("- dl_explanation_summary.json")


if __name__ == "__main__":
    main()
