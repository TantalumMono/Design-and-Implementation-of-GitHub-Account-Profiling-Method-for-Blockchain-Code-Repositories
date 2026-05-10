# -*- coding: utf-8 -*-
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home"
os.environ["HF_HUB_CACHE"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home\hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\st_cache"

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
FAMILY_DIR = SCRIPT_DIR / "dl_st_mlp_fixed_params_family_validation_outputs"
OUTPUT_DIR = SCRIPT_DIR / "final_dl_oof_test_explain_outputs"
CACHE_DIR = SCRIPT_DIR / "dl_st_embedding_cache_offline"

README_TEXT_COL = "readme_text_clean"
DESCRIPTION_TEXT_COL = "description_text_clean"
TOPICS_TEXT_COL = "topics_text_clean"

RANDOM_STATE = 42
OUTER_N_SPLITS = 5
ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST = True
TOP_CASES_PER_BUCKET = 3
TOP_TEXT_UNITS = 15
TOP_STRUCT = 15

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


def get_aux_text(description: str, topics: str) -> str:
    return (str(description or "") + " " + str(topics or "")).strip()


def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    pieces = re.split(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+|\n+', text)
    pieces = [p.strip() for p in pieces if p and p.strip()]
    dedup = []
    for p in pieces:
        if p not in dedup:
            dedup.append(p)
    return dedup


def make_positive_family_folds(real_pos_df: pd.DataFrame, n_splits: int, seed: int):
    families = np.array(sorted(real_pos_df["family_id"].dropna().astype(str).unique()))
    n_splits = min(n_splits, len(families))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_fam_idx, te_fam_idx in kf.split(families):
        train_families = set(families[tr_fam_idx])
        test_families = set(families[te_fam_idx])
        folds.append((train_families, test_families))
    return folds


def make_index_folds(indices: np.ndarray, n_splits: int, seed: int):
    n_splits = min(n_splits, len(indices))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in kf.split(indices):
        folds.append((indices[tr_idx], indices[te_idx]))
    return folds


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


def predict_prob(model, x_row: np.ndarray) -> float:
    with torch.no_grad():
        logit = float(model(torch.tensor(x_row[None, :], dtype=torch.float32)).item())
    return float(1.0 / (1.0 + np.exp(-logit)))


def plot_bar(df_plot: pd.DataFrame, title: str, save_path: Path):
    if df_plot.empty:
        return
    plt.figure(figsize=(10, 8))
    y = np.arange(len(df_plot))
    plt.barh(y, df_plot["value"].values)
    plt.yticks(y, df_plot["label"].values, fontsize=8)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)
    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int)
    df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int)
    df["family_id"] = df["family_id"].fillna("UNKNOWN").astype(str)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)
    structured_cols = [c for c in df.columns if not is_leaky_feature_col(c)]

    oof_df = pd.read_csv(FAMILY_DIR / "dl_st_mlp_fixed_params_oof_predictions.csv")

    real_pos_mask = df["is_real_positive"] == 1
    neg_mask = df["label"] == 0
    real_pos_df = df.loc[real_pos_mask].copy()
    neg_indices_all = df.index[neg_mask].to_numpy()
    outer_pos_family_folds = make_positive_family_folds(real_pos_df, OUTER_N_SPLITS, RANDOM_STATE)
    outer_neg_folds = make_index_folds(neg_indices_all, len(outer_pos_family_folds), RANDOM_STATE)

    bundle1 = torch.load(FAMILY_DIR / "outer_fold_1_bundle.pt", map_location="cpu")
    st_model_path = bundle1["st_model_path"]

    readme_cache = CACHE_DIR / "readme_embeddings.npy"
    aux_cache = CACHE_DIR / "aux_embeddings.npy"
    readme_emb_all = encode_texts_cached(df[README_TEXT_COL].tolist(), readme_cache, st_model_path)
    aux_texts = [get_aux_text(d, t) for d, t in zip(df[DESCRIPTION_TEXT_COL], df[TOPICS_TEXT_COL])]
    aux_emb_all = encode_texts_cached(aux_texts, aux_cache, st_model_path)

    row_indices = oof_df["row_index"].values.astype(int)
    top_score_pos = oof_df.loc[df.loc[row_indices, "is_real_positive"].values == 1].sort_values("y_proba", ascending=False).head(TOP_CASES_PER_BUCKET)
    hard_real_pos = oof_df.loc[df.loc[row_indices, "is_real_positive"].values == 1].sort_values("y_proba", ascending=True).head(TOP_CASES_PER_BUCKET)
    hard_neg = oof_df.loc[df.loc[row_indices, "label"].values == 0].sort_values("y_proba", ascending=False).head(TOP_CASES_PER_BUCKET)

    selected = pd.concat([
        top_score_pos.assign(case_type="high_score_real_positive"),
        hard_real_pos.assign(case_type="hard_real_positive"),
        hard_neg.assign(case_type="hard_negative"),
    ], axis=0)
    selected.to_csv(OUTPUT_DIR / "selected_oof_cases.csv", index=False, encoding="utf-8-sig")

    st_model = SentenceTransformer(
        st_model_path,
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME", None),
        local_files_only=True,
        device="cpu",
    )

    summaries = []

    zipped_folds = list(zip(outer_pos_family_folds, outer_neg_folds))

    for _, row in selected.iterrows():
        row_index = int(row["row_index"])
        fold_id = int(row["outer_fold"])
        ((_, outer_test_fams), (_, outer_test_neg_idx)) = zipped_folds[fold_id - 1]

        outer_train_mask = (
            (~df.index.isin(outer_test_neg_idx))
            & (~((df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
            & (~((df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
        )
        train_df = df.loc[outer_train_mask].copy()

        X_tr_num = train_df[structured_cols].copy()
        X_case_num = df.loc[[row_index], structured_cols].copy()
        for col in structured_cols:
            X_tr_num[col] = pd.to_numeric(X_tr_num[col], errors="coerce")
            X_case_num[col] = pd.to_numeric(X_case_num[col], errors="coerce")
        fill_values = X_tr_num.median(numeric_only=True)
        X_tr_num = X_tr_num.fillna(fill_values).fillna(0)
        X_case_num = X_case_num.fillna(fill_values).fillna(0)
        scaler = StandardScaler()
        scaler.fit(X_tr_num.values.astype(np.float32))
        x_num_scaled = scaler.transform(X_case_num.values.astype(np.float32))[0].astype(np.float32)

        bundle = torch.load(FAMILY_DIR / f"outer_fold_{fold_id}_bundle.pt", map_location="cpu")
        model = FusionMLP(
            input_dim=int(bundle["input_dim"]),
            hidden_dim=int(bundle["fixed_params"]["hidden_dim"]),
            num_layers=int(bundle["fixed_params"]["num_layers"]),
            dropout=float(bundle["fixed_params"]["dropout"]),
        )
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()
        threshold = float(bundle["threshold"])

        readme_dim = readme_emb_all.shape[1]
        aux_dim = aux_emb_all.shape[1]
        x_base = np.concatenate([readme_emb_all[row_index], aux_emb_all[row_index], x_num_scaled], axis=0).astype(np.float32)
        base_prob = predict_prob(model, x_base)

        branch_rows = []
        for name, which in [("remove_readme_branch", "readme"), ("remove_aux_branch", "aux"), ("remove_structured_branch", "struct")]:
            x_ab = x_base.copy()
            if which == "readme":
                x_ab[:readme_dim] = 0.0
            elif which == "aux":
                x_ab[readme_dim:readme_dim + aux_dim] = 0.0
            else:
                x_ab[readme_dim + aux_dim:] = 0.0
            p_ab = predict_prob(model, x_ab)
            branch_rows.append({"ablation": name, "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        repo_name = str(row.get("repo_full_name", f"sample_{row_index}"))
        safe_name = repo_name.replace("/", "__").replace("\\", "__")[:80]
        pd.DataFrame(branch_rows).to_csv(OUTPUT_DIR / f"oof_branch_ablation_{safe_name}.csv", index=False, encoding="utf-8-sig")

        readme_text = df.loc[row_index, README_TEXT_COL]
        aux_text = aux_texts[row_index]
        readme_sents = split_sentences(readme_text)
        aux_sents = split_sentences(aux_text)

        text_rows = []
        for s in readme_sents:
            ablated = " ".join([x for x in readme_sents if x != s]).strip()
            ab_readme_emb = st_model.encode([ablated], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]
            x_ab = np.concatenate([ab_readme_emb, aux_emb_all[row_index], x_num_scaled], axis=0).astype(np.float32)
            p_ab = predict_prob(model, x_ab)
            text_rows.append({"branch": "README", "text_unit": s, "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        for s in aux_sents:
            ablated = " ".join([x for x in aux_sents if x != s]).strip()
            ab_aux_emb = st_model.encode([ablated], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]
            x_ab = np.concatenate([readme_emb_all[row_index], ab_aux_emb, x_num_scaled], axis=0).astype(np.float32)
            p_ab = predict_prob(model, x_ab)
            text_rows.append({"branch": "AUX", "text_unit": s, "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        text_df = pd.DataFrame(text_rows).sort_values("prob_drop", ascending=False)
        text_df.to_csv(OUTPUT_DIR / f"oof_text_sentence_ablation_{safe_name}.csv", index=False, encoding="utf-8-sig")
        plot_df = text_df.head(TOP_TEXT_UNITS).copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["branch"] + " | " + plot_df["text_unit"].str.slice(0, 80)
            plot_bar(plot_df.rename(columns={"prob_drop": "value"})[["label", "value"]], f"OOF top text units | {safe_name}", OUTPUT_DIR / f"oof_text_sentence_ablation_{safe_name}.png")

        x_tensor = torch.tensor(x_base[None, :], dtype=torch.float32, requires_grad=True)
        model.zero_grad()
        out = model(x_tensor)
        out.backward()
        grads = x_tensor.grad.detach().numpy()[0]
        gx = grads * x_base
        struct_vals = x_base[readme_dim + aux_dim:]
        struct_grads = grads[readme_dim + aux_dim:]
        struct_gx = gx[readme_dim + aux_dim:]

        struct_df = pd.DataFrame({
            "feature": structured_cols,
            "feature_value_scaled": struct_vals,
            "gradient": struct_grads,
            "grad_x_input": struct_gx,
            "abs_grad_x_input": np.abs(struct_gx),
        }).sort_values("abs_grad_x_input", ascending=False).head(TOP_STRUCT)
        struct_df.to_csv(OUTPUT_DIR / f"oof_structured_gradxinput_{safe_name}.csv", index=False, encoding="utf-8-sig")
        if not struct_df.empty:
            plot_bar(struct_df.rename(columns={"feature": "label", "grad_x_input": "value"})[["label", "value"]], f"OOF structured grad×input | {safe_name}", OUTPUT_DIR / f"oof_structured_gradxinput_{safe_name}.png")

        summaries.append({
            "row_index": row_index,
            "outer_fold": fold_id,
            "repo_full_name": repo_name,
            "case_type": str(row["case_type"]),
            "label": int(row["label"]),
            "y_proba_oof": float(row["y_proba"]),
            "y_pred_oof": int(row["y_pred"]),
            "fold_threshold": threshold,
            "base_prob_recomputed": base_prob,
            "top_text_branch": str(text_df.iloc[0]["branch"]) if len(text_df) else None,
            "top_text_unit": str(text_df.iloc[0]["text_unit"]) if len(text_df) else None,
            "top_text_prob_drop": float(text_df.iloc[0]["prob_drop"]) if len(text_df) else None,
            "top_struct_feature": str(struct_df.iloc[0]["feature"]) if len(struct_df) else None,
            "top_struct_gradxinput": float(struct_df.iloc[0]["grad_x_input"]) if len(struct_df) else None,
            "max_branch_prob_drop": float(max(r["prob_drop"] for r in branch_rows)) if branch_rows else None,
        })

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "oof_case_explanation_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "oof_case_explanation_overview.json", "w", encoding="utf-8") as f:
        json.dump({"note": "These explanations are generated on outer-test / OOF cases using the corresponding fold model and fold threshold."}, f, ensure_ascii=False, indent=2)

    print("Saved OOF/test case explanation outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
