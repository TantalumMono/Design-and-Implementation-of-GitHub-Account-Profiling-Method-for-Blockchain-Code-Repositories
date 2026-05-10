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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
BUNDLE_PATH = SCRIPT_DIR / "final_dl_st_mlp_artifacts" / "final_dl_st_mlp_bundle.pt"
OUTPUT_DIR = SCRIPT_DIR / "final_dl_text_evidence_outputs"
CACHE_DIR = SCRIPT_DIR / "dl_st_embedding_cache_offline"

TOP_CASES_PER_BUCKET = 3
TOP_SENTENCES = 15
README_TEXT_COL = "readme_text_clean"
DESCRIPTION_TEXT_COL = "description_text_clean"
TOPICS_TEXT_COL = "topics_text_clean"

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


def split_markdown_sections(text: str):
    text = (text or "").strip()
    if not text:
        return []
    lines = text.splitlines()
    sections = []
    current_title = "ROOT"
    current_lines = []
    for ln in lines:
        if re.match(r'^\s{0,3}#{1,6}\s+', ln):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = re.sub(r'^\s{0,3}#{1,6}\s+', '', ln).strip()
            current_lines = []
        else:
            current_lines.append(ln)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, s) for t, s in sections if s]


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


def make_feature_row(idx, readme_emb_all, aux_emb_all, X_num_scaled, override_readme_emb=None, override_aux_emb=None):
    readme = override_readme_emb if override_readme_emb is not None else readme_emb_all[idx]
    aux = override_aux_emb if override_aux_emb is not None else aux_emb_all[idx]
    num = X_num_scaled[idx]
    return np.concatenate([readme, aux, num], axis=0).astype(np.float32)


def plot_sentence_importance(df_plot: pd.DataFrame, title: str, save_path: Path):
    if df_plot.empty:
        return
    plt.figure(figsize=(10, 8))
    y = np.arange(len(df_plot))
    plt.barh(y, df_plot["prob_drop"].values)
    plt.yticks(y, df_plot["label"].values, fontsize=8)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Probability drop after ablation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)
    bundle = torch.load(BUNDLE_PATH, map_location="cpu")
    st_model_path = bundle["st_model_path"]
    threshold = float(bundle["threshold"])
    fixed_params = bundle["fixed_params"]
    structured_cols = bundle["structured_cols"]
    fill_values = pd.Series(bundle["fill_values"])
    readme_dim = int(bundle["readme_dim"])
    aux_dim = int(bundle["aux_dim"])
    input_dim = int(bundle["input_dim"])

    df = pd.read_csv(DATA_PATH)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int)
    df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)

    readme_cache = CACHE_DIR / "readme_embeddings.npy"
    aux_cache = CACHE_DIR / "aux_embeddings.npy"
    readme_emb_all = encode_texts_cached(df[README_TEXT_COL].tolist(), readme_cache, st_model_path)
    aux_texts = [get_aux_text(d, t) for d, t in zip(df[DESCRIPTION_TEXT_COL], df[TOPICS_TEXT_COL])]
    aux_emb_all = encode_texts_cached(aux_texts, aux_cache, st_model_path)

    X_num = df[structured_cols].copy()
    for col in structured_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    X_num = X_num.fillna(fill_values).fillna(0)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num.values.astype(np.float32)).astype(np.float32)

    model = FusionMLP(
        input_dim=input_dim,
        hidden_dim=int(fixed_params["hidden_dim"]),
        num_layers=int(fixed_params["num_layers"]),
        dropout=float(fixed_params["dropout"]),
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    X_all = np.concatenate([readme_emb_all, aux_emb_all, X_num_scaled], axis=1).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X_all, dtype=torch.float32)).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "label": df["label"].values,
        "y_proba": probs,
        "y_pred": pred,
        "is_real_positive": df["is_real_positive"].values,
    })
    if "repo_full_name" in df.columns:
        pred_df["repo_full_name"] = df["repo_full_name"].values
    pred_df.to_csv(OUTPUT_DIR / "final_text_evidence_predictions.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name, which in [("remove_readme_branch", "readme"), ("remove_aux_branch", "aux")]:
        X_ab = X_all.copy()
        if which == "readme":
            X_ab[:, :readme_dim] = 0.0
        else:
            X_ab[:, readme_dim:readme_dim + aux_dim] = 0.0
        with torch.no_grad():
            logits_ab = model(torch.tensor(X_ab, dtype=torch.float32)).numpy()
        probs_ab = 1.0 / (1.0 + np.exp(-logits_ab))
        rows.append({
            "ablation": name,
            "mean_prob_drop": float(np.mean(probs - probs_ab)),
            "positive_mean_prob_drop": float(np.mean((probs - probs_ab)[df["label"].values == 1])),
            "negative_mean_prob_drop": float(np.mean((probs - probs_ab)[df["label"].values == 0])),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "global_text_branch_ablation.csv", index=False, encoding="utf-8-sig")

    real_pos_mask = df["is_real_positive"].values == 1
    neg_mask = df["label"].values == 0
    top_score_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=False).head(TOP_CASES_PER_BUCKET)
    hard_real_pos = pred_df.loc[real_pos_mask].sort_values("y_proba", ascending=True).head(TOP_CASES_PER_BUCKET)
    hard_neg = pred_df.loc[neg_mask].sort_values("y_proba", ascending=False).head(TOP_CASES_PER_BUCKET)
    selected = pd.concat([
        top_score_pos.assign(case_type="high_score_real_positive"),
        hard_real_pos.assign(case_type="hard_real_positive"),
        hard_neg.assign(case_type="hard_negative"),
    ], axis=0)
    selected.to_csv(OUTPUT_DIR / "selected_cases_for_text_evidence.csv", index=False, encoding="utf-8-sig")

    st_model = SentenceTransformer(
        st_model_path,
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME", None),
        local_files_only=True,
        device="cpu",
    )

    summaries = []
    for idx in selected.index.tolist():
        repo_name = str(selected.loc[idx].get("repo_full_name", f"sample_{idx}"))
        safe_name = repo_name.replace("/", "__").replace("\\", "__")[:80]
        base_x = make_feature_row(idx, readme_emb_all, aux_emb_all, X_num_scaled)
        base_prob = predict_prob(model, base_x)

        readme_text = df.loc[idx, README_TEXT_COL]
        readme_sents = split_sentences(readme_text)
        readme_rows = []
        for s in readme_sents:
            ablated = " ".join([x for x in readme_sents if x != s]).strip()
            ab_readme_emb = st_model.encode([ablated], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]
            x_ab = make_feature_row(idx, readme_emb_all, aux_emb_all, X_num_scaled, override_readme_emb=ab_readme_emb)
            p_ab = predict_prob(model, x_ab)
            readme_rows.append({"branch": "README", "text_unit": s, "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        aux_text = aux_texts[idx]
        aux_sents = split_sentences(aux_text)
        aux_rows = []
        for s in aux_sents:
            ablated = " ".join([x for x in aux_sents if x != s]).strip()
            ab_aux_emb = st_model.encode([ablated], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]
            x_ab = make_feature_row(idx, readme_emb_all, aux_emb_all, X_num_scaled, override_aux_emb=ab_aux_emb)
            p_ab = predict_prob(model, x_ab)
            aux_rows.append({"branch": "AUX", "text_unit": s, "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        section_rows = []
        for title, sec_text in split_markdown_sections(readme_text):
            ablated_text = readme_text.replace(sec_text, " ")
            ab_readme_emb = st_model.encode([ablated_text], convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)[0]
            x_ab = make_feature_row(idx, readme_emb_all, aux_emb_all, X_num_scaled, override_readme_emb=ab_readme_emb)
            p_ab = predict_prob(model, x_ab)
            section_rows.append({"section_title": title, "section_text": sec_text[:500], "base_prob": base_prob, "ablated_prob": p_ab, "prob_drop": base_prob - p_ab})

        sent_df = pd.DataFrame(readme_rows + aux_rows).sort_values("prob_drop", ascending=False)
        sec_df = pd.DataFrame(section_rows).sort_values("prob_drop", ascending=False)
        sent_df.to_csv(OUTPUT_DIR / f"text_sentence_ablation_{safe_name}.csv", index=False, encoding="utf-8-sig")
        sec_df.to_csv(OUTPUT_DIR / f"readme_section_ablation_{safe_name}.csv", index=False, encoding="utf-8-sig")

        plot_df = sent_df.head(TOP_SENTENCES).copy()
        if not plot_df.empty:
            plot_df["label"] = plot_df["branch"] + " | " + plot_df["text_unit"].str.slice(0, 80)
            plot_sentence_importance(plot_df[["label", "prob_drop"]], f"Top text units by probability drop | {safe_name}", OUTPUT_DIR / f"text_sentence_ablation_{safe_name}.png")

        summaries.append({
            "row_index": int(idx),
            "repo_full_name": repo_name,
            "case_type": str(selected.loc[idx]["case_type"]),
            "label": int(selected.loc[idx]["label"]),
            "y_proba": float(selected.loc[idx]["y_proba"]),
            "y_pred": int(selected.loc[idx]["y_pred"]),
            "top_text_branch": str(sent_df.iloc[0]["branch"]) if len(sent_df) else None,
            "top_text_unit": str(sent_df.iloc[0]["text_unit"]) if len(sent_df) else None,
            "top_text_prob_drop": float(sent_df.iloc[0]["prob_drop"]) if len(sent_df) else None,
            "top_section_title": str(sec_df.iloc[0]["section_title"]) if len(sec_df) else None,
            "top_section_prob_drop": float(sec_df.iloc[0]["prob_drop"]) if len(sec_df) else None,
        })

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "text_evidence_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "text_evidence_overview.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "note": "Text evidence is produced on the final full-data model and should be treated as qualitative evidence."}, f, ensure_ascii=False, indent=2)

    print("Saved final text evidence outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
