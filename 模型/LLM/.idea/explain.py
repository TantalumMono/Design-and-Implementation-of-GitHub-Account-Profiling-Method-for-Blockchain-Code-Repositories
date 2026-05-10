import json
import os
import re
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from build_llm_inputs import build_prompt


def split_chunks(text: str, chunk_size=320, stride=180):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
        i += stride
    return chunks


def predict_single_prob(trainer, tokenizer, text, cfg):
    from train_llm_classifier import predict_proba
    prob = predict_proba(trainer, [text], [0], tokenizer, cfg)[0]
    return float(prob)


def build_prompt_without_feature(row, feature_names, cfg, feature_name):
    row2 = row.copy()
    if feature_name in row2.index:
        row2[feature_name] = np.nan
    return build_prompt(row2, feature_names, cfg)


def feature_ablation(trainer, tokenizer, row_df, feature_names, cfg):
    row = row_df.iloc[0]
    base_prompt = build_prompt(row, feature_names, cfg)
    base_prob = predict_single_prob(trainer, tokenizer, base_prompt, cfg)

    cand_features = [f for f in feature_names if f in row.index and not str(row[f]) == "nan"]
    scores = []
    for f in cand_features:
        masked = build_prompt_without_feature(row, feature_names, cfg, f)
        p = predict_single_prob(trainer, tokenizer, masked, cfg)
        delta = base_prob - p
        scores.append({
            "feature": f,
            "feature_value": None if str(row[f]) == "nan" else float(row[f]) if isinstance(row[f], (int, float)) else str(row[f]),
            "ablation_delta": float(delta),
            "direction": "push_to_suspicious" if delta > 0 else "push_to_benign"
        })
    scores = sorted(scores, key=lambda x: abs(x["ablation_delta"]), reverse=True)
    return scores[: int(cfg["explain"]["topk_feature_ablation"])]


def text_chunk_ablation(trainer, tokenizer, row_df, feature_names, cfg):
    row = row_df.iloc[0]
    base_prompt = build_prompt(row, feature_names, cfg)
    base_prob = predict_single_prob(trainer, tokenizer, base_prompt, cfg)

    raw_text = " ".join([
        str(row.get("description_text", "")),
        str(row.get("topics_text", "")),
        str(row.get("readme_text", "")),
    ])
    chunks = split_chunks(
        raw_text,
        chunk_size=int(cfg["explain"]["chunk_size"]),
        stride=int(cfg["explain"]["chunk_stride"])
    )

    out = []
    for chunk in chunks:
        reduced_text = raw_text.replace(chunk, " ")
        row2 = row.copy()
        row2["readme_text"] = reduced_text
        row2["description_text"] = ""
        row2["topics_text"] = ""
        masked_prompt = build_prompt(row2, feature_names, cfg)
        p = predict_single_prob(trainer, tokenizer, masked_prompt, cfg)
        out.append({
            "chunk": chunk,
            "ablation_delta": float(base_prob - p),
            "direction": "push_to_suspicious" if (base_prob - p) > 0 else "push_to_benign"
        })
    out = sorted(out, key=lambda x: abs(x["ablation_delta"]), reverse=True)
    return out[: int(cfg["explain"]["topk_text_chunks"])]


def get_hidden_embeddings(trainer, tokenizer, texts, cfg):
    model = trainer.model
    model.eval()
    device = model.device
    all_vecs = []

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=int(cfg["model"]["max_length"]),
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.base_model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            attn = enc["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
            all_vecs.append(pooled.detach().cpu().numpy())
    return np.concatenate(all_vecs, axis=0)


def nearest_cases(trainer, tokenizer, target_prompt, train_prompts, train_meta_df, cfg):
    target_vec = get_hidden_embeddings(trainer, tokenizer, [target_prompt], cfg)
    train_vecs = get_hidden_embeddings(trainer, tokenizer, train_prompts, cfg)
    sims = cosine_similarity(target_vec, train_vecs)[0]
    order = np.argsort(-sims)

    pos = []
    neg = []
    for idx in order:
        rec = {
            "sample_id": train_meta_df.iloc[idx]["sample_id"],
            "label": int(train_meta_df.iloc[idx]["label"]),
            "family": train_meta_df.iloc[idx]["family"],
            "repo_full_name": train_meta_df.iloc[idx]["repo_full_name"],
            "source_type": train_meta_df.iloc[idx]["source_type"],
            "similarity": float(sims[idx]),
        }
        if rec["label"] == 1 and len(pos) < int(cfg["explain"]["topk_neighbors"]):
            pos.append(rec)
        if rec["label"] == 0 and len(neg) < int(cfg["explain"]["topk_neighbors"]):
            neg.append(rec)
        if len(pos) >= int(cfg["explain"]["topk_neighbors"]) and len(neg) >= int(cfg["explain"]["topk_neighbors"]):
            break
    return {"nearest_positive": pos, "nearest_negative": neg}


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")