import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================================================
# 说明
# 1) 兼容新版 positive 样本格式：augmented_300_with_meta.json
#    - family 信息优先读取 meta.family_label
#    - 原始/合成信息优先读取 meta.is_synthetic 与 meta.augmentation_method
# 2) 负样本默认没有 family，统一按样本唯一编号分组，避免数据泄漏
# 3) 默认优先使用 combined_text 作为 TF-IDF 主文本输入；若为空则回退到 readme+description+topics
# 4) 输出：
#    - family_dataset_full.csv：保留较完整信息
#    - family_dataset_train_ready.csv：更适合直接给 XGBoost + TF-IDF 使用
#    - positive_family_summary.csv：正样本 family 分布与原始/合成构成
#    - numeric_feature_columns.json：结构化数值特征列名
#    - text_column_summary.json：文本列说明
# =========================================================

OUTPUT_DIR = "family_prepared_without_F3_web3_platform_app"

POS_CANDIDATES = [
    'augmented_300_without_F3_web3_platform_app.json',
]

NEG_CANDIDATES = [
    'negative_github_dataset_cleaned.json',
]

FEATURES_COL = "features"
GEN_SRC_COL = "generation_source"
META_COL = "meta"

BASE_STOPWORDS = set(ENGLISH_STOP_WORDS)
CUSTOM_STOPWORDS = {
    "project",
    "github",
    "repository",
    "repo",
    "source",
    "code",
}
ALL_STOPWORDS = BASE_STOPWORDS | CUSTOM_STOPWORDS


# =========================
# 通用工具
# =========================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_first_existing(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层应为 list 或 dict")
    return data


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def remove_stopwords(text: str, stopwords=ALL_STOPWORDS) -> str:
    if not text:
        return ""
    tokens = text.split()
    filtered_tokens = []
    for tok in tokens:
        if tok in stopwords:
            continue
        if len(tok) <= 1:
            continue
        filtered_tokens.append(tok)
    return " ".join(filtered_tokens)


# =========================
# 文本清洗
# =========================
def clean_text_for_tfidf(text: Any) -> str:
    text = safe_text(text)
    if not text:
        return ""

    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\((.*?)\)", r" \1 ", text)
    text = re.sub(r"\[([^\]]+)\]\((.*?)\)", r" \1 ", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    for ch in ["`", "#", "*", ">", "|", "="]:
        text = text.replace(ch, " ")

    text = text.replace("_", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")

    text = re.sub(r"[^a-zA-Z0-9\s\.\+\-]", " ", text)
    text = text.lower()

    replacements = {
        "erc 20": "erc20",
        "erc 721": "erc721",
        "erc 1155": "erc1155",
        "erc 4626": "erc4626",
        "layer 2": "layer2",
        "cross chain": "crosschain",
        "wallet connect": "walletconnect",
        "meta mask": "metamask",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()
    text = remove_stopwords(text, stopwords=ALL_STOPWORDS)
    return text


# =========================
# 数值转换
# =========================
def to_number_if_possible(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return v
    if v is None:
        return np.nan
    try:
        s = str(v).strip()
        if s == "":
            return np.nan
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?\d*\.\d+", s):
            return float(s)
    except Exception:
        pass
    return v


# =========================
# 元信息读取
# =========================
def get_generation_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    gen = item.get(GEN_SRC_COL, {})
    return gen if isinstance(gen, dict) else {}


def get_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    meta = item.get(META_COL, {})
    return meta if isinstance(meta, dict) else {}


def choose_best_text(item: Dict[str, Any]) -> Tuple[str, str]:
    combined = safe_text(item.get("combined_text"))
    if combined:
        return combined, "combined_text"

    parts = [
        safe_text(item.get("repo_name")),
        safe_text(item.get("description_text")),
        safe_text(item.get("topics_text")),
        safe_text(item.get("readme_text")),
    ]
    merged = "\n".join([p for p in parts if p]).strip()
    if merged:
        return merged, "fallback_merge"

    return "", "empty"


def infer_positive_family(item: Dict[str, Any], feats: Dict[str, Any], meta: Dict[str, Any], gen: Dict[str, Any], idx: int) -> str:
    family = (
        meta.get("family_label")
        or item.get("family_label")
        or item.get("family_id")
        or item.get("family")
        or gen.get("source_family_reconstructed")
        or gen.get("source_repo_full_name_reconstructed")
        or item.get("source_family")
        or item.get("source")
        or feats.get("repo_full_name")
        or f"pos_family_{idx:04d}"
    )
    return str(family)


def infer_positive_provenance(item: Dict[str, Any], meta: Dict[str, Any], gen: Dict[str, Any]) -> str:
    if "is_synthetic" in meta:
        return "synthetic_positive" if int(meta.get("is_synthetic", 0)) == 1 else "original_real_positive"

    aug_method = safe_text(meta.get("augmentation_method")).lower()
    if aug_method:
        return "original_real_positive" if aug_method == "original" else "synthetic_positive"

    if gen.get("provenance_type"):
        return str(gen["provenance_type"])
    if item.get("provenance_type"):
        return str(item["provenance_type"])
    return "unknown_positive"


def build_positive_flags(provenance: str) -> Tuple[int, int]:
    is_real_positive = int(provenance == "original_real_positive")
    is_generated_positive = int(provenance in {"synthetic_positive", "generated_positive", "unknown_positive"} and not is_real_positive)
    return is_real_positive, is_generated_positive


# =========================
# 行抽取
# =========================
def extract_positive_row(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    feats = item.get(FEATURES_COL, {}) or {}
    gen = get_generation_meta(item)
    meta = get_meta(item)

    provenance = infer_positive_provenance(item, meta, gen)
    is_real_positive, is_generated_positive = build_positive_flags(provenance)
    family = infer_positive_family(item, feats, meta, gen, idx)
    raw_text_for_tfidf, text_source_name = choose_best_text(item)

    readme_raw = safe_text(item.get("readme_text"))
    desc_raw = safe_text(item.get("description_text"))
    topics_raw = safe_text(item.get("topics_text"))
    combined_raw = safe_text(item.get("combined_text"))

    row = {
        "sample_id": f"pos_{idx:05d}",
        "label": int(item.get("label", 1)),
        "sample_source": "positive_augmented" if is_generated_positive else "positive_original",
        "provenance_type": provenance,
        "is_real_positive": is_real_positive,
        "is_generated_positive": is_generated_positive,
        "is_real_negative": 0,
        "family_id": family,
        "group_id": f"POS::{family}",
        "repo_full_name": feats.get("repo_full_name") or safe_text(item.get("repo_full_name")) or f"pos_repo_{idx:05d}",
        "source_repo_full_name": meta.get("source_repo_full_name") or gen.get("source_repo_full_name_reconstructed") or item.get("source_repo_full_name"),
        "source_family": meta.get("family_label") or meta.get("source_family") or gen.get("source_family_reconstructed") or item.get("source_family") or item.get("family") or item.get("family_id"),
        "generated_from": item.get("generated_from"),
        "raw_file_name": feats.get("raw_file_name") or item.get("raw_file_name"),
        "source_raw_file_name": meta.get("source_raw_file_name") or item.get("source_raw_file_name"),
        "augmentation_method": meta.get("augmentation_method") or item.get("augmentation_method"),
        "meta_is_synthetic": to_number_if_possible(meta.get("is_synthetic")),
        "readme_text_raw": readme_raw,
        "description_text_raw": desc_raw,
        "topics_text_raw": topics_raw,
        "combined_text_raw": combined_raw,
        "text_for_tfidf_raw": raw_text_for_tfidf,
        "text_source_name": text_source_name,
    }

    row["readme_text_clean"] = clean_text_for_tfidf(readme_raw)
    row["description_text_clean"] = clean_text_for_tfidf(desc_raw)
    row["topics_text_clean"] = clean_text_for_tfidf(topics_raw)
    row["combined_text_clean"] = clean_text_for_tfidf(combined_raw)
    row["text_for_tfidf_clean"] = clean_text_for_tfidf(raw_text_for_tfidf)

    row["readme_raw_len"] = len(readme_raw)
    row["description_raw_len"] = len(desc_raw)
    row["topics_raw_len"] = len(topics_raw)
    row["combined_raw_len"] = len(combined_raw)
    row["text_for_tfidf_raw_len"] = len(raw_text_for_tfidf)

    row["readme_clean_len"] = len(row["readme_text_clean"])
    row["description_clean_len"] = len(row["description_text_clean"])
    row["topics_clean_len"] = len(row["topics_text_clean"])
    row["combined_clean_len"] = len(row["combined_text_clean"])
    row["text_for_tfidf_clean_len"] = len(row["text_for_tfidf_clean"])

    for k, v in feats.items():
        row[k] = to_number_if_possible(v)

    for k, v in gen.items():
        row[f"genmeta__{k}"] = v
    for k, v in meta.items():
        row[f"meta__{k}"] = v

    return row


def extract_negative_row(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    feats = item.get(FEATURES_COL, {}) or {}
    raw_text_for_tfidf, text_source_name = choose_best_text(item)

    readme_raw = safe_text(item.get("readme_text"))
    desc_raw = safe_text(item.get("description_text"))
    topics_raw = safe_text(item.get("topics_text"))
    combined_raw = safe_text(item.get("combined_text"))

    row = {
        "sample_id": f"neg_{idx:05d}",
        "label": int(item.get("label", 0)),
        "sample_source": "raw_negative",
        "provenance_type": "real_negative",
        "is_real_positive": 0,
        "is_generated_positive": 0,
        "is_real_negative": 1,
        "family_id": f"NEG_{idx:05d}",
        "group_id": f"NEG::{idx:05d}",
        "repo_full_name": feats.get("repo_full_name") or safe_text(item.get("repo_full_name")) or f"neg_repo_{idx:05d}",
        "source_repo_full_name": None,
        "source_family": None,
        "generated_from": None,
        "raw_file_name": feats.get("raw_file_name") or item.get("raw_file_name"),
        "source_raw_file_name": None,
        "augmentation_method": None,
        "meta_is_synthetic": np.nan,
        "readme_text_raw": readme_raw,
        "description_text_raw": desc_raw,
        "topics_text_raw": topics_raw,
        "combined_text_raw": combined_raw,
        "text_for_tfidf_raw": raw_text_for_tfidf,
        "text_source_name": text_source_name,
    }

    row["readme_text_clean"] = clean_text_for_tfidf(readme_raw)
    row["description_text_clean"] = clean_text_for_tfidf(desc_raw)
    row["topics_text_clean"] = clean_text_for_tfidf(topics_raw)
    row["combined_text_clean"] = clean_text_for_tfidf(combined_raw)
    row["text_for_tfidf_clean"] = clean_text_for_tfidf(raw_text_for_tfidf)

    row["readme_raw_len"] = len(readme_raw)
    row["description_raw_len"] = len(desc_raw)
    row["topics_raw_len"] = len(topics_raw)
    row["combined_raw_len"] = len(combined_raw)
    row["text_for_tfidf_raw_len"] = len(raw_text_for_tfidf)

    row["readme_clean_len"] = len(row["readme_text_clean"])
    row["description_clean_len"] = len(row["description_text_clean"])
    row["topics_clean_len"] = len(row["topics_text_clean"])
    row["combined_clean_len"] = len(row["combined_text_clean"])
    row["text_for_tfidf_clean_len"] = len(row["text_for_tfidf_clean"])

    for k, v in feats.items():
        row[k] = to_number_if_possible(v)

    return row


# =========================
# 主流程
# =========================
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    pos_path = resolve_first_existing(POS_CANDIDATES)
    neg_path = resolve_first_existing(NEG_CANDIDATES)

    if not pos_path:
        raise FileNotFoundError("未找到正样本文件。请把 augmented_300_with_meta.json 放在脚本同目录，或修改 POS_CANDIDATES。")
    if not neg_path:
        raise FileNotFoundError("未找到负样本文件。请把 cleaned 后的负样本 JSON 放在脚本同目录，或修改 NEG_CANDIDATES。")

    pos_data = load_json(pos_path)
    neg_data = load_json(neg_path)

    pos_rows = [extract_positive_row(item, i) for i, item in enumerate(pos_data)]
    neg_rows = [extract_negative_row(item, i) for i, item in enumerate(neg_data)]
    df = pd.DataFrame(pos_rows + neg_rows)

    meta_cols = {
        "sample_id", "label", "sample_source", "provenance_type",
        "is_real_positive", "is_generated_positive", "is_real_negative",
        "family_id", "group_id", "repo_full_name", "source_repo_full_name",
        "source_family", "generated_from", "raw_file_name", "source_raw_file_name",
        "augmentation_method", "meta_is_synthetic",
        "readme_text_raw", "description_text_raw", "topics_text_raw", "combined_text_raw",
        "text_for_tfidf_raw", "text_source_name",
        "readme_text_clean", "description_text_clean", "topics_text_clean", "combined_text_clean",
        "text_for_tfidf_clean",
    }

    for c in df.columns:
        if c in meta_cols or c.startswith(("genmeta__", "meta__")):
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        non_null_mask = df[c].notna()
        if non_null_mask.sum() == 0 or converted[non_null_mask].notna().all():
            df[c] = converted

    numeric_feature_cols = []
    for c in df.columns:
        if c in meta_cols or c.startswith(("genmeta__", "meta__")):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_feature_cols.append(c)
    numeric_feature_cols = sorted(numeric_feature_cols)

    train_meta_cols = [
        "sample_id", "label", "sample_source", "provenance_type",
        "is_real_positive", "is_generated_positive", "is_real_negative",
        "family_id", "group_id", "repo_full_name", "source_repo_full_name",
        "source_family", "generated_from", "raw_file_name", "source_raw_file_name",
        "augmentation_method", "meta_is_synthetic",
        "text_source_name",
        "text_for_tfidf_raw", "text_for_tfidf_clean",
        "readme_text_clean", "description_text_clean", "topics_text_clean", "combined_text_clean",
        "text_for_tfidf_raw_len", "text_for_tfidf_clean_len",
    ]
    train_meta_cols = [c for c in train_meta_cols if c in df.columns]
    train_ready_df = df[train_meta_cols + numeric_feature_cols].copy()

    full_csv = os.path.join(OUTPUT_DIR, "family_dataset_full.csv")
    train_ready_csv = os.path.join(OUTPUT_DIR, "family_dataset_train_ready.csv")
    family_summary_csv = os.path.join(OUTPUT_DIR, "positive_family_summary.csv")
    summary_json = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    numeric_cols_json = os.path.join(OUTPUT_DIR, "numeric_feature_columns.json")
    text_cols_json = os.path.join(OUTPUT_DIR, "text_column_summary.json")

    df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    train_ready_df.to_csv(train_ready_csv, index=False, encoding="utf-8-sig")

    pos_df = df[df["label"] == 1].copy()
    family_summary = (
        pos_df.groupby(["family_id", "provenance_type"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    family_pivot = family_summary.pivot_table(
        index="family_id", columns="provenance_type", values="count", fill_value=0
    ).reset_index()
    family_pivot.to_csv(family_summary_csv, index=False, encoding="utf-8-sig")

    dataset_summary = {
        "input_positive_file": pos_path,
        "input_negative_file": neg_path,
        "n_total": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "n_real_positive": int(df["is_real_positive"].sum()),
        "n_generated_positive": int(df["is_generated_positive"].sum()),
        "n_positive_families": int(pos_df["family_id"].nunique()) if len(pos_df) > 0 else 0,
        "numeric_feature_count": int(len(numeric_feature_cols)),
        "tfidf_main_text_column": "text_for_tfidf_clean",
        "tfidf_text_priority": ["combined_text", "fallback_merge(readme+description+topics)", "empty"],
        "negative_family_strategy": "each negative sample gets a unique family_id/group_id",
        "stopword_filtering": True,
        "custom_stopwords": sorted(list(CUSTOM_STOPWORDS)),
        "output_full_csv": full_csv,
        "output_train_ready_csv": train_ready_csv,
    }

    text_column_summary = {
        "recommended_tfidf_column": "text_for_tfidf_clean",
        "available_clean_text_columns": [
            "readme_text_clean",
            "description_text_clean",
            "topics_text_clean",
            "combined_text_clean",
            "text_for_tfidf_clean",
        ],
        "family_rule": {
            "positive": "优先读取 meta.family_label；若缺失则回退到 family_id/family/source_repo_full_name 等字段。",
            "negative": "负样本没有 family，统一使用 NEG_样本编号 作为 family_id，group_id 也是唯一值。",
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
    with open(numeric_cols_json, "w", encoding="utf-8") as f:
        json.dump(numeric_feature_cols, f, ensure_ascii=False, indent=2)
    with open(text_cols_json, "w", encoding="utf-8") as f:
        json.dump(text_column_summary, f, ensure_ascii=False, indent=2)

    print("数据准备完成（已兼容 augmented_300_with_meta.json 的 family 与 meta 格式）")
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    print("输出目录:", OUTPUT_DIR)
    print("- family_dataset_full.csv")
    print("- family_dataset_train_ready.csv")
    print("- positive_family_summary.csv")
    print("- numeric_feature_columns.json")
    print("- text_column_summary.json")
    print("- dataset_summary.json")


if __name__ == "__main__":
    main()
