import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def get_by_path(obj: Dict[str, Any], path: str, default=None):
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list.")
    return data


def infer_feature_names(samples: List[Dict]) -> List[str]:
    names = set()
    for s in samples:
        feat = s.get("features", {})
        if not isinstance(feat, dict):
            continue
        for k, v in feat.items():
            if is_num(v) or isinstance(v, bool):
                names.add(k)
    return sorted(names)


def build_one_record(
        obj: Dict,
        label: int,
        idx: int,
        cfg: Dict,
        feature_names: List[str],
        is_positive: bool,
):
    rec = {
        "sample_id": f"{'pos' if is_positive else 'neg'}_{idx}",
        "label": int(label),
        "readme_text": safe_str(obj.get("readme_text", "")),
        "description_text": safe_str(obj.get("description_text", "")),
        "topics_text": safe_str(obj.get("topics_text", "")),
        "combined_text": safe_str(obj.get("combined_text", "")),
        "repo_full_name": safe_str(get_by_path(obj, cfg["fields"]["repo_full_name_path"], "")),
    }

    if is_positive:
        family = get_by_path(obj, cfg["fields"]["positive_family_path"], None)
        is_syn = int(get_by_path(obj, cfg["fields"]["positive_is_synthetic_path"], 0))
        src_raw = safe_str(get_by_path(obj, cfg["fields"]["positive_source_raw_path"], ""))
        src_repo = safe_str(get_by_path(obj, cfg["fields"]["positive_source_repo_path"], ""))
        if family is None:
            raise ValueError(f"Positive sample {idx} missing family.")
        rec["family"] = str(family)
        rec["group_id"] = f"POS::{family}"
        rec["source_type"] = "synthetic" if is_syn == 1 else "original"
        rec["source_raw_file_name"] = src_raw
        rec["source_repo_full_name"] = src_repo
    else:
        raw_name = safe_str(get_by_path(obj, cfg["fields"]["negative_group_path"], ""))
        if not raw_name:
            raw_name = rec["repo_full_name"] or rec["sample_id"]
        rec["family"] = f"NEG::{raw_name}"
        rec["group_id"] = f"NEG::{raw_name}"
        rec["source_type"] = "real_negative"
        rec["source_raw_file_name"] = raw_name
        rec["source_repo_full_name"] = rec["repo_full_name"]

    feats = obj.get("features", {})
    feats = feats if isinstance(feats, dict) else {}
    for name in feature_names:
        v = feats.get(name, np.nan)
        if isinstance(v, bool):
            v = int(v)
        rec[name] = v

    return rec


def load_dataset(cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    pos = load_json_list(cfg["data"]["positive_path"])
    neg = load_json_list(cfg["data"]["negative_path"])
    feature_names = infer_feature_names(pos + neg)

    rows = []
    for i, obj in enumerate(pos):
        rows.append(build_one_record(obj, 1, i, cfg, feature_names, True))
    for i, obj in enumerate(neg):
        rows.append(build_one_record(obj, 0, i, cfg, feature_names, False))

    df = pd.DataFrame(rows)

    # 检查正样本来源是否跨 family
    pos_df = df[df["label"] == 1].copy()
    g = pos_df.groupby("source_raw_file_name")["family"].nunique()
    bad = g[g > 1]
    if len(bad) > 0:
        raise ValueError(f"Leakage risk: source_raw_file_name maps to multiple families: {bad.index.tolist()[:10]}")

    return df, feature_names