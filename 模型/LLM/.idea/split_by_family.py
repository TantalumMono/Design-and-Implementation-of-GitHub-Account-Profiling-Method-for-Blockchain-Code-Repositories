import hashlib
import json
import os
from typing import Dict, List

import pandas as pd


def stable_bucket(text: str, n_buckets: int = 5, seed: int = 42) -> int:
    s = f"{seed}::{text}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h, 16) % n_buckets


def build_five_splits(df: pd.DataFrame, seed: int = 42) -> List[Dict]:
    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()

    pos_families = sorted(pos_df["family"].unique().tolist())
    if len(pos_families) != 5:
        raise ValueError(f"Expected exactly 5 positive families, got {len(pos_families)}")

    neg_groups = sorted(neg_df["group_id"].unique().tolist())
    neg_bucket = {g: stable_bucket(g, 5, seed) for g in neg_groups}

    splits = []
    for i in range(5):
        pos_test = pos_families[i]
        pos_val = pos_families[(i + 1) % 5]
        pos_train = [x for x in pos_families if x not in {pos_test, pos_val}]

        train_idx = df[
            ((df["label"] == 1) & (df["family"].isin(pos_train))) |
            ((df["label"] == 0) & (df["group_id"].map(neg_bucket) >= 0) &
             (~df["group_id"].map(neg_bucket).isin([i, (i + 1) % 5])))
            ].index.tolist()

        val_idx = df[
            ((df["label"] == 1) & (df["family"] == pos_val)) |
            ((df["label"] == 0) & (df["group_id"].map(neg_bucket) == ((i + 1) % 5)))
            ].index.tolist()

        test_idx = df[
            ((df["label"] == 1) & (df["family"] == pos_test)) |
            ((df["label"] == 0) & (df["group_id"].map(neg_bucket) == i))
            ].index.tolist()

        splits.append({
            "fold": i,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "pos_train_families": pos_train,
            "pos_val_family": pos_val,
            "pos_test_family": pos_test,
        })
    return splits


def subset_stats(sub_df: pd.DataFrame):
    pos = sub_df[sub_df["label"] == 1]
    return {
        "n": int(len(sub_df)),
        "n_pos": int((sub_df["label"] == 1).sum()),
        "n_neg": int((sub_df["label"] == 0).sum()),
        "pos_rate": float((sub_df["label"] == 1).mean()) if len(sub_df) > 0 else 0.0,
        "n_original_pos": int((pos["source_type"] == "original").sum()),
        "n_synthetic_pos": int((pos["source_type"] == "synthetic").sum()),
        "families": sorted(sub_df["family"].unique().tolist())[:50],
        "n_groups": int(sub_df["group_id"].nunique()),
    }


def summarize_splits(df: pd.DataFrame, splits: List[Dict]):
    rows = []
    for sp in splits:
        rows.append({
            "fold": sp["fold"],
            "train": subset_stats(df.loc[sp["train_idx"]]),
            "val": subset_stats(df.loc[sp["val_idx"]]),
            "test": subset_stats(df.loc[sp["test_idx"]]),
            "pos_train_families": sp["pos_train_families"],
            "pos_val_family": sp["pos_val_family"],
            "pos_test_family": sp["pos_test_family"],
        })
    return rows


def save_split_stats(path: str, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)