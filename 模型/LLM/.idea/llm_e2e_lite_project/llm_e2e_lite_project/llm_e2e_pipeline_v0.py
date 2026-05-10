import argparse
import copy
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_by_path(obj: Dict[str, Any], path: str, default=None):
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def safe_str(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", safe_str(x)).strip()


# ------------------------------
# Data loading
# ------------------------------
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")
    return data


def infer_feature_names(samples: List[Dict[str, Any]]) -> List[str]:
    feat_names = set()
    for s in samples:
        feat = s.get("features", {})
        if not isinstance(feat, dict):
            continue
        for k, v in feat.items():
            if is_num(v) or isinstance(v, bool):
                feat_names.add(k)
    return sorted(feat_names)


def build_record(obj: Dict[str, Any], idx: int, label: int, cfg: Dict, feature_names: List[str], is_positive: bool) -> Dict[str, Any]:
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
            raise ValueError(f"Positive sample {idx} missing family_label.")
        rec["family"] = str(family)
        rec["group_id"] = f"POS::{family}"
        rec["source_type"] = "synthetic" if is_syn == 1 else "original"
        rec["source_raw_file_name"] = src_raw
        rec["source_repo_full_name"] = src_repo
    else:
        neg_group = safe_str(get_by_path(obj, cfg["fields"]["negative_group_path"], ""))
        if not neg_group:
            neg_group = rec["repo_full_name"] or rec["sample_id"]
        rec["family"] = f"NEG::{neg_group}"
        rec["group_id"] = f"NEG::{neg_group}"
        rec["source_type"] = "real_negative"
        rec["source_raw_file_name"] = neg_group
        rec["source_repo_full_name"] = rec["repo_full_name"]

    feat = obj.get("features", {})
    feat = feat if isinstance(feat, dict) else {}
    for fn in feature_names:
        v = feat.get(fn, np.nan)
        if isinstance(v, bool):
            v = int(v)
        rec[fn] = v
    return rec


def load_dataset(cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    pos = load_json_list(cfg["data"]["positive_path"])
    neg = load_json_list(cfg["data"]["negative_path"])
    feature_names = infer_feature_names(pos + neg)

    rows = []
    for i, obj in enumerate(pos):
        rows.append(build_record(obj, i, 1, cfg, feature_names, True))
    for i, obj in enumerate(neg):
        rows.append(build_record(obj, i, 0, cfg, feature_names, False))

    df = pd.DataFrame(rows)

    # Positive leakage guard
    pos_df = df[df["label"] == 1].copy()
    g = pos_df.groupby("source_raw_file_name")["family"].nunique()
    bad = g[g > 1]
    if len(bad) > 0:
        raise ValueError(f"Leakage risk: one positive source_raw_file_name maps to multiple families: {bad.index.tolist()[:10]}")

    return df, feature_names


# ------------------------------
# Splits
# ------------------------------
def stable_bucket(text: str, n_buckets: int = 5, seed: int = 42) -> int:
    raw = f"{seed}::{text}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest(), 16) % n_buckets


def build_five_splits(df: pd.DataFrame, seed: int = 42) -> List[Dict[str, Any]]:
    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()
    pos_families = sorted(pos_df["family"].unique().tolist())
    if len(pos_families) != 5:
        raise ValueError(f"Expected 5 positive families, got {len(pos_families)}")

    neg_groups = sorted(neg_df["group_id"].unique().tolist())
    neg_group_bucket = {g: stable_bucket(g, 5, seed) for g in neg_groups}

    splits = []
    for i in range(5):
        pos_test = pos_families[i]
        pos_val = pos_families[(i + 1) % 5]
        pos_train = [x for x in pos_families if x not in {pos_test, pos_val}]

        train_idx = df[
            ((df["label"] == 1) & (df["family"].isin(pos_train))) |
            ((df["label"] == 0) & (~df["group_id"].map(neg_group_bucket).isin([i, (i + 1) % 5])))
        ].index.tolist()

        val_idx = df[
            ((df["label"] == 1) & (df["family"] == pos_val)) |
            ((df["label"] == 0) & (df["group_id"].map(neg_group_bucket) == ((i + 1) % 5)))
        ].index.tolist()

        test_idx = df[
            ((df["label"] == 1) & (df["family"] == pos_test)) |
            ((df["label"] == 0) & (df["group_id"].map(neg_group_bucket) == i))
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


def subset_stats(sub_df: pd.DataFrame) -> Dict[str, Any]:
    pos = sub_df[sub_df["label"] == 1]
    return {
        "n": int(len(sub_df)),
        "n_pos": int((sub_df["label"] == 1).sum()),
        "n_neg": int((sub_df["label"] == 0).sum()),
        "pos_rate": float((sub_df["label"] == 1).mean()) if len(sub_df) else 0.0,
        "n_original_pos": int((pos["source_type"] == "original").sum()),
        "n_synthetic_pos": int((pos["source_type"] == "synthetic").sum()),
        "families": sorted(sub_df["family"].unique().tolist())[:50],
        "n_groups": int(sub_df["group_id"].nunique()),
    }


def summarize_splits(df: pd.DataFrame, splits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# ------------------------------
# Prompt building: USE ALL structured features + ALL text fields
# ------------------------------
def serialize_all_structured_features(row: pd.Series, feature_names: List[str]) -> str:
    lines = []
    for fn in feature_names:
        if fn not in row.index:
            continue
        v = row[fn]
        if pd.isna(v):
            continue
        if isinstance(v, float):
            v = round(float(v), 8)
        lines.append(f"{fn} = {v}")
    return "\n".join(lines)


def take_with_budget(text: str, budget: int) -> str:
    text = clean_text(text)
    if budget <= 0:
        return ""
    return text[:budget]


def allocate_text_budgets(row: pd.Series, cfg: Dict) -> Dict[str, int]:
    # Ensure every text field is represented.
    total_budget = int(cfg["input_build"]["total_text_char_budget"])
    minimums = cfg["input_build"]["text_min_budgets"]

    fields = {
        "repo_full_name": clean_text(row.get("repo_full_name", "")),
        "description_text": clean_text(row.get("description_text", "")),
        "topics_text": clean_text(row.get("topics_text", "")),
        "readme_text": clean_text(row.get("readme_text", "")),
        "combined_text": clean_text(row.get("combined_text", "")),
    }

    budgets = {k: min(minimums.get(k, 0), len(v)) for k, v in fields.items()}
    used = sum(budgets.values())
    remain = max(0, total_budget - used)

    lengths = {k: max(0, len(v) - budgets[k]) for k, v in fields.items()}
    total_remain_len = sum(lengths.values())
    if total_remain_len > 0 and remain > 0:
        for k in fields.keys():
            add = int(remain * (lengths[k] / total_remain_len))
            budgets[k] += add

    # Put leftover on README first, then combined_text.
    used2 = sum(budgets.values())
    left = max(0, total_budget - used2)
    for k in ["readme_text", "combined_text", "description_text", "topics_text", "repo_full_name"]:
        if left <= 0:
            break
        spare = max(0, len(fields[k]) - budgets[k])
        gain = min(left, spare)
        budgets[k] += gain
        left -= gain
    return budgets


def build_prompt(row: pd.Series, feature_names: List[str], cfg: Dict) -> str:
    budgets = allocate_text_budgets(row, cfg)
    repo_name = take_with_budget(row.get("repo_full_name", ""), budgets["repo_full_name"])
    desc = take_with_budget(row.get("description_text", ""), budgets["description_text"])
    topics = take_with_budget(row.get("topics_text", ""), budgets["topics_text"])
    readme = take_with_budget(row.get("readme_text", ""), budgets["readme_text"])
    combined = take_with_budget(row.get("combined_text", ""), budgets["combined_text"])
    structured = serialize_all_structured_features(row, feature_names)

    prompt = f"""You are a security analyst for blockchain GitHub repositories and accounts.
Determine whether the following sample is suspicious.

[Repository Name]
{repo_name}

[Description]
{desc}

[Topics]
{topics}

[README]
{readme}

[Combined Text]
{combined}

[ALL Structured Features]
{structured}

[Label Definition]
0 = benign / normal repository
1 = suspicious repository / account

Return the classification based on all evidence."""
    return prompt.strip()


def build_prompts(df: pd.DataFrame, feature_names: List[str], cfg: Dict) -> List[str]:
    return [build_prompt(row, feature_names, cfg) for _, row in df.iterrows()]


# ------------------------------
# Metrics & threshold
# ------------------------------
def calc_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def search_threshold(y_true, y_prob, start: float, end: float, step: float) -> Tuple[float, pd.DataFrame]:
    rows = []
    ths = np.arange(start, end + 1e-12, step)
    for th in ths:
        rows.append(calc_metrics(y_true, y_prob, float(th)))
    df = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).reset_index(drop=True)
    return float(df.iloc[0]["threshold"]), df


# ------------------------------
# Dataset & trainer
# ------------------------------
class RepoClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_type="weighted_ce", pos_weight=1.0, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor([1.0, float(self.pos_weight)], dtype=logits.dtype, device=logits.device)

        if self.loss_type == "focal":
            ce_each = F.cross_entropy(logits, labels, weight=weights, reduction="none")
            pt = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            focal = ((1 - pt).pow(float(self.focal_gamma)) * ce_each).mean()
            loss = focal
        else:
            loss = F.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def load_model_and_tokenizer(cfg: Dict):
    model_name = cfg["model"]["model_name"]
    use_4bit = bool(cfg["model"]["use_4bit"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg["model"]["lora_r"]),
        lora_alpha=int(cfg["model"]["lora_alpha"]),
        lora_dropout=float(cfg["model"]["lora_dropout"]),
        target_modules=list(cfg["model"]["lora_target_modules"]),
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def fit_one_fold(train_texts: List[str], train_labels: List[int], val_texts: List[str], val_labels: List[int], cfg: Dict, out_dir: str):
    set_seed(int(cfg["seed"]))
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds = RepoClsDataset(train_texts, train_labels, tokenizer, int(cfg["model"]["max_length"]))
    val_ds = RepoClsDataset(val_texts, val_labels, tokenizer, int(cfg["model"]["max_length"]))
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=int(cfg["model"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["model"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["model"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["model"]["learning_rate"]),
        num_train_epochs=float(cfg["model"]["num_train_epochs"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
        warmup_ratio=float(cfg["model"]["warmup_ratio"]),
        lr_scheduler_type=str(cfg["model"]["lr_scheduler_type"]),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        loss_type=str(cfg["loss"]["loss_type"]),
        pos_weight=float(cfg["loss"]["positive_class_weight"]),
        focal_gamma=float(cfg["loss"]["focal_gamma"]),
    )
    trainer.train()
    return trainer, tokenizer


def predict_proba(trainer: Trainer, tokenizer, texts: List[str], cfg: Dict, labels: List[int] = None) -> np.ndarray:
    if labels is None:
        labels = [0] * len(texts)
    ds = RepoClsDataset(texts, labels, tokenizer, int(cfg["model"]["max_length"]))
    out = trainer.predict(ds)
    logits = out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    return probs


# ------------------------------
# Hyperparameter tuning
# ------------------------------
def make_trials(search_cfg: Dict) -> List[Dict[str, Any]]:
    keys = [
        "model_name",
        "max_length",
        "learning_rate",
        "num_train_epochs",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "gradient_accumulation_steps",
        "loss_type",
        "positive_class_weight",
        "focal_gamma",
    ]
    values = [search_cfg[k] for k in keys]
    all_trials = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    rng = np.random.RandomState(int(search_cfg.get("random_state", 42)))
    max_trials = int(search_cfg.get("max_trials", len(all_trials)))
    if len(all_trials) > max_trials:
        idx = rng.choice(len(all_trials), size=max_trials, replace=False)
        all_trials = [all_trials[i] for i in idx]
    return all_trials


def apply_trial(base_cfg: Dict, trial: Dict) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in trial.items():
        if k in cfg["model"]:
            cfg["model"][k] = v
        elif k in cfg["loss"]:
            cfg["loss"][k] = v
    return cfg


def tune(df: pd.DataFrame, feature_names: List[str], splits: List[Dict[str, Any]], cfg: Dict):
    trials = make_trials(cfg["search"])
    rows = []

    for tid, trial in enumerate(trials):
        logging.info(f"Trial {tid + 1}/{len(trials)}: {trial}")
        trial_cfg = apply_trial(cfg, trial)
        fold_scores = []

        for sp in splits:
            fold = sp["fold"]
            fold_dir = os.path.join(cfg["output_dir"], "search_runs", f"trial_{tid}", f"fold_{fold}")
            train_df = df.loc[sp["train_idx"]].reset_index(drop=True)
            val_df = df.loc[sp["val_idx"]].reset_index(drop=True)
            train_prompts = build_prompts(train_df, feature_names, trial_cfg)
            val_prompts = build_prompts(val_df, feature_names, trial_cfg)

            trainer, tokenizer = fit_one_fold(
                train_prompts,
                train_df["label"].tolist(),
                val_prompts,
                val_df["label"].tolist(),
                trial_cfg,
                fold_dir,
            )
            val_prob = predict_proba(trainer, tokenizer, val_prompts, trial_cfg, val_df["label"].tolist())
            m = calc_metrics(val_df["label"].values, val_prob, threshold=0.5)
            fold_scores.append(m)

            # Aggressive cleanup between folds on small GPUs
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows.append({
            "trial_id": tid,
            **trial,
            "mean_val_f1": float(np.mean([x["f1"] for x in fold_scores])),
            "std_val_f1": float(np.std([x["f1"] for x in fold_scores])),
            "mean_val_recall": float(np.mean([x["recall"] for x in fold_scores])),
            "mean_val_precision": float(np.mean([x["precision"] for x in fold_scores])),
        })

    cv_df = pd.DataFrame(rows).sort_values(
        ["mean_val_f1", "std_val_f1", "mean_val_recall", "mean_val_precision"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)
    best = cv_df.iloc[0].to_dict()
    return best, cv_df


# ------------------------------
# Main pipeline
# ------------------------------
def run_pipeline(cfg: Dict, do_tune: bool = True):
    setup_logger(cfg["output_dir"])
    set_seed(int(cfg["seed"]))

    logging.info("Loading dataset...")
    df, feature_names = load_dataset(cfg)
    logging.info(f"Loaded {len(df)} samples with {len(feature_names)} structured features.")

    splits = build_five_splits(df, seed=int(cfg["seed"]))
    split_stats = summarize_splits(df, splits)
    save_json(os.path.join(cfg["output_dir"], "split_stats.json"), split_stats)

    final_cfg = copy.deepcopy(cfg)
    if do_tune:
        logging.info("Starting hyperparameter tuning...")
        best_trial, cv_df = tune(df, feature_names, splits, cfg)
        cv_df.to_csv(os.path.join(cfg["output_dir"], "cv_results.csv"), index=False)
        best_params = {k: v for k, v in best_trial.items() if k not in ["trial_id", "mean_val_f1", "std_val_f1", "mean_val_recall", "mean_val_precision"]}
        save_json(os.path.join(cfg["output_dir"], "best_params.json"), best_params)
        for k, v in best_params.items():
            if k in final_cfg["model"]:
                final_cfg["model"][k] = v
            elif k in final_cfg["loss"]:
                final_cfg["loss"][k] = v
    else:
        logging.info("Skip tuning; use config as final params.")
        save_json(os.path.join(cfg["output_dir"], "best_params.json"), {
            **cfg["model"], **cfg["loss"]
        })

    # 5-fold final runs
    logging.info("Running 5-fold training/evaluation...")
    val_rows, test_rows = [], []
    for sp in splits:
        fold = sp["fold"]
        fold_dir = os.path.join(final_cfg["output_dir"], f"fold_{fold}")
        train_df = df.loc[sp["train_idx"]].reset_index(drop=True)
        val_df = df.loc[sp["val_idx"]].reset_index(drop=True)
        test_df = df.loc[sp["test_idx"]].reset_index(drop=True)

        train_prompts = build_prompts(train_df, feature_names, final_cfg)
        val_prompts = build_prompts(val_df, feature_names, final_cfg)
        test_prompts = build_prompts(test_df, feature_names, final_cfg)

        trainer, tokenizer = fit_one_fold(
            train_prompts, train_df["label"].tolist(),
            val_prompts, val_df["label"].tolist(),
            final_cfg, fold_dir,
        )
        val_prob = predict_proba(trainer, tokenizer, val_prompts, final_cfg, val_df["label"].tolist())
        test_prob = predict_proba(trainer, tokenizer, test_prompts, final_cfg, test_df["label"].tolist())

        v = val_df[["sample_id", "label", "family", "repo_full_name", "source_type"]].copy()
        v["fold"] = fold
        v["split"] = "val"
        v["prob"] = val_prob
        val_rows.append(v)

        t = test_df[["sample_id", "label", "family", "repo_full_name", "source_type"]].copy()
        t["fold"] = fold
        t["split"] = "test"
        t["prob"] = test_prob
        test_rows.append(t)

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_all = pd.concat(val_rows, axis=0).reset_index(drop=True)
    test_all = pd.concat(test_rows, axis=0).reset_index(drop=True)
    val_all.to_csv(os.path.join(final_cfg["output_dir"], "oof_val_predictions.csv"), index=False)
    test_all.to_csv(os.path.join(final_cfg["output_dir"], "all_test_predictions.csv"), index=False)

    best_th, th_df = search_threshold(
        val_all["label"].values,
        val_all["prob"].values,
        float(final_cfg["threshold"]["start"]),
        float(final_cfg["threshold"]["end"]),
        float(final_cfg["threshold"]["step"]),
    )
    th_df.to_csv(os.path.join(final_cfg["output_dir"], "threshold_search.csv"), index=False)
    save_json(os.path.join(final_cfg["output_dir"], "selected_threshold.json"), {"best_threshold": best_th})

    fold_metrics = []
    for fold in sorted(test_all["fold"].unique()):
        sub = test_all[test_all["fold"] == fold]
        m = calc_metrics(sub["label"].values, sub["prob"].values, threshold=best_th)
        m["fold"] = int(fold)
        fold_metrics.append(m)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(os.path.join(final_cfg["output_dir"], "fold_test_metrics.csv"), index=False)

    pos_test = test_all[test_all["label"] == 1].copy()
    subset_report = {
        "all": calc_metrics(test_all["label"].values, test_all["prob"].values, best_th)
    }
    if len(pos_test):
        orig = pos_test[pos_test["source_type"] == "original"]
        syn = pos_test[pos_test["source_type"] == "synthetic"]
        if len(orig):
            subset_report["original_positive_subset"] = calc_metrics(orig["label"].values, orig["prob"].values, best_th)
        if len(syn):
            subset_report["synthetic_positive_subset"] = calc_metrics(syn["label"].values, syn["prob"].values, best_th)

    final_metrics = {
        "overall_test": calc_metrics(test_all["label"].values, test_all["prob"].values, best_th),
        "subset_report": subset_report,
        "fold_mean_std": {
            "precision_mean": float(fold_metrics_df["precision"].mean()),
            "precision_std": float(fold_metrics_df["precision"].std()),
            "recall_mean": float(fold_metrics_df["recall"].mean()),
            "recall_std": float(fold_metrics_df["recall"].std()),
            "f1_mean": float(fold_metrics_df["f1"].mean()),
            "f1_std": float(fold_metrics_df["f1"].std()),
            "accuracy_mean": float(fold_metrics_df["accuracy"].mean()),
            "accuracy_std": float(fold_metrics_df["accuracy"].std()),
            "roc_auc_mean": float(fold_metrics_df["roc_auc"].mean()) if fold_metrics_df["roc_auc"].notna().any() else None,
            "roc_auc_std": float(fold_metrics_df["roc_auc"].std()) if fold_metrics_df["roc_auc"].notna().any() else None,
            "pr_auc_mean": float(fold_metrics_df["pr_auc"].mean()) if fold_metrics_df["pr_auc"].notna().any() else None,
            "pr_auc_std": float(fold_metrics_df["pr_auc"].std()) if fold_metrics_df["pr_auc"].notna().any() else None,
        }
    }
    save_json(os.path.join(final_cfg["output_dir"], "final_metrics.json"), final_metrics)

    suspicious_topk = test_all.sort_values("prob", ascending=False).head(100).copy()
    suspicious_topk["pred_label"] = (suspicious_topk["prob"] >= best_th).astype(int)
    suspicious_topk.to_csv(os.path.join(final_cfg["output_dir"], "suspicious_topk.csv"), index=False)

    err = test_all.copy()
    err["pred_label"] = (err["prob"] >= best_th).astype(int)
    err["error_type"] = "correct"
    err.loc[(err["label"] == 1) & (err["pred_label"] == 0), "error_type"] = "false_negative"
    err.loc[(err["label"] == 0) & (err["pred_label"] == 1), "error_type"] = "false_positive"
    err.to_csv(os.path.join(final_cfg["output_dir"], "error_analysis.csv"), index=False)

    save_json(os.path.join(final_cfg["output_dir"], "final_used_config.json"), final_cfg)
    logging.info("Done.")


# ------------------------------
# CLI
# ------------------------------
def inspect(cfg: Dict):
    df, feature_names = load_dataset(cfg)
    out = {
        "n_samples": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "n_structured_features": int(len(feature_names)),
        "structured_feature_names": feature_names,
        "text_fields_used": ["repo_full_name", "description_text", "topics_text", "readme_text", "combined_text"],
        "positive_families": sorted(df[df["label"] == 1]["family"].unique().tolist()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["inspect", "tune_and_run", "run_only"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.mode == "inspect":
        inspect(cfg)
    elif args.mode == "tune_and_run":
        run_pipeline(cfg, do_tune=True)
    elif args.mode == "run_only":
        run_pipeline(cfg, do_tune=False)


if __name__ == "__main__":
    main()
