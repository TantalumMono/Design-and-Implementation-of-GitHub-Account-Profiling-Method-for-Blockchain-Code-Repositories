
# -*- coding: utf-8 -*-
import os

# 必须放在 import sentence_transformers / transformers 之前
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home"
os.environ["HF_HUB_CACHE"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\hf_home\hub"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\st_cache"

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "dl_st_mlp_temp_scaled_family_validation_outputs"
CACHE_DIR = SCRIPT_DIR / "dl_st_embedding_cache_offline"

# 改成你本地实际模型目录
ST_MODEL_PATH = Path(r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\models\all-MiniLM-L6-v2")

RANDOM_STATE = 42
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST = True

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

ST_BATCH_SIZE = 64
REAL_POS_WEIGHT = 4.5
GEN_POS_WEIGHT = 1.0
NEG_WEIGHT = 1.25

MAX_EPOCHS = 40
PATIENCE = 6
MIN_EPOCHS = 5
CALIB_RATIO = 0.15
BETA = 2.0

# 固定参数：沿用前一版 fixed params
FIXED_PARAMS = {
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.25,
    "lr": 1e-3,
    "weight_decay": 2e-4,
    "batch_size": 32,
    "pos_weight_scale": 1.10,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def calc_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def select_best_threshold_f1_priority(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0):
    thresholds = np.unique(np.round(y_proba, 10))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    best = None
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f_beta = calc_fbeta(p, r, beta=beta)

        cand = {
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "f_beta": float(f_beta),
        }

        if best is None:
            best = cand
            continue

        better = (
            (cand["f1"] > best["f1"])
            or (np.isclose(cand["f1"], best["f1"]) and cand["recall"] > best["recall"])
            or (np.isclose(cand["f1"], best["f1"]) and np.isclose(cand["recall"], best["recall"]) and cand["precision"] > best["precision"])
            or (np.isclose(cand["f1"], best["f1"]) and np.isclose(cand["recall"], best["recall"]) and np.isclose(cand["precision"], best["precision"]) and cand["f_beta"] > best["f_beta"])
            or (np.isclose(cand["f1"], best["f1"]) and np.isclose(cand["recall"], best["recall"]) and np.isclose(cand["precision"], best["precision"]) and np.isclose(cand["f_beta"], best["f_beta"]) and cand["threshold"] > best["threshold"])
        )
        if better:
            best = cand

    return best["threshold"], best


def is_leaky_feature_col(col: str) -> bool:
    if col in NON_STRUCTURED_COLS:
        return True
    if any(col.startswith(p) for p in LEAKY_PREFIXES):
        return True
    if any(col.startswith(p) for p in META_LEAKY_PREFIXES):
        return True
    return False


def get_aux_text_series(df: pd.DataFrame) -> pd.Series:
    desc = df.get(DESCRIPTION_TEXT_COL, "").fillna("").astype(str)
    topics = df.get(TOPICS_TEXT_COL, "").fillna("").astype(str)
    return (desc + " " + topics).str.strip()


def make_positive_family_folds(real_pos_df: pd.DataFrame, n_splits: int, seed: int):
    families = np.array(sorted(real_pos_df["family_id"].dropna().astype(str).unique()))
    if len(families) < 2:
        raise ValueError("Need at least 2 real-positive families for grouped CV.")
    n_splits = min(n_splits, len(families))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_fam_idx, te_fam_idx in kf.split(families):
        train_families = set(families[tr_fam_idx])
        test_families = set(families[te_fam_idx])
        folds.append((train_families, test_families))
    return folds


def make_index_folds(indices: np.ndarray, n_splits: int, seed: int):
    if len(indices) < 2:
        raise ValueError("Need at least 2 negative samples for fold splitting.")
    n_splits = min(n_splits, len(indices))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in kf.split(indices):
        folds.append((indices[tr_idx], indices[te_idx]))
    return folds


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {
        "label", "family_id", "is_real_positive", "is_generated_positive",
        README_TEXT_COL, DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int)
    df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int)
    df["family_id"] = df["family_id"].fillna("UNKNOWN").astype(str)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)

    structured_cols = [c for c in df.columns if not is_leaky_feature_col(c)]
    return df, structured_cols


def build_numeric_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, structured_cols: List[str]):
    X_tr = train_df[structured_cols].copy()
    X_te = test_df[structured_cols].copy()

    for col in structured_cols:
        X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
        X_te[col] = pd.to_numeric(X_te[col], errors="coerce")

    fill_values = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_te = X_te.fillna(fill_values).fillna(0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.values.astype(np.float32))
    X_te_scaled = scaler.transform(X_te.values.astype(np.float32))

    artifacts = {"fill_values": fill_values.to_dict(), "structured_cols": structured_cols}
    return X_tr_scaled.astype(np.float32), X_te_scaled.astype(np.float32), artifacts


def encode_texts_cached(texts: List[str], cache_path: Path) -> np.ndarray:
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        return np.load(cache_path)

    if not ST_MODEL_PATH.exists():
        raise FileNotFoundError(f"本地模型目录不存在: {ST_MODEL_PATH}")

    model = SentenceTransformer(
        str(ST_MODEL_PATH),
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME", None),
        local_files_only=True,
        device="cpu",
    )

    emb = model.encode(
        texts,
        batch_size=ST_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    np.save(cache_path, emb)
    return emb


def build_full_cached_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    readme_cache = CACHE_DIR / "readme_embeddings.npy"
    aux_cache = CACHE_DIR / "aux_embeddings.npy"
    readme_emb = encode_texts_cached(df[README_TEXT_COL].fillna("").astype(str).tolist(), readme_cache)
    aux_emb = encode_texts_cached(get_aux_text_series(df).fillna("").astype(str).tolist(), aux_cache)
    return readme_emb, aux_emb


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.full(len(df), NEG_WEIGHT, dtype=np.float32)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = REAL_POS_WEIGHT
    w[gen_mask] = GEN_POS_WEIGHT
    return w


def stratified_train_calib_split(y: np.ndarray, calib_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    def sample_subset(indices):
        if len(indices) <= 1:
            return np.array([], dtype=int)
        n = max(1, int(round(len(indices) * calib_ratio)))
        n = min(n, len(indices) - 1)
        return np.sort(rng.choice(indices, size=n, replace=False))

    calib_pos = sample_subset(pos_idx)
    calib_neg = sample_subset(neg_idx)
    calib_idx = np.sort(np.concatenate([calib_pos, calib_neg]))

    train_mask = np.ones(len(y), dtype=bool)
    train_mask[calib_idx] = False
    train_idx = np.where(train_mask)[0]

    if len(train_idx) == 0 or len(calib_idx) == 0:
        # fallback: simple split
        perm = rng.permutation(len(y))
        split = max(1, int(round(len(y) * calib_ratio)))
        calib_idx = np.sort(perm[:split])
        train_idx = np.sort(perm[split:])
        if len(train_idx) == 0:
            train_idx = calib_idx.copy()

    return train_idx, calib_idx


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


@dataclass
class TrainConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    pos_weight_scale: float


def make_loader(X: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(X_tr, y_tr, w_tr, X_val, y_val, config: TrainConfig, device: torch.device):
    model = FusionMLP(X_tr.shape[1], config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    pos_weight = torch.tensor(
        [(REAL_POS_WEIGHT / max(NEG_WEIGHT, 1e-8)) * config.pos_weight_scale],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    train_loader = make_loader(X_tr, y_tr, w_tr, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, np.ones_like(y_val, dtype=np.float32), max(256, config.batch_size), shuffle=False)

    best_state = None
    best_score = -np.inf
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                val_probs.append(probs)
                val_true.append(yb.numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        try:
            val_score = average_precision_score(val_true, val_probs)
        except Exception:
            val_score = -np.inf

        if val_score > best_score + 1e-6:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch + 1 >= MIN_EPOCHS and patience_count >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=max(256, batch_size), shuffle=False)
    model.eval()
    logits = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits.append(model(xb).cpu().numpy())
    return np.concatenate(logits)


def fit_temperature(logits_calib: np.ndarray, y_calib: np.ndarray, device: torch.device) -> float:
    logits_calib = np.asarray(logits_calib, dtype=np.float32)
    y_calib = np.asarray(y_calib, dtype=np.float32)

    # 至少要有两个类别，否则校准没有意义
    if len(np.unique(y_calib)) < 2:
        return 1.0

    log_temp = torch.nn.Parameter(torch.zeros(1, device=device))  # temp = exp(log_temp)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss()

    logits_t = torch.tensor(logits_calib, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_calib, dtype=torch.float32, device=device)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_t / temp, y_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        temp = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).detach().cpu().item())
    except Exception:
        temp = 1.0

    if not np.isfinite(temp) or temp <= 0:
        temp = 1.0
    return temp


def apply_temperature_to_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-(logits / temperature)))


def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CACHE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Offline local ST model path:", ST_MODEL_PATH)
    print("Fixed params:", FIXED_PARAMS)

    df, structured_cols = load_data()
    readme_emb_all, aux_emb_all = build_full_cached_embeddings(df)

    real_pos_mask = df["is_real_positive"] == 1
    gen_pos_mask = df["is_generated_positive"] == 1
    neg_mask = df["label"] == 0
    real_pos_df = df.loc[real_pos_mask].copy()
    neg_indices_all = df.index[neg_mask].to_numpy()

    print("Total rows:", len(df))
    print("Real positives:", int(real_pos_mask.sum()))
    print("Generated positives:", int(gen_pos_mask.sum()))
    print("Negatives:", int(neg_mask.sum()))
    print("Unique real-positive families:", real_pos_df["family_id"].nunique())
    print("Structured feature count:", len(structured_cols))

    outer_pos_family_folds = make_positive_family_folds(real_pos_df, OUTER_N_SPLITS, RANDOM_STATE)
    outer_neg_folds = make_index_folds(neg_indices_all, len(outer_pos_family_folds), RANDOM_STATE)

    fold_metrics = []
    oof_records = []

    config = TrainConfig(
        hidden_dim=int(FIXED_PARAMS["hidden_dim"]),
        num_layers=int(FIXED_PARAMS["num_layers"]),
        dropout=float(FIXED_PARAMS["dropout"]),
        lr=float(FIXED_PARAMS["lr"]),
        weight_decay=float(FIXED_PARAMS["weight_decay"]),
        batch_size=int(FIXED_PARAMS["batch_size"]),
        pos_weight_scale=float(FIXED_PARAMS["pos_weight_scale"]),
    )

    for outer_fold_id, ((_, outer_test_fams), (_, outer_test_neg_idx)) in enumerate(zip(outer_pos_family_folds, outer_neg_folds), start=1):
        print(f"\\n===== OUTER FOLD {outer_fold_id} / {len(outer_pos_family_folds)} =====")
        print("Test families:", sorted(list(outer_test_fams)))

        outer_test_real_pos_idx = df.index[(df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))].to_numpy()

        if ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST:
            outer_test_gen_pos_idx = df.index[(df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))].to_numpy()
        else:
            outer_test_gen_pos_idx = np.array([], dtype=int)

        outer_test_idx = np.concatenate([outer_test_real_pos_idx, outer_test_gen_pos_idx, outer_test_neg_idx])

        outer_train_mask = (
            (~df.index.isin(outer_test_neg_idx))
            & (~((df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
            & (~((df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
        )
        outer_train_df = df.loc[outer_train_mask].copy()
        outer_train_real_pos = outer_train_df.loc[outer_train_df["is_real_positive"] == 1].copy()
        outer_train_neg_idx = outer_train_df.index[outer_train_df["label"] == 0].to_numpy()

        inner_pos_family_folds = make_positive_family_folds(outer_train_real_pos, INNER_N_SPLITS, RANDOM_STATE + outer_fold_id)
        inner_neg_folds = make_index_folds(outer_train_neg_idx, len(inner_pos_family_folds), RANDOM_STATE + outer_fold_id)

        pooled_rows = []
        pooled_probs = []
        pooled_temps = []

        for inner_i, ((_, inner_val_fams), (_, inner_val_neg_idx)) in enumerate(zip(inner_pos_family_folds, inner_neg_folds), start=1):
            val_df = outer_train_df.loc[
                ((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams)))
                | (outer_train_df.index.isin(inner_val_neg_idx))
            ].copy()

            train_df = outer_train_df.loc[
                (~outer_train_df.index.isin(inner_val_neg_idx))
                & (~((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
                & (~((outer_train_df["is_generated_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
            ].copy()

            X_tr_num, X_val_num, _ = build_numeric_matrix(train_df, val_df, structured_cols)
            tr_idx = train_df.index.to_numpy()
            val_idx = val_df.index.to_numpy()

            X_tr_full = np.concatenate([readme_emb_all[tr_idx], aux_emb_all[tr_idx], X_tr_num], axis=1).astype(np.float32)
            X_val = np.concatenate([readme_emb_all[val_idx], aux_emb_all[val_idx], X_val_num], axis=1).astype(np.float32)

            y_tr_full = train_df["label"].values.astype(np.float32)
            y_val = val_df["label"].values.astype(np.float32)
            w_tr_full = build_sample_weights(train_df)

            train_sub_idx, calib_idx = stratified_train_calib_split(y_tr_full, CALIB_RATIO, seed=RANDOM_STATE + outer_fold_id * 100 + inner_i)

            X_train_sub = X_tr_full[train_sub_idx]
            y_train_sub = y_tr_full[train_sub_idx]
            w_train_sub = w_tr_full[train_sub_idx]

            X_calib = X_tr_full[calib_idx]
            y_calib = y_tr_full[calib_idx]

            # 若 calib 集过小，则退化为用全部训练集中的一小块作 early stopping/temperature
            if len(calib_idx) == 0 or len(train_sub_idx) == 0:
                perm = np.random.default_rng(RANDOM_STATE + inner_i).permutation(len(X_tr_full))
                split = max(1, int(0.15 * len(X_tr_full)))
                calib_idx = perm[:split]
                train_sub_idx = perm[split:] if split < len(X_tr_full) else perm[:]
                X_train_sub = X_tr_full[train_sub_idx]
                y_train_sub = y_tr_full[train_sub_idx]
                w_train_sub = w_tr_full[train_sub_idx]
                X_calib = X_tr_full[calib_idx]
                y_calib = y_tr_full[calib_idx]

            model = train_one_fold(
                X_train_sub, y_train_sub, w_train_sub,
                X_calib, y_calib,
                config, device
            )

            calib_logits = predict_logits(model, X_calib, config.batch_size, device)
            temperature = fit_temperature(calib_logits, y_calib, device)
            pooled_temps.append(temperature)

            val_logits = predict_logits(model, X_val, config.batch_size, device)
            val_probs = apply_temperature_to_probs(val_logits, temperature)

            tmp = val_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
            tmp["row_index"] = val_df.index.values
            pooled_rows.append(tmp)
            pooled_probs.append(val_probs)

        pooled_df = pd.concat(pooled_rows, axis=0, ignore_index=True)
        pooled_df["y_proba"] = np.concatenate(pooled_probs)

        chosen_threshold, chosen_thr_info = select_best_threshold_f1_priority(
            pooled_df["label"].values,
            pooled_df["y_proba"].values,
            beta=BETA,
        )
        median_temperature = float(np.median(pooled_temps)) if len(pooled_temps) > 0 else 1.0

        print(
            f"Chosen threshold from inner OOF = {chosen_threshold:.6f} | "
            f"T_med={median_temperature:.4f} | "
            f"P={chosen_thr_info['precision']:.4f} R={chosen_thr_info['recall']:.4f} F1={chosen_thr_info['f1']:.4f}"
        )

        # full outer_train refit -> calibrate on training-side calib split -> outer_test
        test_df = df.loc[outer_test_idx].copy()
        train_df = outer_train_df.copy()

        X_tr_num, X_te_num, numeric_artifacts = build_numeric_matrix(train_df, test_df, structured_cols)
        tr_idx = train_df.index.to_numpy()
        te_idx = test_df.index.to_numpy()

        X_tr_full = np.concatenate([readme_emb_all[tr_idx], aux_emb_all[tr_idx], X_tr_num], axis=1).astype(np.float32)
        X_te = np.concatenate([readme_emb_all[te_idx], aux_emb_all[te_idx], X_te_num], axis=1).astype(np.float32)

        y_tr_full = train_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)
        w_tr_full = build_sample_weights(train_df)

        train_sub_idx, calib_idx = stratified_train_calib_split(y_tr_full, CALIB_RATIO, seed=RANDOM_STATE + outer_fold_id * 1000)

        X_train_sub = X_tr_full[train_sub_idx]
        y_train_sub = y_tr_full[train_sub_idx]
        w_train_sub = w_tr_full[train_sub_idx]

        X_calib = X_tr_full[calib_idx]
        y_calib = y_tr_full[calib_idx]

        if len(calib_idx) == 0 or len(train_sub_idx) == 0:
            perm = np.random.default_rng(RANDOM_STATE + outer_fold_id).permutation(len(X_tr_full))
            split = max(1, int(0.15 * len(X_tr_full)))
            calib_idx = perm[:split]
            train_sub_idx = perm[split:] if split < len(X_tr_full) else perm[:]
            X_train_sub = X_tr_full[train_sub_idx]
            y_train_sub = y_tr_full[train_sub_idx]
            w_train_sub = w_tr_full[train_sub_idx]
            X_calib = X_tr_full[calib_idx]
            y_calib = y_tr_full[calib_idx]

        model = train_one_fold(
            X_train_sub, y_train_sub, w_train_sub,
            X_calib, y_calib,
            config, device
        )

        calib_logits = predict_logits(model, X_calib, config.batch_size, device)
        final_temperature = fit_temperature(calib_logits, y_calib, device)

        test_logits = predict_logits(model, X_te, config.batch_size, device)
        test_proba = apply_temperature_to_probs(test_logits, final_temperature)
        test_pred = (test_proba >= chosen_threshold).astype(int)

        test_ap = average_precision_score(y_te, test_proba)
        test_roc = roc_auc_score(y_te, test_proba)
        test_precision = precision_score(y_te, test_pred, zero_division=0)
        test_recall = recall_score(y_te, test_pred, zero_division=0)
        test_f1 = f1_score(y_te, test_pred, zero_division=0)
        test_fbeta = calc_fbeta(test_precision, test_recall, beta=BETA)
        test_cm = confusion_matrix(y_te, test_pred, labels=[0, 1])

        fold_metrics.append({
            "outer_fold": outer_fold_id,
            "model": "SentenceTransformer+MLP (fixed params + temperature scaling, offline)",
            "n_test_total": int(len(test_df)),
            "n_test_pos_real": int((test_df["is_real_positive"] == 1).sum()),
            "n_test_pos_generated": int((test_df["is_generated_positive"] == 1).sum()),
            "n_test_pos_total": int((test_df["label"] == 1).sum()),
            "n_test_neg": int((test_df["label"] == 0).sum()),
            "threshold_from_inner_oof": float(chosen_threshold),
            "temperature": float(final_temperature),
            "inner_precision_at_thr": float(chosen_thr_info["precision"]),
            "inner_recall_at_thr": float(chosen_thr_info["recall"]),
            "inner_f1_at_thr": float(chosen_thr_info["f1"]),
            "test_pr_auc": float(test_ap),
            "test_roc_auc": float(test_roc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "test_fbeta": float(test_fbeta),
            "tn": int(test_cm[0, 0]),
            "fp": int(test_cm[0, 1]),
            "fn": int(test_cm[1, 0]),
            "tp": int(test_cm[1, 1]),
        })

        tmp = test_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
        tmp["row_index"] = test_df.index.values
        tmp["outer_fold"] = outer_fold_id
        tmp["y_proba"] = test_proba
        tmp["y_pred"] = test_pred
        tmp["threshold"] = chosen_threshold
        tmp["temperature"] = final_temperature
        tmp["model"] = "SentenceTransformer+MLP (fixed params + temperature scaling, offline)"
        if "repo_full_name" in test_df.columns:
            tmp["repo_full_name"] = test_df["repo_full_name"].values
        oof_records.append(tmp)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": int(X_tr_full.shape[1]),
                "fixed_params": FIXED_PARAMS,
                "threshold": chosen_threshold,
                "temperature": final_temperature,
                "numeric_artifacts": numeric_artifacts,
                "st_model_path": str(ST_MODEL_PATH),
            },
            OUTPUT_DIR / f"outer_fold_{outer_fold_id}_bundle.pt",
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUTPUT_DIR / "dl_st_mlp_temp_scaled_fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    metric_cols = ["threshold_from_inner_oof", "temperature", "test_pr_auc", "test_roc_auc", "test_precision", "test_recall", "test_f1", "test_fbeta"]
    for c in metric_cols:
        summary_rows.append({
            "metric": c,
            "mean": float(fold_metrics_df[c].mean()),
            "std": float(fold_metrics_df[c].std(ddof=1)) if len(fold_metrics_df) > 1 else 0.0,
            "min": float(fold_metrics_df[c].min()),
            "max": float(fold_metrics_df[c].max()),
        })
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "dl_st_mlp_temp_scaled_summary_mean_std.csv", index=False, encoding="utf-8-sig")

    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    oof_df.to_csv(OUTPUT_DIR / "dl_st_mlp_temp_scaled_oof_predictions.csv", index=False, encoding="utf-8-sig")

    y_oof = oof_df["label"].values
    p_oof = oof_df["y_proba"].values
    pred_foldwise = oof_df["y_pred"].values.astype(int)
    foldwise_cm = confusion_matrix(y_oof, pred_foldwise, labels=[0, 1])

    pooled_global_threshold, pooled_global_info = select_best_threshold_f1_priority(y_oof, p_oof, beta=BETA)
    pred_global = (p_oof >= pooled_global_threshold).astype(int)
    global_cm = confusion_matrix(y_oof, pred_global, labels=[0, 1])

    report = {
        "config": {
            "data_path": str(DATA_PATH),
            "outer_n_splits": OUTER_N_SPLITS,
            "inner_n_splits": INNER_N_SPLITS,
            "allow_generated_positives_in_outer_test": ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST,
            "model": "SentenceTransformer+MLP (fixed params + temperature scaling, offline)",
            "encoder_local_path": str(ST_MODEL_PATH),
            "fixed_params": FIXED_PARAMS,
            "calib_ratio": CALIB_RATIO,
        },
        "pooled_oof_foldwise_threshold": {
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_foldwise, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_foldwise, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_foldwise, zero_division=0)),
            "fbeta": float(calc_fbeta(
                precision_score(y_oof, pred_foldwise, zero_division=0),
                recall_score(y_oof, pred_foldwise, zero_division=0),
                beta=BETA,
            )),
            "confusion_matrix": foldwise_cm.tolist(),
        },
        "pooled_oof_global_threshold": {
            "threshold": float(pooled_global_threshold),
            "threshold_info": pooled_global_info,
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_global, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_global, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_global, zero_division=0)),
            "fbeta": float(calc_fbeta(
                precision_score(y_oof, pred_global, zero_division=0),
                recall_score(y_oof, pred_global, zero_division=0),
                beta=BETA,
            )),
            "confusion_matrix": global_cm.tolist(),
        },
    }

    with open(OUTPUT_DIR / "dl_st_mlp_temp_scaled_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\\nSaved outputs to:", OUTPUT_DIR)
    print("- dl_st_mlp_temp_scaled_fold_metrics.csv")
    print("- dl_st_mlp_temp_scaled_summary_mean_std.csv")
    print("- dl_st_mlp_temp_scaled_oof_predictions.csv")
    print("- dl_st_mlp_temp_scaled_report.json")
    print("- outer_fold_*_bundle.pt")


if __name__ == "__main__":
    main()
