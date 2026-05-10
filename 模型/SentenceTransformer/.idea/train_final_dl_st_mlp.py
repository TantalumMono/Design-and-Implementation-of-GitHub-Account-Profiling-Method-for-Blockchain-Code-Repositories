# -*- coding: utf-8 -*-
import os

# Must be set before importing sentence_transformers / transformers
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "final_dl_st_mlp_artifacts"
CACHE_DIR = SCRIPT_DIR / "dl_st_embedding_cache_offline"

# Change to your actual local model folder
ST_MODEL_PATH = Path(r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\models\all-MiniLM-L6-v2")

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

RANDOM_STATE = 42
ST_BATCH_SIZE = 64
REAL_POS_WEIGHT = 4.5
GEN_POS_WEIGHT = 1.0
NEG_WEIGHT = 1.25

# Final fixed params selected from family validation consensus
FINAL_PARAMS = {
    "hidden_dim": 256,
    "num_layers": 3,
    "dropout": 0.25,
    "lr": 1e-3,
    "weight_decay": 2e-4,
    "batch_size": 32,
    "pos_weight_scale": 1.10,
}
FINAL_THRESHOLD = 0.005434935446828604

MAX_EPOCHS = 40
PATIENCE = 6
MIN_EPOCHS = 5
EARLYSTOP_RATIO = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {
        "label", "is_real_positive", "is_generated_positive",
        README_TEXT_COL, DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int)
    df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)

    structured_cols = [c for c in df.columns if not is_leaky_feature_col(c)]
    return df, structured_cols


def build_numeric_matrix_full(df: pd.DataFrame, structured_cols: List[str]):
    X = df[structured_cols].copy()
    for col in structured_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32)).astype(np.float32)
    return X_scaled, {"fill_values": fill_values.to_dict(), "structured_cols": structured_cols}


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
    readme_emb = encode_texts_cached(df[README_TEXT_COL].tolist(), readme_cache)
    aux_emb = encode_texts_cached(get_aux_text_series(df).tolist(), aux_cache)
    return readme_emb, aux_emb


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.full(len(df), NEG_WEIGHT, dtype=np.float32)
    real_mask = df["is_real_positive"].values == 1
    gen_mask = df["is_generated_positive"].values == 1
    w[real_mask] = REAL_POS_WEIGHT
    w[gen_mask] = GEN_POS_WEIGHT
    return w


def stratified_train_val_split(y: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    def sample_subset(indices):
        if len(indices) <= 1:
            return np.array([], dtype=int)
        n = max(1, int(round(len(indices) * val_ratio)))
        n = min(n, len(indices) - 1)
        return np.sort(rng.choice(indices, size=n, replace=False))

    val_pos = sample_subset(pos_idx)
    val_neg = sample_subset(neg_idx)
    val_idx = np.sort(np.concatenate([val_pos, val_neg]))
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        perm = rng.permutation(len(y))
        split = max(1, int(round(len(y) * val_ratio)))
        val_idx = np.sort(perm[:split])
        train_idx = np.sort(perm[split:])
        if len(train_idx) == 0:
            train_idx = val_idx.copy()

    return train_idx, val_idx


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


def train_with_early_stopping(X_tr, y_tr, w_tr, X_val, y_val, config: TrainConfig, device: torch.device):
    model = FusionMLP(X_tr.shape[1], config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    pos_weight = torch.tensor(
        [(REAL_POS_WEIGHT / max(NEG_WEIGHT, 1e-8)) * config.pos_weight_scale],
        dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    train_loader = make_loader(X_tr, y_tr, w_tr, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, np.ones_like(y_val, dtype=np.float32), max(256, config.batch_size), shuffle=False)

    best_state = None
    best_pr_auc = -np.inf
    best_epoch = 1
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
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
        with torch.no_grad():
            for xb, _, _ in val_loader:
                xb = xb.to(device)
                val_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
        val_probs = np.concatenate(val_probs)

        try:
            val_pr_auc = average_precision_score(y_val, val_probs)
        except Exception:
            val_pr_auc = -np.inf

        if val_pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if epoch >= MIN_EPOCHS and patience_count >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_pr_auc


def train_fixed_epochs(X_all, y_all, w_all, config: TrainConfig, n_epochs: int, device: torch.device):
    model = FusionMLP(X_all.shape[1], config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    pos_weight = torch.tensor(
        [(REAL_POS_WEIGHT / max(NEG_WEIGHT, 1e-8)) * config.pos_weight_scale],
        dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    loader = make_loader(X_all, y_all, w_all, config.batch_size, shuffle=True)

    for _ in range(max(1, int(n_epochs))):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss_vec = criterion(logits, yb)
            loss = (loss_vec * wb).mean()
            loss.backward()
            optimizer.step()

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


def main():
    set_seed(RANDOM_STATE)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CACHE_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Offline local ST model path:", ST_MODEL_PATH)
    print("Final fixed params:", FINAL_PARAMS)

    df, structured_cols = load_data()
    readme_emb_all, aux_emb_all = build_full_cached_embeddings(df)
    X_num_scaled, numeric_artifacts = build_numeric_matrix_full(df, structured_cols)

    X_all = np.concatenate([readme_emb_all, aux_emb_all, X_num_scaled], axis=1).astype(np.float32)
    y_all = df["label"].values.astype(np.float32)
    w_all = build_sample_weights(df)

    readme_dim = readme_emb_all.shape[1]
    aux_dim = aux_emb_all.shape[1]
    struct_dim = X_num_scaled.shape[1]

    train_idx, val_idx = stratified_train_val_split(y_all, EARLYSTOP_RATIO, RANDOM_STATE)
    config = TrainConfig(**FINAL_PARAMS)

    model_es, best_epoch, best_pr_auc = train_with_early_stopping(
        X_all[train_idx], y_all[train_idx], w_all[train_idx],
        X_all[val_idx], y_all[val_idx],
        config, device
    )

    print(f"Best epoch from early stopping subset: {best_epoch}")
    print(f"Best validation PR-AUC: {best_pr_auc:.6f}")

    final_model = train_fixed_epochs(X_all, y_all, w_all, config, best_epoch, device)

    final_logits = predict_logits(final_model, X_all, config.batch_size, device)
    final_probs = 1.0 / (1.0 + np.exp(-final_logits))
    final_pred = (final_probs >= FINAL_THRESHOLD).astype(int)

    precision = precision_score(y_all, final_pred, zero_division=0)
    recall = recall_score(y_all, final_pred, zero_division=0)
    f1 = f1_score(y_all, final_pred, zero_division=0)
    pr_auc = average_precision_score(y_all, final_probs)
    roc_auc = roc_auc_score(y_all, final_probs)

    out_df = pd.DataFrame({
        "label": y_all.astype(int),
        "y_logit": final_logits,
        "y_proba": final_probs,
        "y_pred": final_pred,
    })
    if "repo_full_name" in df.columns:
        out_df["repo_full_name"] = df["repo_full_name"].values
    if "family_id" in df.columns:
        out_df["family_id"] = df["family_id"].astype(str).values
    if "is_real_positive" in df.columns:
        out_df["is_real_positive"] = df["is_real_positive"].astype(int).values
    if "is_generated_positive" in df.columns:
        out_df["is_generated_positive"] = df["is_generated_positive"].astype(int).values
    out_df.to_csv(OUTPUT_DIR / "final_dl_st_mlp_trainset_predictions.csv", index=False, encoding="utf-8-sig")

    bundle = {
        "model_state_dict": final_model.state_dict(),
        "input_dim": int(X_all.shape[1]),
        "threshold": float(FINAL_THRESHOLD),
        "best_epoch": int(best_epoch),
        "fixed_params": FINAL_PARAMS,
        "st_model_path": str(ST_MODEL_PATH),
        "README_TEXT_COL": README_TEXT_COL,
        "DESCRIPTION_TEXT_COL": DESCRIPTION_TEXT_COL,
        "TOPICS_TEXT_COL": TOPICS_TEXT_COL,
        "structured_cols": structured_cols,
        "fill_values": numeric_artifacts["fill_values"],
        "readme_dim": int(readme_dim),
        "aux_dim": int(aux_dim),
        "struct_dim": int(struct_dim),
    }
    torch.save(bundle, OUTPUT_DIR / "final_dl_st_mlp_bundle.pt")

    summary = {
        "data_path": str(DATA_PATH),
        "n_rows": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "threshold": float(FINAL_THRESHOLD),
        "best_epoch": int(best_epoch),
        "final_params": FINAL_PARAMS,
        "trainset_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
        },
    }
    with open(OUTPUT_DIR / "final_dl_st_mlp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved final deep model artifacts to:", OUTPUT_DIR)
    print("- final_dl_st_mlp_bundle.pt")
    print("- final_dl_st_mlp_trainset_predictions.csv")
    print("- final_dl_st_mlp_summary.json")


if __name__ == "__main__":
    main()
