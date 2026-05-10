
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================================================
# 0. CONFIG
# =========================================================
BEST_PARAMS_JSON = "./train_tfidf_bayes_f1_weighted_outputs/best_params.json"
OUTPUT_DIR = "train_tfidf_bayes_f1_weighted_outputs/family_cv_fixed_params_outputs"
RANDOM_STATE = 42

# Family-level outer CV on REAL positive families only
OUTER_N_SPLITS = 5

# Threshold strategy:
# "fixed_from_json" -> use best_user_attrs.val_best_threshold directly
# "fixed_05"        -> always use 0.5
THRESHOLD_STRATEGY = "fixed_from_json"

# Fillna
FILLNA_STRATEGY = "median"

# Metadata / non-structured columns
NON_STRUCTURED_COLS = {
    "label",
    "family_id",
    "repo_full_name",
    "readme_text",
    "readme_text_raw",
    "readme_text_clean",
    "sample_id",
    "collected_at",
    "provenance_type",
    "is_real_positive",
    "is_generated_positive",
    "is_real_negative",
    "sample_source",
    "group_id",
    "source_repo_full_name",
    "source_family",
    "generated_from",
}
LEAKY_PREFIXES = (
    "genmeta__",
    "rewrite_meta__",
    "ablation__",
)

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b"


# =========================================================
# 1. UTILS
# =========================================================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_ngram_range(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    return (1, 2)


def prepare_structured_features(df: pd.DataFrame, structured_cols: list[str]) -> pd.DataFrame:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def fillna_by_train_statistics(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    strategy: str = "median",
):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy must be 'median' or 'zero'")

    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_te = X_te.fillna(fill_values).fillna(0)
    return X_tr, X_te, fill_values


def build_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    structured_cols: list[str],
    text_col: str,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_ngram_range: tuple[int, int],
):
    X_tr_num = prepare_structured_features(train_df, structured_cols)
    X_te_num = prepare_structured_features(test_df, structured_cols)
    X_tr_num, X_te_num, fill_values = fillna_by_train_statistics(
        X_tr_num, X_te_num, strategy=FILLNA_STRATEGY
    )

    X_tr_num_sp = csr_matrix(X_tr_num.to_numpy(dtype=np.float32))
    X_te_num_sp = csr_matrix(X_te_num.to_numpy(dtype=np.float32))

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
        ngram_range=tfidf_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )

    tr_text = train_df[text_col].fillna("").astype(str)
    te_text = test_df[text_col].fillna("").astype(str)

    X_tr_text = vectorizer.fit_transform(tr_text)
    X_te_text = vectorizer.transform(te_text)

    X_tr_all = hstack([X_tr_num_sp, X_tr_text], format="csr")
    X_te_all = hstack([X_te_num_sp, X_te_text], format="csr")
    return X_tr_all, X_te_all, vectorizer, fill_values


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


# =========================================================
# 2. MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_path = cfg["data_path"]
    text_col = cfg.get("text_col", "text_for_tfidf_clean")
    structured_cols = cfg["structured_cols"]
    best_params = cfg["best_params"]
    best_user_attrs = cfg.get("best_user_attrs", {})

    tfidf_max_features = int(best_params["tfidf_max_features"])
    tfidf_min_df = int(best_params["tfidf_min_df"])
    tfidf_ngram_range = parse_ngram_range(best_params["tfidf_ngram_range"])

    threshold_from_json = float(best_user_attrs.get("val_best_threshold", 0.5))
    if THRESHOLD_STRATEGY == "fixed_from_json":
        fixed_threshold = threshold_from_json
    else:
        fixed_threshold = 0.5

    actual_w_real_pos = float(best_user_attrs.get("actual_w_real_pos", 3.0))
    actual_w_generated_pos = float(best_user_attrs.get("actual_w_generated_pos", 1.25))
    actual_w_neg = float(best_user_attrs.get("actual_w_neg", 1.0))

    df = pd.read_csv(data_path)

    required = {"label", "family_id", text_col, "is_real_positive", "is_generated_positive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int)
    df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int)
    df["family_id"] = df["family_id"].fillna("UNKNOWN").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)

    existing_structured_cols = [
        c for c in structured_cols
        if c in df.columns
        and c not in NON_STRUCTURED_COLS
        and not any(c.startswith(p) for p in LEAKY_PREFIXES)
    ]

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
    print("Structured feature count actually used:", len(existing_structured_cols))
    print("Fixed threshold:", fixed_threshold)
    print("Weights -> real pos:", actual_w_real_pos,
          "| gen pos:", actual_w_generated_pos,
          "| neg:", actual_w_neg)

    outer_pos_family_folds = make_positive_family_folds(real_pos_df, OUTER_N_SPLITS, RANDOM_STATE)
    outer_neg_folds = make_index_folds(neg_indices_all, len(outer_pos_family_folds), RANDOM_STATE)

    fold_metrics = []
    oof_records = []

    xgb_fixed_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "importance_type": "gain",
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_weight": int(best_params["min_child_weight"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "gamma": float(best_params["gamma"]),
        "max_delta_step": float(best_params["max_delta_step"]),
    }

    for outer_fold_id, ((outer_train_fams, outer_test_fams), (_, outer_test_neg_idx)) in enumerate(
        zip(outer_pos_family_folds, outer_neg_folds), start=1
    ):
        print(f"\n===== OUTER FOLD {outer_fold_id} / {len(outer_pos_family_folds)} =====")
        print("Test families:", sorted(list(outer_test_fams)))

        outer_test_pos_idx = df.index[
            (df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))
        ].to_numpy()
        outer_test_idx = np.concatenate([outer_test_pos_idx, outer_test_neg_idx])

        # Outer train:
        # 1) remove outer test negatives
        # 2) remove held-out real-positive families
        # 3) also remove generated positives from held-out families to prevent family leakage
        outer_train_mask = (
            (~df.index.isin(outer_test_neg_idx))
            & (~((df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
            & (~((df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
        )

        train_df = df.loc[outer_train_mask].copy()
        test_df = df.loc[outer_test_idx].copy()

        leak_rows = train_df.loc[
            (train_df["is_generated_positive"] == 1) & (train_df["family_id"].isin(outer_test_fams))
        ]
        if len(leak_rows) > 0:
            raise RuntimeError(f"Leakage detected in fold {outer_fold_id}: generated positives from test families remain in train.")

        X_tr, X_te, vectorizer, fill_values = build_matrices(
            train_df=train_df,
            test_df=test_df,
            structured_cols=existing_structured_cols,
            text_col=text_col,
            tfidf_max_features=tfidf_max_features,
            tfidf_min_df=tfidf_min_df,
            tfidf_ngram_range=tfidf_ngram_range,
        )

        y_tr = train_df["label"].values
        y_te = test_df["label"].values

        sample_weight = np.full(len(train_df), actual_w_neg, dtype=float)
        sample_weight[train_df["is_real_positive"].values == 1] = actual_w_real_pos
        sample_weight[train_df["is_generated_positive"].values == 1] = actual_w_generated_pos

        model = XGBClassifier(**xgb_fixed_params)
        model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)

        test_proba = model.predict_proba(X_te)[:, 1]
        test_pred = (test_proba >= fixed_threshold).astype(int)

        test_ap = average_precision_score(y_te, test_proba)
        test_roc = roc_auc_score(y_te, test_proba)
        test_precision = precision_score(y_te, test_pred, zero_division=0)
        test_recall = recall_score(y_te, test_pred, zero_division=0)
        test_f1 = f1_score(y_te, test_pred, zero_division=0)
        test_cm = confusion_matrix(y_te, test_pred, labels=[0, 1])

        fold_metrics.append(
            {
                "outer_fold": outer_fold_id,
                "test_families": "|".join(sorted(list(outer_test_fams))),
                "n_train_total": int(len(train_df)),
                "n_train_real_pos": int((train_df["is_real_positive"] == 1).sum()),
                "n_train_generated_pos": int((train_df["is_generated_positive"] == 1).sum()),
                "n_train_neg": int((train_df["label"] == 0).sum()),
                "n_test_total": int(len(test_df)),
                "n_test_pos_real": int((test_df["is_real_positive"] == 1).sum()),
                "n_test_neg": int((test_df["label"] == 0).sum()),
                "threshold": float(fixed_threshold),
                "test_pr_auc": float(test_ap),
                "test_roc_auc": float(test_roc),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "test_f1": float(test_f1),
                "tn": int(test_cm[0, 0]),
                "fp": int(test_cm[0, 1]),
                "fn": int(test_cm[1, 0]),
                "tp": int(test_cm[1, 1]),
                "tfidf_vocab_size": int(len(vectorizer.get_feature_names_out())),
            }
        )

        tmp = test_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
        tmp["row_index"] = test_df.index.values
        tmp["outer_fold"] = outer_fold_id
        tmp["y_proba"] = test_proba
        tmp["y_pred"] = test_pred
        tmp["threshold"] = fixed_threshold
        if "repo_full_name" in test_df.columns:
            tmp["repo_full_name"] = test_df["repo_full_name"].values
        oof_records.append(tmp)

        joblib.dump(
            {
                "model": model,
                "vectorizer": vectorizer,
                "fill_values": fill_values.to_dict(),
                "structured_cols": existing_structured_cols,
                "threshold": fixed_threshold,
                "params": xgb_fixed_params,
                "text_col": text_col,
                "outer_test_families": sorted(list(outer_test_fams)),
            },
            os.path.join(OUTPUT_DIR, f"outer_fold_{outer_fold_id}_fixed_bundle.joblib"),
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "family_fixedparam_fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    metric_cols = [
        "threshold",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    for c in metric_cols:
        summary_rows.append(
            {
                "metric": c,
                "mean": float(fold_metrics_df[c].mean()),
                "std": float(fold_metrics_df[c].std(ddof=1)) if len(fold_metrics_df) > 1 else 0.0,
                "min": float(fold_metrics_df[c].min()),
                "max": float(fold_metrics_df[c].max()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "family_fixedparam_summary_mean_std.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "family_fixedparam_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    y_oof = oof_df["label"].values
    p_oof = oof_df["y_proba"].values
    pred_oof = (p_oof >= fixed_threshold).astype(int)
    pooled_cm = confusion_matrix(y_oof, pred_oof, labels=[0, 1])

    report = {
        "config": {
            "best_params_json": BEST_PARAMS_JSON,
            "data_path": data_path,
            "text_col": text_col,
            "outer_n_splits": OUTER_N_SPLITS,
            "tfidf_max_features": tfidf_max_features,
            "tfidf_min_df": tfidf_min_df,
            "tfidf_ngram_range": list(tfidf_ngram_range),
            "threshold_strategy": THRESHOLD_STRATEGY,
            "fixed_threshold": fixed_threshold,
            "actual_w_real_pos": actual_w_real_pos,
            "actual_w_generated_pos": actual_w_generated_pos,
            "actual_w_neg": actual_w_neg,
            "structured_feature_count": len(existing_structured_cols),
        },
        "xgb_fixed_params": xgb_fixed_params,
        "pooled_oof": {
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_oof, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_oof, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_oof, zero_division=0)),
            "confusion_matrix": pooled_cm.tolist(),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "family_fixedparam_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("- family_fixedparam_fold_metrics.csv")
    print("- family_fixedparam_summary_mean_std.csv")
    print("- family_fixedparam_oof_predictions.csv")
    print("- family_fixedparam_report.json")
    print("- outer_fold_*_fixed_bundle.joblib")


if __name__ == "__main__":
    main()
