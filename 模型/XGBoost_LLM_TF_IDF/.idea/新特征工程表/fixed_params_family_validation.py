
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

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "fixed_params_family_validation_outputs"
RANDOM_STATE = 42
OUTER_N_SPLITS = 5

ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST = True

README_TEXT_COL = "readme_text_clean"
DESCRIPTION_TEXT_COL = "description_text_clean"
TOPICS_TEXT_COL = "topics_text_clean"
FILLNA_STRATEGY = "median"

NON_STRUCTURED_COLS = {
    "label","family_id","repo_full_name","readme_text","readme_text_raw","readme_text_clean",
    "description_text","topics_text","combined_text","combined_text_clean","description_text_clean",
    "topics_text_clean","text_for_tfidf_clean","sample_id","collected_at","provenance_type",
    "is_real_positive","is_generated_positive","is_real_negative","sample_source","group_id",
    "source_repo_full_name","source_family","generated_from",
}
LEAKY_PREFIXES = ("genmeta__","rewrite_meta__","ablation__")
META_LEAKY_PREFIXES = ("meta_","meta__","source_","generated_","augmentation_")
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b"

FINAL_PARAMS = {
    "readme_max_features": 400,
    "readme_min_df": 3,
    "readme_ngram_range": (1, 1),
    "aux_max_features": 200,
    "aux_min_df": 2,
    "aux_ngram_range": (1, 2),
    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "importance_type": "gain",
        "n_estimators": 600,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.02,
        "reg_lambda": 0.25,
        "gamma": 2.0,
        "max_delta_step": 1.8,
    },
    "weights": {
        "real_positive": 4.5,
        "generated_positive": 1.0,
        "negative": 1.25,
    },
    "threshold": 0.12565019726753235,
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def calc_fbeta(precision, recall, beta=2.0):
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def is_leaky_feature_col(col):
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


def prepare_structured_features(df: pd.DataFrame, structured_cols):
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def fillna_by_train_statistics(X_tr, X_te, strategy="median"):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy must be 'median' or 'zero'")
    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_te = X_te.fillna(fill_values).fillna(0)
    return X_tr, X_te, fill_values


def build_sample_weights(df, w_real_pos, w_generated_pos, w_neg):
    w = np.full(len(df), w_neg, dtype=float)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = w_real_pos
    w[gen_mask] = w_generated_pos
    return w


def build_matrices(train_df, test_df, structured_cols, params):
    X_tr_num = prepare_structured_features(train_df, structured_cols)
    X_te_num = prepare_structured_features(test_df, structured_cols)
    X_tr_num, X_te_num, fill_values = fillna_by_train_statistics(X_tr_num, X_te_num, strategy=FILLNA_STRATEGY)
    X_tr_num_sp = csr_matrix(X_tr_num.to_numpy(dtype=np.float32))
    X_te_num_sp = csr_matrix(X_te_num.to_numpy(dtype=np.float32))

    readme_vectorizer = TfidfVectorizer(
        max_features=params["readme_max_features"],
        min_df=params["readme_min_df"],
        ngram_range=params["readme_ngram_range"],
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    tr_readme = train_df[README_TEXT_COL].fillna("").astype(str)
    te_readme = test_df[README_TEXT_COL].fillna("").astype(str)
    X_tr_readme = readme_vectorizer.fit_transform(tr_readme)
    X_te_readme = readme_vectorizer.transform(te_readme)

    aux_vectorizer = TfidfVectorizer(
        max_features=params["aux_max_features"],
        min_df=params["aux_min_df"],
        ngram_range=params["aux_ngram_range"],
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    tr_aux = get_aux_text_series(train_df)
    te_aux = get_aux_text_series(test_df)
    X_tr_aux = aux_vectorizer.fit_transform(tr_aux)
    X_te_aux = aux_vectorizer.transform(te_aux)

    X_tr_all = hstack([X_tr_num_sp, X_tr_readme, X_tr_aux], format="csr")
    X_te_all = hstack([X_te_num_sp, X_te_readme, X_te_aux], format="csr")
    vectorizers = {"readme_vectorizer": readme_vectorizer, "aux_vectorizer": aux_vectorizer}
    return X_tr_all, X_te_all, vectorizers, fill_values


def make_positive_family_folds(real_pos_df, n_splits, seed):
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


def make_index_folds(indices, n_splits, seed):
    if len(indices) < 2:
        raise ValueError("Need at least 2 negative samples for fold splitting.")
    n_splits = min(n_splits, len(indices))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in kf.split(indices):
        folds.append((indices[tr_idx], indices[te_idx]))
    return folds


def main():
    ensure_dir(OUTPUT_DIR)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"label","family_id","is_real_positive","is_generated_positive",README_TEXT_COL,DESCRIPTION_TEXT_COL,TOPICS_TEXT_COL}
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
    leakage_candidates = [c for c in df.columns if is_leaky_feature_col(c)]

    real_pos_mask = df["is_real_positive"] == 1
    gen_pos_mask = df["is_generated_positive"] == 1
    neg_mask = df["label"] == 0

    real_pos_df = df.loc[real_pos_mask].copy()
    neg_indices_all = df.index[neg_mask].to_numpy()

    print("Total rows:", len(df))
    print("Real positives:", int(real_pos_mask.sum()))
    print("Generated positives:", int(gen_pos_mask.sum()))
    print("Negatives:", int(neg_mask.sum()))
    print("ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST:", ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST)
    print("Unique real-positive families:", real_pos_df["family_id"].nunique())
    print("Structured feature count:", len(structured_cols))
    print("Excluded leakage-prone cols:", leakage_candidates)

    outer_pos_family_folds = make_positive_family_folds(real_pos_df, OUTER_N_SPLITS, RANDOM_STATE)
    outer_neg_folds = make_index_folds(neg_indices_all, len(outer_pos_family_folds), RANDOM_STATE)

    fold_metrics = []
    oof_records = []

    for outer_fold_id, ((outer_train_fams, outer_test_fams), (_, outer_test_neg_idx)) in enumerate(
        zip(outer_pos_family_folds, outer_neg_folds), start=1
    ):
        print(f"\\n===== OUTER FOLD {outer_fold_id} / {len(outer_pos_family_folds)} =====")
        print("Test families:", sorted(list(outer_test_fams)))

        outer_test_real_pos_idx = df.index[
            (df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))
        ].to_numpy()
        if ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST:
            outer_test_gen_pos_idx = df.index[
                (df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))
            ].to_numpy()
        else:
            outer_test_gen_pos_idx = np.array([], dtype=int)

        outer_test_idx = np.concatenate([outer_test_real_pos_idx, outer_test_gen_pos_idx, outer_test_neg_idx])

        outer_train_mask = (
            (~df.index.isin(outer_test_neg_idx))
            & (~((df["is_real_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
            & (~((df["is_generated_positive"] == 1) & (df["family_id"].isin(outer_test_fams))))
        )
        train_df = df.loc[outer_train_mask].copy()
        test_df = df.loc[outer_test_idx].copy()

        X_tr, X_te, vectorizers, fill_values = build_matrices(train_df, test_df, structured_cols, FINAL_PARAMS)
        y_tr = train_df["label"].values
        y_te = test_df["label"].values

        sw = build_sample_weights(
            train_df,
            w_real_pos=FINAL_PARAMS["weights"]["real_positive"],
            w_generated_pos=FINAL_PARAMS["weights"]["generated_positive"],
            w_neg=FINAL_PARAMS["weights"]["negative"],
        )

        model = XGBClassifier(**FINAL_PARAMS["xgb_params"])
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)

        test_proba = model.predict_proba(X_te)[:, 1]
        test_pred = (test_proba >= FINAL_PARAMS["threshold"]).astype(int)

        test_ap = average_precision_score(y_te, test_proba)
        test_roc = roc_auc_score(y_te, test_proba)
        test_precision = precision_score(y_te, test_pred, zero_division=0)
        test_recall = recall_score(y_te, test_pred, zero_division=0)
        test_f1 = f1_score(y_te, test_pred, zero_division=0)
        test_fbeta = calc_fbeta(test_precision, test_recall, beta=2.0)
        test_cm = confusion_matrix(y_te, test_pred, labels=[0, 1])

        fold_metrics.append({
            "outer_fold": outer_fold_id,
            "text_mode": "dual_text_fixed_params",
            "n_test_total": int(len(test_df)),
            "n_test_pos_real": int((test_df["is_real_positive"] == 1).sum()),
            "n_test_pos_generated": int((test_df["is_generated_positive"] == 1).sum()),
            "n_test_pos_total": int((test_df["label"] == 1).sum()),
            "n_test_neg": int((test_df["label"] == 0).sum()),
            "threshold": float(FINAL_PARAMS["threshold"]),
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
        tmp["threshold"] = FINAL_PARAMS["threshold"]
        tmp["text_mode"] = "dual_text_fixed_params"
        if "repo_full_name" in test_df.columns:
            tmp["repo_full_name"] = test_df["repo_full_name"].values
        oof_records.append(tmp)

        joblib.dump(
            {
                "model": model,
                "vectorizers": vectorizers,
                "fill_values": fill_values.to_dict(),
                "structured_cols": structured_cols,
                "threshold": FINAL_PARAMS["threshold"],
                "fixed_params": FINAL_PARAMS,
            },
            os.path.join(OUTPUT_DIR, f"outer_fold_{outer_fold_id}_bundle.joblib"),
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "fixed_params_family_fold_metrics.csv"), index=False, encoding="utf-8-sig")

    summary_rows = []
    metric_cols = ["threshold","test_pr_auc","test_roc_auc","test_precision","test_recall","test_f1","test_fbeta"]
    for c in metric_cols:
        summary_rows.append({
            "metric": c,
            "mean": float(fold_metrics_df[c].mean()),
            "std": float(fold_metrics_df[c].std(ddof=1)) if len(fold_metrics_df) > 1 else 0.0,
            "min": float(fold_metrics_df[c].min()),
            "max": float(fold_metrics_df[c].max()),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "fixed_params_family_summary_mean_std.csv"), index=False, encoding="utf-8-sig")

    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    oof_df.to_csv(os.path.join(OUTPUT_DIR, "fixed_params_family_oof_predictions.csv"), index=False, encoding="utf-8-sig")

    y_oof = oof_df["label"].values
    pred_oof = oof_df["y_pred"].values.astype(int)
    p_oof = oof_df["y_proba"].values
    pooled_cm = confusion_matrix(y_oof, pred_oof, labels=[0, 1])

    report = {
        "config": {
            "data_path": str(DATA_PATH),
            "outer_n_splits": OUTER_N_SPLITS,
            "allow_generated_positives_in_outer_test": ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST,
            "text_mode": "dual_text_fixed_params",
            "readme_text_col": README_TEXT_COL,
            "aux_text_cols": [DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL],
            "meta_leaky_prefixes": list(META_LEAKY_PREFIXES),
        },
        "final_fixed_params": {
            "readme_max_features": FINAL_PARAMS["readme_max_features"],
            "readme_min_df": FINAL_PARAMS["readme_min_df"],
            "readme_ngram_range": list(FINAL_PARAMS["readme_ngram_range"]),
            "aux_max_features": FINAL_PARAMS["aux_max_features"],
            "aux_min_df": FINAL_PARAMS["aux_min_df"],
            "aux_ngram_range": list(FINAL_PARAMS["aux_ngram_range"]),
            "xgb_params": FINAL_PARAMS["xgb_params"],
            "weights": FINAL_PARAMS["weights"],
            "threshold": FINAL_PARAMS["threshold"],
        },
        "fixed_params_pooled_oof": {
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_oof, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_oof, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_oof, zero_division=0)),
            "fbeta": float(calc_fbeta(
                precision_score(y_oof, pred_oof, zero_division=0),
                recall_score(y_oof, pred_oof, zero_division=0),
                beta=2.0,
            )),
            "confusion_matrix": pooled_cm.tolist(),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "fixed_params_family_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\\nSaved outputs to:", OUTPUT_DIR)
    print("- fixed_params_family_fold_metrics.csv")
    print("- fixed_params_family_summary_mean_std.csv")
    print("- fixed_params_family_oof_predictions.csv")
    print("- fixed_params_family_report.json")
    print("- outer_fold_*_bundle.joblib")


if __name__ == "__main__":
    main()
