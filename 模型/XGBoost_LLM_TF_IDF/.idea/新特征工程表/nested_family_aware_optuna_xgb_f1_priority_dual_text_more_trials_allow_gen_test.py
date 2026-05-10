
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
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
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "nested_family_optuna_outputs_f1_priority_dual_text"
RANDOM_STATE = 42

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
N_TRIALS = 150

# 是否允许生成正样本进入 outer test set
ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST = True

# 双文本并联
README_TEXT_COL = "readme_text_clean"
DESCRIPTION_TEXT_COL = "description_text_clean"
TOPICS_TEXT_COL = "topics_text_clean"

# 目标函数保持不变
MIN_F1_TARGET = 0.80
F1_SHORTFALL_PENALTY = 2.0
OBJ_WEIGHT_F1 = 1.00
OBJ_WEIGHT_PR_AUC = 0.03

BASE_WEIGHT_REAL_POS = 3.0
BASE_WEIGHT_GENERATED_POS = 1.25
BASE_WEIGHT_NEG = 1.0

FILLNA_STRATEGY = "median"

NON_STRUCTURED_COLS = {
    "label",
    "family_id",
    "repo_full_name",
    "readme_text",
    "readme_text_raw",
    "readme_text_clean",
    "description_text",
    "topics_text",
    "combined_text",
    "combined_text_clean",
    "description_text_clean",
    "topics_text_clean",
    "text_for_tfidf_clean",
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
META_LEAKY_PREFIXES = (
    "meta_",
    "meta__",
    "source_",
    "generated_",
    "augmentation_",
)

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b"


# =========================================================
# 1. UTILS
# =========================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def prepare_structured_features(df: pd.DataFrame, structured_cols: list[str]) -> pd.DataFrame:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def fillna_by_train_statistics(X_tr, X_val, X_test, strategy="median"):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy must be 'median' or 'zero'")

    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_val = X_val.fillna(fill_values).fillna(0)
    X_test = X_test.fillna(fill_values).fillna(0)
    return X_tr, X_val, X_test, fill_values


def build_sample_weights(df, w_real_pos, w_generated_pos, w_neg):
    w = np.full(len(df), w_neg, dtype=float)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = w_real_pos
    w[gen_mask] = w_generated_pos
    return w


def build_matrices(
    train_df,
    val_df,
    test_df,
    structured_cols,
    readme_max_features,
    readme_min_df,
    readme_ngram_range,
    aux_max_features,
    aux_min_df,
    aux_ngram_range,
):
    # numeric
    X_tr_num = prepare_structured_features(train_df, structured_cols)
    X_val_num = prepare_structured_features(val_df, structured_cols)
    X_test_num = prepare_structured_features(test_df, structured_cols)

    X_tr_num, X_val_num, X_test_num, fill_values = fillna_by_train_statistics(
        X_tr_num, X_val_num, X_test_num, strategy=FILLNA_STRATEGY
    )

    X_tr_num_sp = csr_matrix(X_tr_num.to_numpy(dtype=np.float32))
    X_val_num_sp = csr_matrix(X_val_num.to_numpy(dtype=np.float32))
    X_test_num_sp = csr_matrix(X_test_num.to_numpy(dtype=np.float32))

    # README TF-IDF
    readme_vectorizer = TfidfVectorizer(
        max_features=readme_max_features,
        min_df=readme_min_df,
        ngram_range=readme_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    tr_readme = train_df[README_TEXT_COL].fillna("").astype(str)
    val_readme = val_df[README_TEXT_COL].fillna("").astype(str)
    test_readme = test_df[README_TEXT_COL].fillna("").astype(str)

    X_tr_readme = readme_vectorizer.fit_transform(tr_readme)
    X_val_readme = readme_vectorizer.transform(val_readme)
    X_test_readme = readme_vectorizer.transform(test_readme)

    # AUX TF-IDF: description + topics
    aux_vectorizer = TfidfVectorizer(
        max_features=aux_max_features,
        min_df=aux_min_df,
        ngram_range=aux_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    tr_aux = get_aux_text_series(train_df)
    val_aux = get_aux_text_series(val_df)
    test_aux = get_aux_text_series(test_df)

    X_tr_aux = aux_vectorizer.fit_transform(tr_aux)
    X_val_aux = aux_vectorizer.transform(val_aux)
    X_test_aux = aux_vectorizer.transform(test_aux)

    # concat
    X_tr_all = hstack([X_tr_num_sp, X_tr_readme, X_tr_aux], format="csr")
    X_val_all = hstack([X_val_num_sp, X_val_readme, X_val_aux], format="csr")
    X_test_all = hstack([X_test_num_sp, X_test_readme, X_test_aux], format="csr")

    vectorizers = {
        "readme_vectorizer": readme_vectorizer,
        "aux_vectorizer": aux_vectorizer,
    }
    return X_tr_all, X_val_all, X_test_all, vectorizers, fill_values


def build_xgb_params(trial: optuna.Trial | None) -> dict:
    if trial is None:
        return {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "importance_type": "gain",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "gamma": 1.0,
            "max_delta_step": 0.5,
        }

    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "importance_type": "gain",
        "n_estimators": trial.suggest_int("n_estimators", 400, 1400, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 8.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 6.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 2.0),
    }


def compute_trial_score(y_true, y_proba):
    # 目标函数不调整
    pr_auc = average_precision_score(y_true, y_proba)
    thr, thr_info = select_best_threshold_f1_priority(y_true, y_proba, beta=2.0)
    f1 = thr_info["f1"]
    score = (
        OBJ_WEIGHT_F1 * f1
        + OBJ_WEIGHT_PR_AUC * pr_auc
        - F1_SHORTFALL_PENALTY * max(0.0, MIN_F1_TARGET - f1)
    )
    info = {
        "pr_auc": float(pr_auc),
        "best_threshold": float(thr),
        "precision": float(thr_info["precision"]),
        "recall": float(thr_info["recall"]),
        "f1": float(f1),
        "f_beta": float(thr_info["f_beta"]),
        "score": float(score),
    }
    return score, info


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
    leakage_candidates = [c for c in df.columns if is_leaky_feature_col(c)]

    real_pos_mask = df["is_real_positive"] == 1
    gen_pos_mask = df["is_generated_positive"] == 1
    neg_mask = df["label"] == 0

    real_pos_df = df.loc[real_pos_mask].copy()
    neg_indices_all = df.index[neg_mask].to_numpy()

    print("Total rows:", len(df))
    print("Real positives:", int(real_pos_mask.sum()))
    print("Generated positives:", int(gen_pos_mask.sum()))
    print("ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST:", ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST)
    print("Negatives:", int(neg_mask.sum()))
    print("Dual text streams:", [README_TEXT_COL, f"{DESCRIPTION_TEXT_COL}+{TOPICS_TEXT_COL}"])
    print("Unique real-positive families:", real_pos_df["family_id"].nunique())
    print("Structured feature count:", len(structured_cols))
    print("Excluded leakage-prone cols:", leakage_candidates)

    outer_pos_family_folds = make_positive_family_folds(real_pos_df, OUTER_N_SPLITS, RANDOM_STATE)
    outer_neg_folds = make_index_folds(neg_indices_all, len(outer_pos_family_folds), RANDOM_STATE)

    fold_metrics = []
    oof_records = []
    best_params_by_fold = []

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
        outer_train_df = df.loc[outer_train_mask].copy()

        overlap_rows = outer_train_df.loc[
            (outer_train_df["is_generated_positive"] == 1)
            & (outer_train_df["family_id"].isin(outer_test_fams))
        ]
        if len(overlap_rows) > 0:
            raise RuntimeError(f"Leakage detected: {len(overlap_rows)} generated positives from test families still in outer_train")

        outer_train_real_pos = outer_train_df.loc[outer_train_df["is_real_positive"] == 1].copy()
        outer_train_neg_idx = outer_train_df.index[outer_train_df["label"] == 0].to_numpy()

        inner_pos_family_folds = make_positive_family_folds(
            outer_train_real_pos, INNER_N_SPLITS, RANDOM_STATE + outer_fold_id
        )
        inner_neg_folds = make_index_folds(
            outer_train_neg_idx, len(inner_pos_family_folds), RANDOM_STATE + outer_fold_id
        )

        def objective(trial):
            # TF-IDF 范围微调：README 容量中等，AUX 容量较小更稳
            readme_max_features = trial.suggest_int("readme_max_features", 250, 650, step=50)
            readme_min_df = trial.suggest_int("readme_min_df", 2, 4)
            readme_ngram_high = trial.suggest_int("readme_ngram_high", 1, 2)
            readme_ngram_range = (1, readme_ngram_high)

            aux_max_features = trial.suggest_int("aux_max_features", 50, 250, step=50)
            aux_min_df = trial.suggest_int("aux_min_df", 1, 3)
            aux_ngram_high = trial.suggest_int("aux_ngram_high", 1, 2)
            aux_ngram_range = (1, aux_ngram_high)

            # 权重区间微调：收窄到更稳健的范围
            real_pos_weight_mult = trial.suggest_float("real_pos_weight_mult", 1.2, 2.2)
            gen_pos_weight_mult = trial.suggest_float("gen_pos_weight_mult", 0.5, 1.2)
            neg_weight_mult = trial.suggest_float("neg_weight_mult", 0.9, 1.5)

            w_real_pos = BASE_WEIGHT_REAL_POS * real_pos_weight_mult
            w_generated_pos = BASE_WEIGHT_GENERATED_POS * gen_pos_weight_mult
            w_neg = BASE_WEIGHT_NEG * neg_weight_mult

            params = build_xgb_params(trial)

            pooled_val_rows = []
            pooled_val_proba = []

            for inner_id, ((inner_train_fams, inner_val_fams), (_, inner_val_neg_idx)) in enumerate(
                zip(inner_pos_family_folds, inner_neg_folds), start=1
            ):
                val_df = outer_train_df.loc[
                    ((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams)))
                    | (outer_train_df.index.isin(inner_val_neg_idx))
                ].copy()

                train_df = outer_train_df.loc[
                    (~outer_train_df.index.isin(inner_val_neg_idx))
                    & (~((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
                    & (~((outer_train_df["is_generated_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
                ].copy()

                X_tr, X_val, _, _, _ = build_matrices(
                    train_df,
                    val_df,
                    val_df,
                    structured_cols=structured_cols,
                    readme_max_features=readme_max_features,
                    readme_min_df=readme_min_df,
                    readme_ngram_range=readme_ngram_range,
                    aux_max_features=aux_max_features,
                    aux_min_df=aux_min_df,
                    aux_ngram_range=aux_ngram_range,
                )
                y_tr = train_df["label"].values
                sw = build_sample_weights(train_df, w_real_pos, w_generated_pos, w_neg)

                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
                val_proba = model.predict_proba(X_val)[:, 1]

                tmp = val_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
                tmp["row_index"] = val_df.index.values
                pooled_val_rows.append(tmp)
                pooled_val_proba.append(val_proba)

            pooled_val_df = pd.concat(pooled_val_rows, axis=0, ignore_index=True)
            pooled_val_df["y_proba"] = np.concatenate(pooled_val_proba)

            score, info = compute_trial_score(
                pooled_val_df["label"].values,
                pooled_val_df["y_proba"].values
            )

            trial.set_user_attr("text_mode", "dual_text")
            trial.set_user_attr("val_best_threshold", float(info["best_threshold"]))
            trial.set_user_attr("val_f1", float(info["f1"]))
            trial.set_user_attr("val_precision", float(info["precision"]))
            trial.set_user_attr("val_recall", float(info["recall"]))
            trial.set_user_attr("val_ap", float(info["pr_auc"]))
            trial.set_user_attr("actual_w_real_pos", float(w_real_pos))
            trial.set_user_attr("actual_w_generated_pos", float(w_generated_pos))
            trial.set_user_attr("actual_w_neg", float(w_neg))
            return float(score)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + outer_fold_id),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial

        print("Best inner objective:", round(best_value, 6))
        print("Best params:", best_params)
        print("Best attrs:", best_trial.user_attrs)

        readme_max_features = int(best_params["readme_max_features"])
        readme_min_df = int(best_params["readme_min_df"])
        readme_ngram_range = (1, int(best_params["readme_ngram_high"]))

        aux_max_features = int(best_params["aux_max_features"])
        aux_min_df = int(best_params["aux_min_df"])
        aux_ngram_range = (1, int(best_params["aux_ngram_high"]))

        w_real_pos = float(best_trial.user_attrs["actual_w_real_pos"])
        w_generated_pos = float(best_trial.user_attrs["actual_w_generated_pos"])
        w_neg = float(best_trial.user_attrs["actual_w_neg"])

        xgb_best_params = {
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

        pooled_val_rows = []
        pooled_val_proba = []

        for inner_id, ((inner_train_fams, inner_val_fams), (_, inner_val_neg_idx)) in enumerate(
            zip(inner_pos_family_folds, inner_neg_folds), start=1
        ):
            val_df = outer_train_df.loc[
                ((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams)))
                | (outer_train_df.index.isin(inner_val_neg_idx))
            ].copy()

            train_df = outer_train_df.loc[
                (~outer_train_df.index.isin(inner_val_neg_idx))
                & (~((outer_train_df["is_real_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
                & (~((outer_train_df["is_generated_positive"] == 1) & (outer_train_df["family_id"].isin(inner_val_fams))))
            ].copy()

            X_tr, X_val, _, _, _ = build_matrices(
                train_df,
                val_df,
                val_df,
                structured_cols=structured_cols,
                readme_max_features=readme_max_features,
                readme_min_df=readme_min_df,
                readme_ngram_range=readme_ngram_range,
                aux_max_features=aux_max_features,
                aux_min_df=aux_min_df,
                aux_ngram_range=aux_ngram_range,
            )
            y_tr = train_df["label"].values
            sw = build_sample_weights(train_df, w_real_pos, w_generated_pos, w_neg)

            model = XGBClassifier(**xgb_best_params)
            model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
            val_proba = model.predict_proba(X_val)[:, 1]

            tmp = val_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
            tmp["row_index"] = val_df.index.values
            pooled_val_rows.append(tmp)
            pooled_val_proba.append(val_proba)

        pooled_val_df = pd.concat(pooled_val_rows, axis=0, ignore_index=True)
        pooled_val_df["y_proba"] = np.concatenate(pooled_val_proba)

        pooled_threshold, pooled_thr_info = select_best_threshold_f1_priority(
            pooled_val_df["label"].values,
            pooled_val_df["y_proba"].values,
            beta=2.0,
        )
        print(
            f"Chosen threshold from pooled inner val = {pooled_threshold:.6f} | "
            f"P={pooled_thr_info['precision']:.4f} R={pooled_thr_info['recall']:.4f} F1={pooled_thr_info['f1']:.4f}"
        )

        test_df = df.loc[outer_test_idx].copy()
        train_df = outer_train_df.copy()

        X_tr_final, _, X_te_final, vectorizers, fill_values = build_matrices(
            train_df,
            test_df,
            test_df,
            structured_cols=structured_cols,
            readme_max_features=readme_max_features,
            readme_min_df=readme_min_df,
            readme_ngram_range=readme_ngram_range,
            aux_max_features=aux_max_features,
            aux_min_df=aux_min_df,
            aux_ngram_range=aux_ngram_range,
        )
        y_tr_final = train_df["label"].values
        y_te = test_df["label"].values
        sw_final = build_sample_weights(train_df, w_real_pos, w_generated_pos, w_neg)

        final_model = XGBClassifier(**xgb_best_params)
        final_model.fit(X_tr_final, y_tr_final, sample_weight=sw_final, verbose=False)
        test_proba = final_model.predict_proba(X_te_final)[:, 1]
        test_pred = (test_proba >= pooled_threshold).astype(int)

        test_ap = average_precision_score(y_te, test_proba)
        test_roc = roc_auc_score(y_te, test_proba)
        test_precision = precision_score(y_te, test_pred, zero_division=0)
        test_recall = recall_score(y_te, test_pred, zero_division=0)
        test_f1 = f1_score(y_te, test_pred, zero_division=0)
        test_fbeta = calc_fbeta(test_precision, test_recall, beta=2.0)
        test_cm = confusion_matrix(y_te, test_pred, labels=[0, 1])

        fold_metrics.append(
            {
                "outer_fold": outer_fold_id,
                "text_mode": "dual_text",
                "n_test_total": int(len(test_df)),
                "n_test_pos_real": int((test_df["is_real_positive"] == 1).sum()),
                "n_test_pos_generated": int((test_df["is_generated_positive"] == 1).sum()),
                "n_test_pos_total": int((test_df["label"] == 1).sum()),
                "n_test_neg": int((test_df["label"] == 0).sum()),
                "best_inner_objective": float(best_value),
                "threshold": float(pooled_threshold),
                "val_precision_at_thr": float(pooled_thr_info["precision"]),
                "val_recall_at_thr": float(pooled_thr_info["recall"]),
                "val_f1_at_thr": float(pooled_thr_info["f1"]),
                "val_fbeta_at_thr": float(pooled_thr_info["f_beta"]),
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
            }
        )

        best_params_by_fold.append(
            {
                "outer_fold": outer_fold_id,
                "text_mode": "dual_text",
                "readme_text_col": README_TEXT_COL,
                "aux_text_cols": [DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL],
                "readme_max_features": readme_max_features,
                "readme_min_df": readme_min_df,
                "readme_ngram_range": list(readme_ngram_range),
                "aux_max_features": aux_max_features,
                "aux_min_df": aux_min_df,
                "aux_ngram_range": list(aux_ngram_range),
                "xgb_best_params": xgb_best_params,
                "best_inner_objective": float(best_value),
                "threshold": float(pooled_threshold),
                "weights": {
                    "real_positive": float(w_real_pos),
                    "generated_positive": float(w_generated_pos),
                    "negative": float(w_neg),
                },
                "optuna_user_attrs": best_trial.user_attrs,
            }
        )

        tmp = test_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
        tmp["row_index"] = test_df.index.values
        tmp["outer_fold"] = outer_fold_id
        tmp["y_proba"] = test_proba
        tmp["y_pred"] = test_pred
        tmp["threshold"] = pooled_threshold
        tmp["text_mode"] = "dual_text"
        if "repo_full_name" in test_df.columns:
            tmp["repo_full_name"] = test_df["repo_full_name"].values
        oof_records.append(tmp)

        joblib.dump(
            {
                "model": final_model,
                "vectorizers": vectorizers,
                "fill_values": fill_values.to_dict(),
                "structured_cols": structured_cols,
                "threshold": pooled_threshold,
                "text_mode": "dual_text",
                "params": xgb_best_params,
                "weights": {
                    "real_positive": w_real_pos,
                    "generated_positive": w_generated_pos,
                    "negative": w_neg,
                },
            },
            os.path.join(OUTPUT_DIR, f"outer_fold_{outer_fold_id}_bundle.joblib"),
        )

        study.trials_dataframe().to_csv(
            os.path.join(OUTPUT_DIR, f"outer_fold_{outer_fold_id}_optuna_trials.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "nested_family_fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    metric_cols = [
        "best_inner_objective",
        "threshold",
        "val_f1_at_thr",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fbeta",
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
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "nested_family_summary_mean_std.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    oof_df.to_csv(
        os.path.join(OUTPUT_DIR, "nested_family_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    y_oof = oof_df["label"].values
    p_oof = oof_df["y_proba"].values
    pooled_threshold, pooled_info = select_best_threshold_f1_priority(y_oof, p_oof, beta=2.0)
    pred_oof = (p_oof >= pooled_threshold).astype(int)
    pooled_cm = confusion_matrix(y_oof, pred_oof, labels=[0, 1])

    report = {
        "config": {
            "data_path": str(DATA_PATH),
            "outer_n_splits": OUTER_N_SPLITS,
            "inner_n_splits": INNER_N_SPLITS,
            "n_trials": N_TRIALS,
            "allow_generated_positives_in_outer_test": ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST,
            "text_mode": "dual_text",
            "readme_text_col": README_TEXT_COL,
            "aux_text_cols": [DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL],
            "min_f1_target": MIN_F1_TARGET,
            "base_weights": {
                "real_positive": BASE_WEIGHT_REAL_POS,
                "generated_positive": BASE_WEIGHT_GENERATED_POS,
                "negative": BASE_WEIGHT_NEG,
            },
            "meta_leaky_prefixes": list(META_LEAKY_PREFIXES),
        },
        "pooled_oof": {
            "threshold": float(pooled_threshold),
            "threshold_info": pooled_info,
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_oof, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_oof, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_oof, zero_division=0)),
            "fbeta": float(
                calc_fbeta(
                    precision_score(y_oof, pred_oof, zero_division=0),
                    recall_score(y_oof, pred_oof, zero_division=0),
                    beta=2.0,
                )
            ),
            "confusion_matrix": pooled_cm.tolist(),
        },
        "best_params_by_fold": best_params_by_fold,
    }

    with open(os.path.join(OUTPUT_DIR, "nested_family_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\\nSaved outputs to:", OUTPUT_DIR)
    print("- nested_family_fold_metrics.csv")
    print("- nested_family_summary_mean_std.csv")
    print("- nested_family_oof_predictions.csv")
    print("- nested_family_report.json")
    print("- outer_fold_*_optuna_trials.csv")
    print("- outer_fold_*_bundle.joblib")


if __name__ == "__main__":
    main()
