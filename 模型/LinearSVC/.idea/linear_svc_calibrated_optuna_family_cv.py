
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# =========================================================
# 0. CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "linear_svc_calibrated_optuna_outputs"

RANDOM_STATE = 42
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
N_TRIALS = 80

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

# 保持与你现有主模型一致的样本权重逻辑，便于公平比较
REAL_POS_WEIGHT = 4.5
GEN_POS_WEIGHT = 1.0
NEG_WEIGHT = 1.25
BETA = 2.0


# =========================================================
# 1. UTILS
# =========================================================
def ensure_dir(path: Path):
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


def prepare_structured_features(df: pd.DataFrame, structured_cols: list[str]) -> pd.DataFrame:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def fillna_and_scale_numeric(X_tr: pd.DataFrame, X_te: pd.DataFrame, strategy: str = "median"):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy must be 'median' or 'zero'")

    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_te = X_te.fillna(fill_values).fillna(0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.values.astype(np.float32))
    X_te_scaled = scaler.transform(X_te.values.astype(np.float32))
    return csr_matrix(X_tr_scaled), csr_matrix(X_te_scaled), fill_values, scaler


def build_sample_weights(df: pd.DataFrame):
    w = np.full(len(df), NEG_WEIGHT, dtype=float)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = REAL_POS_WEIGHT
    w[gen_mask] = GEN_POS_WEIGHT
    return w


def build_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    structured_cols: list[str],
    readme_max_features: int,
    readme_min_df: int,
    readme_ngram_range: tuple,
    aux_max_features: int,
    aux_min_df: int,
    aux_ngram_range: tuple,
):
    X_tr_num = prepare_structured_features(train_df, structured_cols)
    X_te_num = prepare_structured_features(test_df, structured_cols)
    X_tr_num_sp, X_te_num_sp, fill_values, scaler = fillna_and_scale_numeric(
        X_tr_num, X_te_num, strategy=FILLNA_STRATEGY
    )

    readme_vectorizer = TfidfVectorizer(
        max_features=readme_max_features,
        min_df=readme_min_df,
        ngram_range=readme_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    X_tr_readme = readme_vectorizer.fit_transform(train_df[README_TEXT_COL].fillna("").astype(str))
    X_te_readme = readme_vectorizer.transform(test_df[README_TEXT_COL].fillna("").astype(str))

    aux_vectorizer = TfidfVectorizer(
        max_features=aux_max_features,
        min_df=aux_min_df,
        ngram_range=aux_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    X_tr_aux = aux_vectorizer.fit_transform(get_aux_text_series(train_df))
    X_te_aux = aux_vectorizer.transform(get_aux_text_series(test_df))

    X_tr_all = hstack([X_tr_num_sp, X_tr_readme, X_tr_aux], format="csr")
    X_te_all = hstack([X_te_num_sp, X_te_readme, X_te_aux], format="csr")

    artifacts = {
        "readme_vectorizer": readme_vectorizer,
        "aux_vectorizer": aux_vectorizer,
        "fill_values": fill_values.to_dict(),
        "numeric_scaler": scaler,
        "structured_cols": structured_cols,
    }
    return X_tr_all, X_te_all, artifacts


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


def fit_calibrated_svc(X_tr, y_tr, sample_weight, C_value: float, calibration_method: str, calibration_cv: int):
    base = LinearSVC(
        C=C_value,
        penalty="l2",
        loss="squared_hinge",
        dual="auto",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    clf = CalibratedClassifierCV(
        estimator=base,
        method=calibration_method,
        cv=calibration_cv,
    )
    try:
        clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    except TypeError:
        clf.fit(X_tr, y_tr)
    return clf


# =========================================================
# 2. MAIN
# =========================================================
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
    print("Negatives:", int(neg_mask.sum()))
    print("ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST:", ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST)
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

        outer_train_real_pos = outer_train_df.loc[outer_train_df["is_real_positive"] == 1].copy()
        outer_train_neg_idx = outer_train_df.index[outer_train_df["label"] == 0].to_numpy()

        inner_pos_family_folds = make_positive_family_folds(
            outer_train_real_pos, INNER_N_SPLITS, RANDOM_STATE + outer_fold_id
        )
        inner_neg_folds = make_index_folds(
            outer_train_neg_idx, len(inner_pos_family_folds), RANDOM_STATE + outer_fold_id
        )

        def objective(trial: optuna.Trial):
            C_value = trial.suggest_float("C", 1e-3, 20.0, log=True)
            calibration_method = trial.suggest_categorical("calibration_method", ["sigmoid", "isotonic"])

            readme_max_features = trial.suggest_int("readme_max_features", 200, 800, step=100)
            readme_min_df = trial.suggest_int("readme_min_df", 2, 5)
            readme_ngram_high = trial.suggest_int("readme_ngram_high", 1, 2)

            aux_max_features = trial.suggest_int("aux_max_features", 50, 300, step=50)
            aux_min_df = trial.suggest_int("aux_min_df", 1, 3)
            aux_ngram_high = trial.suggest_int("aux_ngram_high", 1, 2)

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

                X_tr, X_val, _ = build_matrices(
                    train_df=train_df,
                    test_df=val_df,
                    structured_cols=structured_cols,
                    readme_max_features=readme_max_features,
                    readme_min_df=readme_min_df,
                    readme_ngram_range=(1, readme_ngram_high),
                    aux_max_features=aux_max_features,
                    aux_min_df=aux_min_df,
                    aux_ngram_range=(1, aux_ngram_high),
                )
                y_tr = train_df["label"].values
                sw = build_sample_weights(train_df)

                clf = fit_calibrated_svc(
                    X_tr, y_tr, sw,
                    C_value=C_value,
                    calibration_method=calibration_method,
                    calibration_cv=3,
                )
                val_proba = clf.predict_proba(X_val)[:, 1]

                tmp = val_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
                tmp["row_index"] = val_df.index.values
                pooled_val_rows.append(tmp)
                pooled_val_proba.append(val_proba)

            pooled_val_df = pd.concat(pooled_val_rows, axis=0, ignore_index=True)
            pooled_val_df["y_proba"] = np.concatenate(pooled_val_proba)

            thr, thr_info = select_best_threshold_f1_priority(
                pooled_val_df["label"].values,
                pooled_val_df["y_proba"].values,
                beta=BETA,
            )
            score = thr_info["f1"]

            trial.set_user_attr("threshold", float(thr))
            trial.set_user_attr("precision", float(thr_info["precision"]))
            trial.set_user_attr("recall", float(thr_info["recall"]))
            trial.set_user_attr("f1", float(thr_info["f1"]))
            return float(score)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE + outer_fold_id),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_trial = study.best_trial
        best_params = best_trial.params

        best_C = float(best_params["C"])
        best_calibration_method = best_params["calibration_method"]
        best_threshold = float(best_trial.user_attrs["threshold"])

        readme_max_features = int(best_params["readme_max_features"])
        readme_min_df = int(best_params["readme_min_df"])
        readme_ngram_range = (1, int(best_params["readme_ngram_high"]))
        aux_max_features = int(best_params["aux_max_features"])
        aux_min_df = int(best_params["aux_min_df"])
        aux_ngram_range = (1, int(best_params["aux_ngram_high"]))

        print(
            f"Best trial | C={best_C:.6f} | calib={best_calibration_method} | "
            f"thr={best_threshold:.6f} | P={best_trial.user_attrs['precision']:.4f} "
            f"R={best_trial.user_attrs['recall']:.4f} F1={best_trial.user_attrs['f1']:.4f}"
        )

        test_df = df.loc[outer_test_idx].copy()
        train_df = outer_train_df.copy()

        X_tr_final, X_te_final, artifacts = build_matrices(
            train_df=train_df,
            test_df=test_df,
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
        sw_final = build_sample_weights(train_df)

        final_clf = fit_calibrated_svc(
            X_tr_final, y_tr_final, sw_final,
            C_value=best_C,
            calibration_method=best_calibration_method,
            calibration_cv=3,
        )
        test_proba = final_clf.predict_proba(X_te_final)[:, 1]
        test_pred = (test_proba >= best_threshold).astype(int)

        test_ap = average_precision_score(y_te, test_proba)
        test_roc = roc_auc_score(y_te, test_proba)
        test_precision = precision_score(y_te, test_pred, zero_division=0)
        test_recall = recall_score(y_te, test_pred, zero_division=0)
        test_f1 = f1_score(y_te, test_pred, zero_division=0)
        test_fbeta = calc_fbeta(test_precision, test_recall, beta=BETA)
        test_cm = confusion_matrix(y_te, test_pred, labels=[0, 1])

        fold_metrics.append({
            "outer_fold": outer_fold_id,
            "model": "LinearSVC+CalibratedClassifierCV+Optuna",
            "n_test_total": int(len(test_df)),
            "n_test_pos_real": int((test_df["is_real_positive"] == 1).sum()),
            "n_test_pos_generated": int((test_df["is_generated_positive"] == 1).sum()),
            "n_test_pos_total": int((test_df["label"] == 1).sum()),
            "n_test_neg": int((test_df["label"] == 0).sum()),
            "best_C": best_C,
            "calibration_method": best_calibration_method,
            "threshold": best_threshold,
            "readme_max_features": readme_max_features,
            "readme_min_df": readme_min_df,
            "readme_ngram_range_high": readme_ngram_range[1],
            "aux_max_features": aux_max_features,
            "aux_min_df": aux_min_df,
            "aux_ngram_range_high": aux_ngram_range[1],
            "val_precision_at_thr": float(best_trial.user_attrs["precision"]),
            "val_recall_at_thr": float(best_trial.user_attrs["recall"]),
            "val_f1_at_thr": float(best_trial.user_attrs["f1"]),
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

        best_params_by_fold.append({
            "outer_fold": outer_fold_id,
            "model": "LinearSVC+CalibratedClassifierCV+Optuna",
            "best_C": best_C,
            "calibration_method": best_calibration_method,
            "threshold": best_threshold,
            "text_params": {
                "readme_max_features": readme_max_features,
                "readme_min_df": readme_min_df,
                "readme_ngram_range": [1, readme_ngram_range[1]],
                "aux_max_features": aux_max_features,
                "aux_min_df": aux_min_df,
                "aux_ngram_range": [1, aux_ngram_range[1]],
            },
        })

        tmp = test_df[["family_id", "label", "is_real_positive", "is_generated_positive"]].copy()
        tmp["row_index"] = test_df.index.values
        tmp["outer_fold"] = outer_fold_id
        tmp["y_proba"] = test_proba
        tmp["y_pred"] = test_pred
        tmp["threshold"] = best_threshold
        tmp["model"] = "LinearSVC+CalibratedClassifierCV+Optuna"
        if "repo_full_name" in test_df.columns:
            tmp["repo_full_name"] = test_df["repo_full_name"].values
        oof_records.append(tmp)

        joblib.dump(
            {
                "model": final_clf,
                "threshold": best_threshold,
                "best_C": best_C,
                "calibration_method": best_calibration_method,
                "artifacts": artifacts,
                "text_params": {
                    "readme_max_features": readme_max_features,
                    "readme_min_df": readme_min_df,
                    "readme_ngram_range": readme_ngram_range,
                    "aux_max_features": aux_max_features,
                    "aux_min_df": aux_min_df,
                    "aux_ngram_range": aux_ngram_range,
                },
            },
            OUTPUT_DIR / f"outer_fold_{outer_fold_id}_bundle.joblib",
        )

        study.trials_dataframe().to_csv(
            OUTPUT_DIR / f"outer_fold_{outer_fold_id}_optuna_trials.csv",
            index=False,
            encoding="utf-8-sig",
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(OUTPUT_DIR / "linear_svc_optuna_fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    metric_cols = ["best_C","threshold","test_pr_auc","test_roc_auc","test_precision","test_recall","test_f1","test_fbeta"]
    for c in metric_cols:
        summary_rows.append({
            "metric": c,
            "mean": float(fold_metrics_df[c].mean()),
            "std": float(fold_metrics_df[c].std(ddof=1)) if len(fold_metrics_df) > 1 else 0.0,
            "min": float(fold_metrics_df[c].min()),
            "max": float(fold_metrics_df[c].max()),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "linear_svc_optuna_summary_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_df = pd.concat(oof_records, axis=0, ignore_index=True)
    oof_df.to_csv(OUTPUT_DIR / "linear_svc_optuna_oof_predictions.csv", index=False, encoding="utf-8-sig")

    y_oof = oof_df["label"].values
    p_oof = oof_df["y_proba"].values
    pooled_threshold, pooled_info = select_best_threshold_f1_priority(y_oof, p_oof, beta=BETA)
    pred_oof = (p_oof >= pooled_threshold).astype(int)
    pooled_cm = confusion_matrix(y_oof, pred_oof, labels=[0, 1])

    report = {
        "config": {
            "data_path": str(DATA_PATH),
            "outer_n_splits": OUTER_N_SPLITS,
            "inner_n_splits": INNER_N_SPLITS,
            "n_trials": N_TRIALS,
            "allow_generated_positives_in_outer_test": ALLOW_GENERATED_POSITIVES_IN_OUTER_TEST,
            "model": "LinearSVC+CalibratedClassifierCV+Optuna",
            "readme_text_col": README_TEXT_COL,
            "aux_text_cols": [DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL],
        },
        "pooled_oof": {
            "threshold": float(pooled_threshold),
            "threshold_info": pooled_info,
            "pr_auc": float(average_precision_score(y_oof, p_oof)),
            "roc_auc": float(roc_auc_score(y_oof, p_oof)),
            "precision": float(precision_score(y_oof, pred_oof, zero_division=0)),
            "recall": float(recall_score(y_oof, pred_oof, zero_division=0)),
            "f1": float(f1_score(y_oof, pred_oof, zero_division=0)),
            "fbeta": float(calc_fbeta(
                precision_score(y_oof, pred_oof, zero_division=0),
                recall_score(y_oof, pred_oof, zero_division=0),
                beta=BETA,
            )),
            "confusion_matrix": pooled_cm.tolist(),
        },
        "best_params_by_fold": best_params_by_fold,
    }

    with open(OUTPUT_DIR / "linear_svc_optuna_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\\nSaved outputs to:", OUTPUT_DIR)
    print("- linear_svc_optuna_fold_metrics.csv")
    print("- linear_svc_optuna_summary_mean_std.csv")
    print("- linear_svc_optuna_oof_predictions.csv")
    print("- linear_svc_optuna_report.json")
    print("- outer_fold_*_optuna_trials.csv")
    print("- outer_fold_*_bundle.joblib")


if __name__ == "__main__":
    main()
