
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "family_prepared_v3" / "family_dataset_train_ready.csv"
OUTPUT_DIR = SCRIPT_DIR / "final_linear_svc_calibrated_artifacts"

RANDOM_STATE = 42

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
    "readme_max_features": 300,
    "readme_min_df": 3,
    "readme_ngram_range": (1, 2),
    "aux_max_features": 200,
    "aux_min_df": 3,
    "aux_ngram_range": (1, 2),
    "linear_svc": {
        "C": 0.0015,
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": "auto",
        "fit_intercept": True,
        "max_iter": 10000,
        "random_state": RANDOM_STATE,
    },
    "calibration": {
        "method": "sigmoid",
        "cv": 3,
    },
    "weights": {
        "real_positive": 4.5,
        "generated_positive": 1.0,
        "negative": 1.25,
    },
    "threshold": 0.2379615693,
}


def ensure_dir(path: Path):
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


def prepare_structured_features(df: pd.DataFrame, structured_cols: list[str]) -> pd.DataFrame:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def fillna_and_scale_numeric(X_tr: pd.DataFrame, strategy: str = "median"):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy must be 'median' or 'zero'")

    X_tr = X_tr.fillna(fill_values).fillna(0)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.values.astype(np.float32))
    return csr_matrix(X_tr_scaled), fill_values, scaler


def build_sample_weights(df: pd.DataFrame, w_real_pos: float, w_generated_pos: float, w_neg: float):
    w = np.full(len(df), w_neg, dtype=float)
    real_mask = df.get("is_real_positive", 0).fillna(0).astype(int).values == 1
    gen_mask = df.get("is_generated_positive", 0).fillna(0).astype(int).values == 1
    w[real_mask] = w_real_pos
    w[gen_mask] = w_generated_pos
    return w


def main():
    ensure_dir(OUTPUT_DIR)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到训练数据: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"label", README_TEXT_COL, DESCRIPTION_TEXT_COL, TOPICS_TEXT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df[README_TEXT_COL] = df[README_TEXT_COL].fillna("").astype(str)
    df[DESCRIPTION_TEXT_COL] = df[DESCRIPTION_TEXT_COL].fillna("").astype(str)
    df[TOPICS_TEXT_COL] = df[TOPICS_TEXT_COL].fillna("").astype(str)

    structured_cols = [c for c in df.columns if not is_leaky_feature_col(c)]
    leakage_candidates = [c for c in df.columns if is_leaky_feature_col(c)]

    print("Total rows:", len(df))
    print("Positives:", int((df["label"] == 1).sum()))
    print("Negatives:", int((df["label"] == 0).sum()))
    print("Structured feature count:", len(structured_cols))
    print("Excluded leakage-prone cols:", leakage_candidates)

    X_num = prepare_structured_features(df, structured_cols)
    X_num_sp, fill_values, scaler = fillna_and_scale_numeric(X_num, strategy=FILLNA_STRATEGY)

    readme_vectorizer = TfidfVectorizer(
        max_features=FINAL_PARAMS["readme_max_features"],
        min_df=FINAL_PARAMS["readme_min_df"],
        ngram_range=FINAL_PARAMS["readme_ngram_range"],
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    X_readme = readme_vectorizer.fit_transform(df[README_TEXT_COL])

    aux_vectorizer = TfidfVectorizer(
        max_features=FINAL_PARAMS["aux_max_features"],
        min_df=FINAL_PARAMS["aux_min_df"],
        ngram_range=FINAL_PARAMS["aux_ngram_range"],
        sublinear_tf=True,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
    )
    X_aux = aux_vectorizer.fit_transform(get_aux_text_series(df))

    X_all = hstack([X_num_sp, X_readme, X_aux], format="csr")
    y_all = df["label"].values

    sample_weight = build_sample_weights(
        df,
        w_real_pos=FINAL_PARAMS["weights"]["real_positive"],
        w_generated_pos=FINAL_PARAMS["weights"]["generated_positive"],
        w_neg=FINAL_PARAMS["weights"]["negative"],
    )

    base = LinearSVC(**FINAL_PARAMS["linear_svc"])
    clf = CalibratedClassifierCV(
        estimator=base,
        method=FINAL_PARAMS["calibration"]["method"],
        cv=FINAL_PARAMS["calibration"]["cv"],
    )

    try:
        clf.fit(X_all, y_all, sample_weight=sample_weight)
    except TypeError:
        clf.fit(X_all, y_all)

    y_proba = clf.predict_proba(X_all)[:, 1]
    y_pred = (y_proba >= FINAL_PARAMS["threshold"]).astype(int)

    out_df = pd.DataFrame({
        "label": y_all,
        "y_proba": y_proba,
        "y_pred": y_pred,
    })
    if "repo_full_name" in df.columns:
        out_df["repo_full_name"] = df["repo_full_name"].values
    if "family_id" in df.columns:
        out_df["family_id"] = df["family_id"].astype(str).values
    if "is_real_positive" in df.columns:
        out_df["is_real_positive"] = pd.to_numeric(df["is_real_positive"], errors="coerce").fillna(0).astype(int).values
    if "is_generated_positive" in df.columns:
        out_df["is_generated_positive"] = pd.to_numeric(df["is_generated_positive"], errors="coerce").fillna(0).astype(int).values

    out_df.to_csv(OUTPUT_DIR / "final_linear_svc_trainset_predictions.csv", index=False, encoding="utf-8-sig")

    feature_names = (
        structured_cols
        + [f"tfidf_readme::{t}" for t in readme_vectorizer.get_feature_names_out()]
        + [f"tfidf_aux::{t}" for t in aux_vectorizer.get_feature_names_out()]
    )

    bundle = {
        "model": clf,
        "readme_vectorizer": readme_vectorizer,
        "aux_vectorizer": aux_vectorizer,
        "fill_values": fill_values.to_dict(),
        "numeric_scaler": scaler,
        "structured_cols": structured_cols,
        "feature_names": feature_names,
        "threshold": FINAL_PARAMS["threshold"],
        "final_params": FINAL_PARAMS,
        "README_TEXT_COL": README_TEXT_COL,
        "DESCRIPTION_TEXT_COL": DESCRIPTION_TEXT_COL,
        "TOPICS_TEXT_COL": TOPICS_TEXT_COL,
    }
    joblib.dump(bundle, OUTPUT_DIR / "final_linear_svc_calibrated_bundle.joblib")

    summary = {
        "data_path": str(DATA_PATH),
        "n_rows": int(len(df)),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "threshold": FINAL_PARAMS["threshold"],
        "structured_feature_count": len(structured_cols),
        "readme_vocab_size": int(len(readme_vectorizer.get_feature_names_out())),
        "aux_vocab_size": int(len(aux_vectorizer.get_feature_names_out())),
        "final_params": FINAL_PARAMS,
    }
    with open(OUTPUT_DIR / "final_linear_svc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved final LinearSVC artifacts to:", OUTPUT_DIR)
    print("- final_linear_svc_calibrated_bundle.joblib")
    print("- final_linear_svc_trainset_predictions.csv")
    print("- final_linear_svc_summary.json")


if __name__ == "__main__":
    main()
