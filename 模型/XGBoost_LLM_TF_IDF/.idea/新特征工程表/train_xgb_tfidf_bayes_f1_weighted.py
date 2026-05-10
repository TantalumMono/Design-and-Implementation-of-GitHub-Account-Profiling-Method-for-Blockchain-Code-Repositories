import os
import json
import math
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    auc,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================================================
# 0. 全局配置
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
FILLNA_STRATEGY = "median"

# 数据路径：优先读取上一步 prepare_family_dataset_xgb_tfidf_v3.py 的输出
DATA_CANDIDATES = [
    r"family_prepared_v3/family_dataset_train_ready.csv",
    r"family_prepared_v3/family_dataset_full.csv",
    r"family_prepared/family_dataset_train_ready.csv",
    r"family_prepared/family_dataset_full.csv",
    r"dataset_full_with_text.csv",
]

OUTPUT_DIR = r"./train_tfidf_bayes_f1_weighted_outputs"

# 文本列优先级：更推荐直接使用 prepare 脚本产出的 text_for_tfidf_clean
TEXT_COL_CANDIDATES = [
    "text_for_tfidf_clean",
    "combined_text_clean",
    "readme_text_clean",
    "combined_text",
    "readme_text",
]

# 多次贝叶斯搜索
N_STUDIES = 3
N_TRIALS_PER_STUDY = 35
STUDY_SEEDS = [42, 2024, 3407]

# 样本权重基础值：你可以先改这里，再让贝叶斯在此基础上继续搜 multiplier
BASE_WEIGHT_REAL_POS = 3.0
BASE_WEIGHT_GENERATED_POS = 1.25
BASE_WEIGHT_NEG = 1.0

# TF-IDF 搜索空间
TFIDF_MAX_FEATURES_LOW = 200
TFIDF_MAX_FEATURES_HIGH = 1200
TFIDF_MAX_FEATURES_STEP = 50
TFIDF_MIN_DF_CHOICES = [1, 2, 3, 4, 5]
TFIDF_NGRAM_CHOICES = [(1, 1), (1, 2), (1, 3)]

# F1 优先：objective 以 best-F1 为主，PR-AUC 只做很小的稳定性 tie-break
F1_PRIMARY_WEIGHT = 1.0
AP_TIEBREAK_WEIGHT = 0.02


# =========================================================
# 1. 工具函数
# =========================================================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def resolve_first_existing(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "未找到可用训练数据。请先运行 prepare_family_dataset_xgb_tfidf_v3.py，"
        "或手动修改 DATA_CANDIDATES。"
    )



def choose_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"未找到可用文本列，候选列为: {TEXT_COL_CANDIDATES}")



def load_numeric_feature_columns(csv_path: str, df: pd.DataFrame) -> Optional[List[str]]:
    dataset_dir = Path(csv_path).resolve().parent
    candidates = [
        dataset_dir / "numeric_feature_columns.json",
        Path("numeric_feature_columns.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                cols = json.loads(p.read_text(encoding="utf-8"))
                cols = [c for c in cols if c in df.columns]
                if cols:
                    return cols
            except Exception:
                pass
    return None



def infer_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容不同 prepare 版本：补齐 is_real_positive / is_generated_positive / is_real_negative
    """
    out = df.copy()

    if "is_real_negative" not in out.columns:
        out["is_real_negative"] = (pd.to_numeric(out.get("label", 0), errors="coerce").fillna(0).astype(int) == 0).astype(int)

    if "is_real_positive" not in out.columns:
        if "provenance_type" in out.columns:
            out["is_real_positive"] = (out["provenance_type"].astype(str) == "original_real_positive").astype(int)
        else:
            out["is_real_positive"] = 0

    if "is_generated_positive" not in out.columns:
        if "provenance_type" in out.columns:
            prov = out["provenance_type"].astype(str)
            out["is_generated_positive"] = prov.isin(["synthetic_positive", "generated_positive", "unknown_positive"]).astype(int)
        else:
            out["is_generated_positive"] = 0

    # 兜底：label=1 但两个都没标上的，默认按生成正样本处理，避免漏掉增强样本
    label_pos = pd.to_numeric(out.get("label", 0), errors="coerce").fillna(0).astype(int) == 1
    unlabeled_pos = label_pos & (out["is_real_positive"] == 0) & (out["is_generated_positive"] == 0)
    out.loc[unlabeled_pos, "is_generated_positive"] = 1

    return out



def infer_structured_cols(df: pd.DataFrame, text_col: str, explicit_numeric_cols: Optional[List[str]] = None) -> List[str]:
    if explicit_numeric_cols:
        return explicit_numeric_cols

    exclude_cols = {
        "label",
        "sample_id",
        "sample_source",
        "provenance_type",
        "family_id",
        "group_id",
        "repo_full_name",
        "source_repo_full_name",
        "source_family",
        "generated_from",
        "readme_text_raw",
        "readme_text_clean",
        "description_text_raw",
        "description_text_clean",
        "topics_text_raw",
        "topics_text_clean",
        "combined_text_raw",
        "combined_text_clean",
        "text_for_tfidf_raw",
        "text_for_tfidf_clean",
        "text_source_used",
        text_col,
    }

    numeric_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if str(c).startswith("genmeta__"):
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        non_null_mask = df[c].notna()
        if non_null_mask.sum() == 0:
            continue
        if converted[non_null_mask].notna().all():
            numeric_cols.append(c)
    return numeric_cols



def prepare_structured_features(df: pd.DataFrame, structured_cols: List[str]) -> pd.DataFrame:
    X = df[structured_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X



def fillna_by_train_statistics(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "median",
):
    if strategy == "median":
        fill_values = X_tr.median(numeric_only=True)
    elif strategy == "zero":
        fill_values = pd.Series(0, index=X_tr.columns)
    else:
        raise ValueError("fillna_strategy 仅支持 'median' 或 'zero'")

    X_tr = X_tr.fillna(fill_values).fillna(0)
    X_val = X_val.fillna(fill_values).fillna(0)
    X_test = X_test.fillna(fill_values).fillna(0)
    return X_tr, X_val, X_test, fill_values



def safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, zero_division=0)



def choose_best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # sklearn 的 precision/recall 长度 = thresholds + 1
    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
    }

    # 加入边界阈值
    full_thresholds = np.concatenate(([0.0], thresholds, [1.0]))

    for t in np.unique(np.round(full_thresholds, 10)):
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = safe_f1(y_true, pred)

        better = (
            (f1 > best["f1"]) or
            (np.isclose(f1, best["f1"]) and r > best["recall"]) or
            (np.isclose(f1, best["f1"]) and np.isclose(r, best["recall"]) and p > best["precision"]) or
            (np.isclose(f1, best["f1"]) and np.isclose(r, best["recall"]) and np.isclose(p, best["precision"]) and t > best["threshold"])
        )
        if better:
            best = {
                "threshold": float(t),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }

    return best["threshold"], best



def build_sample_weight(df_part: pd.DataFrame, w_real_pos: float, w_generated_pos: float, w_neg: float) -> np.ndarray:
    weights = np.ones(len(df_part), dtype=np.float32)

    if "label" not in df_part.columns:
        return weights

    label = pd.to_numeric(df_part["label"], errors="coerce").fillna(0).astype(int)
    is_real_pos = pd.to_numeric(df_part.get("is_real_positive", 0), errors="coerce").fillna(0).astype(int)
    is_gen_pos = pd.to_numeric(df_part.get("is_generated_positive", 0), errors="coerce").fillna(0).astype(int)

    weights[label == 0] = w_neg
    weights[(label == 1) & (is_real_pos == 1)] = w_real_pos
    weights[(label == 1) & (is_gen_pos == 1)] = w_generated_pos

    # 兜底：正样本但两个 flag 都没打上的，默认按生成正样本
    unknown_pos = (label == 1) & (is_real_pos == 0) & (is_gen_pos == 0)
    weights[unknown_pos] = w_generated_pos

    return weights



def save_confusion_matrix_figure(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix") -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str) -> None:
    pr_auc = average_precision_score(y_true, y_proba)
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str) -> None:
    roc_auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 2. 读取数据
# =========================================================
ensure_dir(OUTPUT_DIR)
DATA_PATH = resolve_first_existing(DATA_CANDIDATES)
df = pd.read_csv(DATA_PATH)
df = infer_flag_columns(df)

if "label" not in df.columns:
    raise ValueError("数据中缺少 label 列")

df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
text_col = choose_text_col(df)
df[text_col] = df[text_col].fillna("").astype(str)

if "repo_full_name" not in df.columns:
    df["repo_full_name"] = [f"sample_{i}" for i in range(len(df))]

explicit_numeric_cols = load_numeric_feature_columns(DATA_PATH, df)
structured_cols = infer_structured_cols(df, text_col=text_col, explicit_numeric_cols=explicit_numeric_cols)

print("训练数据路径:", DATA_PATH)
print("总样本数:", len(df))
print("正样本数:", int((df["label"] == 1).sum()))
print("负样本数:", int((df["label"] == 0).sum()))
print("文本列:", text_col)
print("结构化特征数:", len(structured_cols))

# =========================================================
# 3. 划分 train / val / test（不做 family 分组验证）
# =========================================================
df_train, df_test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

df_tr, df_val = train_test_split(
    df_train,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_train["label"],
)

print("\n数据划分：")
print("Train:", df_tr.shape, "正样本:", int((df_tr["label"] == 1).sum()))
print("Val  :", df_val.shape, "正样本:", int((df_val["label"] == 1).sum()))
print("Test :", df_test.shape, "正样本:", int((df_test["label"] == 1).sum()))

# =========================================================
# 4. 结构化特征预处理
# =========================================================
X_tr_num = prepare_structured_features(df_tr, structured_cols)
X_val_num = prepare_structured_features(df_val, structured_cols)
X_test_num = prepare_structured_features(df_test, structured_cols)

X_tr_num, X_val_num, X_test_num, fill_values = fillna_by_train_statistics(
    X_tr_num, X_val_num, X_test_num, strategy=FILLNA_STRATEGY
)

X_tr_num_sp = csr_matrix(X_tr_num.to_numpy(dtype=np.float32))
X_val_num_sp = csr_matrix(X_val_num.to_numpy(dtype=np.float32))
X_test_num_sp = csr_matrix(X_test_num.to_numpy(dtype=np.float32))

y_tr = df_tr["label"].values
y_val = df_val["label"].values
y_test = df_test["label"].values


# =========================================================
# 5. Trial 级训练函数
# =========================================================
def train_and_score_trial(trial: optuna.trial.Trial):
    tfidf_max_features = trial.suggest_int(
        "tfidf_max_features",
        TFIDF_MAX_FEATURES_LOW,
        TFIDF_MAX_FEATURES_HIGH,
        step=TFIDF_MAX_FEATURES_STEP,
    )
    tfidf_min_df = trial.suggest_categorical("tfidf_min_df", TFIDF_MIN_DF_CHOICES)
    tfidf_ngram_range = trial.suggest_categorical("tfidf_ngram_range", TFIDF_NGRAM_CHOICES)

    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        min_df=tfidf_min_df,
        ngram_range=tfidf_ngram_range,
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b",
    )

    X_tr_text = vectorizer.fit_transform(df_tr[text_col])
    X_val_text = vectorizer.transform(df_val[text_col])

    X_tr_all = hstack([X_tr_num_sp, X_tr_text], format="csr")
    X_val_all = hstack([X_val_num_sp, X_val_text], format="csr")

    # 样本权重：先用基础值，再让贝叶斯搜 multiplier
    real_pos_mult = trial.suggest_float("real_pos_weight_mult", 0.5, 3.0, log=True)
    gen_pos_mult = trial.suggest_float("gen_pos_weight_mult", 0.4, 3.0, log=True)
    neg_mult = trial.suggest_float("neg_weight_mult", 0.4, 2.0, log=True)

    w_real_pos = BASE_WEIGHT_REAL_POS * real_pos_mult
    w_generated_pos = BASE_WEIGHT_GENERATED_POS * gen_pos_mult
    w_neg = BASE_WEIGHT_NEG * neg_mult

    train_sample_weight = build_sample_weight(df_tr, w_real_pos, w_generated_pos, w_neg)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "importance_type": "gain",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 5.0),
    }

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr_all, y_tr, sample_weight=train_sample_weight, verbose=False)

    val_proba = model.predict_proba(X_val_all)[:, 1]
    best_thr, thr_info = choose_best_threshold_by_f1(y_val, val_proba)
    val_ap = average_precision_score(y_val, val_proba)

    objective_score = F1_PRIMARY_WEIGHT * thr_info["f1"] + AP_TIEBREAK_WEIGHT * val_ap

    trial.set_user_attr("val_best_threshold", float(best_thr))
    trial.set_user_attr("val_f1", float(thr_info["f1"]))
    trial.set_user_attr("val_precision", float(thr_info["precision"]))
    trial.set_user_attr("val_recall", float(thr_info["recall"]))
    trial.set_user_attr("val_ap", float(val_ap))
    trial.set_user_attr("actual_w_real_pos", float(w_real_pos))
    trial.set_user_attr("actual_w_generated_pos", float(w_generated_pos))
    trial.set_user_attr("actual_w_neg", float(w_neg))
    trial.set_user_attr("tfidf_vocab_size", int(len(vectorizer.get_feature_names_out())))

    return objective_score


# =========================================================
# 6. 多次贝叶斯搜索
# =========================================================
all_trial_rows: List[Dict] = []
studies: List[optuna.study.Study] = []

overall_best_value = -np.inf
overall_best_params = None
overall_best_user_attrs = None
overall_best_study_idx = None
overall_best_trial_number = None

print("\n开始多轮贝叶斯优化（F1 优先）...")
for study_idx in range(N_STUDIES):
    seed = STUDY_SEEDS[study_idx % len(STUDY_SEEDS)]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    print(f"\n===== Study {study_idx + 1}/{N_STUDIES} | seed={seed} =====")
    study.optimize(train_and_score_trial, n_trials=N_TRIALS_PER_STUDY, show_progress_bar=True)
    studies.append(study)

    for t in study.trials:
        row = {
            "study_idx": study_idx,
            "trial_number": t.number,
            "value": t.value,
            "state": str(t.state),
        }
        for k, v in t.params.items():
            row[f"param__{k}"] = v
        for k, v in t.user_attrs.items():
            row[f"attr__{k}"] = v
        all_trial_rows.append(row)

    if study.best_value > overall_best_value:
        overall_best_value = study.best_value
        overall_best_params = study.best_params
        overall_best_user_attrs = study.best_trial.user_attrs
        overall_best_study_idx = study_idx
        overall_best_trial_number = study.best_trial.number

print("\n===== 多轮贝叶斯优化完成 =====")
print("Overall Best Objective:", overall_best_value)
print("Overall Best Study:", overall_best_study_idx)
print("Overall Best Trial:", overall_best_trial_number)
print("Overall Best Params:")
print(overall_best_params)
print("Overall Best Trial Attrs:")
print(overall_best_user_attrs)

all_trials_df = pd.DataFrame(all_trial_rows).sort_values("value", ascending=False)
all_trials_df.to_csv(
    os.path.join(OUTPUT_DIR, "optuna_all_trials.csv"),
    index=False,
    encoding="utf-8-sig",
)

with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "data_path": DATA_PATH,
            "text_col": text_col,
            "structured_cols": structured_cols,
            "overall_best_objective": float(overall_best_value),
            "overall_best_study_idx": int(overall_best_study_idx),
            "overall_best_trial_number": int(overall_best_trial_number),
            "best_params": overall_best_params,
            "best_user_attrs": overall_best_user_attrs,
            "base_weights": {
                "real_positive": BASE_WEIGHT_REAL_POS,
                "generated_positive": BASE_WEIGHT_GENERATED_POS,
                "negative": BASE_WEIGHT_NEG,
            },
        },
        f,
        ensure_ascii=False,
        indent=4,
    )


# =========================================================
# 7. 用全局最优参数重训正式模型
# =========================================================
def build_final_vectorizer(best_params: Dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=int(best_params["tfidf_max_features"]),
        min_df=int(best_params["tfidf_min_df"]),
        ngram_range=tuple(best_params["tfidf_ngram_range"]),
        sublinear_tf=True,
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\+\-\.]{1,}\b",
    )


final_vectorizer = build_final_vectorizer(overall_best_params)
X_tr_text = final_vectorizer.fit_transform(df_tr[text_col])
X_val_text = final_vectorizer.transform(df_val[text_col])
X_test_text = final_vectorizer.transform(df_test[text_col])

X_tr_all = hstack([X_tr_num_sp, X_tr_text], format="csr")
X_val_all = hstack([X_val_num_sp, X_val_text], format="csr")
X_test_all = hstack([X_test_num_sp, X_test_text], format="csr")

best_w_real_pos = float(overall_best_user_attrs["actual_w_real_pos"])
best_w_generated_pos = float(overall_best_user_attrs["actual_w_generated_pos"])
best_w_neg = float(overall_best_user_attrs["actual_w_neg"])
train_sample_weight = build_sample_weight(df_tr, best_w_real_pos, best_w_generated_pos, best_w_neg)

final_xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "importance_type": "gain",
}

# 去掉 TF-IDF 和 weight multiplier，只保留 XGB 参数
skip_params = {
    "tfidf_max_features",
    "tfidf_min_df",
    "tfidf_ngram_range",
    "real_pos_weight_mult",
    "gen_pos_weight_mult",
    "neg_weight_mult",
}
for k, v in overall_best_params.items():
    if k not in skip_params:
        final_xgb_params[k] = v

final_model = XGBClassifier(**final_xgb_params)
final_model.fit(X_tr_all, y_tr, sample_weight=train_sample_weight, verbose=False)


# =========================================================
# 8. 验证集选阈值（F1 优先）
# =========================================================
val_proba = final_model.predict_proba(X_val_all)[:, 1]
best_threshold, best_thr_info = choose_best_threshold_by_f1(y_val, val_proba)
val_ap = average_precision_score(y_val, val_proba)
val_roc_auc = roc_auc_score(y_val, val_proba)

print(
    f"\nValidation best threshold = {best_threshold:.4f} | "
    f"Precision = {best_thr_info['precision']:.4f} | "
    f"Recall = {best_thr_info['recall']:.4f} | "
    f"F1 = {best_thr_info['f1']:.4f}"
)


# =========================================================
# 9. 测试集评估
# =========================================================
test_proba = final_model.predict_proba(X_test_all)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

test_roc_auc = roc_auc_score(y_test, test_proba)
test_pr_auc = average_precision_score(y_test, test_proba)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)
test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

print("\n===== Test Metrics =====")
print("Threshold :", best_threshold)
print("ROC-AUC   :", test_roc_auc)
print("PR-AUC    :", test_pr_auc)
print("Precision :", test_precision)
print("Recall    :", test_recall)
print("F1        :", test_f1)
print("\nConfusion Matrix:")
print(test_cm)
print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))

all_feature_names = structured_cols + [f"tfidf::{t}" for t in final_vectorizer.get_feature_names_out()]

metrics_dict = {
    "data_path": DATA_PATH,
    "text_col": text_col,
    "best_objective_score": float(overall_best_value),
    "best_threshold": float(best_threshold),
    "val_ap": float(val_ap),
    "val_roc_auc": float(val_roc_auc),
    "val_precision_at_threshold": float(best_thr_info["precision"]),
    "val_recall_at_threshold": float(best_thr_info["recall"]),
    "val_f1_at_threshold": float(best_thr_info["f1"]),
    "test_roc_auc": float(test_roc_auc),
    "test_pr_auc": float(test_pr_auc),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "test_f1": float(test_f1),
    "test_confusion_matrix": test_cm.tolist(),
    "tfidf_max_features": int(overall_best_params["tfidf_max_features"]),
    "tfidf_min_df": int(overall_best_params["tfidf_min_df"]),
    "tfidf_ngram_range": list(overall_best_params["tfidf_ngram_range"]),
    "structured_feature_count": int(len(structured_cols)),
    "tfidf_feature_count": int(len(final_vectorizer.get_feature_names_out())),
    "total_feature_count": int(X_tr_all.shape[1]),
    "weight_real_positive": float(best_w_real_pos),
    "weight_generated_positive": float(best_w_generated_pos),
    "weight_negative": float(best_w_neg),
}

with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, ensure_ascii=False, indent=4)


# =========================================================
# 10. 保存预测结果
# =========================================================
test_result = pd.DataFrame({
    "repo_full_name": df_test["repo_full_name"].values,
    "y_true": y_test,
    "y_proba": test_proba,
    "y_pred": test_pred,
    "text_len": df_test[text_col].astype(str).str.len().values,
})

test_result.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig",
)

top_risky = test_result.sort_values("y_proba", ascending=False).head(50)
top_risky.to_csv(
    os.path.join(OUTPUT_DIR, "top_50_risky_samples.csv"),
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 11. 保存模型与向量器
# =========================================================
joblib.dump(final_model, os.path.join(OUTPUT_DIR, "xgb_tfidf_model.joblib"))
joblib.dump(final_vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

bundle = {
    "model": final_model,
    "vectorizer": final_vectorizer,
    "structured_cols": structured_cols,
    "fillna_strategy": FILLNA_STRATEGY,
    "fill_values": fill_values.to_dict(),
    "threshold": best_threshold,
    "feature_names": all_feature_names,
    "text_col": text_col,
    "weights": {
        "real_positive": best_w_real_pos,
        "generated_positive": best_w_generated_pos,
        "negative": best_w_neg,
    },
}
joblib.dump(bundle, os.path.join(OUTPUT_DIR, "tfidf_xgb_bundle.joblib"))


# =========================================================
# 12. 特征重要性
# =========================================================
importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_gain.csv"),
    index=False,
    encoding="utf-8-sig",
)

print("\nTop 20 features:")
print(importance_df.head(20))


# =========================================================
# 13. 保存可视化图
# =========================================================
save_confusion_matrix_figure(
    test_cm,
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    title="Confusion Matrix",
)

save_pr_curve(
    y_test,
    test_proba,
    os.path.join(OUTPUT_DIR, "pr_curve.png"),
)

save_roc_curve(
    y_test,
    test_proba,
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
)

print("\n输出目录:", OUTPUT_DIR)
print("已保存文件：")
print("- best_params.json")
print("- optuna_all_trials.csv")
print("- test_metrics.json")
print("- test_predictions.csv")
print("- top_50_risky_samples.csv")
print("- xgb_tfidf_model.joblib")
print("- tfidf_vectorizer.joblib")
print("- tfidf_xgb_bundle.joblib")
print("- feature_importance_gain.csv")
print("- confusion_matrix.png")
print("- pr_curve.png")
print("- roc_curve.png")
print("\nDone.")
