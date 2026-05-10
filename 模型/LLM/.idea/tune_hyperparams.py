import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from build_llm_inputs import build_prompts
from evaluate import calc_metrics
from train_llm_classifier import fit_one_fold, predict_proba


def sample_trials(search_cfg: dict):
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
    trials = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    rng = np.random.RandomState(42)
    max_trials = int(search_cfg["max_trials"])
    if len(trials) > max_trials:
        idx = rng.choice(len(trials), size=max_trials, replace=False)
        trials = [trials[i] for i in idx]
    return trials


def apply_trial(base_cfg, trial):
    cfg = copy.deepcopy(base_cfg)
    for k, v in trial.items():
        if k in cfg["model"]:
            cfg["model"][k] = v
        elif k in cfg["loss"]:
            cfg["loss"][k] = v
    return cfg


def tune_over_folds(df, feature_names, splits, base_cfg):
    trials = sample_trials(base_cfg["search"])
    rows = []

    for tid, trial in enumerate(tqdm(trials, desc="llm_hparam_search")):
        cfg = apply_trial(base_cfg, trial)
        fold_scores = []

        for sp in splits:
            fold = sp["fold"]
            train_df = df.loc[sp["train_idx"]].reset_index(drop=True)
            val_df = df.loc[sp["val_idx"]].reset_index(drop=True)

            train_prompts = build_prompts(train_df, feature_names, cfg)
            val_prompts = build_prompts(val_df, feature_names, cfg)

            trainer, model, tokenizer = fit_one_fold(
                train_prompts,
                train_df["label"].tolist(),
                val_prompts,
                val_df["label"].tolist(),
                cfg,
                f"{cfg['output_dir']}/search_trial_{tid}_fold_{fold}"
            )

            val_prob = predict_proba(trainer, val_prompts, val_df["label"].tolist(), tokenizer, cfg)
            m = calc_metrics(val_df["label"].values, val_prob, threshold=0.5)
            fold_scores.append(m)

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