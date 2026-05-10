import argparse
import copy
import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

from build_llm_inputs import build_prompts, build_prompt
from data_loader import load_dataset
from evaluate import calc_metrics, calc_subset_report, save_json
from explain import feature_ablation, nearest_cases, save_jsonl, text_chunk_ablation
from select_threshold import search_threshold
from split_by_family import build_five_splits, save_split_stats, summarize_splits
from train_llm_classifier import fit_one_fold, predict_proba
from tune_hyperparams import tune_over_folds


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    setup_logger(cfg["output_dir"])

    logging.info("Load dataset...")
    df, feature_names = load_dataset(cfg)

    logging.info("Build strict 5 family-based splits...")
    splits = build_five_splits(df, seed=int(cfg["seed"]))
    split_stats = summarize_splits(df, splits)
    save_split_stats(os.path.join(cfg["output_dir"], "split_stats.json"), split_stats)

    logging.info("Hyperparameter tuning on validation folds only...")
    best_trial, cv_df = tune_over_folds(df, feature_names, splits, cfg)
    cv_df.to_csv(os.path.join(cfg["output_dir"], "cv_results.csv"), index=False)

    best_params = {
        k: v for k, v in best_trial.items()
        if k not in ["trial_id", "mean_val_f1", "std_val_f1", "mean_val_recall", "mean_val_precision"]
    }
    save_json(os.path.join(cfg["output_dir"], "best_params.json"), best_params)

    final_cfg = copy.deepcopy(cfg)
    for k, v in best_params.items():
        if k in final_cfg["model"]:
            final_cfg["model"][k] = v
        elif k in final_cfg["loss"]:
            final_cfg["loss"][k] = v

    logging.info("Run 5 folds with best params...")
    val_rows = []
    test_rows = []
    all_explanations = []

    for sp in splits:
        fold = sp["fold"]
        fold_dir = os.path.join(final_cfg["output_dir"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.loc[sp["train_idx"]].reset_index(drop=True)
        val_df = df.loc[sp["val_idx"]].reset_index(drop=True)
        test_df = df.loc[sp["test_idx"]].reset_index(drop=True)

        train_prompts = build_prompts(train_df, feature_names, final_cfg)
        val_prompts = build_prompts(val_df, feature_names, final_cfg)
        test_prompts = build_prompts(test_df, feature_names, final_cfg)

        trainer, model, tokenizer = fit_one_fold(
            train_prompts,
            train_df["label"].tolist(),
            val_prompts,
            val_df["label"].tolist(),
            final_cfg,
            fold_dir
        )

        val_prob = predict_proba(trainer, val_prompts, val_df["label"].tolist(), tokenizer, final_cfg)
        test_prob = predict_proba(trainer, test_prompts, test_df["label"].tolist(), tokenizer, final_cfg)

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

        # top-20 explanations per fold
        top_df = t.sort_values("prob", ascending=False).head(20)
        fold_exps = []
        for sid in top_df["sample_id"].tolist():
            row_df = test_df[test_df["sample_id"] == sid].reset_index(drop=True)
            target_prompt = build_prompt(row_df.iloc[0], feature_names, final_cfg)

            exp = {
                "fold": fold,
                "sample_id": sid,
                "repo_full_name": row_df.iloc[0]["repo_full_name"],
                "label": int(row_df.iloc[0]["label"]),
                "pred_prob": float(top_df[top_df["sample_id"] == sid]["prob"].iloc[0]),
                "structured_feature_evidence": feature_ablation(trainer, tokenizer, row_df, feature_names, final_cfg),
                "text_chunk_evidence": text_chunk_ablation(trainer, tokenizer, row_df, feature_names, final_cfg),
                "similar_cases": nearest_cases(
                    trainer, tokenizer, target_prompt, train_prompts,
                    train_df[["sample_id", "label", "family", "repo_full_name", "source_type"]].copy(),
                    final_cfg
                ),
            }
            fold_exps.append(exp)
        save_jsonl(os.path.join(fold_dir, "explanations_top20.jsonl"), fold_exps)
        all_explanations.extend(fold_exps)

    val_all = pd.concat(val_rows, axis=0).reset_index(drop=True)
    test_all = pd.concat(test_rows, axis=0).reset_index(drop=True)

    val_all.to_csv(os.path.join(final_cfg["output_dir"], "oof_val_predictions.csv"), index=False)
    test_all.to_csv(os.path.join(final_cfg["output_dir"], "all_test_predictions.csv"), index=False)
    save_jsonl(os.path.join(final_cfg["output_dir"], "explanations.jsonl"), all_explanations)

    best_th, th_df = search_threshold(
        val_all["label"].values,
        val_all["prob"].values,
        start=float(final_cfg["threshold"]["start"]),
        end=float(final_cfg["threshold"]["end"]),
        step=float(final_cfg["threshold"]["step"]),
    )
    th_df.to_csv(os.path.join(final_cfg["output_dir"], "threshold_search.csv"), index=False)
    save_json(os.path.join(final_cfg["output_dir"], "selected_threshold.json"), {"best_threshold": best_th})

    # fold test metrics
    fold_metrics = []
    for fold in sorted(test_all["fold"].unique()):
        sub = test_all[test_all["fold"] == fold]
        m = calc_metrics(sub["label"].values, sub["prob"].values, threshold=best_th)
        m["fold"] = int(fold)
        fold_metrics.append(m)
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(os.path.join(final_cfg["output_dir"], "fold_test_metrics.csv"), index=False)

    final_metrics = {
        "overall_test": calc_metrics(test_all["label"].values, test_all["prob"].values, threshold=best_th),
        "subset_report": calc_subset_report(test_all, "prob", "label", best_th),
        "fold_mean_std": {
            "precision_mean": float(fold_metrics_df["precision"].mean()),
            "precision_std": float(fold_metrics_df["precision"].std()),
            "recall_mean": float(fold_metrics_df["recall"].mean()),
            "recall_std": float(fold_metrics_df["recall"].std()),
            "f1_mean": float(fold_metrics_df["f1"].mean()),
            "f1_std": float(fold_metrics_df["f1"].std()),
            "accuracy_mean": float(fold_metrics_df["accuracy"].mean()),
            "accuracy_std": float(fold_metrics_df["accuracy"].std()),
            "roc_auc_mean": float(fold_metrics_df["roc_auc"].mean()),
            "roc_auc_std": float(fold_metrics_df["roc_auc"].std()),
            "pr_auc_mean": float(fold_metrics_df["pr_auc"].mean()),
            "pr_auc_std": float(fold_metrics_df["pr_auc"].std()),
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

    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()