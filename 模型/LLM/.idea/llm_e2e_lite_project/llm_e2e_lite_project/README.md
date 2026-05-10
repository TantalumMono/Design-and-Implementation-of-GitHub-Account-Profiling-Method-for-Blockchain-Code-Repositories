# Lightweight local LLM end-to-end pipeline

## Quick start (Windows)
1. Run `scripts\setup_env.bat`
2. Run `scripts\run_inspect.bat`
3. Run `scripts\run_tune_tiny.bat`

## Quick start (PowerShell)
1. `powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1`
2. `powershell -ExecutionPolicy Bypass -File .\scripts\run_inspect.ps1`
3. `powershell -ExecutionPolicy Bypass -File .\scripts\run_tune_tiny.ps1`

## Outputs
- split_stats.json
- cv_results.csv
- best_params.json
- oof_val_predictions.csv
- threshold_search.csv
- selected_threshold.json
- all_test_predictions.csv
- fold_test_metrics.csv
- final_metrics.json
- suspicious_topk.csv
- error_analysis.csv
- run.log
