@echo off
call .venv\Scripts\activate.bat
python llm_e2e_pipeline.py --config .\configs\config_lite_1.5b.yaml --mode tune_and_run
