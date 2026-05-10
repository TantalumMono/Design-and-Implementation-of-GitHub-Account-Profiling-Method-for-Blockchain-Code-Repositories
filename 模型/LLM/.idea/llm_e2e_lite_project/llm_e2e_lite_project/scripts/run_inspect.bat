@echo off
call .venv\Scripts\activate.bat
python llm_e2e_pipeline.py --config .\configs\config_tiny_0.5b.yaml --mode inspect
