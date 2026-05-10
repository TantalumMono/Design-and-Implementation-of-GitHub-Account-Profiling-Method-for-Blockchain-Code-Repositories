$ErrorActionPreference = 'Stop'
.\.venv\Scripts\Activate.ps1
python .\llm_e2e_pipeline.py --config .\configs\config_tiny_0.5b.yaml --mode tune_and_run
