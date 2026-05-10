@echo off
set PYTHON=C:\Users\Dell\Desktop\Grade4\毕业设计\模型\LLM\.venv\Scripts\python.exe

"%PYTHON%" -c "import sys, torch; print('python=', sys.executable); print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

"%PYTHON%" llm_e2e_pipeline.py --config configs\config_tiny_0.5b.yaml --mode run_only

pause