$ErrorActionPreference = 'Stop'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-lite.txt
Write-Host 'Environment ready.'
