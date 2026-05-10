@echo off
setlocal enabledelayedexpansion

REM 设置国内镜像源
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_ENABLE_HF_TRANSFER=1

REM 检查参数
if "%~1"=="" (
  echo Usage: download_model.bat "MODEL_ID" "LOCAL_DIR"
  exit /b 1
)
if "%~2"=="" (
  echo Usage: download_model.bat "MODEL_ID" "LOCAL_DIR"
  exit /b 1
)

REM 处理路径（移除单引号）
set "target_dir=%~2"
set "target_dir=!target_dir:'=!"

REM 创建目标目录（如果不存在）
if not exist "!target_dir!" (
  echo Creating directory: !target_dir!
  mkdir "!target_dir!"
)

echo Downloading model: %~1
echo To directory: !target_dir!
echo Using mirror: https://hf-mirror.com
echo.

REM 下载模型
python -m huggingface_hub download "%~1" --local-dir "!target_dir!" --resume-download

if %errorlevel% equ 0 (
  echo.
  echo Download completed successfully!
  echo Model saved to: !target_dir!
) else (
  echo.
  echo Download failed with error code: %errorlevel%
)

endlocal