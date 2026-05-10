import os
import sys
from huggingface_hub import snapshot_download

def download_model(model_id, local_dir):
    # 设置环境变量（国内加速）
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    print(f"Downloading model: {model_id}")
    print(f"To directory: {local_dir}")
    print("Using mirror: https://hf-mirror.com")
    print()

    # 创建目录
    os.makedirs(local_dir, exist_ok=True)

    # 下载模型
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        resume_download=True,
        max_workers=4  # 多线程加速
    )

    print("\nDownload completed successfully!")
    print(f"Model saved to: {local_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_model.py <MODEL_ID> <LOCAL_DIR>")
        sys.exit(1)

    model_id = sys.argv[1]
    local_dir = sys.argv[2]

    download_model(model_id, local_dir)