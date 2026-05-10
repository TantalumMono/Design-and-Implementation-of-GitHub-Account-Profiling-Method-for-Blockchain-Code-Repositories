# download_and_test_st_model.py
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

LOCAL_DIR = Path(r"C:\Users\Dell\Desktop\Grade4\毕业设计\模型\SentenceTransformer\models\all-MiniLM-L6-v2")
HF_TOKEN = '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'# 没有就保持 None

LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)

print("开始下载模型到:", LOCAL_DIR)
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=str(LOCAL_DIR),
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
    resume_download=True,
)

print("下载完成，开始测试本地离线加载...")

os.environ["HF_HUB_OFFLINE"] = "1"

model = SentenceTransformer(
    str(LOCAL_DIR),
    local_files_only=True,
    device="cpu",
)

vec = model.encode(["test sentence"], convert_to_numpy=True)
print("本地加载成功，embedding shape =", vec.shape)