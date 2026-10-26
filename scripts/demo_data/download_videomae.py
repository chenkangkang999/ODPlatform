"""经 hf-mirror 下载 VideoMAE Kinetics 预训练模型（用于动作识别）。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

repo_id = "MCG-NJU/videomae-base-finetuned-kinetics"
local = r"D:\ODPlatform\models\pretrained\videomae-base-kinetics"
print(f"开始下载 {repo_id} -> {local}", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local,
    allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"],
)
print("下载完成:", path, flush=True)
