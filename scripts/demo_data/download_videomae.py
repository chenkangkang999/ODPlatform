"""Download optional VideoMAE weights to a configurable, ignored directory."""
import argparse
import os
from pathlib import Path

from od_platform.common.paths import PRETRAINED_MODELS_DIR


def main(argv=None, *, downloader=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--output", type=Path, default=PRETRAINED_MODELS_DIR / "videomae-base-kinetics")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args(argv)
    os.environ["HF_ENDPOINT"] = args.endpoint
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    output = args.output.resolve()
    print(f"开始下载 {args.repo} -> {output}", flush=True)
    path = downloader(repo_id=args.repo, local_dir=str(output),
                      allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
    print("下载完成:", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
