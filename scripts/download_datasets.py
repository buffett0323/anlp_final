#!/usr/bin/env python3
"""
Download datasets for SDSD ablation experiments.

Usage:
  python scripts/download_datasets.py [--cache-dir CACHE] [--datasets json,humaneval,mbpp,gsm]

Datasets:
  - json-mode-eval: NousResearch/json-mode-eval (JSON schema)
  - humaneval: openai/openai_humaneval (code)
  - mbpp: google-research-datasets/mbpp (Python code)
  - gsm-symbolic: apple/GSM-Symbolic (math reasoning)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Download SDSD evaluation datasets")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache directory")
    parser.add_argument("--datasets", type=str, default="json-mode-eval,humaneval,mbpp,gsm-symbolic",
                        help="Comma-separated: json-mode-eval, humaneval, mbpp, gsm-symbolic")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    cache = args.cache_dir
    names = [s.strip() for s in args.datasets.split(",") if s.strip()]

    configs = [
        ("NousResearch/json-mode-eval", "json-mode-eval", None, "train"),
        ("openai/openai_humaneval", "humaneval", None, "test"),
        ("google-research-datasets/mbpp", "mbpp", None, "test"),
        ("apple/GSM-Symbolic", "gsm-symbolic", "main", "test"),
    ]

    for item in configs:
        hf_id, short_name, config, split = item[0], item[1], item[2], item[3]
        if short_name not in names:
            continue
        print(f"Downloading {short_name} ({hf_id})...")
        try:
            if config:
                load_dataset(hf_id, config, split=split, cache_dir=cache)
            else:
                load_dataset(hf_id, split=split, cache_dir=cache)
            print(f"  OK: {short_name}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\nDone. Datasets cached in ~/.cache/huggingface/datasets/ (or --cache-dir)")


if __name__ == "__main__":
    main()
