"""Set HF_HOME once (large models on E:/ or D:/, not a full C: drive)."""

import os
from pathlib import Path


def ensure_hf_home():
    if "HF_HOME" in os.environ:
        return
    if Path("E:/").exists():
        root = Path("E:/hf_cache")
    elif Path("D:/").exists():
        root = Path("D:/hf_cache")
    else:
        root = Path(__file__).resolve().parent.parent / ".hf_cache"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root.resolve())
