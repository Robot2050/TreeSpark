import os
from pathlib import Path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_symlink(src, dst):
    dst_path = Path(dst)
    if dst_path.is_symlink() or dst_path.exists():
        dst_path.unlink()
    dst_path.symlink_to(Path(src).resolve())
