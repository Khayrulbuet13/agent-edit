"""Fetch what the PMT backend needs: the PMT source code and the COCO model weights.

    python tools/fetch_pmt.py [ROOT] [--no-source] [--no-weights]

ROOT defaults to $AGENT_EDIT_PMT_ROOT, else $AGENT_EDIT_CACHE/pmt. After it ran:

    ROOT/src/image/models/   github.com/tue-mps/pmt at a pinned commit (MIT licence)
    ROOT/weights/model.safetensors   tue-mps/coco_instance_pmt_large_1280_dinov3 on Hugging Face (MIT)
                                     215 MB, checked against its sha256

The DINOv3 backbone is NOT fetched here. Its weights are gated on Hugging Face:
open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m, accept the licence, then
sign in on this machine (`huggingface-cli login`, or export HF_TOKEN). The code loads the model by
the id in $AGENT_EDIT_PMT_BACKBONE (default facebook/dinov3-vitl16-pretrain-lvd1689m); a local folder
that holds the same weights in Hugging Face format works too.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "https://github.com/tue-mps/pmt"
COMMIT = "0e803722aa5737a242b383dec1b90c2c66b86baa"
HF_REPO = "tue-mps/coco_instance_pmt_large_1280_dinov3"
HF_REVISION = "11f7f85547fef36f32ddc8f3dd0acdd3cfc009fd"
WEIGHTS_SHA256 = "9755f2c06f07323896f7114673553c78ecc43e45716876b9b876b93f4cd00ee9"
WEIGHTS_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/model.safetensors"


def default_root() -> Path:
    if os.environ.get("AGENT_EDIT_PMT_ROOT"):
        return Path(os.environ["AGENT_EDIT_PMT_ROOT"])
    if os.environ.get("AGENT_EDIT_CACHE"):
        return Path(os.environ["AGENT_EDIT_CACHE"]) / "pmt"
    sys.exit("give ROOT, or set AGENT_EDIT_PMT_ROOT or AGENT_EDIT_CACHE")


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_source(root: Path) -> None:
    src = root / "src"
    if not (src / ".git").is_dir():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(src)], check=True)
    if git("rev-parse", "HEAD", cwd=src) != COMMIT:
        subprocess.run(["git", "checkout", "-q", "--detach", COMMIT], cwd=src, check=True)
    head = git("rev-parse", "HEAD", cwd=src)
    if head != COMMIT:
        sys.exit(f"{src} is at {head}, wanted {COMMIT}")
    print(f"source ok: {src} at {head}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(root: Path) -> None:
    dest = root / "weights" / "model.safetensors"
    if dest.is_file() and sha256_of(dest) == WEIGHTS_SHA256:
        print(f"weights ok: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"downloading {WEIGHTS_URL}")
    h = hashlib.sha256()
    with urllib.request.urlopen(WEIGHTS_URL) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != WEIGHTS_SHA256:
        part.unlink()
        sys.exit(f"sha256 mismatch: got {h.hexdigest()}, wanted {WEIGHTS_SHA256}")
    part.replace(dest)
    print(f"weights ok: {dest}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", type=Path, help="target folder (default $AGENT_EDIT_PMT_ROOT or $AGENT_EDIT_CACHE/pmt)")
    ap.add_argument("--no-source", action="store_true", help="skip the git clone")
    ap.add_argument("--no-weights", action="store_true", help="skip the weights download")
    a = ap.parse_args(argv)
    root = (a.root or default_root()).resolve()
    if not a.no_source:
        fetch_source(root)
    if not a.no_weights:
        fetch_weights(root)
    print("\nNot fetched: the DINOv3 backbone (gated). Accept the licence at\n"
          "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m and sign in "
          "(`huggingface-cli login` or HF_TOKEN).\nThe code loads it by $AGENT_EDIT_PMT_BACKBONE "
          "(default facebook/dinov3-vitl16-pretrain-lvd1689m).")


if __name__ == "__main__":
    main()
