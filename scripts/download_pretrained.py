"""
Download pretrained backbone weights for NoPoSplat, DepthSplat, and Flash3D.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

Usage:
    # Download all models to pretrained/ (default)
    python scripts/download_pretrained.py

    # Download a specific model.
    python scripts/download_pretrained.py --model noposplat
    python scripts/download_pretrained.py --model depthsplat
    python scripts/download_pretrained.py --model flash3d

    # Download to a custom directory.
    python scripts/download_pretrained.py --out_dir /path/to/dir
"""

import argparse
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_noposplat(out_dir: Path) -> None:
    url = (
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
        "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    )
    dest = out_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading NoPoSplat (MASt3R) -> {dest}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def download_depthsplat(out_dir: Path) -> None:
    filename = "depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
    dest = out_dir / filename
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading DepthSplat -> {dest}")
    hf_hub_download(
        repo_id="haofeixu/depthsplat",
        filename=filename,
        repo_type="model",
        revision="main",
        local_dir=out_dir,
    )


def download_flash3d(out_dir: Path) -> None:
    dest = out_dir / "unidepth-v1-vitl14.bin"
    if dest.exists():
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading Flash3D (UniDepth V1 ViT-L/14) -> {dest}")
    hf_hub_download(
        repo_id="lpiccinelli/unidepth-v1-vitl14",
        filename="pytorch_model.bin",
        repo_type="model",
        revision="main",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    (out_dir / "pytorch_model.bin").rename(dest)


DEFAULT_OUT_DIR = Path(__file__).parent.parent / "pretrained"

MODELS = {
    "noposplat": download_noposplat,
    "depthsplat": download_depthsplat,
    "flash3d": download_flash3d,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained backbone weights."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Which model to download. Downloads all if not specified.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to save weights (default: {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    to_download = [args.model] if args.model else list(MODELS.keys())
    for name in to_download:
        MODELS[name](args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
