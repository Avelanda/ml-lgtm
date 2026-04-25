"""
Download DL3DV scene archives from HuggingFace for preprocessing.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

This script downloads the DL3DV-10K dataset (images + poses) at a specified
resolution from HuggingFace. Each scene is downloaded as a zip file, extracted,
and the zip is removed to save space.

Prerequisites:
    - A HuggingFace account with access to the DL3DV dataset repos.
    - Set your HuggingFace token via:
        export HF_TOKEN=hf_xxxxx
      Or login via:
        hf auth login

Usage:
    # Download all 4K scenes in the "1K" subset.
    python scripts/download_dl3dv.py --output_dir data/dl3dv-4k --subset 1K

    # Download a specific scene.
    python scripts/download_dl3dv.py --output_dir data/dl3dv-4k --scene_id SCENE_HASH

    # Download all subsets at 2K resolution.
    python scripts/download_dl3dv.py --output_dir data/dl3dv-2k --resolution 2K --subset 1K
    python scripts/download_dl3dv.py --output_dir data/dl3dv-2k --resolution 2K --subset 2K
    # ... etc.
"""

import argparse
import os
import shutil
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, HfFileSystem, get_token, login
from tqdm import tqdm

RESOLUTION_TO_REPO = {
    "480P": "DL3DV/DL3DV-ALL-480P",
    "960P": "DL3DV/DL3DV-ALL-960P",
    "2K": "DL3DV/DL3DV-ALL-2K",
    "4K": "DL3DV/DL3DV-ALL-4K",
}

DL3DV_META_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"


def _setup_hf_auth():
    """Authenticate with HuggingFace via HF_TOKEN env var or cached token."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace via HF_TOKEN.")
    elif get_token():
        print("Using cached HuggingFace token (from `hf auth login`).")
    else:
        raise RuntimeError(
            "No HuggingFace credentials found. Either set the HF_TOKEN "
            "environment variable or run `hf auth login`."
        )


def verify_access(repo: str) -> bool:
    """Check if the user has access to the given HuggingFace dataset repo."""
    fs = HfFileSystem()
    try:
        fs.ls(f"datasets/{repo}")
        return True
    except Exception:
        return False


def hf_download_path(
    api: HfApi,
    repo: str,
    rel_path: str,
    output_dir: str,
    max_retries: int = 5,
) -> bool:
    """
    Download a file from HuggingFace with retries.

    Args:
        api: HfApi instance.
        repo: HuggingFace dataset repo ID.
        rel_path: Relative path within the repo.
        output_dir: Local output directory.
        max_retries: Maximum number of download attempts.

    Returns:
        True if download succeeded, False otherwise.
    """
    cache_dir = os.path.join(output_dir, ".cache")
    for attempt in range(max_retries):
        try:
            api.hf_hub_download(
                repo_id=repo,
                filename=rel_path,
                repo_type="dataset",
                local_dir=output_dir,
                cache_dir=cache_dir,
            )
            return True
        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting.")
            raise
        except Exception:
            traceback.print_exc()
            # Clean cache after failure to prevent corruption.
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  Retry {attempt + 1}/{max_retries}")

    print(
        f"ERROR: Download {repo}/{rel_path} failed after {max_retries}"
        " attempts."
    )
    return False


def get_download_list(
    subset: str,
    scene_id: str,
    resolution: str,
    file_type: str,
    output_dir: str,
) -> list[dict]:
    """
    Build the list of files to download.

    Args:
        subset: Subset name (e.g., "1K", "2K").
        scene_id: Specific scene ID, or "" for all scenes in subset.
        resolution: Resolution level (e.g., "4K", "2K").
        file_type: Type of files to download.
        output_dir: Output directory (used for caching meta CSV).

    Returns:
        List of dicts with "repo" and "rel_path" keys.
    """

    def to_download_item(scene_id, resolution, batch, file_type):
        if file_type == "images+poses":
            repo = RESOLUTION_TO_REPO[resolution]
            rel_path = f"{batch}/{scene_id}.zip"
        elif file_type == "video":
            repo = "DL3DV/DL3DV-ALL-video"
            rel_path = f"{batch}/{scene_id}/video.mp4"
        elif file_type == "colmap_cache":
            repo = "DL3DV/DL3DV-ALL-ColmapCache"
            rel_path = f"{batch}/{scene_id}.zip"
        return {"repo": repo, "rel_path": rel_path}

    # Download meta CSV.
    cache_dir = os.path.join(output_dir, ".cache")
    meta_path = os.path.join(cache_dir, "DL3DV-valid.csv")
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(meta_path):
        print(f"Downloading DL3DV metadata CSV to {meta_path} ...")
        response = requests.get(DL3DV_META_URL, timeout=60)
        response.raise_for_status()
        with open(meta_path, "wb") as f:
            f.write(response.content)

    df = pd.read_csv(meta_path)

    # Single scene mode.
    if scene_id:
        if scene_id not in df["hash"].values:
            raise ValueError(f"Scene ID {scene_id} not found in meta CSV.")
        batch = df[df["hash"] == scene_id]["batch"].values[0]
        return [to_download_item(scene_id, resolution, batch, file_type)]

    # Full subset mode.
    items = []
    subdf = df[df["batch"] == subset]
    for _, row in subdf.iterrows():
        items.append(
            to_download_item(row["hash"], resolution, subset, file_type)
        )
    return items


def download(
    api: HfApi,
    download_list: list[dict],
    output_dir: str,
    clean_cache: bool,
) -> bool:
    """
    Download and extract all items in the download list.

    Args:
        api: HfApi instance.
        download_list: List of dicts with "repo" and "rel_path".
        output_dir: Local output directory.
        clean_cache: Whether to clean HuggingFace cache after each download.

    Returns:
        True if all downloads succeeded.
    """
    succ_count = 0
    failed_scenes_log_path = Path(output_dir) / "failed_scenes.txt"

    for item in tqdm(download_list, desc="Downloading"):
        repo = item["repo"]
        rel_path = item["rel_path"]

        # Check if already extracted.
        output_path = os.path.join(output_dir, rel_path).replace(".zip", "")
        if os.path.exists(output_path):
            succ_count += 1
            continue

        succ = hf_download_path(api, repo, rel_path, output_dir)
        if not succ:
            print(f"Download {rel_path} failed.")
            continue

        # Extract zip files.
        if rel_path.endswith(".zip"):
            zip_path = os.path.join(output_dir, rel_path)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    extract_dir = os.path.join(
                        output_dir, os.path.dirname(rel_path)
                    )
                    zf.extractall(extract_dir)
                os.remove(zip_path)
                succ_count += 1
            except Exception as e:
                scene_id = os.path.splitext(os.path.basename(rel_path))[0]
                print(
                    f"WARNING: Failed to unzip {rel_path} for {scene_id}: {e}"
                )
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                extracted_dir = os.path.join(
                    os.path.dirname(zip_path), scene_id
                )
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir, ignore_errors=True)
                with open(failed_scenes_log_path, "a") as f:
                    f.write(f"{scene_id}\n")
        else:
            succ_count += 1

        if clean_cache:
            cache_dir = os.path.join(output_dir, ".cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

    print(
        f"Summary: {succ_count}/{len(download_list)} downloaded successfully."
    )
    return succ_count == len(download_list)


def main():
    parser = argparse.ArgumentParser(
        description="Download DL3DV dataset from HuggingFace."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for downloaded data.",
    )
    parser.add_argument(
        "--subset",
        choices=[
            "1K",
            "2K",
            "3K",
            "4K",
            "5K",
            "6K",
            "7K",
            "8K",
            "9K",
            "10K",
            "11K",
        ],
        default=None,
        help=(
            "Subset to download (e.g., 1K, 2K). Required unless --scene_id is"
            " set."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=["4K", "2K", "960P", "480P"],
        default="4K",
        help="Resolution to download (default: 4K).",
    )
    parser.add_argument(
        "--file_type",
        choices=["images+poses", "video", "colmap_cache"],
        default="images+poses",
        help="File type to download (default: images+poses).",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="",
        help="Download a specific scene by its hash ID (ignores --subset).",
    )
    parser.add_argument(
        "--clean_cache",
        action="store_true",
        help="Clean HuggingFace cache after each download to save space.",
    )
    args = parser.parse_args()

    if not args.subset and not args.scene_id:
        parser.error("Either --subset or --scene_id must be provided.")

    _setup_hf_auth()

    repo = RESOLUTION_TO_REPO[args.resolution]
    if not verify_access(repo):
        print(
            f"Access denied. Visit https://huggingface.co/datasets/{repo} "
            "to request access."
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)

    download_list = get_download_list(
        subset=args.subset or "",
        scene_id=args.scene_id,
        resolution=args.resolution,
        file_type=args.file_type,
        output_dir=args.output_dir,
    )

    api = HfApi()
    if download(api, download_list, args.output_dir, args.clean_cache):
        print(f"Download complete. Data saved to {args.output_dir}")
    else:
        print(
            f"Some downloads failed. Check {args.output_dir}/failed_scenes.txt"
        )


if __name__ == "__main__":
    main()
