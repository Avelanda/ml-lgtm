"""
Preprocess downloaded DL3DV scenes into LGTM .torch chunk files.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

This script takes the raw DL3DV data (downloaded via download_dl3dv.py) and
converts it into .torch chunk files that can be loaded by the LGTM dataloader.

The processing pipeline:
  1. Scan the input directory for scenes (each with transforms.json + images).
  2. Convert PNG images to JPG if necessary (the original HuggingFace data uses PNG).
  3. Split scenes into train/test based on the evaluation index.
  4. Pack scenes into .torch chunk files (~200 MB each if it fits, but 4K scenes are larger).
  5. Generate an index.json mapping scene keys to chunk files.

Usage:
    # Process all downloaded 4K data.
    python scripts/preprocess_dl3dv.py \
        --input_dir data/dl3dv-4k \
        --output_dir data/dl3dv \
        --resolution 4K

    # Process only the "1K" subset for testing.
    python scripts/preprocess_dl3dv.py \
        --input_dir data/dl3dv-4k \
        --output_dir data/dl3dv \
        --resolution 4K \
        --subset 1K

Note:
    - DL3DV/DL3DV-Benchmark contains ~140 test scenes, however, ~130 of them
      are also inside DL3DV/DL3DV-ALL-4K (and other resolutions). This code will
      exclude these scenes from the train set. The test set is built from the
      index json file with the DL3DV/DL3DV-ALL-4K dataset.
    - In DL3DV's terminology, "4K" and "2K" refers to the horizontal resolution
      being approximately 4,000 and 2,000 pixels, respectively. Their "960P"
      (540x960) is actually 540P and their "480P" (270x480) is actually 270P,
      where typically "P" refers to the vertical resolution.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from jaxtyping import UInt8
from natsort import natsorted
from PIL import Image
from torch import Tensor
from tqdm import tqdm

# Target chunk size in bytes (~200 MB)
TARGET_BYTES_PER_CHUNK = int(2e8)

# JPEG quality for PNG -> JPG conversion.
JPG_QUALITY = 70

RESOLUTION_TO_IMAGE_DIR = {
    "480P": "images_8",
    "960P": "images_4",
    "2K": "images_2",
    "4K": "images",
}

RESOLUTION_TO_IMAGE_WH = {
    "480P": (480, 270),
    "960P": (960, 540),
    "2K": (1920, 1080),
    "4K": (3840, 2160),
}


def load_metadata(transforms_path: Path) -> dict:
    """
    Load camera metadata from a DL3DV transforms.json file.

    Converts from OpenGL camera-to-world to OpenCV world-to-camera convention,
    and normalizes intrinsics by image dimensions.

    Returns dict with keys: timestamps, cameras, w, h.
    """

    def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
        c2w = c2w.copy()
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        w2c_opencv = np.linalg.inv(c2w)
        return w2c_opencv

    with open(transforms_path, "r") as f:
        data = json.load(f)

    w = data["w"]
    h = data["h"]

    # FIXME: ignoring k1, k2, p1, p2
    intrinsic = np.array(
        [
            data["fl_x"] / w,
            data["fl_y"] / h,
            data["cx"] / w,
            data["cy"] / h,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    timestamps = []
    cameras = []

    for frame in data["frames"]:
        frame_id = int(frame["file_path"].split("_")[-1].split(".")[0])
        timestamps.append(frame_id)
        extrinsic = np.array(frame["transform_matrix"], dtype=np.float32)
        w2c = opengl_c2w_to_opencv_w2c(extrinsic)
        w2c = w2c[:3, :].flatten()
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

    return {
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(np.stack(cameras), dtype=torch.float32),
        "w": w,
        "h": h,
    }


def load_im_bytes_in_dir(
    im_dir: Path,
    suffix: str,
) -> dict[str, UInt8[Tensor, "..."]]:
    """
    Load all images in a directory as raw bytes tensors (no decoding).

    Returns a dict mapping image stem -> bytes tensor.
    """
    if not im_dir.is_dir():
        raise FileNotFoundError(f"{im_dir} is not a directory.")

    im_paths = natsorted(
        [p for p in im_dir.iterdir() if p.suffix.lower() == suffix]
    )
    result = {}
    for p in im_paths:
        result[p.stem] = torch.tensor(
            np.memmap(p, dtype="uint8", mode="r").copy()
        )
    return result


def convert_png_dir_to_jpg(
    png_dir: Path,
    jpg_dir: Path,
    quality: int = JPG_QUALITY,
) -> None:
    """
    Convert all PNG images in png_dir to JPG in jpg_dir.
    Skips if jpg_dir already has the expected number of files.
    """
    png_paths = natsorted(list(png_dir.glob("*.png")))
    if not png_paths:
        return

    jpg_dir.mkdir(parents=True, exist_ok=True)
    existing_jpgs = list(jpg_dir.glob("*.jpg"))
    if len(existing_jpgs) == len(png_paths):
        return

    for png_path in png_paths:
        jpg_path = jpg_dir / png_path.with_suffix(".jpg").name
        if jpg_path.exists():
            continue
        im = Image.open(png_path)
        im.save(jpg_path, "JPEG", quality=quality)


def get_dir_size_bytes(dir_path: Path) -> int:
    """Get total size of all files in a directory in bytes."""
    total = 0
    for p in dir_path.iterdir():
        if p.is_file():
            total += p.stat().st_size
    return total


def discover_scenes(
    input_dir: Path,
    resolution: str,
    subset: str | None = None,
) -> list[tuple[str, str, Path]]:
    """
    Discover all valid DL3DV scenes in the input directory.

    Returns list of (subset, scene_id, scene_dir) tuples.
    """
    scenes = []
    png_im_dir_name = RESOLUTION_TO_IMAGE_DIR[resolution]

    if subset:
        subsets = [subset]
    else:
        subsets = natsorted(
            [
                d.name
                for d in input_dir.iterdir()
                if d.is_dir() and d.name != ".cache"
            ]
        )

    for sub in subsets:
        subset_dir = input_dir / sub
        if not subset_dir.is_dir():
            print(
                f"Warning: subset directory {subset_dir} not found, skipping."
            )
            continue

        for scene_dir in natsorted(subset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            transforms_path = scene_dir / "transforms.json"
            im_dir = scene_dir / png_im_dir_name

            if not transforms_path.exists():
                continue
            if not im_dir.is_dir():
                continue

            scenes.append((sub, scene_id, scene_dir))

    return scenes


def load_test_scene_ids() -> set[str]:
    """
    Load test scene IDs from the DL3DV evaluation index file.

    The evaluation index file (assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json)
    maps scene_id -> evaluation parameters. The keys are the test scene IDs.

    Raises:
        FileNotFoundError: If the evaluation index file is missing.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    eval_index_path = (
        repo_root / "assets" / "dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
    )

    if not eval_index_path.exists():
        raise FileNotFoundError(
            f"DL3DV evaluation index not found: {eval_index_path}. It is"
            " required to split train vs test; add the file or fix the path."
        )

    with open(eval_index_path, "r") as f:
        eval_index = json.load(f)

    return set(eval_index.keys())


def build_chunks(
    scenes: list[tuple[str, str, Path]],
    output_dir: Path,
    resolution: str,
    split: str,
    test_scene_ids: set[str],
) -> None:
    """
    Build .torch chunk files from discovered scenes.

    Args:
        scenes: List of (subset, scene_id, scene_dir) tuples.
        output_dir: Root output directory for chunks.
        resolution: Resolution string.
        split: "train" or "test".
        test_scene_ids: Set of scene IDs belonging to the test set.
    """
    png_im_dir_name = RESOLUTION_TO_IMAGE_DIR[resolution]
    jpg_im_dir_name = (
        f"{png_im_dir_name}_jpg"
        if png_im_dir_name != "images"
        else "images_jpg"
    )
    expected_w, expected_h = RESOLUTION_TO_IMAGE_WH[resolution]

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenes by split.
    split_scenes = []
    for sub, scene_id, scene_dir in scenes:
        is_test = scene_id in test_scene_ids
        if split == "test" and is_test:
            split_scenes.append((sub, scene_id, scene_dir))
        elif split == "train" and not is_test:
            split_scenes.append((sub, scene_id, scene_dir))

    print(f"[{split}] {len(split_scenes)} scenes to process.")

    # Build chunks.
    key_to_torch = {}
    current_chunk: list[dict] = []
    current_chunk_size = 0
    chunk_index = 0

    for sub, scene_id, scene_dir in tqdm(
        split_scenes, desc=f"Building {split} chunks"
    ):
        scene_key = f"{sub}/{scene_id}"

        # Load metadata.
        transforms_path = scene_dir / "transforms.json"
        try:
            metadata = load_metadata(transforms_path)
        except Exception as e:
            print(f"Warning: failed to load metadata for {scene_key}: {e}")
            continue

        # Check image dimensions.
        if metadata["w"] != expected_w or metadata["h"] != expected_h:
            continue

        # Find image directory: prefer JPG, fall back to PNG (then convert)
        png_dir = scene_dir / png_im_dir_name
        jpg_dir = scene_dir / jpg_im_dir_name

        if jpg_dir.is_dir() and list(jpg_dir.glob("*.jpg")):
            im_dir = jpg_dir
            im_suffix = ".jpg"
        elif png_dir.is_dir() and list(png_dir.glob("*.png")):
            # Convert PNG to JPG.
            convert_png_dir_to_jpg(png_dir, jpg_dir)
            im_dir = jpg_dir
            im_suffix = ".jpg"
        else:
            # Try loading whatever images are available.
            if png_dir.is_dir():
                pngs = list(png_dir.glob("*.png"))
                jpgs = list(png_dir.glob("*.jpg"))
                if jpgs:
                    im_dir = png_dir
                    im_suffix = ".jpg"
                elif pngs:
                    convert_png_dir_to_jpg(png_dir, jpg_dir)
                    im_dir = jpg_dir
                    im_suffix = ".jpg"
                else:
                    print(f"Warning: no images found for {scene_key}")
                    continue
            else:
                print(f"Warning: image directory not found for {scene_key}")
                continue

        # Load image bytes.
        try:
            im_bytes_dict = load_im_bytes_in_dir(im_dir, suffix=im_suffix)
        except Exception as e:
            print(f"Warning: failed to load images for {scene_key}: {e}")
            continue

        # Sort images by timestamp order from metadata.
        im_stems = [f"frame_{ts.item():0>5}" for ts in metadata["timestamps"]]
        try:
            images = [im_bytes_dict[stem] for stem in im_stems]
        except KeyError:
            print(f"Warning: missing images for {scene_key}, skipping.")
            continue

        if len(images) != len(metadata["timestamps"]):
            print(
                f"Warning: image count mismatch for {scene_key}: {len(images)}"
                f" images vs {len(metadata['timestamps'])} timestamps"
            )
            continue

        example = {
            "timestamps": metadata["timestamps"],
            "cameras": metadata["cameras"],
            "key": scene_key,
            "images": images,
        }

        scene_size = get_dir_size_bytes(im_dir)

        # Check if current chunk would exceed target size.
        if (
            current_chunk_size > 0
            and current_chunk_size + scene_size >= TARGET_BYTES_PER_CHUNK
        ):
            # Save current chunk.
            chunk_filename = f"{chunk_index:06d}.torch"
            chunk_path = split_dir / chunk_filename
            torch.save(current_chunk, chunk_path)
            print(
                f"  Saved {chunk_filename}: {len(current_chunk)} scenes, "
                f"{current_chunk_size / 1e6:.1f} MB"
            )
            chunk_index += 1
            current_chunk = []
            current_chunk_size = 0

        current_chunk.append(example)
        current_chunk_size += scene_size
        key_to_torch[scene_key] = f"{chunk_index:06d}.torch"

    # Save final chunk.
    if current_chunk:
        chunk_filename = f"{chunk_index:06d}.torch"
        chunk_path = split_dir / chunk_filename
        torch.save(current_chunk, chunk_path)
        print(
            f"  Saved {chunk_filename}: {len(current_chunk)} scenes, "
            f"{current_chunk_size / 1e6:.1f} MB"
        )

    # Save index.json.
    index_path = split_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(key_to_torch, f, indent=2)
    print(
        f"  Saved index.json with {len(key_to_torch)} entries to {index_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DL3DV data into .torch chunk files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Directory containing downloaded DL3DV data (from"
            " download_dl3dv.py)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .torch chunk files.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["480P", "960P", "2K", "4K"],
        default="4K",
        help="Resolution of the downloaded data (default: 4K).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
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
        help="Process only a specific subset. If not set, process all subsets.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split to build (default: both).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: input directory {input_dir} does not exist.")
        return

    # Discover scenes.
    print(f"Scanning {input_dir} for scenes (resolution={args.resolution}) ...")
    scenes = discover_scenes(input_dir, args.resolution, args.subset)
    print(f"Found {len(scenes)} scenes.")

    if not scenes:
        print("No scenes found. Check --input_dir and --resolution.")
        return

    # Load test scene IDs.
    test_scene_ids = load_test_scene_ids()
    print(f"Loaded {len(test_scene_ids)} test scene IDs from evaluation index.")

    # Build chunks.
    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Building {split} chunks ...")
        print(f"{'='*60}")
        build_chunks(scenes, output_dir, args.resolution, split, test_scene_ids)

    print("\nDone!")


if __name__ == "__main__":
    main()
