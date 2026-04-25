"""
Helpers for LGTM example inference scripts (downloads, batching).

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image

_EXAMPLES_DIR = Path(__file__).resolve().parent
_DATA_DIR = _EXAMPLES_DIR / "data"

DL3DV_SCENE_HASH = (
    "0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
)
_DL3DV_HF_REPO = "DL3DV/DL3DV-Benchmark"
_DL3DV_NUM_FRAMES = 10

_to_tensor = tf.ToTensor()


def download_dl3dv_test_data(
    scene_hash: str = DL3DV_SCENE_HASH,
    num_frames: int = _DL3DV_NUM_FRAMES,
    data_dir: Path | None = None,
) -> Path:
    """
    Download a DL3DV test scene from HuggingFace (verbatim, no conversion).

    Downloads ``transforms.json`` and the first *num_frames* PNG images
    from the ``DL3DV/DL3DV-Benchmark`` dataset.  Files are cached
    locally; subsequent calls skip the download.

    The dataset is gated — users must accept the licence at
    https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark
    and run ``hf auth login`` before the first download.

    Args:
        scene_hash: DL3DV scene hash identifier.
        num_frames: Number of frames to download (first N).
        data_dir: Root directory for cached downloads.  Defaults to
            ``examples/data/``.

    Returns:
        Path to the scene directory containing ``transforms.json``
        and ``images/``.
    """
    import os

    from huggingface_hub import get_token, hf_hub_download, login

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    elif not get_token():
        raise RuntimeError(
            "No HuggingFace credentials found. Either set the HF_TOKEN "
            "environment variable or run `hf auth login`."
        )

    if data_dir is None:
        data_dir = _DATA_DIR
    data_dir = Path(data_dir)

    scene_dir = data_dir / scene_hash / "nerfstudio"
    transforms_path = scene_dir / "transforms.json"
    img_dir = scene_dir / "images"

    # Check if already cached.
    if transforms_path.exists() and img_dir.exists():
        existing = sorted(img_dir.glob("frame_*"))
        if len(existing) >= num_frames:
            print("  DL3DV test data already cached.")
            return scene_dir

    print("  Downloading DL3DV test data from HuggingFace...")
    scene_prefix = f"{scene_hash}/nerfstudio"

    hf_hub_download(
        repo_id=_DL3DV_HF_REPO,
        filename=f"{scene_prefix}/transforms.json",
        repo_type="dataset",
        revision="main",
        local_dir=data_dir,
    )

    with open(transforms_path) as f:
        meta = json.load(f)

    frames = sorted(
        meta["frames"],
        key=lambda fr: int(fr["file_path"].split("_")[-1].split(".")[0]),
    )[:num_frames]

    for frame in frames:
        hf_hub_download(
            repo_id=_DL3DV_HF_REPO,
            filename=f"{scene_prefix}/{frame['file_path']}",
            repo_type="dataset",
            revision="main",
            local_dir=data_dir,
        )

    print(f"  Downloaded {len(frames)} frames.")
    return scene_dir


def load_dl3dv_test_data(
    scene_dir: Path,
    num_frames: int = _DL3DV_NUM_FRAMES,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Load DL3DV test data and convert cameras to OpenCV convention.

    Reads ``transforms.json`` and images from *scene_dir*.  Camera
    poses are converted from the OpenGL / nerfstudio convention to the
    convention expected by the LGTM models, and intrinsics are normalised
    by image dimensions.

    Both PNG (from HuggingFace) and JPEG (from local cache) images are
    supported transparently.

    Args:
        scene_dir: Path containing ``transforms.json`` and ``images/``.
        num_frames: Number of frames to load (first N).

    Returns:
        Tuple of ``(c2w_poses, intrinsics, images)``:

        - ``c2w_poses``:  ``[N, 4, 4]`` float tensor (OpenCV convention).
        - ``intrinsics``: ``[N, 3, 3]`` normalised intrinsic matrices.
        - ``images``:     ``[N, 3, H, W]`` float tensor in ``[0, 1]``.
    """
    transforms_path = scene_dir / "transforms.json"
    img_dir = scene_dir / "images"

    with open(transforms_path) as f:
        meta = json.load(f)

    frames = sorted(
        meta["frames"],
        key=lambda fr: int(fr["file_path"].split("_")[-1].split(".")[0]),
    )[:num_frames]

    # Camera poses: OpenGL c2w → LGTM c2w.
    # Matches the conversion in the training data pipeline.
    # (opengl_c2w_to_opencv_w2c from NoPoSplat, without the final.
    # Matrix inversion that is later undone by convert_poses).
    poses_list = []
    for frame in frames:
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        poses_list.append(c2w)
    poses = torch.from_numpy(np.stack(poses_list))

    # Normalised intrinsics.
    w, h = meta["w"], meta["h"]
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = meta["fl_x"] / w
    K[1, 1] = meta["fl_y"] / h
    K[0, 2] = meta["cx"] / w
    K[1, 2] = meta["cy"] / h
    intrinsics = K.unsqueeze(0).expand(len(frames), -1, -1).clone()

    # Images (supports both .png and .jpg).
    img_tensors = []
    for frame in frames:
        stem = Path(frame["file_path"]).stem
        png_path = img_dir / f"{stem}.png"
        jpg_path = img_dir / f"{stem}.jpg"
        if png_path.exists():
            img_path = png_path
        elif jpg_path.exists():
            img_path = jpg_path
        else:
            raise FileNotFoundError(
                f"Image not found: {png_path} or {jpg_path}"
            )
        img_tensors.append(_to_tensor(Image.open(img_path)))

    images = torch.stack(img_tensors)
    print(
        f"  Loaded {len(frames)} frames at "
        f"{images.shape[-2]}\u00d7{images.shape[-1]}"
    )
    return poses, intrinsics, images
