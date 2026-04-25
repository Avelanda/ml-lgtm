"""
Batching, indexing, and IO helpers for LGTM datasets.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from io import BytesIO

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor

from lgtm.dataset.data_types import BatchedExample
from lgtm.dataset.dataset import DatasetCfg
from lgtm.dataset.shims.crop_shim import rescale_and_crop_bilinear
from lgtm.utils.camera_utils import camera_normalization

_to_tensor = tf.ToTensor()


def convert_poses(
    cameras: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],
    Float[Tensor, "batch 3 3"],
]:
    """
    Convert DL3DV camera format [N, 18] to c2w poses and normalized
    intrinsics.

    Args:
        cameras: Camera parameters tensor of shape [N, 18].

    Returns:
        Tuple of (c2w poses [N, 4, 4], normalized intrinsics [N, 3, 3]).
    """
    b = cameras.shape[0]
    intrinsics = repeat(
        torch.eye(3, dtype=torch.float32), "h w -> b h w", b=b
    ).clone()
    fx, fy, cx, cy = cameras[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(cameras[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(
    images_bytes: list[UInt8[Tensor, "..."]],
) -> Float[Tensor, "batch 3 height width"]:
    """
    Convert a list of JPEG byte tensors to a float tensor [N, 3, H, W].

    Args:
        images_bytes: List of byte tensors, each encoding a JPEG image.

    Returns:
        Float tensor of shape [N, 3, H, W] in [0, 1].
    """
    return torch.stack(
        [
            _to_tensor(Image.open(BytesIO(img.numpy().tobytes())))
            for img in images_bytes
        ]
    )


@torch.no_grad()
def prepare_inference_batch(
    context_images: list[Tensor],
    context_poses: list[Tensor],
    context_intrinsics: list[Tensor],
    dataset_cfg: DatasetCfg | None = None,
    device: torch.device | str = "cpu",
    target_images: list[Tensor] | None = None,
    target_poses: list[Tensor] | None = None,
    target_intrinsics: list[Tensor] | None = None,
    near: float | None = None,
    far: float | None = None,
) -> BatchedExample:
    """
    Prepare a batch for inference from raw tensors.

    Applies pose normalization (normalize_scene_scale + relative_pose)
    and image rescaling/cropping using the same preprocessing as
    the training pipeline, then adds a batch dimension and moves
    everything to the given device.

    Preprocessing parameters (image shape, near/far, pose flags) are
    read from dataset_cfg when provided.

    Args:
        context_images: List of context images [3, H, W] in [0,1].
        context_poses: List of context c2w poses [4, 4].
        context_intrinsics: List of normalized intrinsics [3, 3].
        dataset_cfg: DatasetCfg from the experiment config. When
            None, no rescaling or pose normalization is applied.
        device: Target device for the output tensors.
        target_images: Optional list of target images [3, H, W].
        target_poses: Optional list of target c2w poses [4, 4].
        target_intrinsics: Optional list of target normalized
            intrinsics [3, 3].
        near: Near plane override. If None, uses dataset_cfg.near.
        far: Far plane override. If None, uses dataset_cfg.far.

    Returns:
        BatchedExample dict ready for forward().
    """
    cfg = dataset_cfg
    if near is None:
        near = cfg.near if cfg is not None else 0.1
    if far is None:
        far = cfg.far if cfg is not None else 100.0
    normalize_scene_scale = (
        cfg.normalize_scene_scale if cfg is not None else False
    )
    relative_pose = cfg.relative_pose if cfg is not None else False

    n_ctx = len(context_images)
    ctx_imgs = torch.stack(context_images)
    ctx_poses = torch.stack(context_poses)
    ctx_K = torch.stack(context_intrinsics)

    has_target = (
        target_images is not None
        and target_poses is not None
        and target_intrinsics is not None
    )

    if has_target:
        tgt_imgs = torch.stack(target_images)
        tgt_poses = torch.stack(target_poses)
        tgt_K = torch.stack(target_intrinsics)
        all_poses = torch.cat([ctx_poses, tgt_poses], dim=0)
    else:
        all_poses = ctx_poses

    scale = torch.tensor(1.0)

    if normalize_scene_scale:
        a = ctx_poses[0, :3, 3]
        b = ctx_poses[-1, :3, 3]
        scale = (a - b).norm()
        if scale > 1e-8:
            all_poses = all_poses.clone()
            all_poses[:, :3, 3] = all_poses[:, :3, 3] / scale

    if relative_pose:
        all_poses = camera_normalization(all_poses[0:1], all_poses)

    ctx_poses_norm = all_poses[:n_ctx]
    near_f32 = torch.tensor(near, dtype=torch.float32)
    far_f32 = torch.tensor(far, dtype=torch.float32)
    if scale > 1e-8:
        near_f32 = near_f32 / scale
        far_f32 = far_f32 / scale

    if cfg is not None:
        ctx_imgs, ctx_K = rescale_and_crop_bilinear(
            ctx_imgs, ctx_K, tuple(cfg.input_image_shape)
        )

    batch: BatchedExample = {
        "context": {
            "image": ctx_imgs.unsqueeze(0).to(device),
            "poses": ctx_poses_norm.unsqueeze(0).to(device),
            "intrinsics": ctx_K.unsqueeze(0).to(device),
            "near": near_f32.unsqueeze(0).repeat(1, n_ctx).to(device),
            "far": far_f32.unsqueeze(0).repeat(1, n_ctx).to(device),
        },
    }

    if has_target:
        n_tgt = len(target_images)
        tgt_poses_norm = all_poses[n_ctx:]
        if cfg is not None:
            tgt_imgs, tgt_K = rescale_and_crop_bilinear(
                tgt_imgs, tgt_K, tuple(cfg.input_image_shape)
            )
        batch["target"] = {
            "image": tgt_imgs.unsqueeze(0).to(device),
            "poses": tgt_poses_norm.unsqueeze(0).to(device),
            "intrinsics": tgt_K.unsqueeze(0).to(device),
            "near": near_f32.unsqueeze(0).repeat(1, n_tgt).to(device),
            "far": far_f32.unsqueeze(0).repeat(1, n_tgt).to(device),
        }

    return batch
