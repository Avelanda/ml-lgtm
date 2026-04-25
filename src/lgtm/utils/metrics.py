"""
Novel-view quality metrics (PSNR, SSIM, LPIPS).

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import json
import logging
from functools import cache

import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from natsort import natsorted
from skimage.metrics import structural_similarity
from torch import Tensor

from lgtm.utils.logging_utils import get_exp_output_dir


def downsample_for_metric_or_loss(
    image: Float[Tensor, "batch channel height width"],
    max_dim: int,
) -> Float[Tensor, "batch channel new_height new_width"]:
    """
    Downsample image if max dimension exceeds max_dim
    while preserving aspect ratio.

    Args:
        image: Input image tensor
        max_dim: Maximum dimension for downsampling, 0 means no downsampling

    Returns:
        Downsampled image tensor
    """
    if max_dim == 0:
        return image

    _, _, h, w = image.shape
    input_max_dim = max(h, w)

    if input_max_dim <= max_dim:
        return image

    scale_factor = max_dim / input_max_dim
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # Use bilinear interpolation for downsampling.
    return F.interpolate(
        image,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    lpips_metric_max_dim: int = 0,
) -> Float[Tensor, " batch"]:

    if lpips_metric_max_dim > 0:
        ground_truth = downsample_for_metric_or_loss(
            image=ground_truth,
            max_dim=lpips_metric_max_dim,
        )
        predicted = downsample_for_metric_or_loss(
            image=predicted,
            max_dim=lpips_metric_max_dim,
        )

    value = get_lpips(predicted.device).forward(
        ground_truth, predicted, normalize=True
    )
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(
        cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))
    )
    cos = torch.max(
        cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1
    )

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.mm(R1.T, R2)) - 1) / 2
    cos = torch.clamp(
        cos, -1.0, 1.0
    )  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.acos(cos)))


def angle_error_vec(v1, v2):
    n = torch.norm(v1) * torch.norm(v2)
    cos_theta = torch.dot(v1, v2) / n
    cos_theta = torch.clamp(
        cos_theta, -1.0, 1.0
    )  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.acos(cos_theta))


def compute_translation_error(t1, t2):
    return torch.norm(t1 - t2)


@torch.no_grad()
def compute_pose_error(pose_gt, pose_pred):
    R_gt = pose_gt[:3, :3]
    t_gt = pose_gt[:3, 3]

    R = pose_pred[:3, :3]
    t = pose_pred[:3, 3]

    error_t = angle_error_vec(t, t_gt)
    error_t = torch.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_t_scale = compute_translation_error(t, t_gt)
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_t_scale, error_R


def collect_test_metrics() -> list[dict]:
    """
    Collect test metrics from all scene JSON files and return
    a natsorted list of dicts.

    Returns:
        List of scene metric dicts sorted by scene_id
    """
    test_metrics_dir = get_exp_output_dir() / "test_metrics"

    if not test_metrics_dir.exists():
        return []
    metric_paths = list(test_metrics_dir.glob("**/*.json"))
    if not metric_paths:
        return []

    test_metrics = []
    for metric_path in metric_paths:
        try:
            with open(metric_path, "r") as f:
                test_metric = json.load(f)
            if "scene_id" in test_metric:
                test_metrics.append(test_metric)
        except Exception as e:
            logging.warning("Failed to load metric file %s: %s", metric_path, e)
            continue

    test_metrics = natsorted(test_metrics, key=lambda x: x.get("scene_id", ""))
    return test_metrics


def aggregate_test_metrics() -> dict:
    """
    Aggregate test metrics from all scene JSON files and compute
    averages.

    Returns:
        Dict with keys num_scenes, num_context_views,
        num_target_views, lpips, ssim, psnr. Empty dict if no
        metrics found.
    """
    test_metrics = collect_test_metrics()
    if not test_metrics:
        return {}

    all_target_metrics = []
    num_scenes = 0
    num_context_views = 0
    num_target_views = 0

    for test_metric in test_metrics:
        if "scene_id" in test_metric:
            num_scenes += 1
        if "context_indices" in test_metric:
            num_context_views += len(test_metric["context_indices"])
        if "target_metrics" in test_metric:
            num_target_views += len(test_metric["target_metrics"])
            all_target_metrics.extend(test_metric["target_metrics"].values())

    if not all_target_metrics:
        return {}

    test_metrics_aggregated = {
        "num_scenes": num_scenes,
        "num_context_views": num_context_views,
        "num_target_views": num_target_views,
    }
    for metric_name in ["lpips", "ssim", "psnr"]:
        values = [
            m[metric_name] for m in all_target_metrics if metric_name in m
        ]
        if values:
            test_metrics_aggregated[metric_name] = sum(values) / len(values)

    return test_metrics_aggregated
