"""
Utilities for Gaussian parameter prediction and packing.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def compute_sh_mask(
    sh_degree: int,
    device: torch.device | None = None,
) -> Tensor:
    """
    Create a mask for SH coefficients that biases towards a large DC component
    and small view-dependent components at initialization.
    """
    d_sh = (sh_degree + 1) ** 2
    sh_mask = torch.ones((d_sh,), dtype=torch.float32, device=device)
    for degree in range(1, sh_degree + 1):
        sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
    return sh_mask


def activate_scales(scales: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    Applies a softplus activation to the raw scales.
    """
    activated_scales = 0.001 * F.softplus(scales)
    return activated_scales


def deactivate_scales(
    activated_scales: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """
    Inverse of scale_activate.
    """
    z = 1000.0 * activated_scales
    scales = z + torch.log(-torch.expm1(-z))
    return scales


def activate_texture_alphas(
    texture_alphas: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """
    Applies a sigmoid activation to the raw texture alphas.
    """
    return F.sigmoid(texture_alphas)


def deactivate_texture_alphas(
    activated_texture_alphas: Float[Tensor, "..."],
    eps: float = 1e-6,
) -> Float[Tensor, "..."]:
    """
    Inverse of texture_alpha_activate. Takes sigmoid-activated texture alphas
    and returns raw texture alphas.
    """
    activated_texture_alphas = torch.clamp(
        activated_texture_alphas, min=eps, max=1.0 - eps
    )
    return torch.log(
        activated_texture_alphas / (1.0 - activated_texture_alphas)
    )


def rgbs_to_sh0_coefficients(
    rgbs: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """
    Converts pixel colors [0, 1] to the required raw SH[0] input for the CUDA
    rasterizer, reversing the C0 scaling and offset. raw = (color - 0.5) / C0.
    """
    if not rgbs.shape[-1] == 3:
        raise ValueError(
            f"pixel_colors's last dim must be 3, but got {rgbs.shape}"
        )
    SH_C0 = 0.28209479177387814
    sh0_coefficients = (rgbs - 0.5) / SH_C0
    return sh0_coefficients


def sh0_coefficients_to_rgbs(
    sh0_coefficients: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """
    Converts SH[0] coefficients to RGB values.
    """
    if not sh0_coefficients.shape[-1] == 3:
        raise ValueError(
            "sh0_coefficients's last dim must be 3, but got"
            f" {sh0_coefficients.shape}"
        )
    SH_C0 = 0.28209479177387814
    rgbs = sh0_coefficients * SH_C0 + 0.5
    return rgbs


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )
