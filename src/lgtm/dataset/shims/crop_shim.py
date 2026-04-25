"""
Random and center crop shim for multi-view batches.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import resize

from lgtm.dataset.data_types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    if h_scaled != h_out and w_scaled != w_out:
        raise ValueError(
            f"Scaled dimensions ({h_scaled}, {w_scaled}) do not match "
            f"target ({h_out}, {w_out}) on either axis"
        )

    # Reshape the images to the correct size. Assume we don't have.
    # To worry about changing the intrinsics based on how the images.
    # Are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack(
        [rescale(image, (h_scaled, w_scaled)) for image in images]
    )
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(
    views: AnyViews,
    shape: tuple[int, int],
    *,
    use_torch_bilinear: bool = False,
) -> AnyViews:
    fn = rescale_and_crop_bilinear if use_torch_bilinear else rescale_and_crop
    images, intrinsics = fn(views["image"], views["intrinsics"], shape)
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(
    example: AnyExample,
    shape: tuple[int, int],
    *,
    use_torch_bilinear: bool = False,
) -> AnyExample:
    """
    Crop images in the example.
    """
    return {
        **example,
        "context": apply_crop_shim_to_views(
            example["context"], shape, use_torch_bilinear=use_torch_bilinear
        ),
        "target": apply_crop_shim_to_views(
            example["target"], shape, use_torch_bilinear=use_torch_bilinear
        ),
    }


def rescale_and_crop_bilinear(
    images: Float[Tensor, "n c h_in w_in"],
    intrinsics: Float[Tensor, "n 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "n c h_out w_out"],
    Float[Tensor, "n 3 3"],
]:
    """
    Rescale images to minimally cover shape, then center-crop.

    Uses torchvision bilinear resize (rather than PIL LANCZOS) to match the
    preprocessing used in inference example scripts, ensuring numerical parity
    with the original example outputs.

    Args:
        images: Input images [N, C, H_in, W_in] in [0, 1].
        intrinsics: Normalized intrinsics [N, 3, 3].
        shape: Target (H_out, W_out).

    Returns:
        Tuple of (cropped images [N, C, H_out, W_out],
        adjusted intrinsics [N, 3, 3]).
    """
    _, _, h_in, w_in = images.shape
    h_out, w_out = shape
    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    resized = torch.stack([resize(img, (h_scaled, w_scaled)) for img in images])
    row = (h_scaled - h_out) / 2
    col = (w_scaled - w_out) / 2
    cropped = resized[
        :, :, int(row) : int(row) + h_out, int(col) : int(col) + w_out
    ]

    adjusted = intrinsics.clone()
    adjusted[:, 0, 2] -= col / w_scaled
    adjusted[:, 1, 2] -= row / h_scaled
    adjusted[:, 0, :] *= w_scaled / w_out
    adjusted[:, 1, :] *= h_scaled / h_out
    return cropped, adjusted
