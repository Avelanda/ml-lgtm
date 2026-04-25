"""
Color jitter and grayscale augmentation shim.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from lgtm.dataset.data_types import AnyExample, AnyViews


def reflect_poses(
    poses: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    """
    Reflect c2w poses across the YZ plane (flip X axis).

    Args:
        poses: c2w 4x4 matrices.

    Returns:
        Reflected c2w 4x4 matrices.
    """
    reflect = torch.eye(4, dtype=torch.float32, device=poses.device)
    reflect[0, 0] = -1
    return reflect @ poses @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    return {
        **views,
        "image": views["image"].flip(-1),
        "poses": reflect_poses(views["poses"]),
    }


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """
    Randomly augment the training images.
    """
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }
