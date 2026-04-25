"""
ImageNet normalization shim for RGB tensors.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch

from lgtm.dataset.data_types import BatchedExample


def inverse_normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    return tensor * std + mean


def normalize_image(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    return (tensor - mean) / std


def apply_normalize_shim(
    batch: BatchedExample,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> BatchedExample:
    batch["context"]["image"] = normalize_image(
        batch["context"]["image"], mean, std
    )
    return batch
