"""
Flash3D-specific loss terms.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

from lgtm.dataset.data_types import BatchedExample
from lgtm.loss.loss import Loss
from lgtm.model.gaussian.gaussian_rasterizer import GaussianRasterizerOutput
from lgtm.model.gaussian.gaussians import Gaussians


@dataclass
class LossGaussianScaleCfg:
    weight: float
    thresh: float


@dataclass
class LossGaussianScaleCfgWrapper:
    gauss_scale: LossGaussianScaleCfg


class LossGaussianScale(
    Loss[LossGaussianScaleCfg, LossGaussianScaleCfgWrapper]
):
    def __init__(self, cfg: LossGaussianScaleCfgWrapper):
        super().__init__(cfg)

    def forward(
        self,
        prediction: GaussianRasterizerOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        encoder_info: dict[str, Any] | None = None,
    ) -> Float[Tensor, ""]:
        device = prediction.colors.device
        scaling = encoder_info.get("gauss_scaling") if encoder_info else None
        if scaling is None:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        big_gaussians = torch.where(scaling > self.cfg.thresh)
        if len(big_gaussians[0]) > 0:
            big_gauss_reg_loss = torch.mean(scaling[big_gaussians])
        else:
            big_gauss_reg_loss = torch.tensor(
                0.0, device=device, dtype=torch.float32
            )
        return big_gauss_reg_loss * self.cfg.weight


@dataclass
class LossGaussianOffsetCfg:
    weight: float
    thresh: float


@dataclass
class LossGaussianOffsetCfgWrapper:
    gauss_offset: LossGaussianOffsetCfg


class LossGaussianOffset(
    Loss[LossGaussianOffsetCfg, LossGaussianOffsetCfgWrapper]
):
    def __init__(self, cfg: LossGaussianOffsetCfgWrapper):
        super().__init__(cfg)

    def forward(
        self,
        prediction: GaussianRasterizerOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        encoder_info: dict[str, Any] | None = None,
    ) -> Float[Tensor, ""]:
        device = prediction.colors.device
        offset = encoder_info.get("gauss_offset") if encoder_info else None
        if offset is None:
            return torch.tensor(0.0, device=device, dtype=torch.float32)
        big_offset = torch.where(offset**2 > self.cfg.thresh**2)
        if len(big_offset[0]) > 0:
            big_offset_reg_loss = torch.mean(offset[big_offset] ** 2)
        else:
            big_offset_reg_loss = torch.tensor(
                0.0, device=device, dtype=torch.float32
            )
        return big_offset_reg_loss * self.cfg.weight
