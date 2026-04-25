"""
LPIPS perceptual loss wrapper for LGTM.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from lgtm.dataset.data_types import BatchedExample
from lgtm.loss.loss import Loss
from lgtm.model.gaussian.gaussian_rasterizer import GaussianRasterizerOutput
from lgtm.model.gaussian.gaussians import Gaussians
from lgtm.utils.metrics import downsample_for_metric_or_loss
from lgtm.utils.model_utils import convert_to_buffer


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    lpips_loss_max_dim: int = 0


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: GaussianRasterizerOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        encoder_info: dict[str, Any] | None = None,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        # Check input shapes for robustness.
        if prediction.colors.shape != image.shape:
            raise ValueError(
                "Prediction and target shapes must match:"
                f" {prediction.colors.shape} vs {image.shape}"
            )

        pred_colors = rearrange(prediction.colors, "b v c h w -> (b v) c h w")
        gt_colors = rearrange(image, "b v c h w -> (b v) c h w")

        # Apply downsampling if specified.
        if self.cfg.lpips_loss_max_dim > 0:
            pred_colors = downsample_for_metric_or_loss(
                image=pred_colors,
                max_dim=self.cfg.lpips_loss_max_dim,
            )
            gt_colors = downsample_for_metric_or_loss(
                image=gt_colors,
                max_dim=self.cfg.lpips_loss_max_dim,
            )

        loss = self.lpips.forward(
            pred_colors,
            gt_colors,
            normalize=True,
        )

        return self.cfg.weight * loss.mean()
