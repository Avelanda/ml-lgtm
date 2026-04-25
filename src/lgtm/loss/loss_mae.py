"""
Mean absolute error loss for LGTM.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from lgtm.dataset.data_types import BatchedExample
from lgtm.loss.loss import Loss
from lgtm.model.gaussian.gaussian_rasterizer import GaussianRasterizerOutput
from lgtm.model.gaussian.gaussians import Gaussians


@dataclass
class LossMaeCfg:
    weight: float


@dataclass
class LossMaeCfgWrapper:
    mae: LossMaeCfg


class LossMae(Loss[LossMaeCfg, LossMaeCfgWrapper]):
    def forward(
        self,
        prediction: GaussianRasterizerOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        encoder_info: dict[str, Any] | None = None,
    ) -> Float[Tensor, ""]:
        delta = prediction.colors - batch["target"]["image"]
        return self.cfg.weight * (delta.abs()).mean()
