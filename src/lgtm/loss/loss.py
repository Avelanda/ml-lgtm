"""
Combined training loss for LGTM (reconstruction + auxiliary terms).

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from lgtm.dataset.data_types import BatchedExample
from lgtm.model.gaussian.gaussian_rasterizer import GaussianRasterizerOutput
from lgtm.model.gaussian.gaussians import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction: GaussianRasterizerOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        encoder_info: dict[str, Any] | None = None,
    ) -> Float[Tensor, ""]:
        pass
