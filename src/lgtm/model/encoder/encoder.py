"""
Abstract base and registry for LGTM view encoders.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from torch import nn

from lgtm.dataset.data_types import BatchedViews
from lgtm.model.gaussian.gaussians import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target: Optional[BatchedViews] = None,
    ) -> tuple[Gaussians, dict[Any, Any]]:
        pass
