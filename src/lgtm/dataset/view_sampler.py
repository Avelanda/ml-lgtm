"""
Train/test sampling of context and target views.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int, Int64
from torch import Tensor

from lgtm.dataset.data_types import DataStage
from lgtm.utils.training_utils import StepTracker


def add_third_context_index(
    indices: Int[Tensor, "*batch 2"],
) -> Int[Tensor, "*batch 3"]:
    left, right = indices.unbind(dim=-1)
    return torch.stack((left, (left + right) // 2, right), dim=-1)


T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    data_stage: DataStage
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        data_stage: DataStage,
        step_tracker: StepTracker | None,
    ) -> None:
        self.cfg = cfg
        self.data_stage = data_stage
        self.step_tracker = step_tracker

    @abstractmethod
    def sample(
        self,
        scene: str,
        poses: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        """
        Sample context and target view indices from the scene.

        Args:
            scene: Scene identifier string.
            poses: c2w 4x4 matrices [view, 4, 4].
            intrinsics: Normalized intrinsic 3x3 matrices [view, 3, 3].
            device: Target device for output tensors.
        """
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    sort_view_indices: bool = True


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        poses: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = poses.shape

        # Compute the context view spacing based on the current global step.
        if self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        index_context_right = index_context_left + context_gap

        # Pick the target view indices.
        # When training or validating (visualizing), pick at random.
        index_target = torch.randint(
            index_context_left + self.cfg.min_distance_to_context_views,
            index_context_right + 1 - self.cfg.min_distance_to_context_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        # If more than two context views are desired, pick extra context views.
        # Between the left and right ones.
        if self.cfg.num_context_views > 2:
            num_extra_views = self.cfg.num_context_views - 2
            extra_views = []
            while len(set(extra_views)) != num_extra_views:
                extra_views = torch.randint(
                    index_context_left + 1,
                    index_context_right,
                    (num_extra_views,),
                ).tolist()
        else:
            extra_views = []

        overlap = torch.tensor(
            [0.5], dtype=torch.float32, device=device
        )  # dummy

        if self.cfg.sort_view_indices:
            # Sort target indices for consistent network supervision.
            index_target_sorted, _ = torch.sort(index_target)
        else:
            index_target_sorted = index_target

        return (
            torch.tensor(
                (index_context_left, *extra_views, index_context_right)
            ),
            index_target_sorted,
            overlap,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]
    # Choose from ["small", "medium", "large"] or a float number indicates the.
    # Overlap ratio.
    overlap: Optional[str | float] = None


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int
    sort_view_indices: bool = True


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        data_stage: DataStage,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, data_stage, step_tracker)

        self.cfg = cfg

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: (
                    None
                    if v is None
                    else from_dict(IndexEntry, v, dacite_config)
                )
                for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        poses: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        # Match 1: First try exact match.
        entry = self.index.get(scene)

        # Match 2: If not found, try to handle prefix mismatch.
        # (e.g., "2K/hash" vs "hash")
        if entry is None and "/" in scene:
            # Split by "/" and try matching with the hash part (last part)
            scene_without_prefix = scene.split("/")[-1]
            entry = self.index.get(scene_without_prefix)

        # Match 3: If still not found, try partial matching.
        if entry is None:
            # Look for any key in the index that ends with the.
            # Scene or contains it.
            matching_keys = []
            for key in self.index.keys():
                if (
                    scene.endswith(key)
                    or key.endswith(scene)
                    or scene in key
                    or key in scene
                ):
                    matching_keys.append(key)

            if len(matching_keys) == 1:
                entry = self.index[matching_keys[0]]
            elif len(matching_keys) > 1:
                # Multiple matches found, try to pick the best one.
                # Prefer exact suffix match over partial match.
                exact_suffix_matches = [
                    k
                    for k in matching_keys
                    if scene.endswith(k) or k.endswith(scene)
                ]
                if len(exact_suffix_matches) == 1:
                    entry = self.index[exact_suffix_matches[0]]
                else:
                    print(
                        f"Warning: Multiple matching scenes found for {scene}:"
                        f" {matching_keys}. Using first match:"
                        f" {matching_keys[0]}"
                    )
                    entry = self.index[matching_keys[0]]

        if entry is None:
            raise ValueError(
                f"No indices available for scene {scene}. Available keys:"
                f" {list(self.index.keys())[:10]}..."
            )
        context_indices = torch.tensor(
            entry.context, dtype=torch.int64, device=device
        )
        target_indices = torch.tensor(
            entry.target, dtype=torch.int64, device=device
        )

        overlap = (
            entry.overlap
            if isinstance(entry.overlap, float)
            else 0.75 if entry.overlap == "large" else 0.25
        )
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        # Handle 2-view index for 3 views.
        v = self.num_context_views
        if v > len(context_indices) and v == 3:
            context_indices = add_third_context_index(context_indices)

        if self.cfg.sort_view_indices:
            # Sort context indices for consistent network input.
            context_indices_sorted, _ = torch.sort(context_indices)
        else:
            context_indices_sorted = context_indices

        if self.cfg.sort_view_indices:
            # Sort target indices for consistent network supervision.
            target_indices_sorted, _ = torch.sort(target_indices)
        else:
            target_indices_sorted = target_indices

        return context_indices_sorted, target_indices_sorted, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0


VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
}

ViewSamplerCfg = ViewSamplerBoundedCfg | ViewSamplerEvaluationCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    data_stage: DataStage,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        data_stage,
        step_tracker,
    )
