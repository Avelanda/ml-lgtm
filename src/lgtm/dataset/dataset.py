"""
Multi-view scene dataset loading from .torch chunks.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float, UInt8
from torch import Tensor
from torch.utils.data import IterableDataset

import lgtm.storage as storage
from lgtm.dataset.data_types import (
    DataStage,
    Mode,
)
from lgtm.dataset.shims.augmentation_shim import apply_augmentation_shim
from lgtm.dataset.shims.crop_shim import (
    apply_crop_shim,
    apply_crop_shim_to_views,
)
from lgtm.dataset.view_sampler import (
    ViewSampler,
    ViewSamplerCfg,
    get_view_sampler,
)
from lgtm.utils.camera_utils import camera_normalization
from lgtm.utils.geometry_utils import get_fov
from lgtm.utils.logging_utils import safe_log_to_file
from lgtm.utils.training_utils import StepTracker


@dataclass
class DatasetCfgCommon:
    original_image_shape: list[int]
    input_image_shape: list[int]
    background_color: list[float]
    view_sampler: ViewSamplerCfg


@dataclass
class DatasetCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    normalize_scene_scale: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    view_sampler_on_test: Optional[ViewSamplerCfg] = None
    view_sampler_on_val: Optional[ViewSamplerCfg] = None
    skip_bad_scenes: list[str] = field(default_factory=list)
    train_num_repeat_scenes_in_chunk: int = 1
    include_context_view: bool = False
    near: float = 0.1
    far: float = 100.0


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetCfg


DatasetCfgWrapper = DatasetDL3DVCfgWrapper


class Dataset(IterableDataset):
    cfg: DatasetCfg
    mode: Mode
    data_stage: DataStage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetCfg,
        mode: Mode,
        data_stage: DataStage,
        view_sampler: ViewSampler,
        target_image_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.data_stage = data_stage
        self.view_sampler = view_sampler
        self.target_image_shape = target_image_shape
        self.to_tensor = tf.ToTensor()
        self.near = cfg.near  # for self.get_bound("near", ...)
        self.far = cfg.far  # for self.get_bound("far", ...)

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = storage.resolve_path(root)
            root = root / self.data_folder

            root_chunks = storage.list_children(root, suffix=".torch")
            self.chunks.extend(root_chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def repeat_scenes_in_chunk(self, chunk: list, num_repeats: int) -> list:
        """
        Shallow repeat each example in the chunk num_repeats times.
        Interleaves repeats: [A, B] with num_repeats=2 -> [A, B, A, B]
        """
        if num_repeats == 1:
            return chunk

        # Interleave repeats instead of consecutive duplicates.
        duplicated = []
        for repeat_idx in range(num_repeats):
            for example in chunk:
                duplicated.append(example)  # Reference only, no data copying
        return duplicated

    def __iter__(self):
        # Shuffle chunks only during training mode (cfg.mode == "train")
        # This applies to both training data and validation data during training.
        # Do not shuffle during test mode.
        if self.mode == "train":
            self.chunks = self.shuffle(self.chunks)

        # Get GPU index, worker index, and the total numbers.
        if dist.is_available() and dist.is_initialized():
            gpu_idx = dist.get_rank()
            num_gpus = dist.get_world_size()
        else:
            gpu_idx = 0
            num_gpus = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_idx = worker_info.id
            num_workers = worker_info.num_workers  # Per GPU
        else:
            worker_idx = 0
            num_workers = 1

        if self.mode == "test":
            original_chunk_count = len(self.chunks)

            # Distribute chunks across GPUs.
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % num_gpus == gpu_idx
            ]

            # Print GPU-level distribution.
            if worker_idx == 0:
                gpu_distribution_msg = (
                    f"GPU[{gpu_idx}]: assigned "
                    f"{len(self.chunks)}/{original_chunk_count} chunks"
                )
                safe_log_to_file(
                    message=gpu_distribution_msg,
                    rel_log_path="test_dataloader.log",
                    mode="a",
                )
                print(gpu_distribution_msg)

            # Distribute chunks across workers for this GPU.
            # Printed by all workers on each GPU.
            gpu_chunk_count = len(self.chunks)
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % num_workers == worker_idx
            ]
            worker_distribution_msg = (
                f"GPU[{gpu_idx}]-Worker[{worker_idx}]: "
                f"assigned {len(self.chunks)}/{gpu_chunk_count} chunks"
            )
            safe_log_to_file(
                message=worker_distribution_msg,
                rel_log_path="test_dataloader.log",
                mode="a",
            )

        # target_image_shape is passed in at construction time from train/test config.
        # It controls the resolution at which target views are cropped for rasterization.
        for chunk_idx, chunk_path in enumerate(self.chunks):
            if self.mode == "test":
                log_message = (
                    f"GPU[{gpu_idx}]-Worker[{worker_idx}]: "
                    f"loading chunks[{chunk_idx}]: {chunk_path.name}"
                )
                safe_log_to_file(
                    message=log_message,
                    rel_log_path="test_dataloader.log",
                    mode="a",
                )

            # Fetch chunk (no-op for local paths)
            chunk_path = storage.fetch_to_local(chunk_path)

            # Load the chunk.
            # These are data, not weights, so weights_only=False.
            chunk = torch.load(chunk_path, weights_only=False)

            # Shuffle examples within each chunk during training mode.
            if self.mode == "train":
                chunk = self.shuffle(chunk)

            # Repeat examples for multiple sampling per scene (training only)
            if (
                self.mode == "train"
                and self.data_stage == "train"
                and self.cfg.train_num_repeat_scenes_in_chunk > 1
            ):
                chunk = self.repeat_scenes_in_chunk(
                    chunk=chunk,
                    num_repeats=self.cfg.train_num_repeat_scenes_in_chunk,
                )

            for example in chunk:
                poses, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                # Skip scenes that are in the skip_bad_scenes list.
                if scene in self.cfg.skip_bad_scenes:
                    print(f"Skipped bad scene (skip_bad_scenes): {scene}.")
                    continue

                try:
                    context_indices, target_indices, overlap = (
                        self.view_sampler.sample(
                            scene,
                            poses,
                            intrinsics,
                        )
                    )
                    if self.cfg.include_context_view and self.mode == "train":
                        target_indices = torch.cat(
                            [context_indices[0:1], target_indices]
                        )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                try:
                    context_images = [
                        example["images"][index.item()]
                        for index in context_indices
                    ]
                    context_images = self.convert_images(context_images)
                    target_images = [
                        example["images"][index.item()]
                        for index in target_indices
                    ]
                    target_images = self.convert_images(target_images)
                except IndexError:
                    continue
                except OSError:
                    print(f"Skipped bad example {example['key']}.")
                    continue

                # Check: images don't have the expected shape.
                context_image_invalid = context_images.shape[1:] != (
                    3,
                    *self.cfg.original_image_shape,
                )
                target_image_invalid = target_images.shape[1:] != (
                    3,
                    *self.cfg.original_image_shape,
                )
                if self.cfg.skip_bad_shape and (
                    context_image_invalid or target_image_invalid
                ):
                    print(
                        f"Skipped bad example {example['key']}. Context shape"
                        f" was {context_images.shape} and target shape was"
                        f" {target_images.shape}."
                    )
                    continue

                # Check: NaN in pose determinants.
                if any(
                    torch.isnan(torch.det(poses[context_indices][:, :3, :3]))
                ):
                    print(
                        f"Skipped {scene} because of NaN in context pose"
                        " determinant"
                    )
                    continue
                if any(
                    torch.isnan(torch.det(poses[target_indices][:, :3, :3]))
                ):
                    print(
                        f"Skipped {scene} because of NaN in target pose"
                        " determinant"
                    )
                    continue

                # Check: large camera translations.
                # https://github.com/DL3DV-10K/Dataset/issues/34
                if (poses[context_indices][:, :3, 3] > 1e3).any():
                    print(
                        f"Skipped {scene} because of extremely large context"
                        " camera translation"
                    )
                    continue
                if (poses[target_indices][:, :3, 3] > 1e3).any():
                    print(
                        f"Skipped {scene} because of extremely large target"
                        " camera translation"
                    )
                    continue

                # Check: rotation matrix determinant close to 1.
                if not torch.allclose(
                    torch.det(poses[context_indices][:, :3, :3]),
                    torch.det(poses[context_indices][:, :3, :3]).new_tensor(1),
                ):
                    print(
                        f"Skipped {scene} because context rotation matrix"
                        " determinant not close to 1"
                    )
                    continue
                if not torch.allclose(
                    torch.det(poses[target_indices][:, :3, :3]),
                    torch.det(poses[target_indices][:, :3, :3]).new_tensor(1),
                ):
                    print(
                        f"Skipped {scene} because target rotation matrix"
                        " determinant not close to 1"
                    )
                    continue

                # Check: singular camera matrices.
                try:
                    # Inverse the camera matrices to check for singularity.
                    _ = poses.inverse()

                    # Invert the make_scale_invariant poses.
                    # See cuda_splatting.py.
                    poses_clone = poses.clone()
                    poses_clone[..., :3, 3] = poses_clone[..., :3, 3] * (
                        1.0 / self.near
                    )
                    _ = poses_clone.inverse()
                except torch._C._LinAlgError:
                    print(f"Skipped {example['key']} due to singular camera")
                    continue
                except Exception as e:
                    print(f"Skipped {example['key']} due to {e}")
                    continue

                # Resize the world to make the baseline 1
                context_poses = poses[context_indices]
                if self.cfg.normalize_scene_scale:
                    a, b = context_poses[0, :3, 3], context_poses[-1, :3, 3]
                    scale = (a - b).norm()
                    if (
                        scale < self.cfg.baseline_min
                        or scale > self.cfg.baseline_max
                    ):
                        print(
                            f"Skipped {scene} because of baseline out of range:"
                            f" {scale:.6f}"
                        )
                        continue
                    poses[:, :3, 3] /= scale
                else:
                    scale = 1

                if self.cfg.relative_pose:
                    poses = camera_normalization(
                        poses[context_indices][0:1], poses
                    )

                example = {
                    "context": {
                        "poses": poses[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": (
                            self.get_bound("near", len(context_indices)) / scale
                        ),
                        "far": (
                            self.get_bound("far", len(context_indices)) / scale
                        ),
                        "index": context_indices,
                        "overlap": overlap,
                    },
                    "target": {
                        "poses": poses[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": (
                            self.get_bound("near", len(target_indices)) / scale
                        ),
                        "far": (
                            self.get_bound("far", len(target_indices)) / scale
                        ),
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                # Apply augmentation during training mode.
                # - train with training dataloader: augment.
                # - train with validation dataloader: no augment.
                # - test: no augment.
                if (
                    self.mode == "train"
                    and self.data_stage == "train"
                    and self.cfg.augment
                ):
                    example = apply_augmentation_shim(example)

                # Handle training-time (inc. validation)
                # target_image_shape override.
                # - low-res.
                # - Shape  : input_image_shape.
                # - Applied: context gt views.
                # - Applied: shape received by the encoder network.
                # - High-res.
                # - Shape  : train_target_image_shape.
                # - Applied: target gt views.
                # - Applied: target pd views.
                # - Applied: shape received by the rasterizer.
                if self.target_image_shape is not None:
                    if self.mode == "train":
                        context_cropped = apply_crop_shim_to_views(
                            example["context"],
                            shape=tuple(self.cfg.input_image_shape),
                        )
                        target_cropped = apply_crop_shim_to_views(
                            example["target"],
                            shape=self.target_image_shape,
                        )
                        yield {
                            **example,
                            "context": context_cropped,
                            "target": target_cropped,
                        }
                        continue

                    if self.mode == "test":
                        original_shape = tuple(self.cfg.input_image_shape)
                        example_original = apply_crop_shim(
                            example=example,
                            shape=original_shape,
                            use_torch_bilinear=True,
                        )
                        example_hi_res = apply_crop_shim(
                            example=example,
                            shape=self.target_image_shape,
                            use_torch_bilinear=True,
                        )
                        example_original["test_target"] = example_hi_res[
                            "target"
                        ]
                        yield example_original
                        continue

                # Default yield case.
                yield apply_crop_shim(
                    example=example,
                    shape=tuple(self.cfg.input_image_shape),
                )

    def convert_poses(
        self,
        cameras: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],
        Float[Tensor, "batch 3 3"],
    ]:
        from lgtm.dataset.data_utils import convert_poses

        return convert_poses(cameras)

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        from lgtm.dataset.data_utils import convert_images

        return convert_images(images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_folder(self) -> DataStage:
        return self.data_stage


DATASETS: dict[str, type[Dataset]] = {
    "dl3dv": Dataset,
}


def get_dataset(
    dataset_cfgs: list[DatasetCfgWrapper],
    mode: Mode,
    data_stage: DataStage,
    step_tracker: StepTracker | None,
    target_image_shape: tuple[int, int] | None = None,
) -> list[Dataset]:
    datasets = []
    for dataset_cfg in dataset_cfgs:
        (field_info,) = fields(type(dataset_cfg))
        dataset_cfg = getattr(dataset_cfg, field_info.name)

        if (
            mode == "test"
            and hasattr(dataset_cfg, "view_sampler_on_test")
            and dataset_cfg.view_sampler_on_test is not None
        ):
            view_sampler_cfg = dataset_cfg.view_sampler_on_test
            print(
                f"Using view_sampler_on_test view sampler: {view_sampler_cfg}"
            )
        elif (
            mode == "train"
            and data_stage == "test"
            and hasattr(dataset_cfg, "view_sampler_on_val")
            and dataset_cfg.view_sampler_on_val is not None
        ):
            view_sampler_cfg = dataset_cfg.view_sampler_on_val
            print(f"Using view_sampler_on_val view sampler: {view_sampler_cfg}")
        else:
            view_sampler_cfg = dataset_cfg.view_sampler

        view_sampler = get_view_sampler(
            cfg=view_sampler_cfg,
            data_stage=data_stage,
            step_tracker=step_tracker,
        )
        dataset = DATASETS[dataset_cfg.name](
            cfg=dataset_cfg,
            mode=mode,
            data_stage=data_stage,
            view_sampler=view_sampler,
            target_image_shape=target_image_shape,
        )
        datasets.append(dataset)

    return datasets
