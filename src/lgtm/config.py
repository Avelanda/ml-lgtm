"""
Hydra/dataclass configuration for LGTM experiments.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar

from dacite import Config, from_dict
from gsplat.cuda._wrapper import TextureAlphaMode
from omegaconf import DictConfig, OmegaConf

from lgtm.dataset.data_module import DataLoaderCfg
from lgtm.dataset.data_types import Mode
from lgtm.dataset.dataset import DatasetCfgWrapper
from lgtm.loss import LossCfgWrapper
from lgtm.model.encoder import EncoderCfg
from lgtm.model.encoder.encoder_depthsplat import EncoderDepthSplatCfg
from lgtm.model.encoder.encoder_flash3d import EncoderFlash3DCfg
from lgtm.model.encoder.encoder_noposplat import EncoderNoPoSplatCfg
from lgtm.model.gaussian.gaussian_rasterizer import RasterizerCfg
from lgtm.utils.pose_opt import PoseOptConfig

_cfg: Optional[DictConfig] = None


def set_cfg(new_cfg: DictConfig) -> None:
    global _cfg
    _cfg = new_cfg


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_mult: Optional[float] = None
    non_head_lr_mult: Optional[float] = None
    non_2dgs_lr_mult: Optional[float] = None


@dataclass
class TrainCfg:
    print_log_every_n_steps: int
    enable_dry_run: bool = False
    validation_lpips_metric_max_dim: int = 0
    train_target_image_shape: Optional[list[int]] = None  # [h, w]


@dataclass
class TestCfg:
    enable_pose_opt: bool
    compute_scores: bool

    lpips_metric_max_dim: int = 0
    test_target_image_shape: Optional[list[int]] = None  # [h, w]


@dataclass
class CheckpointingCfg:
    test_ckpt_path: Optional[str]
    every_n_train_steps: int
    save_top_k: int
    save_weights_only: bool
    maybe_use_remote_ckpt: bool = False
    sync_to_remote_after_ckpt_save: bool = False


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_nodes: int = 1
    log_every_n_steps: int = 50


@dataclass
class RootCfg:
    wandb: dict
    mode: Mode
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    encoder: EncoderCfg
    rasterizer: RasterizerCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    pose_opt: PoseOptConfig
    seed: int
    sync_to_remote_on_finish: bool


_TEXTURE_ALPHA_MODE_MAP = {
    "TEXTURE": TextureAlphaMode.TEXTURE,
    "GAUSSIAN": TextureAlphaMode.GAUSSIAN,
    "MULTIPLY": TextureAlphaMode.MULTIPLY,
    0: TextureAlphaMode.TEXTURE,
    1: TextureAlphaMode.GAUSSIAN,
    2: TextureAlphaMode.MULTIPLY,
}


def _parse_texture_alpha_mode(value):
    if isinstance(value, TextureAlphaMode):
        return value
    if value in _TEXTURE_ALPHA_MODE_MAP:
        return _TEXTURE_ALPHA_MODE_MAP[value]
    raise ValueError(
        f"Invalid texture_alpha_mode: {value!r}. "
        "Use a string (TEXTURE/GAUSSIAN/MULTIPLY) or int (0/1/2)."
    )


_ENCODER_CFG_MAP = {
    "noposplat": EncoderNoPoSplatCfg,
    "depthsplat": EncoderDepthSplatCfg,
    "unidepth": EncoderFlash3DCfg,
}

_BASE_TYPE_HOOKS = {
    Path: Path,
    TextureAlphaMode: _parse_texture_alpha_mode,
}


def _parse_encoder_cfg(value):
    if not isinstance(value, dict):
        return value
    name = value.get("name")
    if name not in _ENCODER_CFG_MAP:
        raise ValueError(f"Unknown encoder name: {name}")
    config = Config(
        type_hooks=_BASE_TYPE_HOOKS,
        check_types=False,
        strict_unions_match=False,
        cast=[float],
    )
    return from_dict(_ENCODER_CFG_MAP[name], value, config=config)


_TYPE_HOOKS = {
    **_BASE_TYPE_HOOKS,
    EncoderCfg: _parse_encoder_cfg,
}

T = TypeVar("T")


def _load_typed(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(
            type_hooks={**_TYPE_HOOKS, **extra_type_hooks},
            check_types=False,
            strict_unions_match=False,
            cast=[float],
        ),
    )


def _separate_loss_cfgs(joined: dict) -> list[LossCfgWrapper]:
    @dataclass
    class _W:
        item: LossCfgWrapper

    return [
        _load_typed(DictConfig({"item": {k: v}}), _W).item
        for k, v in joined.items()
        if v is not None
    ]


def _separate_dataset_cfgs(joined: dict) -> list[DatasetCfgWrapper]:
    @dataclass
    class _W:
        item: DatasetCfgWrapper

    return [
        _load_typed(DictConfig({"item": {k: v}}), _W).item
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return _load_typed(
        cfg,
        RootCfg,
        {
            list[LossCfgWrapper]: _separate_loss_cfgs,
            list[DatasetCfgWrapper]: _separate_dataset_cfgs,
        },
    )
