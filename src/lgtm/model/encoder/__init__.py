"""
Encoder subpackage exports.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from typing import Union

from lgtm.model.encoder.encoder import Encoder
from lgtm.model.encoder.encoder_depthsplat import (
    EncoderDepthSplat,
    EncoderDepthSplatCfg,
)
from lgtm.model.encoder.encoder_flash3d import EncoderFlash3D, EncoderFlash3DCfg
from lgtm.model.encoder.encoder_noposplat import (
    EncoderNoPoSplat,
    EncoderNoPoSplatCfg,
)

ENCODERS = {
    "noposplat": EncoderNoPoSplat,
    "depthsplat": EncoderDepthSplat,
    "unidepth": EncoderFlash3D,
}

EncoderCfg = Union[EncoderNoPoSplatCfg, EncoderDepthSplatCfg, EncoderFlash3DCfg]


def get_encoder(cfg: EncoderCfg) -> Encoder:
    if cfg.name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {cfg.name}")
    return ENCODERS[cfg.name](cfg)
