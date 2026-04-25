"""
Loss subpackage exports.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from lgtm.loss.loss import Loss
from lgtm.loss.loss_flash3d import (
    LossGaussianOffset,
    LossGaussianOffsetCfgWrapper,
    LossGaussianScale,
    LossGaussianScaleCfgWrapper,
)
from lgtm.loss.loss_lpips import LossLpips, LossLpipsCfgWrapper
from lgtm.loss.loss_mae import LossMae, LossMaeCfgWrapper
from lgtm.loss.loss_mse import LossMse, LossMseCfgWrapper
from lgtm.loss.loss_ssim import LossFlash3DSSIM, LossFlash3DSSIMCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossMaeCfgWrapper: LossMae,
    LossGaussianScaleCfgWrapper: LossGaussianScale,
    LossGaussianOffsetCfgWrapper: LossGaussianOffset,
    LossFlash3DSSIMCfgWrapper: LossFlash3DSSIM,
}

LossCfgWrapper = (
    LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossMaeCfgWrapper
    | LossGaussianScaleCfgWrapper
    | LossGaussianOffsetCfgWrapper
    | LossFlash3DSSIMCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
