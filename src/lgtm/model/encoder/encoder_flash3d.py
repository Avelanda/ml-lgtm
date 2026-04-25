"""
Flash3D monocular view encoder.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._wrapper import TextureAlphaMode
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms as tvt

from lgtm.model.backbone.backbone_flash3d import GaussianPredictor
from lgtm.model.encoder.encoder import Encoder
from lgtm.model.gaussian.gaussian_utils import build_covariance
from lgtm.model.gaussian.gaussians import GaussianCfg, Gaussians
from lgtm.model.head.head_flash3d import GSHeadFlash3D
from lgtm.utils.camera_utils import normalized_K_to_K


@dataclass
class EncoderFlash3DDepthCfg:
    version: Literal["v1"]
    backbone: Literal["vitl14"]
    pretrained_checkpoint_path: str | None


@dataclass
class EncoderFlash3DBackboneCfg:
    name: Literal["resnet"]
    num_layers: int
    num_ch_dec: list[int]
    resnet_bn_order: Literal["pre_bn"]
    weights_init: Literal["pretrained"]
    upsample_mode: Literal["nearest"]
    depth_cond: bool


@dataclass
class EncoderFlash3DDatasetCfg:
    height: int
    width: int
    pad_border_aug: int


@dataclass
class EncoderFlash3DCfg:
    name: Literal["unidepth"]

    # Configs for texture head.
    gaussian: GaussianCfg

    # Configs from flash3d.
    depth: EncoderFlash3DDepthCfg
    backbone: EncoderFlash3DBackboneCfg
    dataset: EncoderFlash3DDatasetCfg

    scales: list[int]
    min_depth: float
    max_depth: float

    # Gaussian parameters.
    gaussians_per_pixel: int
    gaussian_rendering: bool
    max_sh_degree: int
    scaled_offset: bool
    one_gauss_decoder: bool
    predict_offset: bool
    shift_rays_half_pixel: Literal["forward", "backward"]

    depth_type: Literal["depth_inc", "disp", "disp_inc", "depth"]
    depth_scale: float
    xyz_scale: float
    opacity_scale: float
    scale_scale: float
    sh_scale: float

    scale_lambda: float
    depth_bias: float
    xyz_bias: float
    opacity_bias: float
    scale_bias: float

    # Common configs with noposplat.
    pretrained_checkpoint_path: str = ""


def build_gaussians_flash3d(
    means: Float[Tensor, "batch g 3"],
    scales: Float[Tensor, "batch g 3"],
    xyzws: Float[Tensor, "batch g 4"],
    opacities: Float[Tensor, "batch g"],
    features_dc: Float[Tensor, "batch g 3 1"],
    features_rest: Float[Tensor, "batch g 3 sh_rest"],
) -> Gaussians:
    """
    Convert Flash3D model outputs to Gaussians.

    Unlike NoPoSplat/DepthSplat, Flash3D outputs are already activated
    (scales, xyzws, opacities). This function builds covariance matrices,
    concatenates SH harmonics, and returns the final Gaussians.
    """
    covariances = build_covariance(scales, xyzws)
    harmonics = torch.cat([features_dc, features_rest], dim=-1)
    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        xyzws=xyzws,
    )


class EncoderFlash3D(Encoder[EncoderFlash3DCfg]):
    """
    Define the network of Flash3D.
    """

    def __init__(self, cfg: EncoderFlash3DCfg):
        super().__init__(cfg)

        # Currently load the cfg in a tricky way.
        cfg.model = cfg
        self.gaussian_predictor = GaussianPredictor(cfg)

        # Pre-compute Gaussian config-derived values.
        self.texture_enabled = cfg.gaussian.texture_enabled
        self.texture_size = (
            cfg.gaussian.texture_size if cfg.gaussian.texture_enabled else 1
        )
        self.texture_alpha_mode = cfg.gaussian.texture_alpha_mode

        # The texture branch to predict per-gaussian texture.
        if cfg.gaussian.texture_enabled:
            # Convert num_ch_enc from numpy array to list for type checking.
            num_ch_enc = self.gaussian_predictor.models[
                "unidepth_extended"
            ].encoder.num_ch_enc
            num_ch_enc_list = (
                num_ch_enc.tolist()
                if hasattr(num_ch_enc, "tolist")
                else list(num_ch_enc)
            )

            self.gaussian_param_head = GSHeadFlash3D(
                num_ch_enc=num_ch_enc_list,
                texture_size=cfg.gaussian.texture_size,
                texture_enabled=cfg.gaussian.texture_enabled,
                texture_project_enabled=cfg.gaussian.texture_project_enabled,
                texture_alpha_mode=cfg.gaussian.texture_alpha_mode,
                texture_color_sigma=cfg.gaussian.texture_color_sigma,
                texture_project_as_base=cfg.gaussian.texture_project_as_base,
            )

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> tuple[Gaussians, dict[Any, Any]]:
        # Shallow copy to avoid modifying the caller's dict.
        context = context.copy()

        original_context_ims = context["image"]
        b, v, _, raw_h, raw_w = context["image"].shape

        # Downsample context images to reduce the number of Gaussians.
        if self.texture_enabled:
            h = raw_h // self.texture_size
            w = raw_w // self.texture_size

            downsampled_context_ims = F.interpolate(
                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,  # To match Lanczos in dataloader
            )
            downsampled_context_ims = rearrange(
                downsampled_context_ims,
                "(b v) c h w -> b v c h w",
                b=b,
                v=v,
            )
            context["image"] = downsampled_context_ims
        else:
            h = raw_h
            w = raw_w

        inputs = context.copy()

        flash3d_inputs: dict[Any, Any] = {}
        flash3d_inputs[("frame_id", 0)] = "dummy_name"
        frame_name = 0

        input_index = 0
        image = inputs["image"][:, input_index]
        intrinsic = inputs["intrinsics"][:, input_index]
        pose = inputs["poses"][:, input_index]
        im_height, im_width = image.shape[-2:]

        intrinsic = normalized_K_to_K(intrinsic, im_width, im_height)
        flash3d_inputs[("color", frame_name, 0)] = inputs["image"][
            :, input_index
        ]
        if self.cfg.dataset.pad_border_aug > 0:
            pad_border_fn = tvt.Pad(
                (
                    self.cfg.dataset.pad_border_aug,
                    self.cfg.dataset.pad_border_aug,
                )
            )
            color_aug = pad_border_fn(image)
            intrinsic_aug = intrinsic.clone()
            intrinsic_aug[:, 0, 2] += self.cfg.dataset.pad_border_aug
            intrinsic_aug[:, 1, 2] += self.cfg.dataset.pad_border_aug
        else:
            color_aug = image
            intrinsic_aug = intrinsic.clone()
        flash3d_inputs[("color_aug", frame_name, 0)] = color_aug

        flash3d_inputs[("K_src", frame_name)] = intrinsic_aug
        flash3d_inputs[("inv_K_src", frame_name)] = torch.linalg.inv(
            intrinsic_aug
        )

        outputs = self.gaussian_predictor(flash3d_inputs)
        self.gaussian_predictor.compute_gauss_means(flash3d_inputs, outputs)

        # Rearrange Flash3D outputs to (b, g, ...) format.
        gaussians_per_pixel = self.cfg.gaussians_per_pixel
        gaussian_scales = rearrange(
            outputs["gauss_scaling"],
            "(b n) c h w -> b (n h w) c",
            n=gaussians_per_pixel,
            c=3,
        )
        gaussian_xyzws = rearrange(
            outputs["gauss_rotation"],
            "(b n) c h w -> b (n h w) c",
            n=gaussians_per_pixel,
            c=4,
        )
        gaussian_opacities = rearrange(
            outputs["gauss_opacity"],
            "(b n) c h w -> b (n h w) c",
            n=gaussians_per_pixel,
            c=1,
        ).squeeze(-1)
        gaussian_feature_dc = rearrange(
            outputs["gauss_features_dc"],
            "(b n) c h w -> b (n h w) c 1",
            n=gaussians_per_pixel,
            c=3,
        )
        sh_degree = 3**self.cfg.max_sh_degree
        gaussian_feature_rest = rearrange(
            outputs["gauss_features_rest"],
            "(b n) (sh c) h w -> b (n h w) c sh",
            n=gaussians_per_pixel,
            sh=sh_degree,
            c=3,
        )
        gaussian_means = rearrange(
            outputs["gauss_means"][:, :3],
            "(b n) c hw -> b (n hw) c",
            n=gaussians_per_pixel,
            c=3,
        )
        gaussians = build_gaussians_flash3d(
            means=gaussian_means,
            scales=gaussian_scales,
            xyzws=gaussian_xyzws,
            opacities=gaussian_opacities,
            features_dc=gaussian_feature_dc,
            features_rest=gaussian_feature_rest,
        )

        if self.texture_enabled:
            gaussians_shape = outputs["gauss_scaling"].shape[-2:]
            # Should use the intrinsic of the original context images.
            original_height, original_width = original_context_ims.shape[-2:]
            original_intrinsic = normalized_K_to_K(
                inputs["intrinsics"][:, input_index],
                original_width,
                original_height,
            )
            gaussian_head_out = self.gaussian_param_head(
                outputs["encoder_features"],
                original_context_ims[:, input_index],
                original_intrinsic,
                pose,
                gaussians,
                gaussians_shape,
            )
            texture_alpha_mode = self.texture_alpha_mode
            texture_size = self.texture_size
            if texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
                per_gaussian_channel = 3
            else:
                per_gaussian_channel = 4
            texture_colors = rearrange(
                gaussian_head_out,
                "b (c t1 t2) h w -> b (h w) c t1 t2",
                c=per_gaussian_channel,
                t1=texture_size,
                t2=texture_size,
            )
            if texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
                texture_alphas = None
            else:
                texture_alphas = torch.sigmoid(texture_colors[:, :, -1])
                texture_colors = texture_colors[:, :, :-1]

            gaussians = Gaussians(
                means=gaussians.means,
                covariances=gaussians.covariances,
                harmonics=gaussians.harmonics,
                opacities=gaussians.opacities,
                scales=gaussians.scales,
                xyzws=gaussians.xyzws,
                texture_colors=texture_colors,
                texture_alphas=texture_alphas,
            )

        # Only return gauss_offset and gauss_scaling for loss supervision.
        encoder_info = {"gauss_scaling": outputs["gauss_scaling"]}
        if "gauss_offset" in outputs:
            encoder_info["gauss_offset"] = outputs["gauss_offset"]
        return gaussians, encoder_info
