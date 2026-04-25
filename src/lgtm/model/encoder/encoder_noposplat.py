"""
NoPoSplat encoder.

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
from torch import Tensor, nn

from lgtm.dataset.shims.normalize_shim import normalize_image
from lgtm.model.backbone.backbone_noposplat import (
    BackboneCfg,
    get_backbone,
    transpose_to_landscape,
)
from lgtm.model.encoder.encoder import Encoder
from lgtm.model.gaussian.gaussian_utils import (
    activate_scales,
    activate_texture_alphas,
    build_covariance,
    compute_sh_mask,
)
from lgtm.model.gaussian.gaussians import GaussianCfg, Gaussians
from lgtm.model.head.head_noposplat import GSHeadNoPoSplat, Pts3dHead

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat"]
    backbone: BackboneCfg
    gaussian: GaussianCfg
    opacity_mapping: OpacityMappingCfg
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_checkpoint_path: str = ""
    pretrained_checkpoint_ignored_prefixes: list[str] | None = None


def build_gaussians_noposplat(
    means: Float[Tensor, "*#batch 3"],
    opacities: Float[Tensor, "*#batch"],
    raw_gaussians: Float[Tensor, "*#batch _"],
    sh_mask: Float[Tensor, "d_sh"],
    cfg: GaussianCfg,
    eps: float = 1e-8,
) -> Gaussians:
    """
    Convert raw head outputs to Gaussians for NoPoSplat.

    All inputs should share the same batch dimensions (typically (b, g) after
    the encoder flattens views/pixels/surfaces into a single g dim). Means are
    provided directly from pts3d prediction. Scales use softplus activation.
    """
    d_sh = cfg.d_sh
    d_texture = cfg.d_texture

    if cfg.texture_enabled:
        scales, xyzws, sh, texture_features = raw_gaussians.split(
            (3, 4, 3 * d_sh, d_texture), dim=-1
        )

        texture_size = cfg.texture_size

        if cfg.texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
            texture_features = rearrange(
                texture_features,
                "... (c h w) -> ... c h w",
                c=3,
                h=texture_size,
                w=texture_size,
            )
            texture_colors = texture_features
            texture_alphas = None
        else:
            texture_features = rearrange(
                texture_features,
                "... (c h w) -> ... c h w",
                c=(3 + 1),
                h=texture_size,
                w=texture_size,
            )
            texture_colors = texture_features[..., :3, :, :]
            texture_alphas = texture_features[..., 3, :, :]
            texture_alphas = activate_texture_alphas(texture_alphas)
    else:
        scales, xyzws, sh = raw_gaussians.split((3, 4, 3 * d_sh), dim=-1)
        texture_colors = None
        texture_alphas = None

    # NOTE: NoPoSplat and DepthSplat use different scale activations. The scale.
    # Clamp values are not directly comparable between encoders.
    scales = activate_scales(scales)
    scales = scales.clamp(
        min=cfg.gaussian_scale_min,
        max=cfg.gaussian_scale_max,
    )

    # Normalize the quaternion features to yield a valid quaternion (xyzw).
    xyzws = xyzws / (xyzws.norm(dim=-1, keepdim=True) + eps)

    sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
    sh = sh.broadcast_to((*opacities.shape, 3, d_sh)) * sh_mask

    covariances = build_covariance(scales, xyzws)

    return Gaussians(
        means=means,
        covariances=covariances,
        harmonics=sh,
        opacities=opacities,
        scales=scales,
        xyzws=xyzws.broadcast_to((*scales.shape[:-1], 4)),
        texture_colors=texture_colors,
        texture_alphas=texture_alphas,
    )


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        # Pre-compute Gaussian config-derived values.
        self.gaussian_cfg = cfg.gaussian
        self.texture_enabled = cfg.gaussian.texture_enabled
        self.texture_size = (
            cfg.gaussian.texture_size if cfg.gaussian.texture_enabled else 1
        )
        self.raw_gs_dim = 1 + cfg.gaussian.d_in  # 1 for opacity.

        # SH mask (non-persistent buffer, not saved in checkpoints).
        self.register_buffer(
            "sh_mask",
            compute_sh_mask(cfg.gaussian.sh_degree),
            persistent=False,
        )

        self.patch_size = self.backbone.patch_embed.patch_size[0]

        # Pts3d center heads.
        self.downstream_head1 = Pts3dHead(
            self.backbone,
            has_conf=False,
            depth_mode=("exp", -inf, inf),
            conf_mode=None,
        )
        self.downstream_head2 = Pts3dHead(
            self.backbone,
            has_conf=False,
            depth_mode=("exp", -inf, inf),
            conf_mode=None,
        )
        self.head1 = transpose_to_landscape(
            self.downstream_head1, activate=True
        )
        self.head2 = transpose_to_landscape(
            self.downstream_head2, activate=True
        )

        # GS parameter heads.
        texture_project_enabled = cfg.gaussian.texture_project_enabled

        self.gaussian_param_head = GSHeadNoPoSplat(
            self.backbone,
            out_nchan=self.raw_gs_dim,
            texture_size=self.texture_size,
            texture_enabled=cfg.gaussian.texture_enabled,
            texture_project_enabled=texture_project_enabled,
            texture_project_as_base=cfg.gaussian.texture_project_as_base,
            texture_alpha_mode=cfg.gaussian.texture_alpha_mode,
            texture_color_sigma=cfg.gaussian.texture_color_sigma,
        )
        self.gaussian_param_head2 = GSHeadNoPoSplat(
            self.backbone,
            out_nchan=self.raw_gs_dim,
            texture_size=self.texture_size,
            texture_enabled=cfg.gaussian.texture_enabled,
            texture_project_enabled=texture_project_enabled,
            texture_project_as_base=cfg.gaussian.texture_project_as_base,
            texture_alpha_mode=cfg.gaussian.texture_alpha_mode,
            texture_color_sigma=cfg.gaussian.texture_color_sigma,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        """
        When initial == 0.0 and final == 0.0, this is an identity mapping.
        """
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (
            cfg.final - cfg.initial
        )
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> tuple[Gaussians, dict[Any, Any]]:
        # Shallow copy to avoid modifying the caller's dict.
        context = context.copy()

        original_context_ims = context["image"]

        # Normalize [0, 1] → [-1, 1] for CroCo (writes to context dict directly).
        context["image"] = normalize_image(
            context["image"],
            mean=self.cfg.input_mean,
            std=self.cfg.input_std,
        )

        # Preserve high-res normalized images for the GS param head.
        original_normalized_context_ims = context["image"]

        device = context["image"].device
        b, v, _, raw_h, raw_w = context["image"].shape

        # Downsample context images to reduce the number of Gaussians.
        if self.texture_enabled:
            h = raw_h // self.texture_size
            w = raw_w // self.texture_size

            downsampled_normalized_context_ims = F.interpolate(
                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,  # To match Lanczos in dataloader
            )
            downsampled_normalized_context_ims = rearrange(
                downsampled_normalized_context_ims,
                "(b v) c h w -> b v c h w",
                b=b,
                v=v,
            )

            context["image"] = downsampled_normalized_context_ims
        else:
            h = raw_h
            w = raw_w

        # Backbone: encode context images → decoder tokens for each view.
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(
            context, return_views=True
        )

        res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        gs_head_out1 = self.gaussian_param_head(
            x=[tok.float() for tok in dec1],
            depths=res1["pts3d"].permute(0, 3, 1, 2),
            imgs=original_normalized_context_ims[:, 0, :, :, :],
            img_info=shape1[0].cpu().tolist(),
            intrinsics=context["intrinsics"][:, 0],
            poses=context["poses"][:, 0],
            global_step=global_step,
        )
        gs_head_out2 = self.gaussian_param_head2(
            x=[tok.float() for tok in dec2],
            depths=res2["pts3d"].permute(0, 3, 1, 2),
            imgs=original_normalized_context_ims[:, 1, :, :, :],
            img_info=shape2[0].cpu().tolist(),
            intrinsics=context["intrinsics"][:, 1],
            poses=context["poses"][:, 1],
            global_step=global_step,
        )

        GS_res1 = rearrange(gs_head_out1, "b d h w -> b (h w) d")
        GS_res2 = rearrange(gs_head_out2, "b d h w -> b (h w) d")

        pts3d1 = res1["pts3d"]
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2["pts3d"]
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)  # (b, v, h*w, 3)

        raw_gaussians = torch.stack([GS_res1, GS_res2], dim=1)  # (b, v, h*w, D)
        densities = raw_gaussians[..., 0].sigmoid()  # (b, v, h*w)
        opacities = self.map_pdf_to_opacity(densities, global_step)
        raw_gs = raw_gaussians[..., 1:]  # (b, v, h*w, c_gs)

        means = rearrange(pts_all, "b v r xyz -> b (v r) xyz")
        opacities = rearrange(opacities, "b v r -> b (v r)")
        raw_gs = rearrange(raw_gs, "b v r c -> b (v r) c")

        gaussians = build_gaussians_noposplat(
            means=means,
            opacities=opacities,
            raw_gaussians=raw_gs,
            sh_mask=self.sh_mask,
            cfg=self.gaussian_cfg,
        )

        return gaussians, {}
