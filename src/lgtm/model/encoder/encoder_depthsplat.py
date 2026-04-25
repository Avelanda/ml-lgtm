"""
DepthSplat encoder.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._wrapper import TextureAlphaMode
from jaxtyping import Float
from torch import Tensor, nn

from lgtm.dataset.data_types import BatchedExample
from lgtm.dataset.shims.patch_shim import apply_patch_shim
from lgtm.model.backbone.backbone_depthsplat import DPTHead, MultiViewUniMatch
from lgtm.model.encoder.encoder import Encoder
from lgtm.model.gaussian.gaussian_utils import (
    build_covariance,
    compute_sh_mask,
    rgbs_to_sh0_coefficients,
)
from lgtm.model.gaussian.gaussians import GaussianCfg, Gaussians
from lgtm.model.head.head_depthsplat import GSHeadDepthSplat
from lgtm.utils.camera_utils import normalized_K_to_K
from lgtm.utils.geometry_utils import (
    get_world_rays_depthsplat,
    sample_image_grid,
)


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    gaussian: GaussianCfg
    downscale_factor: int
    shim_patch_size: int

    # Mv_unimatch.
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # Depthsplat color branch.
    init_sh_input_img: bool
    gaussian_regressor_channels: int

    # Loss config.
    supervise_intermediate_depth: bool
    return_depth: bool

    # Only depth.
    train_depth_only: bool

    # Monodepth config.
    monodepth_vit_type: str

    # Multi-view matching.
    local_mv_match: int

    # Config required for lgtm.
    pretrained_checkpoint_path: str = ""
    pretrained_checkpoint_ignored_prefixes: list[str] | None = None
    freeze_depth_predictor: bool = False


def build_gaussians_depthsplat(
    poses: Float[Tensor, "batch views 4 4"],
    intrinsics: Float[Tensor, "batch views 3 3"],
    coordinates: Float[Tensor, "batch views r 2"],
    depths: Float[Tensor, "batch views r"],
    opacities: Float[Tensor, "batch views r"],
    raw_gaussians: Float[Tensor, "batch views r _"],
    sh_mask: Float[Tensor, "d_sh"],
    cfg: GaussianCfg,
    eps: float = 1e-8,
    input_images: Tensor | None = None,
) -> Gaussians:
    """
    Convert raw head outputs to Gaussians for DepthSplat.

    Args:
        poses: c2w 4x4 matrices [batch, views, 4, 4].
        intrinsics: Normalized intrinsic 3x3 matrices [batch, views, 3, 3].

    Inputs have per-view structure (b, v, r, ...) because means are
    computed from per-view ray-depth and covariances are rotated by
    per-view c2w. All spatial dims (views, pixels) are flattened into
    a single 'g' dim in the returned Gaussians.
    """
    d_sh = cfg.d_sh

    scales, xyzws, sh = raw_gaussians.split((3, 4, 3 * d_sh), dim=-1)

    scales = torch.clamp(
        F.softplus(scales - 4.0),
        min=cfg.gaussian_scale_min,
        max=cfg.gaussian_scale_max,
    )

    if input_images is None:
        raise ValueError("input_images must not be None")

    # Normalize the quaternion features to yield a valid quaternion (xyzw).
    xyzws = xyzws / (xyzws.norm(dim=-1, keepdim=True) + eps)

    sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
    sh = sh.broadcast_to(*opacities.shape, 3, d_sh) * sh_mask

    if input_images is not None:
        # input_images: (b, v, c, h, w) → (b, v, h*w, c)
        imgs = rearrange(input_images, "b v c h w -> b v (h w) c")
        sh[..., 0] = sh[..., 0] + rgbs_to_sh0_coefficients(imgs)

    # Create world-space covariance matrices.
    covariances = build_covariance(scales, xyzws)
    # Broadcast c2w rotation over spatial dim r: (b, v, 1, 3, 3)
    c2w_rotations = poses[:, :, None, :3, :3]
    covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

    # Compute Gaussian means from ray-depth.
    # Broadcast poses/intrinsics over spatial dim r: (b, v, 1, ...)
    origins, directions = get_world_rays_depthsplat(
        coordinates,
        poses[:, :, None],
        intrinsics[:, :, None],
    )
    means = origins + directions * depths[..., None]

    # Flatten (views, r) → g.
    return Gaussians(
        means=means.flatten(1, 2),
        covariances=covariances.flatten(1, 2),
        harmonics=sh.flatten(1, 2),
        opacities=opacities.flatten(1, 2),
        # NOTE: These aren't yet rotated into world space, but they're only.
        # Used for exporting Gaussians to ply files. This needs fixing.
        scales=scales.flatten(1, 2),
        xyzws=xyzws.broadcast_to(*scales.shape[:-1], 4).flatten(1, 2),
    )


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # Upsample features to the original resolution.
        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 96,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 128,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.feature_upsampler = DPTHead(
            **model_configs[cfg.monodepth_vit_type],
            downsample_factor=cfg.upsample_factor,
            return_feature=True,
            num_scales=cfg.num_scales,
        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type][
            "features"
        ]

        # Pre-compute Gaussian config-derived values.
        self.gaussian_cfg = cfg.gaussian
        self.texture_enabled = cfg.gaussian.texture_enabled
        self.texture_size = (
            cfg.gaussian.texture_size if cfg.gaussian.texture_enabled else 1
        )

        # SH mask (non-persistent buffer, not saved in checkpoints)
        self.register_buffer(
            "sh_mask",
            compute_sh_mask(cfg.gaussian.sh_degree),
            persistent=False,
        )

        # Concat(img, depth, match_prob, features).
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # Conv regressor.
        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # Predict 2DGS Gaussian parameters: scale, q, sh, offset, opacity minus.
        # The texture parameters because we predict them in gaussian_param_head.
        num_gaussian_parameters = (
            cfg.gaussian.d_in + 2 + 1 - cfg.gaussian.d_texture
        )

        # Concat(img, features, regressor_out, match_prob).
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_gaussian_parameters,
                3,
                1,
                1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                num_gaussian_parameters,
                num_gaussian_parameters,
                3,
                1,
                1,
                padding_mode="replicate",
            ),
        )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # Init scale.
        # First 3: opacity, offset_xy.
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

        # The texture branch to predict per-gaussian texture.
        if cfg.gaussian.texture_enabled:
            decoder_feature_dim = model_configs[cfg.monodepth_vit_type][
                "features"
            ]
            decoder = DPTHead(
                **model_configs[cfg.monodepth_vit_type],
                downsample_factor=cfg.upsample_factor,
                return_feature=True,
                num_scales=cfg.num_scales,
            )
            self.gaussian_param_head = GSHeadDepthSplat(
                decoder=decoder,
                decoder_feature_dim=decoder_feature_dim,
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
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
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

        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["poses"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[
                    :, :, : (self.cfg.local_mv_match + 1)
                ]
        else:
            cameras_dist_index = None

        # Depth prediction.
        if self.cfg.freeze_depth_predictor:
            with torch.no_grad():
                results_dict = self.depth_predictor(
                    context["image"],
                    attn_splits_list=[2],
                    min_depth=1.0 / context["far"],
                    max_depth=1.0 / context["near"],
                    intrinsics=context["intrinsics"],
                    poses=context["poses"],
                    nn_matrix=cameras_dist_index,
                )
        else:
            results_dict = self.depth_predictor(
                context["image"],
                attn_splits_list=[2],
                min_depth=1.0 / context["far"],
                max_depth=1.0 / context["near"],
                intrinsics=context["intrinsics"],
                poses=context["poses"],
                nn_matrix=cameras_dist_index,
            )

        # List of [B, V, H, W], with all the intermediate depths.
        depth_preds = results_dict["depth_preds"]

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # Convert format.
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[: (num_depths - 1)], dim=0
                )
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()"
                )

                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            depths = (
                rearrange(depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
                .squeeze(-1)
                .squeeze(-1)
            )

            return {"gaussians": None, "depths": depths}

        # Features [BV, C, H, W].
        features = self.feature_upsampler(
            results_dict["features_mono_intermediate"],
            cnn_features=results_dict["features_cnn_all_scales"][::-1],
            mv_features=(
                results_dict["features_mv"][0]
                if self.cfg.num_scales == 1
                else results_dict["features_mv"][::-1]
            ),
        )

        # Match prob from softmax.
        # [BV, D, H, W] in feature resolution.
        match_prob = results_dict["match_probs"][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0
        ]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode="nearest"
        )

        concat = torch.cat(
            (
                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                rearrange(depth, "b v h w -> (b v) () h w"),
                match_prob,
                features,
            ),
            dim=1,
        )

        out = self.gaussian_regressor(concat)

        concat = [
            out,
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            features,
            match_prob,
        ]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w)")

        # Densities from match_prob (not used in Gaussian construction).
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w)", b=b, v=v
        )
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            num_depths = len(depth_preds)
            intermediate_depths = torch.cat(
                depth_preds[: (num_depths - 1)], dim=0
            )
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w)"
            )
            depths = torch.cat((intermediate_depths, depths), dim=0)
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat([raw_gaussians] * num_depths, dim=0)
            b *= num_depths

        # Extract opacity from first channel.
        opacities = raw_gaussians[..., 0].sigmoid()  # (b, v, h*w)
        raw_gaussians = raw_gaussians[..., 1:]

        # Coordinate offset handling.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) xy")  # (h*w, 2)
        offset_xy = raw_gaussians[..., :2].sigmoid()  # (b, v, h*w, 2)
        pixel_size = 1 / torch.tensor(
            (w, h), dtype=torch.float32, device=device
        )
        coordinates = xy_ray + (offset_xy - 0.5) * pixel_size
        raw_gs = raw_gaussians[..., 2:]

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_poses = torch.cat(
                [context["poses"]] * len(depth_preds), dim=0
            )
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0
            )
            gaussians = build_gaussians_depthsplat(
                context_poses,
                context_intrinsics,
                coordinates,
                depths,
                opacities,
                raw_gs,
                sh_mask=self.sh_mask,
                cfg=self.gaussian_cfg,
                input_images=(
                    sh_input_images.repeat(len(depth_preds), 1, 1, 1, 1)
                    if self.cfg.init_sh_input_img
                    else None
                ),
            )
        else:
            gaussians = build_gaussians_depthsplat(
                context["poses"],
                context["intrinsics"],
                coordinates,
                depths,
                opacities,
                raw_gs,
                sh_mask=self.sh_mask,
                cfg=self.gaussian_cfg,
                input_images=(
                    sh_input_images if self.cfg.init_sh_input_img else None
                ),
            )

        if visualization_dump is not None:
            visualization_dump["depth"] = (
                rearrange(depths, "b v (h w) -> b v h w", h=h, w=w)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            visualization_dump["scales"] = gaussians.scales
            visualization_dump["xyzws"] = gaussians.xyzws

        texture_colors = None
        texture_alphas = None
        if self.texture_enabled:
            # Should use the intrinsic of the original context images.
            original_height, original_width = original_context_ims.shape[-2:]
            original_intrinsic = normalized_K_to_K(
                context["intrinsics"],
                original_width,
                original_height,
            )
            gaussians_shape = (
                original_height // self.texture_size,
                original_width // self.texture_size,
            )
            # Un-flatten Gaussians to per-view for texture head.
            v_ctx = context["poses"].shape[1]
            perv_means = rearrange(
                gaussians.means, "b (v r) xyz -> b v r xyz", v=v_ctx
            )
            perv_scales = rearrange(
                gaussians.scales, "b (v r) xyz -> b v r xyz", v=v_ctx
            )
            perv_xyzws = rearrange(
                gaussians.xyzws, "b (v r) xyzw -> b v r xyzw", v=v_ctx
            )
            gaussian_head_out = self.gaussian_param_head(
                [],
                original_context_ims,
                intrinsics=original_intrinsic,
                poses=context["poses"],
                gaussians_means=perv_means,
                gaussians_scales=perv_scales,
                gaussians_xyzws=perv_xyzws,
                gaussians_shape=gaussians_shape,
                unimatch_return_dict=results_dict,
            )
            texture_alpha_mode = self.gaussian_cfg.texture_alpha_mode
            texture_size = self.texture_size
            if texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
                per_gaussian_channel = 3
            else:
                per_gaussian_channel = 4
            texture_colors = rearrange(
                gaussian_head_out,
                "b v (c t1 t2) h w -> b (v h w) c t1 t2",
                c=per_gaussian_channel,
                t1=texture_size,
                t2=texture_size,
            )
            if texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
                texture_alphas = None
            else:
                texture_alphas = torch.sigmoid(texture_colors[:, :, -1])
                texture_colors = texture_colors[:, :, :-1]

        if self.texture_enabled:
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

        if self.cfg.return_depth:
            depths = rearrange(depths, "b v (h w) -> b v h w", h=h, w=w)
            return {"gaussians": gaussians, "depths": depths}

        return gaussians, {}

    def get_data_shim(self):
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
