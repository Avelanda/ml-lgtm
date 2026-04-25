"""
DepthSplat Gaussian prediction head.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._wrapper import TextureAlphaMode

from lgtm.model.head.head_common import (
    compute_texture_dims,
    init_texture_modules,
)
from lgtm.utils.texture_project import project_texture_colors


class GSHeadDepthSplat(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        decoder_feature_dim=64,
        texture_size=1,
        texture_enabled=False,
        texture_project_enabled=False,
        texture_alpha_mode=TextureAlphaMode.TEXTURE,
        texture_color_sigma=1.0,
        texture_project_as_base=False,
    ):
        super().__init__()
        self.texture_enabled = texture_enabled
        self.texture_size = texture_size
        self.texture_project_enabled = texture_project_enabled
        self.texture_alpha_mode = texture_alpha_mode
        self.texture_color_sigma = texture_color_sigma
        self.texture_project_as_base = texture_project_as_base

        d_texture, d_texture_colors_only = compute_texture_dims(
            texture_size, texture_alpha_mode
        )
        self.d_texture = d_texture
        self.d_texture_colors_only = d_texture_colors_only

        self.input_merger = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )

        self.decoder = decoder
        self.decoder_out_conv = nn.Conv2d(
            decoder_feature_dim, 256, kernel_size=1
        )

        if texture_enabled:
            init_texture_modules(
                self,
                texture_size=texture_size,
                d_texture=d_texture,
                d_texture_colors_only=d_texture_colors_only,
                texture_project_enabled=texture_project_enabled,
            )

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        imgs: torch.Tensor,
        intrinsics: torch.Tensor,
        poses: torch.Tensor,
        gaussians_means: torch.Tensor,
        gaussians_scales: torch.Tensor,
        gaussians_xyzws: torch.Tensor,
        gaussians_shape: tuple[int, int],
        unimatch_return_dict: dict[str, Any] | None = None,
    ):
        """
        Args:
            encoder_features: List[torch.Tensor], features from
                the depth predictor.
            imgs: torch.Tensor, high-res RGB images [batch, views, 3, H, W].
            intrinsics: Pixel-scale intrinsic 3x3 matrices [batch, views, 3, 3].
            poses: c2w 4x4 matrices [batch, views, 4, 4].
        """
        device = imgs.device
        batch_size, num_views = imgs.shape[:2]
        flatten_imgs = rearrange(imgs, "b v c h w -> (b v) c h w")
        # Downsample imgs for shape compatibility with the 2DGS branch.
        downsampled_imgs = F.interpolate(
            flatten_imgs,
            size=(
                imgs.shape[-2] // self.texture_size,
                imgs.shape[-1] // self.texture_size,
            ),
            mode="bilinear",
            align_corners=False,
            antialias=True,  # To match Lanczos in dataloader.
        )
        feat_downsampled_imgs = self.input_merger(downsampled_imgs)

        path_1 = self.decoder(
            unimatch_return_dict["features_mono_intermediate"],
            cnn_features=unimatch_return_dict["features_cnn_all_scales"][::-1],
            mv_features=unimatch_return_dict["features_mv"][::-1],
        )
        path_1 = self.decoder_out_conv(path_1)
        feat_2dgs = path_1 + feat_downsampled_imgs

        if self.texture_project_enabled:
            h_high, w_high = imgs.shape[-2:]
            H, W = gaussians_shape
            Ts = torch.linalg.inv(poses)  # c2w to w2c

            # Texture projection (projected result is used as network input).
            texture_color_projected_list = []
            for i_view in range(imgs.shape[1]):
                texture_colors_projected = project_texture_colors(
                    means=gaussians_means[:, i_view],  # (B, H*W, 3)
                    scales=gaussians_scales[:, i_view],  # (B, H*W, 3)
                    xyzws=gaussians_xyzws[:, i_view],  # (B, H*W, 4)
                    ims=imgs[:, i_view],  # (B, 3, h_high, w_high)
                    Ks=intrinsics[
                        :, i_view
                    ],  # (B, 3, 3) pixel-scale intrinsics
                    Ts=Ts[:, i_view],  # (B, 4, 4) w2c
                    texture_size=self.texture_size,
                    texture_color_sigma=self.texture_color_sigma,
                )

                # Reshape for conv: flatten texture to channels.
                # (B, 3*tex_size^2, H, W)
                texture_colors_projected = rearrange(
                    texture_colors_projected,
                    "b (h w) c th tw -> b (c th tw) h w",
                    h=H,
                    w=W,
                )
                texture_color_projected_list.append(texture_colors_projected)

            texture_colors_projected = torch.cat(
                texture_color_projected_list, dim=0
            )  # (BV, 3*tex_size^2, H, W)

            feat_textures = self.direct_2dgs_baked_texture_processor(
                texture_colors_projected=texture_colors_projected,
                imgs=flatten_imgs,
                h_high=h_high,
                w_high=w_high,
            )
            # (b, 3, h_high, w_high) → (b, 256, h_low, w_low)
            feat_imgs = self.direct_2dgs_imgs_patchify(flatten_imgs)

            # 3x (b, 256, h_low, w_low) -> (b, 768, h_low, w_low)
            feat_fused = torch.cat([feat_imgs, feat_2dgs, feat_textures], dim=1)

            feat_texture = self.direct_2dgs_conv_block(feat_fused)

            # (b, d_texture_colors, h_low, w_low)
            out_texture_colors = self.direct_2dgs_texture_head(feat_texture)

            # Use projected colors as base and learn delta (if enabled).
            if self.texture_project_as_base:
                if self.texture_alpha_mode == TextureAlphaMode.TEXTURE:
                    bv, _, height, width = texture_colors_projected.shape
                    channel = (
                        out_texture_colors.shape[1]
                        - texture_colors_projected.shape[1]
                    )
                    device = texture_colors_projected.device
                    texture_colors_projected = torch.cat(
                        [
                            texture_colors_projected,
                            torch.zeros(bv, channel, height, width).to(device),
                        ],
                        dim=1,
                    )
                out_texture_colors = (
                    texture_colors_projected + out_texture_colors
                )

            out_bbsplat = out_texture_colors.view(
                batch_size, num_views, *out_texture_colors.shape[1:]
            )
        else:
            feat_imgs = self.direct_2dgs_imgs_patchify(flatten_imgs)

            feat_texture = self.direct_2dgs_conv_block(feat_imgs + feat_2dgs)

            # 1x1 conv: (1, 256, H_low, W_low) -> (1, d_texture, H_low, W_low).
            out_texture = self.direct_2dgs_texture_head(feat_texture)

            out_bbsplat = out_texture.view(
                batch_size, num_views, *out_texture.shape[1:]
            )

        return out_bbsplat
