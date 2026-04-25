"""
Flash3D Gaussian prediction head.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._wrapper import TextureAlphaMode

from lgtm.model.backbone.backbone_flash3d import (
    ConvBlock,
    upsample,
)
from lgtm.model.gaussian.gaussians import Gaussians
from lgtm.model.head.head_common import (
    compute_texture_dims,
    init_texture_modules,
)
from lgtm.utils.texture_project import project_texture_colors


class ResnetDecoder(nn.Module):
    """
    Pytorch module for a resnet decoder.
    """

    def __init__(self, num_ch_enc, dim_out):
        super().__init__()
        self.use_skips = True
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [32, 32, 64, 128, 256]

        self.num_output_channels = dim_out

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # Upconv_0
            num_ch_in = (
                self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # Upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.out = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, 1)

    def forward(self, input_features):
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode="nearest")]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)

        x = self.out(x)
        return x


class GSHeadFlash3D(nn.Module):
    """
    Flash3D head specified for Gaussian Splatting parameter generation.

        Input:
        ├── encoder_features: List[torch.Tensor] (multi-level ResNet features)
        └── imgs: torch.Tensor (high-res RGB images)

        Processing:
        1. Image Downsampling:
        └── imgs → downsampled_imgs (bilinear interpolation by texture_size)

        2. Feature Fusion:
        ├── encoder_features → ResNet Decoder → path_1
        ├── downsampled_imgs → input_merger → feat_downsampled_imgs
        └── path_1 (cropped) + feat_downsampled_imgs → feat_2dgs

        3. Texture Generation (if texture_enabled):
        ├── imgs → patchify_block → feat_imgs
        ├── feat_imgs + feat_2dgs → conv_block → feat_texture
        └── feat_texture → texture_head → out_texture

        Output:
        └── out_texture: torch.Tensor (gaussian texture parameters)

        Key Components:
        - ResNet Decoder: Fuses multi-level encoder features
        - Input Merger: Processes downsampled RGB images
        - Patchify Block: ViT-like patchification of high-res images
        - Conv Block: Feature fusion (add or concat modes)
        - Texture Head: Final 1x1 conv for texture parameter generation
    """

    def __init__(
        self,
        num_ch_enc: List[int],
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

        self.resnet_decoder = ResnetDecoder(num_ch_enc=num_ch_enc, dim_out=256)

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
        gaussians: Gaussians,
        gaussians_shape: tuple[int, int],
    ):
        """
        Args:
            encoder_features: List[torch.Tensor], features from
                Flash3D ResNet encoder.
            imgs: torch.Tensor, high-res RGB images.
            intrinsics: Pixel-scale intrinsic 3x3 matrix
                [batch, 3, 3].
            poses: c2w 4x4 matrix
                [batch, 4, 4].
        """
        # Downsample imgs for shape compatibility with the 2DGS branch.
        downsampled_imgs = F.interpolate(
            imgs,
            size=(
                imgs.shape[-2] // self.texture_size,
                imgs.shape[-1] // self.texture_size,
            ),
            mode="bilinear",
            align_corners=False,
            antialias=True,  # To match Lanczos in dataloader.
        )
        feat_downsampled_imgs = self.input_merger(downsampled_imgs)

        # Instead of using DPT to fuse features, we reuse the ResNet decoder.
        # In Flash3D to fuse the encoder features.
        path_1 = self.resnet_decoder(encoder_features)
        if path_1.shape[-2:] != feat_downsampled_imgs.shape[-2:]:
            raise ValueError(
                "Shape mismatch: do you turn on Flash3D's padding?"
            )
        feat_2dgs = path_1 + feat_downsampled_imgs

        if self.texture_project_enabled:
            _, _, h_high, w_high = imgs.shape
            H, W = gaussians_shape
            Ts = torch.linalg.inv(poses)  # c2w to w2c

            # Texture projection (projected result is used as network input).
            texture_colors_projected = project_texture_colors(
                means=gaussians.means,  # (B, H*W, 3)
                scales=gaussians.scales,  # (B, H*W, 3)
                xyzws=gaussians.xyzws,  # (B, H*W, 4)
                ims=imgs,  # (B, 3, h_high, w_high)
                Ks=intrinsics,  # (B, 3, 3) pixel-scale intrinsics
                Ts=Ts,  # (B, 4, 4) w2c
                texture_size=self.texture_size,
                texture_color_sigma=self.texture_color_sigma,
            ).detach()

            # Reshape for conv: flatten texture to channels.
            # (B, 3*tex_size^2, H, W)
            texture_colors_projected = rearrange(
                texture_colors_projected,
                "b (h w) c th tw -> b (c th tw) h w",
                h=H,
                w=W,
            )

            feat_textures = self.direct_2dgs_baked_texture_processor(
                texture_colors_projected=texture_colors_projected,
                imgs=imgs,
                h_high=h_high,
                w_high=w_high,
            )
            # (b, 3, h_high, w_high) → (b, 256, h_low, w_low)
            feat_imgs = self.direct_2dgs_imgs_patchify(imgs)

            # 3x (b, 256, h_low, w_low) -> (b, 768, h_low, w_low)
            feat_fused = torch.cat([feat_imgs, feat_2dgs, feat_textures], dim=1)

            feat_texture = self.direct_2dgs_conv_block(feat_fused)

            # (b, d_texture_colors, h_low, w_low)
            out_texture_colors = self.direct_2dgs_texture_head(feat_texture)

            # Use projected colors as base and learn delta (if enabled).
            if self.texture_project_as_base:
                if self.texture_alpha_mode == TextureAlphaMode.TEXTURE:
                    batch_size, _, height, width = (
                        texture_colors_projected.shape
                    )
                    channel = (
                        out_texture_colors.shape[1]
                        - texture_colors_projected.shape[1]
                    )
                    device = texture_colors_projected.device
                    texture_colors_projected = torch.cat(
                        [
                            texture_colors_projected,
                            torch.zeros(batch_size, channel, height, width).to(
                                device
                            ),
                        ],
                        dim=1,
                    )
                out_texture_colors = (
                    texture_colors_projected + out_texture_colors
                )

            out_bbsplat = out_texture_colors
        else:
            feat_imgs = self.direct_2dgs_imgs_patchify(imgs)

            feat_texture = self.direct_2dgs_conv_block(feat_imgs + feat_2dgs)

            # 1x1 conv: (1, 256, H_low, W_low) -> (1, d_texture, H_low, W_low).
            out_texture = self.direct_2dgs_texture_head(feat_texture)

            out_bbsplat = out_texture

        return out_bbsplat
