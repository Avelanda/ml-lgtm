"""
NoPoSplat Gaussian prediction head.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._wrapper import TextureAlphaMode

from lgtm.model.gaussian.gaussian_utils import activate_scales
from lgtm.model.head.head_common import (
    compute_texture_dims,
    init_texture_modules,
)
from lgtm.utils.camera_utils import normalized_K_to_K
from lgtm.utils.texture_project import project_texture_colors


class Pts3dHead(nn.Module):
    """
    DPT-based pts3d center prediction head for NoPoSplat.

    Predicts 3D point positions (and optionally confidence) from encoder tokens.
    Takes a CroCo backbone to extract architecture dimensions.
    """

    def __init__(
        self,
        backbone,
        has_conf: bool = False,
        out_nchan: int = 3,
        depth_mode: tuple = ("exp", -float("inf"), float("inf")),
        conf_mode: tuple | None = None,
    ):
        super().__init__()
        self.return_all_layers = True
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        if backbone.dec_depth <= 9:
            raise ValueError(
                f"backbone.dec_depth must be > 9, got {backbone.dec_depth}"
            )
        l2 = backbone.dec_depth
        ed = backbone.enc_embed_dim
        dd = backbone.dec_embed_dim

        last_dim = FEATURE_DIM // 2  # 128
        self.hooks = [0, l2 * 2 // 4, l2 * 3 // 4, l2]
        dim_tokens = [ed, dd, dd, dd]
        num_channels = out_nchan + has_conf

        # All DPT modules under self.dpt for checkpoint compatibility.
        # (state_dict keys: downstream_head1.dpt.scratch.*, etc.)
        self.dpt = nn.Module()
        self.dpt.scratch = _build_scratch_and_refinenets()

        # Regression head (pts3d prediction)
        self.dpt.head = nn.Sequential(
            nn.Conv2d(FEATURE_DIM, FEATURE_DIM // 2, 3, 1, 1),
            _Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(FEATURE_DIM // 2, last_dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(last_dim, num_channels, 1, 1, 0),
        )

        self.dpt.act_postprocess = _build_act_postprocess(dim_tokens)

    def forward(self, x, img_info, ray_embedding=None):
        image_size = (img_info[0], img_info[1])
        path_1 = _dpt_fuse(
            x,
            self.hooks,
            self.dpt.act_postprocess,
            self.dpt.scratch,
            image_size,
        )
        out = self.dpt.head(path_1)
        return _postprocess(out, self.depth_mode, self.conf_mode)


class GSHeadNoPoSplat(nn.Module):
    """
    DPT-based GS parameter prediction head for NoPoSplat.

    Predicts Gaussian splatting parameters from encoder tokens, with optional
    texture support. Takes a CroCo backbone to extract architecture dimensions.
    """

    def __init__(
        self,
        backbone,
        out_nchan: int,
        texture_size: int = 1,
        texture_enabled: bool = False,
        texture_project_enabled: bool = False,
        texture_alpha_mode: TextureAlphaMode = TextureAlphaMode.TEXTURE,
        texture_color_sigma: float = 1.0,
        texture_project_as_base: bool = False,
    ):
        super().__init__()
        self.return_all_layers = True

        if backbone.dec_depth <= 9:
            raise ValueError(
                f"backbone.dec_depth must be > 9, got {backbone.dec_depth}"
            )
        l2 = backbone.dec_depth
        ed = backbone.enc_embed_dim
        dd = backbone.dec_embed_dim

        self.hooks = [0, l2 * 2 // 4, l2 * 3 // 4, l2]
        dim_tokens = [ed, dd, dd, dd]

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

        # Subtract texture channels (predicted separately).
        if texture_enabled:
            num_channels = out_nchan - d_texture
        else:
            num_channels = out_nchan

        # All DPT modules under self.dpt for checkpoint compatibility.
        # (state_dict keys: gaussian_param_head.dpt.scratch.*, etc.)
        self.dpt = nn.Module()
        self.dpt.scratch = _build_scratch_and_refinenets()

        # GS params head (semseg-style)
        self.dpt.head = nn.Sequential(
            nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 3, padding=1, bias=False),
            nn.Identity(),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(FEATURE_DIM, num_channels, 1),
        )

        self.dpt.act_postprocess = _build_act_postprocess(dim_tokens)

        self.dpt.feat_up = _Interpolate(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.dpt.input_merger = nn.Sequential(
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )

        if texture_enabled:
            init_texture_modules(
                self.dpt,
                texture_size=texture_size,
                d_texture=d_texture,
                d_texture_colors_only=d_texture_colors_only,
                texture_project_enabled=texture_project_enabled,
            )

    def forward(
        self,
        x,
        depths,
        imgs,
        img_info,
        conf=None,
        intrinsics=None,
        poses=None,  # c2w 4x4 matrix [batch, 4, 4]
        global_step=0,
    ):
        image_size = (img_info[0], img_info[1])
        if self.texture_enabled:
            return self._forward_texture(
                x=x,
                depths=depths,
                imgs=imgs,
                image_size=image_size,
                conf=conf,
                intrinsics=intrinsics,
                poses=poses,
                global_step=global_step,
            )
        else:
            return self._forward_default(
                x=x,
                depths=depths,
                imgs=imgs,
                image_size=image_size,
                conf=conf,
            )

    def _forward_default(
        self,
        x,
        depths,
        imgs,
        image_size,
        conf=None,
    ):
        path_1 = _dpt_fuse(
            x,
            self.hooks,
            self.dpt.act_postprocess,
            self.dpt.scratch,
            image_size,
        )

        path_1 = self.dpt.feat_up(path_1)

        direct_img_feat = self.dpt.input_merger(imgs)
        path_1 = path_1 + direct_img_feat

        out = self.dpt.head(path_1)

        return out

    def _forward_texture(
        self,
        x,
        depths,
        imgs,
        image_size,
        conf=None,
        intrinsics=None,
        poses=None,  # c2w 4x4 matrix [batch, 4, 4]
        global_step=0,
    ):
        path_1 = _dpt_fuse(
            x,
            self.hooks,
            self.dpt.act_postprocess,
            self.dpt.scratch,
            image_size,
        )

        # Downsample imgs for shape compatibility with the DPT output.
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
        feat_downsampled_imgs = self.dpt.input_merger(downsampled_imgs)

        path_1 = self.dpt.feat_up(path_1)
        feat_2dgs = path_1 + feat_downsampled_imgs

        out_2dgs = self.dpt.head(feat_2dgs)

        H, W = image_size

        if self.texture_project_enabled:
            B, C, H_out, W_out = out_2dgs.shape
            _, _, h_high, w_high = imgs.shape

            # Extract and activate scales/xyzws for projection.
            # Out_2dgs channels: [opacity(1), scales(3), xyzws(4), sh(3*d_sh)]
            opacity = out_2dgs[:, 0:1]  # (B, 1, H, W)
            raw_scales = out_2dgs[:, 1:4]  # (B, 3, H, W)
            raw_xyzws = out_2dgs[:, 4:8]  # (B, 4, H, W)
            sh_coeffs = out_2dgs[:, 8:]  # (B, 3*d_sh, H, W)

            raw_scales = rearrange(raw_scales, "b c h w -> b (h w) c")
            raw_xyzws = rearrange(raw_xyzws, "b c h w -> b (h w) c")
            means = rearrange(depths, "b c h w -> b (h w) c")  # pts3d

            activated_scales = activate_scales(raw_scales)
            activated_scales = activated_scales.clamp_max(0.3)

            eps = 1e-8
            activated_xyzws = raw_xyzws / (
                raw_xyzws.norm(dim=-1, keepdim=True) + eps
            )

            # Texture projection uses pixel-scale Ks and w2c matrices.
            Ks = normalized_K_to_K(
                intrinsics,
                im_width=w_high,
                im_height=h_high,
            )
            Ts = torch.linalg.inv(poses)  # c2w to w2c

            # Texture projection (projected result is used as network input).
            texture_colors_projected = project_texture_colors(
                means=means,  # (B, H*W, 3)
                scales=activated_scales,  # (B, H*W, 3)
                xyzws=activated_xyzws,  # (B, H*W, 4)
                ims=imgs,  # (B, 3, h_high, w_high)
                Ks=Ks,  # (B, 3, 3) pixel-scale intrinsics
                Ts=Ts,  # (B, 4, 4) w2c
                texture_size=self.texture_size,
                texture_color_sigma=self.texture_color_sigma,
            )

            # Reshape for conv: flatten texture to channels.
            # (B, 3*tex_size^2, H, W)
            texture_colors_projected = rearrange(
                texture_colors_projected,
                "b (h w) c th tw -> b (c th tw) h w",
                h=H_out,
                w=W_out,
            )

            feat_textures = self.dpt.direct_2dgs_baked_texture_processor(
                texture_colors_projected=texture_colors_projected,
                imgs=imgs,
                h_high=h_high,
                w_high=w_high,
            )

            # (b, 3, h_high, w_high) -> (b, 256, h_low, w_low)
            feat_imgs = self.dpt.direct_2dgs_imgs_patchify(imgs)

            # 3x (b, 256, h_low, w_low) -> (b, 768, h_low, w_low)
            feat_fused = torch.cat([feat_imgs, feat_2dgs, feat_textures], dim=1)

            feat_texture = self.dpt.direct_2dgs_conv_block(feat_fused)

            # (b, d_texture_colors, h_low, w_low)
            out_texture_colors = self.dpt.direct_2dgs_texture_head(feat_texture)

            # Use projected colors as base and learn delta (if enabled).
            if self.texture_project_as_base:
                # [-1, 1] -> [0, 1]
                texture_colors_projected = (texture_colors_projected + 1) / 2

                # GAUSSIAN: colors only. TEXTURE/MULTIPLY: colors + alpha.
                if self.texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
                    out_texture_colors = (
                        texture_colors_projected + out_texture_colors
                    )
                elif (
                    self.texture_alpha_mode == TextureAlphaMode.TEXTURE
                    or self.texture_alpha_mode == TextureAlphaMode.MULTIPLY
                ):
                    # For TEXTURE/MULTIPLY modes, out_texture_colors has shape.
                    # (b, 4*th*tw, h, w) where first 3*th*tw channels are RGB.
                    # And last th*tw channels are alpha.
                    # Texture_colors_projected has shape (b, 3*th*tw, h, w).
                    rgb_channels = 3 * self.texture_size * self.texture_size
                    predicted_rgb = out_texture_colors[:, :rgb_channels, :, :]
                    predicted_alpha = out_texture_colors[:, rgb_channels:, :, :]
                    enhanced_rgb = predicted_rgb + texture_colors_projected
                    out_texture_colors = torch.cat(
                        [enhanced_rgb, predicted_alpha], dim=1
                    )
                else:
                    raise ValueError(
                        f"Invalid texture alpha mode: {self.texture_alpha_mode}"
                    )

            out_bbsplat = torch.cat([out_2dgs, out_texture_colors], dim=1)

        else:
            feat_imgs = self.dpt.direct_2dgs_imgs_patchify(imgs)

            feat_texture = self.dpt.direct_2dgs_conv_block(
                feat_imgs + feat_2dgs
            )

            # 1x1 conv: (b, 256, h_low, w_low) -> (b, d_texture, h_low, w_low).
            out_texture = self.dpt.direct_2dgs_texture_head(feat_texture)

            out_bbsplat = torch.cat([out_2dgs, out_texture], dim=1)

        return out_bbsplat


FEATURE_DIM = 256
LAYER_DIMS = [96, 192, 384, 768]
PATCH_SIZE = 16


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList(
        [
            scratch.layer1_rn,
            scratch.layer2_rn,
            scratch.layer3_rn,
            scratch.layer4_rn,
        ]
    )

    return scratch


class _ResidualConvUnit(nn.Module):
    """
    Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class _FeatureFusionBlock(nn.Module):
    """
    Feature fusion block.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        width_ratio=1,
    ):
        super(_FeatureFusionBlock, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = _ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = _ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(
                    res,
                    size=(output.shape[2], output.shape[3]),
                    mode="bilinear",
                )

            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output = F.interpolate(
                output, size=(2 * output.shape[2], shape), mode="bilinear"
            )
        else:
            output = nn.functional.interpolate(
                output,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn, width_ratio=1, expand=False):
    return _FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=expand,
        align_corners=True,
        width_ratio=width_ratio,
    )


class _Interpolate(nn.Module):
    """
    Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        super(_Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


def _postprocess(out, depth_mode, conf_mode):
    """
    Extract 3D points/confidence from prediction head output.
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=_reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res["conf"] = _reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def _reg_dense_depth(xyz, mode):
    """
    Extract 3D points from prediction head output.
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    # assert no_bounds

    if mode == "range":
        xyz = xyz.sigmoid()
        xyz = (1 - xyz) * vmin + xyz * vmax
        return xyz

    if mode == "linear":
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    if mode == "exp_direct":
        xyz = xyz.expm1()
        return xyz.clip(min=vmin, max=vmax)

    # Distance to origin.
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return xyz * d.square()

    if mode == "exp":
        exp_d = d.expm1()
        if not no_bounds:
            exp_d = exp_d.clip(min=vmin, max=vmax)
        xyz = xyz * exp_d
        return xyz

    raise ValueError(f"bad {mode=}")


def _reg_dense_conf(x, mode):
    """
    Extract confidence from prediction head output.
    """
    mode, vmin, vmax = mode
    if mode == "opacity":
        return x.sigmoid()
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")


def _build_act_postprocess(dim_tokens: List[int]):
    """
    Build activation postprocessing layers that project encoder tokens to the
    expected spatial feature dimensions for DPT fusion.
    """
    act_1 = nn.Sequential(
        nn.Conv2d(dim_tokens[0], LAYER_DIMS[0], 1, 1, 0),
        nn.ConvTranspose2d(LAYER_DIMS[0], LAYER_DIMS[0], 4, 4, 0, bias=True),
    )
    act_2 = nn.Sequential(
        nn.Conv2d(dim_tokens[1], LAYER_DIMS[1], 1, 1, 0),
        nn.ConvTranspose2d(LAYER_DIMS[1], LAYER_DIMS[1], 2, 2, 0, bias=True),
    )
    act_3 = nn.Sequential(
        nn.Conv2d(dim_tokens[2], LAYER_DIMS[2], 1, 1, 0),
    )
    act_4 = nn.Sequential(
        nn.Conv2d(dim_tokens[3], LAYER_DIMS[3], 1, 1, 0),
        nn.Conv2d(LAYER_DIMS[3], LAYER_DIMS[3], 3, 2, 1),
    )
    return nn.ModuleList([act_1, act_2, act_3, act_4])


def _build_scratch_and_refinenets():
    """
    Build the DPT scratch module with feature projection and refinement layers.
    """
    scratch = _make_scratch(LAYER_DIMS, FEATURE_DIM)
    scratch.refinenet1 = _make_fusion_block(FEATURE_DIM, use_bn=False)
    scratch.refinenet2 = _make_fusion_block(FEATURE_DIM, use_bn=False)
    scratch.refinenet3 = _make_fusion_block(FEATURE_DIM, use_bn=False)
    scratch.refinenet4 = _make_fusion_block(FEATURE_DIM, use_bn=False)
    return scratch


def _dpt_fuse(
    encoder_tokens: List[torch.Tensor],
    hooks: List[int],
    act_postprocess: nn.ModuleList,
    scratch: nn.Module,
    image_size: tuple[int, int],
):
    """
    Common DPT fusion pipeline: hook encoder layers, reshape to spatial, run
    activation postprocessing, project, and fuse with refinement.

    Returns path_1 (the final fused feature map before head-specific
    processing).
    """
    H, W = image_size
    N_H = H // PATCH_SIZE
    N_W = W // PATCH_SIZE

    layers = [encoder_tokens[hook] for hook in hooks]
    layers = [
        rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
    ]
    layers = [act_postprocess[idx](l) for idx, l in enumerate(layers)]
    layers = [scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

    path_4 = scratch.refinenet4(layers[3])[
        :, :, : layers[2].shape[2], : layers[2].shape[3]
    ]
    path_3 = scratch.refinenet3(path_4, layers[2])
    path_2 = scratch.refinenet2(path_3, layers[1])
    path_1 = scratch.refinenet1(path_2, layers[0])

    return path_1
