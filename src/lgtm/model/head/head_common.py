"""
Shared head blocks for NoPoSplat, Flash3D, and DepthSplat.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch.nn as nn
from gsplat.cuda._wrapper import TextureAlphaMode


class PatchifyBlock(nn.Module):
    """
    Simple patchify network that converts high-res images to
    low-res feature maps.
    """

    def __init__(self, texture_size):
        super().__init__()
        self.texture_size = texture_size
        self.simplified_block = nn.Sequential(
            nn.Conv2d(
                3,
                256,
                kernel_size=texture_size,
                stride=texture_size,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.simplified_block(x)


class ProjectedTextureProcessor(nn.Module):
    """
    Processes projected texture colors into feature maps for texture prediction.
    """

    def __init__(
        self,
        d_texture_colors_only: int,
        texture_size: int,
    ):
        super().__init__()
        self.d_texture_colors_only = d_texture_colors_only
        self.texture_size = texture_size

        self.baked_texture_processor = nn.Sequential(
            nn.Conv2d(d_texture_colors_only, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Zero-init the final Conv2d layer so output is ~zero initially.
        # This ensures smooth transition when projection starts.
        final_layer = self.baked_texture_processor[-2]
        if isinstance(final_layer, nn.Conv2d):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(self, texture_colors_projected, imgs, h_high, w_high):
        return self.baked_texture_processor(texture_colors_projected)


def compute_texture_dims(
    texture_size: int, texture_alpha_mode: TextureAlphaMode
):
    """
    Compute texture channel dimensions based on texture_alpha_mode.

    Returns:
        d_texture: Total texture channels (colors + optional alphas)
        d_texture_colors_only: Texture channels for colors only (always RGB)
    """
    if texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
        d_texture = texture_size * texture_size * 3
    else:
        d_texture = texture_size * texture_size * (3 + 1)
    d_texture_colors_only = texture_size * texture_size * 3
    return d_texture, d_texture_colors_only


def init_texture_modules(
    module: nn.Module,
    texture_size: int,
    d_texture: int,
    d_texture_colors_only: int,
    texture_project_enabled: bool,
):
    """
    Initialize texture-related sub-modules on the given nn.Module.

    Sets the following attributes on the module:
    - direct_2dgs_imgs_patchify
    - direct_2dgs_conv_block
    - direct_2dgs_texture_head
    - direct_2dgs_baked_texture_processor (only if texture_project_enabled)
    """
    module.direct_2dgs_imgs_patchify = PatchifyBlock(texture_size)

    if texture_project_enabled:
        module.direct_2dgs_baked_texture_processor = ProjectedTextureProcessor(
            d_texture_colors_only=d_texture_colors_only,
            texture_size=texture_size,
        )
        module.direct_2dgs_conv_block = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    else:
        module.direct_2dgs_conv_block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    module.direct_2dgs_texture_head = nn.Conv2d(256, d_texture, kernel_size=1)
    _zero_init_texture_head(module.direct_2dgs_texture_head)


def _zero_init_texture_head(conv: nn.Conv2d):
    """
    Zero-initialize a Conv2d texture head for better training stability.
    """
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
