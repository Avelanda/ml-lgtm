"""
Textured 2D Gaussian scene representation.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from gsplat.cuda._wrapper import TextureAlphaMode
from jaxtyping import Float
from torch import Tensor


@dataclass
class GaussianCfg:
    """
    Configuration for Gaussian parameter generation.

    Controls scale bounds, SH degree, and texture settings. Previously named
    GaussianAdapterCfg; the adapter class has been eliminated in favor of
    stateless utility functions.
    """

    gaussian_scale_min: float = 1e-10
    gaussian_scale_max: float = 15.0
    sh_degree: int = 4

    texture_enabled: bool = False
    texture_size: int = 1
    texture_project_enabled: bool = False
    texture_project_as_base: bool = False
    texture_alpha_mode: TextureAlphaMode = TextureAlphaMode.TEXTURE
    texture_color_sigma: float = 1.0

    def __post_init__(self):
        if self.texture_project_enabled and not self.texture_enabled:
            raise ValueError(
                "texture_project_enabled is True, but texture_enabled is False"
            )
        valid_texture_sizes = [1, 2, 4, 8, 16, 32]
        if (
            self.texture_enabled
            and self.texture_size not in valid_texture_sizes
        ):
            raise ValueError(
                f"texture_size must be one of {valid_texture_sizes}, "
                f"got {self.texture_size}"
            )

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_texture(self) -> int:
        if not self.texture_enabled:
            return 0
        if self.texture_alpha_mode == TextureAlphaMode.GAUSSIAN:
            return self.texture_size * self.texture_size * 3
        else:
            return self.texture_size * self.texture_size * (3 + 1)

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh + self.d_texture


@dataclass
class Gaussians:
    """
    The "g" dimension has been flattened from (v h w). You may reshape it back
    for further processing.
    """

    # Shape equivalences.
    # - (b g 3)
    # - (b (v g_per_v) 3)
    # - (b (v g_per_v 1 1) 3)
    # - (b (v h w 1 1) 3)
    # To unpack, you can treat it as (b (v h w) 3)
    means: Float[Tensor, "b g 3"]
    covariances: Float[Tensor, "b g 3 3"]
    harmonics: Float[Tensor, "b g 3 d_sh"]
    opacities: Float[Tensor, "b g"]

    # Additional properties for 2DGS rendering.
    scales: Optional[Float[Tensor, "b g 3"]] = None
    xyzws: Optional[Float[Tensor, "b g 4"]] = None

    # Additional properties for textured Gaussians.
    texture_colors: Optional[Float[Tensor, "b g 3 texture_h texture_w"]] = None
    texture_alphas: Optional[Float[Tensor, "b g texture_h texture_w"]] = None

    def __repr__(self):
        tensor_repr = lambda t: (
            f"{t.dtype} {tuple(t.shape)}" if t is not None else "None"
        )
        means_repr = tensor_repr(self.means)
        covariances_repr = tensor_repr(self.covariances)
        harmonics_repr = tensor_repr(self.harmonics)
        opacities_repr = tensor_repr(self.opacities)
        scales_repr = tensor_repr(self.scales)
        rotations_repr = tensor_repr(self.xyzws)
        texture_colors_repr = tensor_repr(self.texture_colors)
        texture_alphas_repr = tensor_repr(self.texture_alphas)

        rc = "Gaussians(\n"
        rc += f"  means               : {means_repr},\n"
        rc += f"  covariances         : {covariances_repr},\n"
        rc += f"  harmonics           : {harmonics_repr},\n"
        rc += f"  opacities           : {opacities_repr},\n"
        rc += f"  scales              : {scales_repr},\n"
        rc += f"  xyzws               : {rotations_repr},\n"
        rc += f"  texture_colors      : {texture_colors_repr},\n"
        rc += f"  texture_alphas      : {texture_alphas_repr},\n"
        rc += ")"
        return rc

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index) -> "Gaussians":
        """
        Select Gaussians by index or slice in the "g" dimension.

        Args:
            index: Can be:
                - int: Select single Gaussian
                - slice: Select range of Gaussians
                - list/tuple of ints: Select specific Gaussians
                - Tensor of ints: Select specific Gaussians

        Returns:
            New Gaussians object with selected subset

        Examples:
            gaussians[5]              # Single Gaussian
            gaussians[5:10]           # Slice from 5 to 10
            gaussians[[1, 2, 3, 7, 8]]  # Specific indices
        """
        # Convert various index types to consistent format for fancy indexing.
        if isinstance(index, (list, tuple)):
            index = torch.tensor(index, device=self.means.device)

        # Apply indexing to all tensors along the "g" dimension (dim=1)
        new_means = self.means[:, index]
        new_covariances = self.covariances[:, index]
        new_harmonics = self.harmonics[:, index]
        new_opacities = self.opacities[:, index]

        # Handle optional tensors.
        new_scales = self.scales[:, index] if self.scales is not None else None
        new_xyzws = self.xyzws[:, index] if self.xyzws is not None else None
        new_texture_colors = (
            self.texture_colors[:, index]
            if self.texture_colors is not None
            else None
        )
        new_texture_alphas = (
            self.texture_alphas[:, index]
            if self.texture_alphas is not None
            else None
        )

        return Gaussians(
            means=new_means,
            covariances=new_covariances,
            harmonics=new_harmonics,
            opacities=new_opacities,
            scales=new_scales,
            xyzws=new_xyzws,
            texture_colors=new_texture_colors,
            texture_alphas=new_texture_alphas,
        )

    def get_stats(self) -> Dict[str, float]:
        """
        Calculate statistics of Gaussian parameters.

        Returns:
            Dictionary of statistics as float values
        """
        stats = {}

        # Check if we have any Gaussians - if not, return zeros.
        if self.means.numel() == 0:
            empty_stats = {
                "gs_stats/mean_x": 0.0,
                "gs_stats/mean_y": 0.0,
                "gs_stats/mean_z": 0.0,
                "gs_stats/mean_harmonics_r": 0.0,
                "gs_stats/mean_harmonics_g": 0.0,
                "gs_stats/mean_harmonics_b": 0.0,
                "gs_stats/mean_opacities": 0.0,
                "gs_stats/min_opacities": 0.0,
                "gs_stats/max_opacities": 0.0,
                "gs_stats/median_opacities": 0.0,
                "gs_stats/std_opacities": 0.0,
                "gs_stats/mean_scales": 0.0,
                "gs_stats/min_scales": 0.0,
                "gs_stats/max_scales": 0.0,
                "gs_stats/median_scales": 0.0,
                "gs_stats/std_scales": 0.0,
            }
            # Add texture stats if texture tensors exist (but are empty)
            if self.texture_colors is not None:
                empty_stats.update(
                    {
                        "gs_stats/mean_texture_r": 0.0,
                        "gs_stats/mean_texture_g": 0.0,
                        "gs_stats/mean_texture_b": 0.0,
                    }
                )
            if self.texture_alphas is not None:
                empty_stats.update(
                    {
                        "gs_stats/mean_texture_alphas": 0.0,
                        "gs_stats/min_texture_alphas": 0.0,
                        "gs_stats/max_texture_alphas": 0.0,
                        "gs_stats/median_texture_alphas": 0.0,
                        "gs_stats/std_texture_alphas": 0.0,
                    }
                )
            return empty_stats

        # Position stats.
        stats["gs_stats/mean_x"] = self.means[:, :, 0].mean().item()
        stats["gs_stats/mean_y"] = self.means[:, :, 1].mean().item()
        stats["gs_stats/mean_z"] = self.means[:, :, 2].mean().item()

        # Base color from harmonics (first coefficient)
        base_colors = self.harmonics[:, :, :, 0]  # shape: (batch, gaussian, 3)
        stats["gs_stats/mean_harmonics_r"] = base_colors[:, :, 0].mean().item()
        stats["gs_stats/mean_harmonics_g"] = base_colors[:, :, 1].mean().item()
        stats["gs_stats/mean_harmonics_b"] = base_colors[:, :, 2].mean().item()

        # Opacity stats.
        opacities_flat = self.opacities.flatten()
        stats["gs_stats/mean_opacities"] = opacities_flat.mean().item()
        stats["gs_stats/min_opacities"] = opacities_flat.min().item()
        stats["gs_stats/max_opacities"] = opacities_flat.max().item()
        stats["gs_stats/median_opacities"] = opacities_flat.median().item()
        stats["gs_stats/std_opacities"] = opacities_flat.std().item()

        # Scale stats.
        scales_flat = self.scales.flatten()
        stats["gs_stats/mean_scales"] = scales_flat.mean().item()
        stats["gs_stats/min_scales"] = scales_flat.min().item()
        stats["gs_stats/max_scales"] = scales_flat.max().item()
        stats["gs_stats/median_scales"] = scales_flat.median().item()
        stats["gs_stats/std_scales"] = scales_flat.std().item()

        # Texture stats (if available)
        if self.texture_colors is not None:
            # Average across texture dimensions (h, w)
            texture_r = self.texture_colors[:, :, 0].mean(dim=(-2, -1))
            texture_g = self.texture_colors[:, :, 1].mean(dim=(-2, -1))
            texture_b = self.texture_colors[:, :, 2].mean(dim=(-2, -1))
            stats["gs_stats/mean_texture_r"] = texture_r.mean().item()
            stats["gs_stats/mean_texture_g"] = texture_g.mean().item()
            stats["gs_stats/mean_texture_b"] = texture_b.mean().item()

        if self.texture_alphas is not None:
            texture_alphas_flat = self.texture_alphas.flatten()
            stats["gs_stats/mean_texture_alphas"] = (
                texture_alphas_flat.mean().item()
            )
            stats["gs_stats/min_texture_alphas"] = (
                texture_alphas_flat.min().item()
            )
            stats["gs_stats/max_texture_alphas"] = (
                texture_alphas_flat.max().item()
            )
            stats["gs_stats/median_texture_alphas"] = (
                texture_alphas_flat.median().item()
            )
            stats["gs_stats/std_texture_alphas"] = (
                texture_alphas_flat.std().item()
            )

        return stats
