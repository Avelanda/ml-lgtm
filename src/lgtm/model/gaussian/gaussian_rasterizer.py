"""
Textured 2D Gaussian rasterization wrapper.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

from dataclasses import dataclass
from math import isqrt
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from gsplat.cuda._wrapper import TextureAlphaMode
from gsplat.rendering import rasterization, rasterization_2dgs
from jaxtyping import Float
from torch import Tensor, nn

from lgtm.model.gaussian.gaussians import Gaussians
from lgtm.utils.camera_utils import normalized_K_to_K


@dataclass
class GaussianRasterizerOutput:
    colors: Float[Tensor, "batch view 3 height width"]
    depths: Float[Tensor, "batch view height width"] | None
    render_normals: Float[Tensor, "batch view 3 height width"] | None
    normals_from_depth: Float[Tensor, "batch view 3 height width"] | None

    def __repr__(self):
        tensor_repr = lambda t: (
            f"{t.dtype} {tuple(t.shape)}" if t is not None else "None"
        )
        colors_repr = tensor_repr(self.colors)
        depths_repr = tensor_repr(self.depths)
        render_normals_repr = tensor_repr(self.render_normals)
        normals_from_depth_repr = tensor_repr(self.normals_from_depth)

        rc = "GaussianRasterizerOutput(\n"
        rc += f"  colors            : {colors_repr},\n"
        rc += f"  depths            : {depths_repr},\n"
        rc += f"  render_normals    : {render_normals_repr},\n"
        rc += f"  normals_from_depth: {normals_from_depth_repr},\n"
        rc += ")"
        return rc

    def __str__(self):
        return self.__repr__()


@dataclass
class GaussianRasterizerCfg:
    name: Literal["gsplat"]
    background_color: list[float]
    make_scale_invariant: bool
    backend: Optional[
        Literal[
            "gsplat_3dgs",
            "gsplat_2dgs",
            "gsplat_lgtm",
        ]
    ] = "gsplat_3dgs"

    sanity_check: bool = False

    texture_color_sigma: float = 1.0
    texture_alpha_mode: TextureAlphaMode = TextureAlphaMode.TEXTURE


class GaussianRasterizer(nn.Module):
    cfg: GaussianRasterizerCfg
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GaussianRasterizerCfg,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.make_scale_invariant = cfg.make_scale_invariant
        self.backend = cfg.backend
        self.sanity_check = cfg.sanity_check
        self.texture_color_sigma = cfg.texture_color_sigma
        self.texture_alpha_mode = cfg.texture_alpha_mode
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        poses: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> GaussianRasterizerOutput:
        """
        Render Gaussians using gsplat batched rasterization.

        Args:
            gaussians: 3D Gaussian representation.
            poses: c2w 4x4 matrices [batch, view, 4, 4].
            intrinsics: Normalized intrinsic 3x3 matrices [batch, view, 3, 3].
            near: Near plane distances [batch, view].
            far: Far plane distances [batch, view].
            image_shape: (height, width) of the output image.
            cam_rot_delta: Optional rotation deltas [batch, view, 3].
            cam_trans_delta: Optional translation deltas [batch, view, 3].
        """
        b, v, _, _ = poses.shape
        h, w = image_shape
        _, _, _, n = gaussians.harmonics.shape
        degree = isqrt(n) - 1
        use_sh = True

        if self.backend not in ("gsplat_3dgs", "gsplat_2dgs", "gsplat_lgtm"):
            raise ValueError(f"Unknown backend: {self.backend}")

        if self.sanity_check:
            # Near and far shall be consistent across views for each batch.
            if not torch.allclose(
                near,
                repeat(near[:, 0], "b -> b v", v=v),
                rtol=1e-5,
            ):
                raise ValueError(
                    "near must be constant across views for each batch"
                )
            if not torch.allclose(
                far,
                repeat(far[:, 0], "b -> b v", v=v),
                rtol=1e-5,
            ):
                raise ValueError(
                    "far must be constant across views for each batch"
                )

        if self.make_scale_invariant:
            scale = 1.0 / near  # (b, v)
            poses = poses.clone()
            poses[..., :3, 3] = poses[..., :3, 3] * scale[..., None]

            # Scale covariances: (b, g, 3, 3) * (b, 1, 1, 1)
            gaussian_covariances = gaussians.covariances * (
                rearrange(scale[:, 0], "b -> b 1 1 1") ** 2
            )

            # Scale scales: (b, g, 3) * (b, 1, 1)
            gaussian_scales = gaussians.scales * rearrange(
                scale[:, 0], "b -> b 1 1"
            )

            # Convert xyzw → wxyz for gsplat.
            qx, qy, qz, qw = gaussians.xyzws.unbind(dim=-1)
            gaussian_wxyzs = torch.stack([qw, qx, qy, qz], dim=-1)

            # Scale means: (b, g, 3) * (b, 1, 1)
            gaussian_means = gaussians.means * rearrange(
                scale[:, 0], "b -> b 1 1"
            )

            # Near/far/scale: (b, v)
            near = near * scale
            far = far * scale
        else:
            gaussian_covariances = gaussians.covariances
            gaussian_scales = gaussians.scales
            # Convert xyzw → wxyz for gsplat.
            qx, qy, qz, qw = gaussians.xyzws.unbind(dim=-1)
            gaussian_wxyzs = torch.stack([qw, qx, qy, qz], dim=-1)
            gaussian_means = gaussians.means

        Ks = normalized_K_to_K(
            intrinsics, w, h
        )  # Denormalize to pixels (b, v, 3, 3)
        Ts = poses.inverse()  # c2w to w2c for gsplat

        # Gsplat's background is (C, D) i.e. (v, 3)
        backgrounds = repeat(self.background_color, "c -> v c", v=v)

        # Gsplat's harmonics is (N, K, 3) i.e. (g, d_sh, rgb)
        gaussian_harmonics = (
            rearrange(gaussians.harmonics, "b g rgb d_sh -> b g d_sh rgb")
            if use_sh
            else None
        )

        all_rgbs = []
        all_depths = []
        all_render_normals = []
        all_normals_from_depth = []
        for i in range(b):
            if self.backend == "gsplat_2dgs":
                (
                    render_colors,
                    _render_alphas,
                    render_normals,
                    normals_from_depth,
                    _render_distort,
                    _render_median,
                    _info,
                ) = rasterization_2dgs(
                    means=gaussian_means[i],  # [g, 3]
                    quats=gaussian_wxyzs[i],
                    scales=gaussian_scales[i],
                    opacities=gaussians.opacities[i],  # [g]
                    colors=gaussian_harmonics[i] if use_sh else None,
                    viewmats=Ts[i],  # [v, 4, 4]
                    Ks=Ks[i],  # [v, 3, 3]
                    width=w,
                    height=h,
                    near_plane=near[i, 0].item(),
                    far_plane=far[i, 0].item(),
                    sh_degree=degree if use_sh else None,
                    backgrounds=backgrounds,  # [v, 3]
                    render_mode="RGB+ED",
                )
                batch_render_normals = rearrange(
                    render_normals,
                    "v h w c -> v c h w",
                )
                batch_normals_from_depth = rearrange(
                    normals_from_depth,
                    "v h w c -> v c h w",
                )
            elif self.backend == "gsplat_lgtm":
                if gaussians.texture_colors is None:
                    raise ValueError("gaussians.texture_colors is None")
                if (
                    gaussians.texture_alphas is None
                    and self.texture_alpha_mode != TextureAlphaMode.GAUSSIAN
                ):
                    raise ValueError("gaussians.texture_alphas is None")

                (
                    render_colors,
                    _render_alphas,
                    render_normals,
                    normals_from_depth,
                    _render_distort,
                    _render_median,
                    _info,
                ) = rasterization_2dgs(
                    means=gaussian_means[i],  # [g, 3]
                    quats=gaussian_wxyzs[i],
                    scales=gaussian_scales[i],
                    opacities=gaussians.opacities[i],  # [g]
                    colors=gaussian_harmonics[i] if use_sh else None,
                    viewmats=Ts[i],  # [v, 4, 4]
                    Ks=Ks[i],  # [v, 3, 3]
                    width=w,
                    height=h,
                    near_plane=near[i, 0].item(),
                    far_plane=far[i, 0].item(),
                    sh_degree=degree if use_sh else None,
                    backgrounds=backgrounds,  # [v, 3]
                    render_mode="RGB+ED",
                    texture_colors=gaussians.texture_colors[i],
                    texture_alphas=(
                        gaussians.texture_alphas[i]
                        if gaussians.texture_alphas is not None
                        else None
                    ),
                    texture_scale_multiplier=1.0,
                    texture_color_sigma=self.texture_color_sigma,
                    texture_alpha_mode=self.texture_alpha_mode,
                )
                batch_render_normals = rearrange(
                    render_normals,
                    "v h w c -> v c h w",
                )
                batch_normals_from_depth = rearrange(
                    normals_from_depth,
                    "v h w c -> v c h w",
                )
            elif self.backend == "gsplat_3dgs":
                render_colors, _render_alphas, _meta = rasterization(
                    means=gaussian_means[i],  # [g, 3]
                    quats=gaussian_wxyzs[i],
                    scales=gaussian_scales[i],
                    opacities=gaussians.opacities[i],  # [g]
                    colors=gaussian_harmonics[i] if use_sh else None,
                    viewmats=Ts[i],  # [v, 4, 4]
                    Ks=Ks[i],  # [v, 3, 3]
                    width=w,
                    height=h,
                    near_plane=near[i, 0].item(),
                    far_plane=far[i, 0].item(),
                    sh_degree=degree if use_sh else None,
                    backgrounds=backgrounds,  # [v, 3]
                    render_mode="RGB+ED",
                )
                batch_render_normals = None
                batch_normals_from_depth = None
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            # Extract RGB and depth from render_colors and reshape.
            # [v, h, w, 4] -> [v, 3, h, w] for RGB, [v, h, w] for depth.
            colors = rearrange(render_colors[..., :3], "v h w rgb -> v rgb h w")
            depths = render_colors[..., 3]
            all_rgbs.append(colors)
            all_depths.append(depths)
            all_render_normals.append(batch_render_normals)
            all_normals_from_depth.append(batch_normals_from_depth)

        colors = torch.stack(all_rgbs, dim=0)
        depths = torch.stack(all_depths, dim=0)

        if all_render_normals[0] is not None:
            render_normals = torch.stack(all_render_normals, dim=0)
            normals_from_depth = torch.stack(all_normals_from_depth, dim=0)
        else:
            render_normals = None
            normals_from_depth = None

        return GaussianRasterizerOutput(
            colors, depths, render_normals, normals_from_depth
        )


RASTERIZERS = {
    "gsplat": GaussianRasterizer,
}

RasterizerCfg = GaussianRasterizerCfg


def get_rasterizer(cfg: RasterizerCfg) -> GaussianRasterizer:
    return RASTERIZERS[cfg.name](cfg)
