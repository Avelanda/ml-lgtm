"""
Texture projection for textured 2D Gaussians.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from gsplat.cuda._torch_impl import _quat_scale_to_matrix
from gsplat.cuda._wrapper import fully_fused_projection_2dgs
from jaxtyping import Float
from torch import Tensor

torch.set_printoptions(sci_mode=False)


def _fully_fused_projection_2dgs(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
) -> Tensor:
    """
    PyTorch implementation of
    `gsplat.cuda._wrapper.fully_fused_projection_2dgs()`.
    Only returns the ray transform matrix with other outputs removed.
    """
    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]
    means_c = (
        torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]
    )  # (C, N, 3)
    RS_wl = _quat_scale_to_matrix(quats, scales)
    RS_cl = torch.einsum("cij,njk->cnik", R_cw, RS_wl)  # [C, N, 3, 3]

    # Ray transform matrix, omitting the z rotation.
    T_cl = torch.cat(
        [RS_cl[..., :2], means_c[..., None]], dim=-1
    )  # [C, N, 3, 3]
    T_sl = torch.einsum("cij,cnjk->cnik", Ks[:, :3, :3], T_cl)  # [C, N, 3, 3]
    # In paper notation M = (WH)^T.
    # Later h_u = M @ h_x, h_v = M @ h_y.
    M = torch.transpose(T_sl, -1, -2)  # [C, N, 3, 3]

    # See extern/gsplat/tests/test_2dgs_bbsplat.py.
    M = M.permute((0, 1, 3, 2))

    return M


def uvs_to_im_coords(
    ray_transforms: Float[torch.Tensor, "n 3 3"],
    uvs: Float[torch.Tensor, "m 2"],
) -> Float[torch.Tensor, "n m 2"]:
    """
    Transform NDC coordinates back to pixel coordinates using homogeneous
    coordinates.

    Args:
        ray_transforms: (n, 3, 3) image to uv transform matrices
        uvs: (m, 2) uv (ndc) coordinates of range [-1, 1] but can go any range

    Returns:
        [n, m, 2] pixel coordinates
    """
    # Extract u, v coordinates.
    us = uvs[:, 0:1]  # (m, 1)
    vs = uvs[:, 1:2]  # (m, 1)

    # Homogeneous transform of [u, v, 1]
    # Equivalent to: [u, v, 1] @ trans_mats.transpose(-1, -2)
    trans_mats_T = ray_transforms.transpose(-1, -2)  # (n, 3, 3)
    transformed = (
        us * trans_mats_T[:, 0:1, :]  # (m, 1) * (n, 1, 3) -> (n, m, 3)
        + vs * trans_mats_T[:, 1:2, :]  # (m, 1) * (n, 1, 3) -> (n, m, 3)
        + trans_mats_T[:, 2:3, :]  # (n, 1, 3) -> (n, m, 3)
    )  # (n, m, 3)

    # Convert back from homogeneous coordinates.
    im_coords = transformed[..., :2] / transformed[..., 2:3]  # (n, m, 2)

    return im_coords


def project_texture_colors(
    means: Float[torch.Tensor, "b g 3"],
    scales: Float[torch.Tensor, "b g 3"],
    xyzws: Float[torch.Tensor, "b g 4"],
    ims: Float[torch.Tensor, "b 3 h w"],
    Ks: Float[torch.Tensor, "b 3 3"],
    Ts: Float[torch.Tensor, "b 4 4"],
    texture_size: int,
    texture_color_sigma: float = 1.0,
    use_cuda_projection: bool = False,
) -> Float[torch.Tensor, "b g 3 ts ts"]:
    """
    Perform texture projection for gaussians from sample images.
    Handles batched inputs for integration with DPT-GS head.

    Args:
        means: (b, g, 3) gaussian centers in world coordinates
        scales: (b, g, 3) gaussian scales
        xyzws: (b, g, 4) gaussian quaternions in xyzw convention
        ims: (b, 3, height, width) sample images to project from
        Ks: (b, 3, 3) pixel-scale intrinsic 3x3 matrices.
        Ts: (b, 4, 4) w2c 4x4 matrices.
        texture_size: Size of texture (assumes square texture)
        texture_color_sigma: Controls texture sampling extent (default: 1.0)
        use_cuda_projection: Whether to use CUDA implementation with filtering
            (default: False)

    Returns:
        texture_colors: (b, g, 3, ts, ts) projected texture colors
    """
    bs = means.shape[0]
    g = means.shape[1]
    if means.shape != (bs, g, 3):
        raise ValueError(
            f"Expected means shape ({bs}, {g}, 3), got {means.shape}"
        )
    if scales.shape != (bs, g, 3):
        raise ValueError(
            f"Expected scales shape ({bs}, {g}, 3), got {scales.shape}"
        )
    if xyzws.shape != (bs, g, 4):
        raise ValueError(
            f"Expected xyzws shape ({bs}, {g}, 4), got {xyzws.shape}"
        )
    if ims.shape[:2] != (bs, 3):
        raise ValueError(f"Expected ims shape ({bs}, 3, H, W), got {ims.shape}")
    if Ks.shape != (bs, 3, 3):
        raise ValueError(f"Expected Ks shape ({bs}, 3, 3), got {Ks.shape}")
    if Ts.shape != (bs, 4, 4):
        raise ValueError(f"Expected Ts shape ({bs}, 4, 4), got {Ts.shape}")

    # Sanity check: detect normalized intrinsics.
    # Normalized intrinsics typically have cx, cy < 1 (in normalized.
    # Coordinates) whereas pixel-scale intrinsics have cx, cy in pixel.
    # Coordinates (> 1 for typical images).
    cx_values = Ks[:, 0, 2]  # (b,)
    cy_values = Ks[:, 1, 2]  # (b,)
    if torch.any(cx_values < 1.0) or torch.any(cy_values < 1.0):
        raise ValueError(
            "This function expects pixel-scale intrinsics, but received what"
            " appears to be normalized intrinsics (cx or cy < 1). Please use"
            " normalized_K_to_K() to convert normalized intrinsics to"
            " pixel-scale format before calling this function."
        )

    # Convert images from (b, 3, h, w) to (b, h, w, 3)
    ims = rearrange(ims, "b c h w -> b h w c")

    # Process each batch element.
    texture_colors_list = []
    for b in range(bs):
        if use_cuda_projection:
            texture_colors_b = project_texture_colors_single_cuda(
                means=means[b],  # (g, 3)
                scales=scales[b],  # (g, 3)
                xyzws=xyzws[b],  # (g, 4)
                im=ims[b],  # (h, w, 3)
                K=Ks[b],  # (3, 3)
                T=Ts[b],  # (4, 4)
                texture_size=texture_size,
                texture_color_sigma=texture_color_sigma,
            )
        else:
            texture_colors_b = project_texture_colors_single_python(
                means=means[b],  # (g, 3)
                scales=scales[b],  # (g, 3)
                xyzws=xyzws[b],  # (g, 4)
                im=ims[b],  # (h, w, 3)
                K=Ks[b],  # (3, 3)
                T=Ts[b],  # (4, 4)
                texture_size=texture_size,
                texture_color_sigma=texture_color_sigma,
            )
        texture_colors_list.append(texture_colors_b)

    texture_colors = torch.stack(texture_colors_list, dim=0)
    return texture_colors


def project_texture_colors_single_python(
    means: Float[torch.Tensor, "g 3"],
    scales: Float[torch.Tensor, "g 3"],
    xyzws: Float[torch.Tensor, "g 4"],
    im: Float[torch.Tensor, "h w 3"],
    K: Float[torch.Tensor, "3 3"],
    T: Float[torch.Tensor, "4 4"],
    texture_size: int,
    texture_color_sigma: float = 1.0,
) -> Float[torch.Tensor, "g 3 ts ts"]:
    """
    Perform texture projection for gaussians from sample images (single batch)
    using Python implementation.

    Args:
        means: (g, 3) gaussian centers in world coordinates
        scales: (g, 3) gaussian scales
        xyzws: (g, 4) gaussian quaternions in xyzw convention
        im: (h, w, 3) sample image to project from
        K: (3, 3) camera intrinsic matrix in pixel-scale.
        T: (4, 4) world-to-camera matrix (w2c)
        texture_size: Size of texture (assumes square texture)
        texture_color_sigma: Controls texture sampling extent (default: 1.0)

    Returns:
        texture_colors: (g, 3, texture_size, texture_size) projected
            texture colors
    """
    # Sanity checks.
    if not means.ndim == 2 or means.shape[1] != 3:
        raise ValueError("means must be a 2D tensor with shape (g, 3)")
    if not scales.ndim == 2 or scales.shape[1] != 3:
        raise ValueError("scales must be a 2D tensor with shape (g, 3)")
    if not xyzws.ndim == 2 or xyzws.shape[1] != 4:
        raise ValueError("xyzws must be a 2D tensor with shape (g, 4)")
    if not im.ndim == 3 or im.shape[2] != 3:
        raise ValueError("im must be a 3D tensor with shape (h, w, 3)")
    if not K.ndim == 2 or K.shape[0] != 3 or K.shape[1] != 3:
        raise ValueError("K must be a 2D tensor with shape (3, 3)")
    if not T.ndim == 2 or T.shape[0] != 4 or T.shape[1] != 4:
        raise ValueError("T must be a 2D tensor with shape (4, 4)")
    if not texture_size > 0:
        raise ValueError("ts must be a positive integer")

    # Normalized intrinsics typically have cx, cy < 1 (in normalized.
    # Coordinates) whereas pixel-scale intrinsics have cx, cy in pixel.
    # Coordinates (> 1 for typical images)
    cx = K[0, 2].item()
    cy = K[1, 2].item()
    if cx < 1.0 or cy < 1.0:
        raise ValueError(
            "This function expects pixel-scale intrinsics, but received what"
            " appears to be normalized intrinsics (cx or cy < 1). Please use"
            " normalized_K_to_K() to convert normalized intrinsics to"
            " pixel-scale format before calling this function."
        )

    device = means.device
    wxyzs = torch.cat([xyzws[..., 3:4], xyzws[..., :3]], dim=-1)
    im_height, im_width = im.shape[0], im.shape[1]

    # Compute pixel to ndc transform matrix using Python implementation.
    ray_transforms = _fully_fused_projection_2dgs(
        means=means,
        quats=wxyzs,
        scales=scales,
        viewmats=rearrange(T, "h w -> 1 h w"),
        Ks=rearrange(K, "h w -> 1 h w"),
    )
    ray_transforms = rearrange(ray_transforms, "1 n h w -> n h w")

    # Create texture UV coordinates.
    # e.g. ts = 4
    # - linspace(0.5, ts - 0.5, ts): [0.5, 1.5, 2.5, 3.5]
    # - linspace(0.5, ts - 0.5, ts) / ts: [0.125, 0.375, 0.625, 0.875]
    #   # Range [0, 1]
    u_coords = (
        torch.linspace(0.5, texture_size - 0.5, texture_size, device=device)
        / texture_size
    )
    v_coords = (
        torch.linspace(0.5, texture_size - 0.5, texture_size, device=device)
        / texture_size
    )  # Same as u_coords
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
    uvs = torch.stack([u_grid, v_grid], dim=-1)  # (ts, ts, 2)
    uvs = rearrange(uvs, "h w d -> (h w) d")  # (ts*ts, 2)

    # UV's valid range is [-sigma, sigma], otherwise, it's GL_CLAMP_TO_EDGE.
    # Convert range [0, 1] -> [-sigma, sigma]
    # e.g. if sigma = 1.0, range [0, 1] -> [-1, 1],
    uvs = (uvs * 2.0 - 1.0) * texture_color_sigma  # (ts*ts, 2)

    # Transform UV coordinates to pixel coordinates using broadcasting.
    # (g, ts*ts, 2)
    im_coords = uvs_to_im_coords(ray_transforms, uvs)
    grid_x = (
        2.0 * im_coords[..., 0] / im_width - 1.0
    )  # Normalize to (-1, 1) range
    grid_y = (
        2.0 * im_coords[..., 1] / im_height - 1.0
    )  # Normalize to (-1, 1) range
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (g, ts*ts, 2)
    grid_coords = rearrange(
        grid_coords, "n tsts d -> 1 (n tsts) 1 d"
    )  # (1, g*ts*ts, 1, 2)

    # Sample colors using grid_sample.
    im_batched = rearrange(
        im, "h w c -> 1 c h w"
    )  # (1, 3, im_height, im_width)
    sampled_colors = F.grid_sample(
        im_batched,
        grid_coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (1, 3, g*ts*ts, 1)
    sampled_colors = rearrange(
        sampled_colors,
        "1 c (n h w) 1 -> n c h w",
        n=means.shape[0],
        h=texture_size,
        w=texture_size,
    )  # (g, 3, ts, ts)

    return sampled_colors


def project_texture_colors_single_cuda(
    means: Float[torch.Tensor, "g 3"],
    scales: Float[torch.Tensor, "g 3"],
    xyzws: Float[torch.Tensor, "g 4"],
    im: Float[torch.Tensor, "h w 3"],
    K: Float[torch.Tensor, "3 3"],
    T: Float[torch.Tensor, "4 4"],
    texture_size: int,
    texture_color_sigma: float = 1.0,
) -> Float[torch.Tensor, "g 3 ts ts"]:
    """
    Perform texture projection for gaussians from sample images (single batch).

    Args:
        means: (g, 3) gaussian centers in world coordinates
        scales: (g, 3) gaussian scales
        xyzws: (g, 4) gaussian quaternions in xyzw convention
        im: (h, w, 3) sample image to project from
        K: (3, 3) camera intrinsic matrix in pixel-scale.
        T: (4, 4) world-to-camera matrix (w2c)
        texture_size: Size of texture (assumes square texture)

    Returns:
        texture_colors: (g, 3, texture_size, texture_size) projected
            texture colors
    """
    # Sanity checks.
    if not means.ndim == 2 or means.shape[1] != 3:
        raise ValueError("means must be a 2D tensor with shape (g, 3)")
    if not scales.ndim == 2 or scales.shape[1] != 3:
        raise ValueError("scales must be a 2D tensor with shape (g, 3)")
    if not xyzws.ndim == 2 or xyzws.shape[1] != 4:
        raise ValueError("xyzws must be a 2D tensor with shape (g, 4)")
    if not im.ndim == 3 or im.shape[2] != 3:
        raise ValueError("im must be a 3D tensor with shape (h, w, 3)")
    if not K.ndim == 2 or K.shape[0] != 3 or K.shape[1] != 3:
        raise ValueError("K must be a 2D tensor with shape (3, 3)")
    if not T.ndim == 2 or T.shape[0] != 4 or T.shape[1] != 4:
        raise ValueError("T must be a 2D tensor with shape (4, 4)")
    if not texture_size > 0:
        raise ValueError("ts must be a positive integer")

    # Normalized intrinsics typically have cx, cy < 1 (in normalized.
    # Coordinates) whereas pixel-scale intrinsics have cx, cy in pixel.
    # Coordinates (> 1 for typical images)
    cx = K[0, 2].item()
    cy = K[1, 2].item()
    if cx < 1.0 or cy < 1.0:
        raise ValueError(
            "This function expects pixel-scale intrinsics, but received what"
            " appears to be normalized intrinsics (cx or cy < 1). Please use"
            " normalized_K_to_K() to convert normalized intrinsics to"
            " pixel-scale format before calling this function."
        )

    device = means.device
    wxyzs = torch.cat([xyzws[..., 3:4], xyzws[..., :3]], dim=-1)
    im_height, im_width = im.shape[0], im.shape[1]

    # Compute pixel to ndc transform matrix.
    radii, _, _, ray_transforms, _ = fully_fused_projection_2dgs(
        means=means,
        quats=wxyzs,
        scales=scales,
        viewmats=rearrange(T, "h w -> 1 h w"),
        Ks=rearrange(K, "h w -> 1 h w"),
        width=im_width,
        height=im_height,
    )

    valid_mask = (radii > 0).all(dim=-1)
    valid_mask = rearrange(valid_mask, "1 n -> n")
    ray_transforms = rearrange(ray_transforms, "1 n h w -> n h w")

    num_valid = valid_mask.sum().item()
    if num_valid == 0:
        return torch.zeros(
            means.shape[0], 3, texture_size, texture_size, device=device
        )
    ray_transforms_valid = ray_transforms[valid_mask]  # (num_valid, 3, 3)

    # Create texture UV coordinates.
    # e.g. ts = 4
    # - linspace(0.5, ts - 0.5, ts): [0.5, 1.5, 2.5, 3.5]
    # - linspace(0.5, ts - 0.5, ts) / ts: [0.125, 0.375, 0.625, 0.875]
    #   # Range [0, 1]
    u_coords = (
        torch.linspace(0.5, texture_size - 0.5, texture_size, device=device)
        / texture_size
    )
    v_coords = (
        torch.linspace(0.5, texture_size - 0.5, texture_size, device=device)
        / texture_size
    )  # Same as u_coords
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
    uvs = torch.stack([u_grid, v_grid], dim=-1)  # (ts, ts, 2)
    uvs = rearrange(uvs, "h w d -> (h w) d")  # (ts*ts, 2)

    # UV's valid range is [-sigma, sigma], otherwise, it's GL_CLAMP_TO_EDGE.
    # Convert range [0, 1] -> [-sigma, sigma]
    # e.g. if sigma = 1.0, range [0, 1] -> [-1, 1],
    uvs = (uvs * 2.0 - 1.0) * texture_color_sigma  # (ts*ts, 2)

    # Transform UV coordinates to pixel coordinates using broadcasting.
    # (num_valid, ts*ts, 2)
    im_coords = uvs_to_im_coords(ray_transforms_valid, uvs)
    grid_x = (
        2.0 * im_coords[..., 0] / im_width - 1.0
    )  # Normalize to (-1, 1) range
    grid_y = (
        2.0 * im_coords[..., 1] / im_height - 1.0
    )  # Normalize to (-1, 1) range
    grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (num_valid, ts*ts, 2)
    grid_coords = rearrange(
        grid_coords, "n tsts d -> 1 (n tsts) 1 d"
    )  # (1, num_valid*ts*ts, 1, 2)

    # Sample colors using grid_sample.
    im_batched = rearrange(
        im, "h w c -> 1 c h w"
    )  # (1, 3, im_height, im_width)
    sampled_colors = F.grid_sample(
        im_batched,
        grid_coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (1, 3, num_valid*ts*ts, 1)
    sampled_colors = rearrange(
        sampled_colors,
        "1 c (n h w) 1 -> n c h w",
        n=num_valid,
        h=texture_size,
        w=texture_size,
    )  # (num_valid, 3, ts, ts)

    # Apply texture colors.
    num_gaussians = means.shape[0]
    texture_colors = torch.zeros(
        num_gaussians, 3, texture_size, texture_size, device=device
    )
    texture_colors[valid_mask] = sampled_colors

    return texture_colors
