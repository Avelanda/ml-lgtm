"""
3D projection, unprojection, and pose utilities.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

3D geometry utilities: projection, ray operations, coordinate transforms,
epipolar geometry, and camera embeddings.
"""

import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int64
from torch import Tensor

from lgtm.utils.sht import rsh_cart_2, rsh_cart_4, rsh_cart_8


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """
    Convert batched points (xyz) to (xyz1).
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """
    Convert batched vectors (xyz) to (xyz0).
    """
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """
    Apply a rigid-body transformation to points or vectors.
    """
    return einsum(
        transformation, homogeneous_coordinates, "... i j, ... j -> ... i"
    )


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    poses: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """
    Transform points from 3D camera coordinates to 3D world coordinates.

    Args:
        homogeneous_coordinates: Homogeneous points or vectors.
        poses: c2w 4x4 matrices.
    """
    return transform_rigid(homogeneous_coordinates, poses)


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """
    Unproject 2D camera coordinates with the given Z values.
    """

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays_depthsplat(
    coordinates: Float[Tensor, "*#batch dim"],
    poses: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    """
    Compute world-space ray origins and directions (DepthSplat variant).

    Args:
        coordinates: 2D pixel coordinates.
        poses: c2w 4x4 matrices.
        intrinsics: Normalized intrinsic 3x3 matrices.
    """
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions[..., -1:]

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, poses)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = poses[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_local_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return directions


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """
    Get normalized (range 0 to 1) coordinates and integer indices for an image.
    """

    # Each entry is a pixel-wise integer coordinate.
    # In the 2D case, each entry is a (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(
        torch.meshgrid(*indices, indexing="ij"), dim=-1
    )

    # Each entry is a floating-point coordinate in the range (0, 1).
    # In the 2D case, each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(
        torch.meshgrid(*coordinates, indexing="xy"), dim=-1
    )

    return coordinates, stacked_indices


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(
            vector, dtype=torch.float32, device=intrinsics.device
        )
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def get_intrinsic_embedding(context, degree=0, downsample=1, merge_hw=False):
    if degree not in [0, 2, 4, 8]:
        raise ValueError(f"degree must be one of [0, 2, 4, 8], got {degree}")

    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_local_rays(
        xy_ray,
        rearrange(context["intrinsics"], "b v i j -> b v () () i j"),
    )

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "b v h w d -> b v (h w) d")
    else:
        directions = rearrange(directions, "b v h w d -> b v d h w")

    return directions
