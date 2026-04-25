"""
Camera intrinsics, extrinsics, and ray helpers.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    """
    Normalize c2w poses so that the pivotal camera becomes identity.

    Args:
        pivotal_pose: c2w 4x4 matrix [1, 4, 4].
        poses: c2w 4x4 matrices [N, 4, 4].

    Returns:
        Normalized c2w 4x4 matrices [N, 4, 4].
    """
    canonical_pose = torch.tensor(
        [
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ],
        dtype=torch.float32,
        device=pivotal_pose.device,
    )
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_pose, pivotal_pose_inv)

    # Normalize all views.
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


def normalized_K_to_K(
    normalized_K: Float[Tensor, "*#batch 3 3"],
    im_width: int | Float[Tensor, "*#batch"],
    im_height: int | Float[Tensor, "*#batch"],
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Convert normalized camera intrinsic matrix back to regular intrinsics.
    Regular K is recovered by:
        K[..., 0, :] = K_norm[..., 0, :] * im_width
        K[..., 1, :] = K_norm[..., 1, :] * im_height
    Args:
        normalized_K: Normalized camera intrinsic matrix with shape
            (*batch, 3, 3)
        im_width: Width of the image in pixels, either scalar or tensor
            matching batch dims
        im_height: Height of the image in pixels, either scalar or tensor
            matching batch dims
    Returns:
        Regular camera intrinsic matrix with shape (*batch, 3, 3)
    """
    K = normalized_K.clone()
    K[..., 0, :] *= (
        im_width.unsqueeze(-1) if torch.is_tensor(im_width) else im_width
    )
    K[..., 1, :] *= (
        im_height.unsqueeze(-1) if torch.is_tensor(im_height) else im_height
    )
    return K


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from
    pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOptModule(torch.nn.Module):
    """
    Camera pose optimization module.
    ref: gsplat/examples/utils.py
    """

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation.
        self.register_buffer(
            "identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """
        Adjust camera pose based on learned deltas.

        Args:
            camtoworlds: c2w 4x4 matrices
                [..., 4, 4].
            embed_ids: Embedding indices [...].

        Returns:
            Updated c2w 4x4 matrices [..., 4, 4].
        """
        if camtoworlds.shape[:-2] != embed_ids.shape:
            raise ValueError(
                f"camtoworlds batch shape {camtoworlds.shape[:-2]} does not "
                f"match embed_ids shape {embed_ids.shape}"
            )
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat(
            (*batch_shape, 1, 1)
        )
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """
    Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """
    Generate a coordinate frame given perpendicular, unit-length Y and Z
    vectors.
    """
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([torch.linalg.cross(y, z), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    """
    Generate a coordinate frame where the Y direction is normal to the plane
    defined by unit vectors a and b. The other axes are arbitrary.
    """
    device = a.device

    # Replace every entry in b that is parallel to the corresponding entry in a
    # with an arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame; the initial cross product defines the plane.
    return generate_coordinate_frame(normalize(torch.linalg.cross(a, b)), a)


def matrix_to_euler(
    rotations: Float[Tensor, "*batch 3 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(
        pattern
    )
    rotations = torch.tensor(
        angles_np, dtype=rotations.dtype, device=rotations.device
    )
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: Float[Tensor, "*batch 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(
        pattern, rotations.detach().cpu().numpy()
    ).as_matrix()
    rotations = torch.tensor(
        matrix_np, dtype=rotations.dtype, device=rotations.device
    )
    return rotations.reshape(*batch, 3, 3)


def poses_to_pivot_parameters(
    poses: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    """
    Convert c2w poses to a 5-DOF pivot parametrization.

    Args:
        poses: c2w 4x4 matrices.
        pivot_coordinate_frame: 3x3 coordinate frame for the pivot.
        pivot_point: 3D pivot point in world coordinates.

    Returns:
        5-DOF parameters: [tx, ty, tz, angle_y, angle_z].
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, poses[..., :3, 2])
    origin = poses[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ poses[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_poses(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    """
    Convert 5-DOF pivot parameters back to c2w poses.

    Args:
        parameters: 5-DOF parameters [*batch, 5].
        pivot_coordinate_frame: 3x3 coordinate frame for the pivot.
        pivot_point: 3D pivot point in world coordinates.

    Returns:
        c2w 4x4 matrices.
    """
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(
        pivot_axis, rotation[..., :3, 2]
    )
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    poses = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    poses = poses.broadcast_to((*batch, 4, 4)).clone()
    poses[..., 3, 3] = 1
    poses[..., :3, :3] = rotation
    poses[..., :3, 3] = origin
    return poses


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_poses(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    """
    Interpolate c2w poses by rotating around their "focus point,"
    which is the least-squares intersection between the look vectors
    of the initial and final poses.

    Args:
        initial: c2w 4x4 matrices.
        final: c2w 4x4 matrices.
        t: Interpolation time steps in [0, 1].

    Returns:
        Interpolated c2w 4x4 matrices [*batch, time_step, 4, 4].
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two.
    # Cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(
        initial_look, final_look, eps=eps
    )
    initial_params = poses_to_pivot_parameters(
        initial, pivot_frame, pivot_point
    )
    final_params = poses_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(
        initial_params, final_params, t
    )

    # Convert back.
    return pivot_parameters_to_poses(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )
