"""
Flash3D backbone.

For third-party code see ACKNOWLEDGMENTS file.

By using the Flash3D backbone, you agree to comply with the licenses of
Flash3D and all associated dependencies, including but not limited to:
- flash3d: https://github.com/eldar/flash3d/blob/main/LICENSE
- unidepth: https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE
- dinov2: https://github.com/facebookresearch/dinov2/blob/main/LICENSE
"""

import contextlib
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from math import ceil, pi
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import Tensor

logger = logging.getLogger("dinov2")

IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)


def max_stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.stack(tensors, dim=-1).max(dim=-1).values


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def generate_rays(
    camera_intrinsics: torch.Tensor,
    image_shape: Tuple[int, int],
    noisy: bool = False,
):
    batch_size, device, dtype = (
        camera_intrinsics.shape[0],
        camera_intrinsics.device,
        camera_intrinsics.dtype,
    )
    height, width = image_shape
    pixel_coords_x = torch.linspace(
        0, width - 1, width, device=device, dtype=dtype
    )
    pixel_coords_y = torch.linspace(
        0, height - 1, height, device=device, dtype=dtype
    )
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()],
        dim=2,
    )
    pixel_coords = pixel_coords + 0.5
    intrinsics_inv = (
        torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    intrinsics_inv[:, 0, 0] = 1.0 / camera_intrinsics[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / camera_intrinsics[:, 1, 1]
    intrinsics_inv[:, 0, 2] = (
        -camera_intrinsics[:, 0, 2] / camera_intrinsics[:, 0, 0]
    )
    intrinsics_inv[:, 1, 2] = (
        -camera_intrinsics[:, 1, 2] / camera_intrinsics[:, 1, 1]
    )
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )
    ray_directions = F.normalize(ray_directions, dim=1)
    ray_directions = ray_directions.permute(0, 2, 1)
    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


def spherical_zbuffer_to_euclidean(
    spherical_tensor: torch.Tensor,
) -> torch.Tensor:
    theta = spherical_tensor[..., 0]
    phi = spherical_tensor[..., 1]
    z = spherical_tensor[..., 2]
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)
    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def flat_interpolate(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(
        0, 3, 1, 2
    )
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(0, 2, 1)
    return flat_tensor_interp.contiguous()


def rsh_cart_8(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 8. Returns (N,...,81)."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    return torch.stack(
        [
            0.282094791773878
            * torch.ones(1, device=xyz.device).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
            2.5033429417967 * xy * (x2 - y2),
            -1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.267618617422916
            * y
            * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            1.48099765681286
            * z
            * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
            - 0.952069922236839 * z2
            + 0.317356640745613,
            0.267618617422916
            * x
            * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            -1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            -0.65638205684017 * y * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (52.5 - 472.5 * z2) * (3.0 * x2 - y2),
            0.0913054625709205 * xy * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z),
            0.241571547304372
            * y
            * (
                2.25 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 9.375 * z2
                - 1.875
            ),
            -1.24747010616985 * z * (1.5 * z2 - 0.5)
            + 1.6840846433293
            * z
            * (
                1.75
                * z
                * (
                    1.66666666666667 * z * (1.5 * z2 - 0.5)
                    - 0.666666666666667 * z
                )
                - 1.125 * z2
                + 0.375
            )
            + 0.498988042467941 * z,
            0.241571547304372
            * x
            * (
                2.25 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 9.375 * z2
                - 1.875
            ),
            0.0456527312854602
            * (x2 - y2)
            * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z),
            0.00931882475114763 * x * (52.5 - 472.5 * z2) * (x2 - 3.0 * y2),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            -0.65638205684017 * x * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            4.09910463115149 * x**4 * xy
            - 13.6636821038383 * xy**3
            + 4.09910463115149 * xy * y**4,
            -2.36661916223175 * yz * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00427144889505798 * xy * (x2 - y2) * (5197.5 * z2 - 472.5),
            0.00584892228263444
            * y
            * (3.0 * x2 - y2)
            * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z),
            0.0701870673916132
            * xy
            * (
                2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                - 91.875 * z2
                + 13.125
            ),
            0.221950995245231
            * y
            * (
                -2.8 * z * (1.5 - 7.5 * z2)
                + 2.2
                * z
                * (
                    2.25
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 9.375 * z2
                    - 1.875
                )
                - 4.8 * z
            ),
            -1.48328138624466
            * z
            * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
            + 1.86469659985043
            * z
            * (
                -1.33333333333333 * z * (1.5 * z2 - 0.5)
                + 1.8
                * z
                * (
                    1.75
                    * z
                    * (
                        1.66666666666667 * z * (1.5 * z2 - 0.5)
                        - 0.666666666666667 * z
                    )
                    - 1.125 * z2
                    + 0.375
                )
                + 0.533333333333333 * z
            )
            + 0.953538034014426 * z2
            - 0.317846011338142,
            0.221950995245231
            * x
            * (
                -2.8 * z * (1.5 - 7.5 * z2)
                + 2.2
                * z
                * (
                    2.25
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 9.375 * z2
                    - 1.875
                )
                - 4.8 * z
            ),
            0.0350935336958066
            * (x2 - y2)
            * (
                2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                - 91.875 * z2
                + 13.125
            ),
            0.00584892228263444
            * x
            * (x2 - 3.0 * y2)
            * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z),
            0.0010678622237645
            * (5197.5 * z2 - 472.5)
            * (-6.0 * x2 * y2 + x4 + y4),
            -2.36661916223175 * xz * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            0.683184105191914 * x2**3
            + 10.2477615778787 * x2 * y4
            - 10.2477615778787 * x4 * y2
            - 0.683184105191914 * y2**3,
            -0.707162732524596
            * y
            * (7.0 * x2**3 + 21.0 * x2 * y4 - 35.0 * x4 * y2 - y2**3),
            2.6459606618019
            * z
            * (6.0 * x**4 * xy - 20.0 * xy**3 + 6.0 * xy * y**4),
            9.98394571852353e-5
            * y
            * (5197.5 - 67567.5 * z2)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00239614697244565
            * xy
            * (x2 - y2)
            * (4.33333333333333 * z * (5197.5 * z2 - 472.5) - 3150.0 * z),
            0.00397356022507413
            * y
            * (3.0 * x2 - y2)
            * (
                3.25
                * z
                * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z)
                + 1063.125 * z2
                - 118.125
            ),
            0.0561946276120613
            * xy
            * (
                -4.8 * z * (52.5 * z2 - 7.5)
                + 2.6
                * z
                * (
                    2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                    - 91.875 * z2
                    + 13.125
                )
                + 48.0 * z
            ),
            0.206472245902897
            * y
            * (
                -2.625 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 2.16666666666667
                * z
                * (
                    -2.8 * z * (1.5 - 7.5 * z2)
                    + 2.2
                    * z
                    * (
                        2.25
                        * z
                        * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                        + 9.375 * z2
                        - 1.875
                    )
                    - 4.8 * z
                )
                - 10.9375 * z2
                + 2.1875
            ),
            1.24862677781952 * z * (1.5 * z2 - 0.5)
            - 1.68564615005635
            * z
            * (
                1.75
                * z
                * (
                    1.66666666666667 * z * (1.5 * z2 - 0.5)
                    - 0.666666666666667 * z
                )
                - 1.125 * z2
                + 0.375
            )
            + 2.02901851395672
            * z
            * (
                -1.45833333333333
                * z
                * (
                    1.66666666666667 * z * (1.5 * z2 - 0.5)
                    - 0.666666666666667 * z
                )
                + 1.83333333333333
                * z
                * (
                    -1.33333333333333 * z * (1.5 * z2 - 0.5)
                    + 1.8
                    * z
                    * (
                        1.75
                        * z
                        * (
                            1.66666666666667 * z * (1.5 * z2 - 0.5)
                            - 0.666666666666667 * z
                        )
                        - 1.125 * z2
                        + 0.375
                    )
                    + 0.533333333333333 * z
                )
                + 0.9375 * z2
                - 0.3125
            )
            - 0.499450711127808 * z,
            0.206472245902897
            * x
            * (
                -2.625 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 2.16666666666667
                * z
                * (
                    -2.8 * z * (1.5 - 7.5 * z2)
                    + 2.2
                    * z
                    * (
                        2.25
                        * z
                        * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                        + 9.375 * z2
                        - 1.875
                    )
                    - 4.8 * z
                )
                - 10.9375 * z2
                + 2.1875
            ),
            0.0280973138060306
            * (x2 - y2)
            * (
                -4.8 * z * (52.5 * z2 - 7.5)
                + 2.6
                * z
                * (
                    2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                    - 91.875 * z2
                    + 13.125
                )
                + 48.0 * z
            ),
            0.00397356022507413
            * x
            * (x2 - 3.0 * y2)
            * (
                3.25
                * z
                * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z)
                + 1063.125 * z2
                - 118.125
            ),
            0.000599036743111412
            * (4.33333333333333 * z * (5197.5 * z2 - 472.5) - 3150.0 * z)
            * (-6.0 * x2 * y2 + x4 + y4),
            9.98394571852353e-5
            * x
            * (5197.5 - 67567.5 * z2)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            2.6459606618019
            * z
            * (x2**3 + 15.0 * x2 * y4 - 15.0 * x4 * y2 - y2**3),
            -0.707162732524596
            * x
            * (x2**3 + 35.0 * x2 * y4 - 21.0 * x4 * y2 - 7.0 * y2**3),
            5.83141328139864
            * xy
            * (x2**3 + 7.0 * x2 * y4 - 7.0 * x4 * y2 - y2**3),
            -2.91570664069932
            * yz
            * (7.0 * x2**3 + 21.0 * x2 * y4 - 35.0 * x4 * y2 - y2**3),
            7.87853281621404e-6
            * (1013512.5 * z2 - 67567.5)
            * (6.0 * x**4 * xy - 20.0 * xy**3 + 6.0 * xy * y**4),
            5.10587282657803e-5
            * y
            * (5.0 * z * (5197.5 - 67567.5 * z2) + 41580.0 * z)
            * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            0.00147275890257803
            * xy
            * (x2 - y2)
            * (
                3.75
                * z
                * (4.33333333333333 * z * (5197.5 * z2 - 472.5) - 3150.0 * z)
                - 14293.125 * z2
                + 1299.375
            ),
            0.0028519853513317
            * y
            * (3.0 * x2 - y2)
            * (
                -7.33333333333333 * z * (52.5 - 472.5 * z2)
                + 3.0
                * z
                * (
                    3.25
                    * z
                    * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z)
                    + 1063.125 * z2
                    - 118.125
                )
                - 560.0 * z
            ),
            0.0463392770473559
            * xy
            * (
                -4.125 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                + 2.5
                * z
                * (
                    -4.8 * z * (52.5 * z2 - 7.5)
                    + 2.6
                    * z
                    * (
                        2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                        - 91.875 * z2
                        + 13.125
                    )
                    + 48.0 * z
                )
                + 137.8125 * z2
                - 19.6875
            ),
            0.193851103820053
            * y
            * (
                3.2 * z * (1.5 - 7.5 * z2)
                - 2.51428571428571
                * z
                * (
                    2.25
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 9.375 * z2
                    - 1.875
                )
                + 2.14285714285714
                * z
                * (
                    -2.625
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 2.16666666666667
                    * z
                    * (
                        -2.8 * z * (1.5 - 7.5 * z2)
                        + 2.2
                        * z
                        * (
                            2.25
                            * z
                            * (
                                2.33333333333333 * z * (1.5 - 7.5 * z2)
                                + 4.0 * z
                            )
                            + 9.375 * z2
                            - 1.875
                        )
                        - 4.8 * z
                    )
                    - 10.9375 * z2
                    + 2.1875
                )
                + 5.48571428571429 * z
            ),
            1.48417251362228
            * z
            * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
            - 1.86581687426801
            * z
            * (
                -1.33333333333333 * z * (1.5 * z2 - 0.5)
                + 1.8
                * z
                * (
                    1.75
                    * z
                    * (
                        1.66666666666667 * z * (1.5 * z2 - 0.5)
                        - 0.666666666666667 * z
                    )
                    - 1.125 * z2
                    + 0.375
                )
                + 0.533333333333333 * z
            )
            + 2.1808249179756
            * z
            * (
                1.14285714285714 * z * (1.5 * z2 - 0.5)
                - 1.54285714285714
                * z
                * (
                    1.75
                    * z
                    * (
                        1.66666666666667 * z * (1.5 * z2 - 0.5)
                        - 0.666666666666667 * z
                    )
                    - 1.125 * z2
                    + 0.375
                )
                + 1.85714285714286
                * z
                * (
                    -1.45833333333333
                    * z
                    * (
                        1.66666666666667 * z * (1.5 * z2 - 0.5)
                        - 0.666666666666667 * z
                    )
                    + 1.83333333333333
                    * z
                    * (
                        -1.33333333333333 * z * (1.5 * z2 - 0.5)
                        + 1.8
                        * z
                        * (
                            1.75
                            * z
                            * (
                                1.66666666666667 * z * (1.5 * z2 - 0.5)
                                - 0.666666666666667 * z
                            )
                            - 1.125 * z2
                            + 0.375
                        )
                        + 0.533333333333333 * z
                    )
                    + 0.9375 * z2
                    - 0.3125
                )
                - 0.457142857142857 * z
            )
            - 0.954110901614325 * z2
            + 0.318036967204775,
            0.193851103820053
            * x
            * (
                3.2 * z * (1.5 - 7.5 * z2)
                - 2.51428571428571
                * z
                * (
                    2.25
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 9.375 * z2
                    - 1.875
                )
                + 2.14285714285714
                * z
                * (
                    -2.625
                    * z
                    * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                    + 2.16666666666667
                    * z
                    * (
                        -2.8 * z * (1.5 - 7.5 * z2)
                        + 2.2
                        * z
                        * (
                            2.25
                            * z
                            * (
                                2.33333333333333 * z * (1.5 - 7.5 * z2)
                                + 4.0 * z
                            )
                            + 9.375 * z2
                            - 1.875
                        )
                        - 4.8 * z
                    )
                    - 10.9375 * z2
                    + 2.1875
                )
                + 5.48571428571429 * z
            ),
            0.0231696385236779
            * (x2 - y2)
            * (
                -4.125 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                + 2.5
                * z
                * (
                    -4.8 * z * (52.5 * z2 - 7.5)
                    + 2.6
                    * z
                    * (
                        2.75 * z * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z)
                        - 91.875 * z2
                        + 13.125
                    )
                    + 48.0 * z
                )
                + 137.8125 * z2
                - 19.6875
            ),
            0.0028519853513317
            * x
            * (x2 - 3.0 * y2)
            * (
                -7.33333333333333 * z * (52.5 - 472.5 * z2)
                + 3.0
                * z
                * (
                    3.25
                    * z
                    * (3.66666666666667 * z * (52.5 - 472.5 * z2) + 280.0 * z)
                    + 1063.125 * z2
                    - 118.125
                )
                - 560.0 * z
            ),
            0.000368189725644507
            * (-6.0 * x2 * y2 + x4 + y4)
            * (
                3.75
                * z
                * (4.33333333333333 * z * (5197.5 * z2 - 472.5) - 3150.0 * z)
                - 14293.125 * z2
                + 1299.375
            ),
            5.10587282657803e-5
            * x
            * (5.0 * z * (5197.5 - 67567.5 * z2) + 41580.0 * z)
            * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
            7.87853281621404e-6
            * (1013512.5 * z2 - 67567.5)
            * (x2**3 + 15.0 * x2 * y4 - 15.0 * x4 * y2 - y2**3),
            -2.91570664069932
            * xz
            * (x2**3 + 35.0 * x2 * y4 - 21.0 * x4 * y2 - 7.0 * y2**3),
            -20.4099464848952 * x2**3 * y2
            - 20.4099464848952 * x2 * y2**3
            + 0.72892666017483 * x4**2
            + 51.0248662122381 * x4 * y4
            + 0.72892666017483 * y4**2,
        ],
        -1,
    )


def _dino_drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class _DinoDropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(_DinoDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _dino_drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def _make_2tuple(x):
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"Expected tuple of length 2, got length {len(x)}")
        return x
    if not isinstance(x, int):
        raise TypeError(f"Expected int or 2-tuple, got {type(x)}")
    return (x, x)


class _DinoPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_HW = _make_2tuple(img_size)
        patch_HW = _make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        if H % patch_H != 0:
            raise ValueError(
                f"Input image height {H} is not a multiple of patch height"
                f" {patch_H}"
            )
        if W % patch_W != 0:
            raise ValueError(
                f"Input image width {W} is not a multiple of patch width:"
                f" {patch_W}"
            )
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


class _DinoMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _DinoSwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU as _XformersSwiGLU

    _XFORMERS_SWIGLU_AVAILABLE = True
except ImportError:
    _XformersSwiGLU = _DinoSwiGLUFFN
    _XFORMERS_SWIGLU_AVAILABLE = False


class _DinoSwiGLUFFNFused(_XformersSwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


try:
    if os.environ.get("XFORMERS_DISABLED"):
        raise ImportError("xFormers disabled via XFORMERS_DISABLED env var")
    from xformers.ops import fmha, memory_efficient_attention, unbind

    _XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    _XFORMERS_AVAILABLE = False

_XFORMERS_AVAILABLE = _XFORMERS_AVAILABLE and torch.cuda.is_available()


class _DinoAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _DinoMemEffAttention(_DinoAttention):
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        if not _XFORMERS_AVAILABLE or x.device.type == "cpu":
            if attn_bias is not None:
                raise RuntimeError(
                    "xFormers is required for nested tensors usage"
                )
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _DinoBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=_DinoAttention,
        ffn_layer=_DinoMlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path1 = (
            _DinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = (
            _DinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = _drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def _drop_add_residual_stochastic_depth(
    x, residual_func, sample_drop_ratio=0.0
):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(
        x_flat,
        0,
        brange,
        residual.to(dtype=x.dtype),
        alpha=residual_scale_factor,
    )
    return x_plus_residual.view_as(x)


def _get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def _add_residual(
    x, brange, residual, residual_scale_factor, scaling_vector=None
):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(
            x_flat,
            0,
            brange,
            residual.to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
    else:
        from xformers.ops import scaled_index_add

        x_plus_residual = scaled_index_add(
            x,
            brange,
            residual.to(dtype=x.dtype),
            scaling=scaling_vector,
            alpha=residual_scale_factor,
        )
    return x_plus_residual


_attn_bias_cache = {}


def _get_attn_bias_and_cat(x_list, branges=None):
    batch_sizes = (
        [b.shape[0] for b in branges]
        if branges is not None
        else [x.shape[0] for x in x_list]
    )
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in _attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        _attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        from xformers.ops import index_select_cat

        cat_tensors = index_select_cat(
            [x.flatten(1) for x in x_list], branges
        ).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
    return _attn_bias_cache[all_shapes], cat_tensors


def _drop_add_residual_stochastic_depth_list(
    x_list, residual_func, sample_drop_ratio=0.0, scaling_vector=None
):
    branges_scales = [
        _get_branges_scales(x, sample_drop_ratio=sample_drop_ratio)
        for x in x_list
    ]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, x_cat = _get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))
    outputs = []
    for x, brange, residual, residual_scale_factor in zip(
        x_list, branges, residual_list, residual_scale_factors
    ):
        outputs.append(
            _add_residual(
                x, brange, residual, residual_scale_factor, scaling_vector
            ).view_as(x)
        )
    return outputs


class _DinoNestedTensorBlock(_DinoBlock):
    def forward_nested(self, x_list):
        if not isinstance(self.attn, _DinoMemEffAttention):
            raise TypeError("attn must be an instance of _DinoMemEffAttention")
        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x, attn_bias=None):
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x, attn_bias=None):
                return self.mlp(self.norm2(x))

            x_list = _drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls1.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            x_list = _drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(
                    self.ls2.gamma if isinstance(self.ls1, LayerScale) else None
                ),
            )
            return x_list
        else:

            def attn_residual_func(x, attn_bias=None):
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x, attn_bias=None):
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = _get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super(_DinoNestedTensorBlock, self).forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not _XFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Please install xFormers for nested tensors usage"
                )
            return self.forward_nested(x_or_x_list)
        else:
            raise TypeError(f"Unexpected input type: {type(x_or_x_list)}")


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        _named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class _BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


def _init_weights_vit_timm(module, name=""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=_DinoPatchEmbed,
        act_layer=nn.GELU,
        block_fn=_DinoNestedTensorBlock,
        ffn_layer="mlp",
        block_chunks=1,
        output_idx=[5, 12, 18, 24],
        checkpoint=False,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.0,
        use_norm=False,
        frozen_stages=0,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.embed_dims = [embed_dim] * output_idx[-1]
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.depths = output_idx
        self.checkpoint = checkpoint
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        if num_register_tokens < 0:
            raise ValueError(
                f"num_register_tokens must be >= 0, got {num_register_tokens}"
            )
        self.register_tokens = nn.Parameter(
            torch.zeros(1, max(1, num_register_tokens), embed_dim)
        )
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if ffn_layer == "mlp":
            ffn_layer = _DinoMlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = _DinoSwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList(
                [_BlockChunk(p) for p in chunked_blocks]
            )
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(embed_dim)
        self.use_norm = use_norm
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.num_register_tokens:
            nn.init.normal_(self.register_tokens, std=1e-6)
        _named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        if N != M * M:
            raise ValueError(f"N={N} must be a perfect square, got M={M}")
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        if (w0, h0) != patch_pos_embed.shape[-2:]:
            raise ValueError(
                f"Expected patch_pos_embed shape[-2:] == ({w0}, {h0}), got"
                f" {tuple(patch_pos_embed.shape[-2:])}"
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1
        ).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        with (
            torch.no_grad()
            if self.frozen_stages > -1
            else contextlib.nullcontext()
        ):
            x = self.patch_embed(x)
        if masks is not None:
            masks = masks.bool().view(B, -1, 1)
            x = torch.where(masks, self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.num_register_tokens:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward(self, x, masks=None):
        shapes = [val // self.patch_size for val in x.shape[-2:]]
        batch_size = x.shape[0]
        x = self.prepare_tokens_with_masks(x, masks)
        outputs = []
        for i, blk in enumerate(self.blocks):
            with (
                torch.no_grad()
                if i < self.frozen_stages
                else contextlib.nullcontext()
            ):
                x = blk(x)
            outputs.append(x)
        if self.use_norm:
            with (
                torch.no_grad()
                if self.frozen_stages >= len(self.blocks)
                else contextlib.nullcontext()
            ):
                outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, :1] for out in outputs]
        outputs = [out[:, self.num_register_tokens + 1 :] for out in outputs]
        outputs = [out.reshape(batch_size, *shapes, -1) for out in outputs]
        return (outputs, class_tokens)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_stages > -1:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        for i, blk in enumerate(self.blocks):
            if i < self.frozen_stages:
                blk.eval()
                for p in blk.parameters():
                    p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = (
                self.frozen_stages <= len(self.blocks) and self.use_norm
            )
        self.cls_token.requires_grad = self.frozen_stages < 1
        self.pos_embed.requires_grad = self.frozen_stages < 1
        self.mask_token.requires_grad = False
        self.register_tokens.requires_grad = False


def _make_dinov2_model(
    *,
    arch_name="vit_large",
    img_size=518,
    patch_size=14,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    pretrained="",
    output_idx=[],
    num_register_tokens=0,
    drop_path_rate=0.0,
    use_norm=False,
    export=False,
    interpolate_offset=0.0,
    frozen_stages=0,
    **kwargs,
):
    compact_arch_name = arch_name.replace("_", "")[:4]
    model_name = f"dinov2_{compact_arch_name}{patch_size}"
    _ARCH_MAP = {
        "vit_large": dict(
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
        ),
    }
    if arch_name not in _ARCH_MAP:
        raise ValueError(
            f"Unknown arch_name '{arch_name}', expected one of"
            f" {list(_ARCH_MAP)}"
        )
    vit_kwargs = dict(
        img_size=img_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        output_idx=output_idx,
        drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        use_norm=use_norm,
        export=export,
        interpolate_offset=interpolate_offset,
        frozen_stages=frozen_stages,
    )
    vit_kwargs.update(**_ARCH_MAP[arch_name])
    vit_kwargs.update(**kwargs)
    vit_kwargs.pop("export", None)
    block_fn = partial(
        _DinoNestedTensorBlock,
        attn_class=_DinoAttention if export else _DinoMemEffAttention,
    )
    model = DinoVisionTransformer(block_fn=block_fn, **vit_kwargs)
    if pretrained == "":
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}"
        if num_register_tokens > 0:
            url += "_reg4"
        url += "_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=False
        )
        info = model.load_state_dict(state_dict, strict=False)
        print(info)
    elif pretrained is not None:
        state_dict = torch.load(
            pretrained, map_location="cpu", weights_only=True
        )
        info = model.load_state_dict(state_dict, strict=False)
        print(f"loading from {pretrained} with:", info)
    else:
        print("Not loading pretrained weights for backbone")
    return model


def dinov2_vitl14(config, **kwargs):
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


class _UnidepthSwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class _UnidepthMLP(nn.Module):
    def __init__(
        self, input_dim, expansion=4, dropout=0.0, gated=False, output_dim=None
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = default(output_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else _UnidepthSwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x


class CvnxtBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        layer_scale=1.0,
        expansion=4,
        dilation=1,
        padding_mode="zeros",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            groups=dim,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones((dim)))
            if layer_scale > 0.0
            else 1.0
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = input + x.permute(0, 3, 1, 2)
        return x


class ConvUpsample(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers=2,
        expansion=4,
        layer_scale=1.0,
        kernel_size=7,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(
                hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class _UnidepthAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        expansion=4,
        dropout=0.0,
        cosine=False,
        gated=False,
        layer_scale=1.0,
        context_dim=None,
        use_bias=True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = _UnidepthMLP(
            dim, expansion=expansion, dropout=dropout, gated=gated
        )
        self.kv = nn.Linear(context_dim, dim * 2, bias=use_bias)
        self.q = nn.Linear(dim, dim, bias=use_bias)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=use_bias)
        self.ls1 = (
            LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        )

    def attn(
        self,
        x,
        attn_bias=None,
        context=None,
        pos_embed=None,
        pos_embed_context=None,
    ):
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context),
            "b n (kv h d) -> b h n d kv",
            h=self.num_heads,
            kv=2,
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        if pos_embed is not None:
            pos_embed = rearrange(
                pos_embed, "b n (h d) -> b h n d", h=self.num_heads
            )
            q = q + pos_embed
        if pos_embed_context is not None:
            pos_embed_context = rearrange(
                pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads
            )
            k = k + pos_embed_context
        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x,
        attn_bias=None,
        context=None,
        pos_embed=None,
        pos_embed_context=None,
    ):
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class _NystromAttention(nn.Module):
    def __init__(self, num_landmarks=64, num_heads=1, dropout=0.0, **kwargs):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v, key_padding_mask=None):
        q = rearrange(q, "b n h d -> b h n d")
        k = rearrange(k, "b n h d -> b h n d")
        v = rearrange(v, "b n h d -> b h n d")
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = rearrange(x, "b h n d -> b n h d")
        return x


class _NystromBlock(_UnidepthAttentionBlock):
    def __init__(
        self,
        dim,
        num_heads=4,
        expansion=4,
        dropout=0.0,
        cosine=False,
        gated=False,
        layer_scale=1.0,
        context_dim=None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = _NystromAttention(
            num_landmarks=128, num_heads=num_heads, dropout=dropout
        )

    def attn(
        self,
        x,
        attn_bias=None,
        context=None,
        pos_embed=None,
        pos_embed_context=None,
        rope=None,
    ):
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context),
            "b n (kv h d) -> b n h d kv",
            h=self.num_heads,
            kv=2,
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context
        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x


class _ListAdapter(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        for input_dim in input_dims:
            self.input_adapters.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, splits.int().tolist(), dim=-1)
        xs = [adapter(x) for x, adapter in zip(xs, self.input_adapters)]
        return torch.cat(xs, dim=-1)


class _CameraHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=8,
        expansion=4,
        depth=4,
        dropout=0.0,
        layer_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        self.aggregate = _UnidepthAttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, 4, hidden_dim), requires_grad=True
        )
        self.layers = nn.ModuleList([])
        self.in_features = _UnidepthMLP(
            hidden_dim, expansion=2, dropout=dropout
        )
        for _ in range(depth):
            blk = _UnidepthAttentionBlock(
                hidden_dim,
                num_heads=num_heads,
                expansion=expansion,
                dropout=dropout,
                layer_scale=layer_scale,
            )
            self.layers.append(blk)
        self.out = _UnidepthMLP(
            hidden_dim, expansion=2, dropout=0.0, output_dim=1
        )
        self.cls_project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, features, cls_tokens, pos_embed):
        features = features.unbind(dim=-1)
        cls_tokens = self.cls_project(cls_tokens)
        features_stack = torch.cat(features, dim=1)
        features_stack = features_stack + pos_embed
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        features_stack = self.in_features(features_stack)
        features = torch.cat((features_stack, cls_tokens), dim=1)
        cls_tokens = self.aggregate(
            cls_tokens, context=features, pos_embed=latents_pos
        )
        for i, layer in enumerate(self.layers):
            cls_tokens = layer(cls_tokens, pos_embed=latents_pos)
        x = self.out(cls_tokens).squeeze(-1)
        camera_intrinsics = torch.zeros(
            x.shape[0], 3, 3, device=x.device, requires_grad=False
        )
        camera_intrinsics[:, 0, 0] = x[:, 0].exp()
        camera_intrinsics[:, 1, 1] = x[:, 1].exp()
        camera_intrinsics[:, 0, 2] = x[:, 2].sigmoid()
        camera_intrinsics[:, 1, 2] = x[:, 3].sigmoid()
        camera_intrinsics[:, 2, 2] = 1.0
        return camera_intrinsics

    def set_shapes(self, shapes):
        self.shapes = shapes


class _DepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        expansion=4,
        depths=4,
        camera_dim=256,
        num_resolutions=4,
        dropout=0.0,
        layer_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(depths, int):
            depths = [depths] * 3
        if len(depths) != 3:
            raise ValueError(f"depths must have length 3, got {len(depths)}")
        self.project_rays16 = _UnidepthMLP(
            camera_dim,
            expansion=expansion,
            dropout=dropout,
            output_dim=hidden_dim,
        )
        self.project_rays8 = _UnidepthMLP(
            camera_dim,
            expansion=expansion,
            dropout=dropout,
            output_dim=hidden_dim // 2,
        )
        self.project_rays4 = _UnidepthMLP(
            camera_dim,
            expansion=expansion,
            dropout=dropout,
            output_dim=hidden_dim // 4,
        )
        self.to_latents = _UnidepthMLP(hidden_dim, expansion=2, dropout=dropout)
        self.features_channel_cat = nn.Linear(
            hidden_dim * num_resolutions, hidden_dim
        )
        self.up8 = ConvUpsample(
            hidden_dim, expansion=expansion, layer_scale=layer_scale
        )
        self.up4 = ConvUpsample(
            hidden_dim // 2, expansion=expansion, layer_scale=layer_scale
        )
        self.up2 = ConvUpsample(
            hidden_dim // 4, expansion=expansion, layer_scale=layer_scale
        )
        self.layers_16 = nn.ModuleList([])
        self.layers_8 = nn.ModuleList([])
        self.layers_4 = nn.ModuleList([])
        self.aggregate_16 = _UnidepthAttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            context_dim=hidden_dim,
        )
        self.prompt_camera = _UnidepthAttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            context_dim=hidden_dim,
        )
        for i, (blk_lst, depth) in enumerate(
            zip([self.layers_16, self.layers_8, self.layers_4], depths)
        ):
            attn_cls = _UnidepthAttentionBlock if i == 0 else _NystromBlock
            for _ in range(depth):
                blk_lst.append(
                    attn_cls(
                        hidden_dim // (2**i),
                        num_heads=num_heads // (2**i),
                        expansion=expansion,
                        dropout=dropout,
                        layer_scale=layer_scale,
                    )
                )
        self.out2 = nn.Conv2d(hidden_dim // 8, 1, 3, padding=1)
        self.out4 = nn.Conv2d(hidden_dim // 4, 1, 3, padding=1)
        self.out8 = nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)

    def set_original_shapes(self, shapes):
        self.original_shapes = shapes

    def set_shapes(self, shapes):
        self.shapes = shapes

    def forward(self, features, rays_hr, pos_embed, level_embed):
        features = features.unbind(dim=-1)
        shapes = self.shapes
        rays_hr = rays_hr.detach()
        rays_embedding_16 = F.normalize(
            flat_interpolate(rays_hr, old=self.original_shapes, new=shapes),
            dim=-1,
        )
        rays_embedding_8 = F.normalize(
            flat_interpolate(
                rays_hr,
                old=self.original_shapes,
                new=tuple(x * 2 for x in shapes),
            ),
            dim=-1,
        )
        rays_embedding_4 = F.normalize(
            flat_interpolate(
                rays_hr,
                old=self.original_shapes,
                new=tuple(x * 4 for x in shapes),
            ),
            dim=-1,
        )
        rays_embedding_16 = self.project_rays16(rsh_cart_8(rays_embedding_16))
        rays_embedding_8 = self.project_rays8(rsh_cart_8(rays_embedding_8))
        rays_embedding_4 = self.project_rays4(rsh_cart_8(rays_embedding_4))
        features_tokens = torch.cat(features, dim=1)
        features_tokens_pos = pos_embed + level_embed
        features_channels = torch.cat(features, dim=-1)
        features_16 = self.features_channel_cat(features_channels)
        latents_16 = self.to_latents(
            flat_interpolate(
                features_16, old=self.shapes, new=shapes, antialias=False
            )
        )
        latents_16 = self.aggregate_16(
            latents_16,
            context=features_tokens,
            pos_embed_context=features_tokens_pos,
        )
        latents_16 = self.prompt_camera(latents_16, context=rays_embedding_16)
        for layer in self.layers_16:
            latents_16 = layer(latents_16, pos_embed=rays_embedding_16)
        latents_8 = self.up8(
            rearrange(
                latents_16 + rays_embedding_16,
                "b (h w) c -> b c h w",
                h=shapes[0],
                w=shapes[1],
            ).contiguous()
        )
        out8 = self.out8(
            rearrange(
                latents_8,
                "b (h w) c -> b c h w",
                h=shapes[0] * 2,
                w=shapes[1] * 2,
            )
        )
        for layer in self.layers_8:
            latents_8 = layer(latents_8, pos_embed=rays_embedding_8)
        latents_4 = self.up4(
            rearrange(
                latents_8 + rays_embedding_8,
                "b (h w) c -> b c h w",
                h=shapes[0] * 2,
                w=shapes[1] * 2,
            ).contiguous()
        )
        out4 = self.out4(
            rearrange(
                latents_4,
                "b (h w) c -> b c h w",
                h=shapes[0] * 4,
                w=shapes[1] * 4,
            )
        )
        for layer in self.layers_4:
            latents_4 = layer(latents_4, pos_embed=rays_embedding_4)
        latents_2 = self.up2(
            rearrange(
                latents_4 + rays_embedding_4,
                "b (h w) c -> b c h w",
                h=shapes[0] * 4,
                w=shapes[1] * 4,
            ).contiguous()
        )
        out2 = self.out2(
            rearrange(
                latents_2,
                "b (h w) c -> b c h w",
                h=shapes[0] * 8,
                w=shapes[1] * 8,
            )
        )
        proj_latents_16 = rearrange(
            latents_16, "b (h w) c -> b c h w", h=shapes[0], w=shapes[1]
        ).contiguous()
        out2 = out2.clamp(-10.0, 10.0).exp()
        out4 = out4.clamp(-10.0, 10.0).exp()
        out8 = out8.clamp(-10.0, 10.0).exp()
        return out8, out4, out2, proj_latents_16


class _Decoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
        self.test_fixed_camera = False
        self.skip_camera = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_adapted_features(self, features_flat, splits):
        features_flat_cat = torch.cat(features_flat, dim=-1)
        features_projected = self.input_adapter(features_flat_cat, splits)
        features = torch.chunk(features_projected, len(splits), dim=-1)
        return features

    def run_camera(
        self, cls_tokens, features, pos_embed, original_shapes, rays
    ):
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in cls_tokens],
            device=features.device,
            requires_grad=False,
            dtype=features.dtype,
        )
        cls_tokens = torch.cat(cls_tokens, dim=-1)
        cls_tokens = self.token_adapter(cls_tokens, cls_tokens_splits)
        cls_tokens = torch.cat(
            torch.chunk(cls_tokens, len(cls_tokens_splits), dim=-1), dim=1
        )
        intrinsics = self.camera_layer(
            features=features, cls_tokens=cls_tokens, pos_embed=pos_embed
        )
        intrinsics[:, 0, 0] = max(original_shapes) / 2 * intrinsics[:, 0, 0]
        intrinsics[:, 1, 1] = max(original_shapes) / 2 * intrinsics[:, 1, 1]
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * original_shapes[1]
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * original_shapes[0]
        if not self.test_fixed_camera:
            rays, _ = generate_rays(intrinsics, original_shapes, noisy=False)
        return intrinsics, rays

    def forward(self, inputs, image_metas):
        B, _, H, W = inputs["image"].shape
        device = inputs["image"].device
        original_encoder_outputs = [
            x.contiguous() for x in inputs["encoder_outputs"]
        ]
        cls_tokens = [x.contiguous() for x in inputs["cls_tokens"]]
        original_encoder_outputs = [
            max_stack(original_encoder_outputs[i:j])
            for i, j in self.slices_encoder_range
        ]
        cls_tokens = [
            cls_tokens[-i - 1].detach()
            for i in range(len(self.slices_encoder_range))
        ]
        resolutions = [
            tuple(sorted([x.shape[1], x.shape[2]]))
            for x in original_encoder_outputs
        ]
        level_shapes = sorted(list(set(resolutions)))[::-1]
        if len(level_shapes) == 1:
            level_shapes = level_shapes * self.num_resolutions
        input_shapes = [
            level_shapes[i]
            for i, (start, end) in enumerate(self.slices_encoder)
            for _ in range(end - start)
        ]
        common_shape = level_shapes[-2]
        features_flat = [
            flat_interpolate(
                rearrange(x, "b h w c -> b (h w) c"),
                old=input_shape,
                new=common_shape,
            )
            for x, input_shape in zip(original_encoder_outputs, input_shapes)
        ]
        features_splits = torch.tensor(
            [x.shape[-1] for x in features_flat],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )
        features = self.get_adapted_features(features_flat, features_splits)
        features = torch.stack(features, dim=-1)
        level_embed = torch.cat(
            [
                self.level_embed_layer(self.level_embeds)[i : i + 1]
                .unsqueeze(0)
                .repeat(B, common_shape[0] * common_shape[1], 1)
                for i in range(self.num_resolutions)
            ],
            dim=1,
        )
        pos_embed = self.pos_embed(
            torch.zeros(
                B,
                1,
                common_shape[0],
                common_shape[1],
                device=device,
                requires_grad=False,
            )
        )
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )
        self.camera_layer.set_shapes(common_shape)
        intrinsics, rays = (
            self.run_camera(
                cls_tokens,
                features=features,
                pos_embed=pos_embed + level_embed,
                original_shapes=(H, W),
                rays=inputs.get("rays", None),
            )
            if not self.skip_camera
            else (inputs["K"], inputs["rays"])
        )
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))
        out8, out4, out2, depth_features = self.depth_layer(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )
        return intrinsics, [out8, out4, out2], depth_features

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def build(self, config):
        depth = config["model"]["pixel_decoder"]["depths"]
        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        num_heads = config["model"]["num_heads"]
        expansion = config["model"]["expansion"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        layer_scale = 1.0
        self.depth = depth
        self.dim = hidden_dim
        self.downsample = 4
        self.num_heads = num_heads
        self.num_resolutions = len(depths_encoder)
        self.depths_encoder = depths_encoder
        self.slices_encoder_single = list(
            zip([d - 1 for d in self.depths_encoder], self.depths_encoder)
        )
        self.slices_encoder_range = list(
            zip([0, *self.depths_encoder[:-1]], self.depths_encoder)
        )
        cls_token_input_dims = [
            input_dims[-i - 1] for i in range(len(depths_encoder))
        ]
        input_dims = [input_dims[d - 1] for d in depths_encoder]
        self.slices_encoder = self.slices_encoder_single
        self.input_adapter = _ListAdapter(input_dims, hidden_dim)
        self.token_adapter = _ListAdapter(cls_token_input_dims, hidden_dim)
        self.camera_layer = _CameraHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depth=2,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.depth_layer = _DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=81,
            num_resolutions=self.num_resolutions,
            layer_scale=layer_scale,
        )
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            torch.randn(len(input_dims), hidden_dim), requires_grad=True
        )
        self.level_embed_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )


def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    pad_left, pad_right, pad_top, pad_bottom = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(
        rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant"
    )
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    pad_left, pad_right, pad_top, pad_bottom = pads
    predictions = sum(
        [
            F.interpolate(
                x.clone(),
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)
    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]
    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio
    return predictions, intrinsics


try:
    from huggingface_hub import PyTorchModelHubMixin as _PyTorchModelHubMixin
except ImportError:
    _PyTorchModelHubMixin = object


class UniDepthV1(
    nn.Module,
    _PyTorchModelHubMixin,
    library_name="UniDepth",
    repo_url="https://github.com/lpiccinelli-eth/UniDepth",
    tags=["monocular-metric-depth-estimation"],
):
    def __init__(self, config, eps=1e-6, **kwargs):
        super().__init__()
        self.build(config)
        self.eps = eps

    @torch.no_grad()
    def infer(self, rgbs, intrinsics=None, skip_camera=False):
        if rgbs.ndim == 3:
            rgbs = rgbs.unsqueeze(0)
        if intrinsics is not None and intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        B, _, H, W = rgbs.shape
        rgbs = rgbs.to(self.device)
        if intrinsics is not None:
            intrinsics = intrinsics.to(self.device)
        if rgbs.max() > 5 or rgbs.dtype == torch.uint8:
            rgbs = rgbs.to(torch.float32).div(255)
        if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
            rgbs = TF.normalize(
                rgbs, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD
            )
        (h, w), ratio = _shapes((H, W), self.image_shape)
        pad_left, pad_right, pad_top, pad_bottom = _paddings(
            (h, w), self.image_shape
        )
        rgbs, gt_intrinsics = _preprocess(
            rgbs,
            intrinsics,
            (h, w),
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            self.image_shape,
        )
        encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
        if "dino" in self.pixel_encoder.__class__.__name__.lower():
            encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(encoder_outputs, cls_tokens)
            ]
        inputs = {}
        inputs["encoder_outputs"] = encoder_outputs
        inputs["cls_tokens"] = cls_tokens
        inputs["image"] = rgbs
        if gt_intrinsics is not None:
            rays, angles = generate_rays(
                gt_intrinsics, self.image_shape, noisy=self.training
            )
            inputs["rays"] = rays
            inputs["angles"] = angles
            inputs["K"] = gt_intrinsics
            self.pixel_decoder.test_fixed_camera = True
            self.pixel_decoder.skip_camera = skip_camera
        pred_intrinsics, predictions, _ = self.pixel_decoder(inputs, {})
        predictions, pred_intrinsics = _postprocess(
            predictions,
            pred_intrinsics,
            self.image_shape,
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            (H, W),
        )
        intrinsics = (
            gt_intrinsics if gt_intrinsics is not None else pred_intrinsics
        )
        angles = generate_rays(intrinsics, (H, W), noisy=False)[-1]
        angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
        points_3d = torch.cat((angles, predictions), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        outputs = {
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": predictions[:, -1:],
        }
        self.pixel_decoder.test_fixed_camera = False
        self.pixel_decoder.skip_camera = False
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        pixel_encoder_config = {
            **config.get("training", {}),
            **config.get("data", {}),
            **config["model"]["pixel_encoder"],
            "interpolate_offset": 0.1,
        }
        pixel_encoder = dinov2_vitl14(pixel_encoder_config)
        config["model"]["pixel_encoder"]["patch_size"] = 14
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"][
            "embed_dims"
        ] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths
        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = _Decoder(config)
        self.image_shape = tuple(config["data"]["image_shape"])


_UNIDEPTH_V1_VITL14_CONFIG = {
    "generic": {"seed": 13},
    "training": {},
    "data": {"image_shape": [462, 616]},
    "model": {
        "name": "UniDepthV1",
        "num_heads": 8,
        "expansion": 4,
        "pixel_decoder": {
            "hidden_dim": 512,
            "depths": [3, 2, 1],
            "dropout": 0.0,
        },
        "pixel_encoder": {"name": "dinov2_vitl14", "pretrained": None},
    },
}


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def upsample(x, mode="nearest"):
    return F.interpolate(x, scale_factor=2, mode=mode)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    def __init__(self, height, width, shift_rays_half_pixel=0):
        super(BackprojectDepth, self).__init__()
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing="xy"
        )
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords)
        ones = torch.ones(1, 1, self.height * self.width)
        pix_coords = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0
        )
        pix_coords = torch.cat([pix_coords + shift_rays_half_pixel, ones], 1)
        self.register_buffer("pix_coords", pix_coords)
        self.register_buffer("id_coords", id_coords)
        self.register_buffer("ones", ones)

    def forward(self, depth, inv_K):
        batch_size = depth.shape[0]
        pixel_coords = self.pix_coords.expand(batch_size, -1, -1).to(
            depth.device
        )
        ones = self.ones.expand(batch_size, -1, -1).to(depth.device)
        cam_points = torch.matmul(inv_K[:, :3, :3], pixel_coords)
        cam_points = depth.view(batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, ones], 1)
        return cam_points


RESNETS = {
    18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, bn_order, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.bn_order = bn_order
        if num_layers not in RESNETS:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers)
            )
        if num_input_images > 1:
            self.encoder = ResNetMultiImageInput(
                {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[
                    num_layers
                ],
                {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers],
                num_input_images=num_input_images,
            )
        else:
            model, weights = RESNETS[num_layers]
            self.encoder = model(weights=weights)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        encoder = self.encoder
        features = []
        x = (input_image - 0.45) / 0.225
        x = encoder.conv1(x)
        if self.bn_order == "pre_bn":
            features.append(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
        elif self.bn_order == "monodepth":
            x = encoder.bn1(x)
            x = encoder.relu(x)
            features.append(x)
        else:
            raise RuntimeError("Unexpected num_input_images configuration")
        features.append(encoder.layer1(encoder.maxpool(x)))
        features.append(encoder.layer2(features[-1]))
        features.append(encoder.layer3(features[-1]))
        features.append(encoder.layer4(features[-1]))
        return features


def get_splits_and_inits(cfg):
    split_dimensions = []
    scale_inits = []
    bias_inits = []
    for g_idx in range(cfg.model.gaussians_per_pixel):
        if cfg.model.predict_offset:
            split_dimensions += [3]
            scale_inits += [cfg.model.xyz_scale]
            bias_inits += [cfg.model.xyz_bias]
        split_dimensions += [1, 3, 4, 3]
        scale_inits += [
            cfg.model.opacity_scale,
            cfg.model.scale_scale,
            1.0,
            5.0,
        ]
        bias_inits += [
            cfg.model.opacity_bias,
            np.log(cfg.model.scale_bias),
            0.0,
            0.0,
        ]
        if cfg.model.max_sh_degree != 0:
            sh_num = (cfg.model.max_sh_degree + 1) ** 2 - 1
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(cfg.model.sh_scale)
            bias_inits.append(0.0)
        if not cfg.model.one_gauss_decoder:
            break
    return split_dimensions, scale_inits, bias_inits


class GaussianDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_lambda = cfg.model.scale_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, split_dimensions=[3, 1, 3, 4, 3, 9]):
        outputs = x.split(split_dimensions, dim=1)
        offset_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        feat_dc_list = []
        feat_rest_list = []
        for i in range(self.cfg.model.gaussians_per_pixel):
            if (
                self.cfg.model.predict_offset
                and self.cfg.model.max_sh_degree != 0
            ):
                (
                    offset_s,
                    opacity_s,
                    scaling_s,
                    rotation_s,
                    feat_dc_s,
                    features_rest_s,
                ) = outputs[i * 6 : (i + 1) * 6]
                offset_list.append(offset_s[:, None, ...])
                feat_rest_list.append(features_rest_s[:, None, ...])
            elif self.cfg.model.predict_offset:
                offset_s, opacity_s, scaling_s, rotation_s, feat_dc_s = outputs[
                    i * 5 : (i + 1) * 5
                ]
                offset_list.append(offset_s[:, None, ...])
            elif self.cfg.model.max_sh_degree != 0:
                opacity_s, scaling_s, rotation_s, feat_dc_s, features_rest_s = (
                    outputs[i * 5 : (i + 1) * 5]
                )
                feat_rest_list.append(features_rest_s[:, None, ...])
            else:
                opacity_s, scaling_s, rotation_s, feat_dc_s = outputs[
                    i * 4 : (i + 1) * 4
                ]
            opacity_list.append(opacity_s[:, None, ...])
            scaling_list.append(scaling_s[:, None, ...])
            rotation_list.append(rotation_s[:, None, ...])
            feat_dc_list.append(feat_dc_s[:, None, ...])
            if not self.cfg.model.one_gauss_decoder:
                break
        opacity = torch.cat(opacity_list, dim=1)
        scaling = torch.cat(scaling_list, dim=1)
        rotation = torch.cat(rotation_list, dim=1)
        feat_dc = torch.cat(feat_dc_list, dim=1)
        out = {
            "gauss_opacity": self.opacity_activation(opacity),
            "gauss_scaling": (
                self.scaling_activation(scaling) * self.scaling_lambda
            ),
            "gauss_rotation": self.rotation_activation(rotation, dim=-3),
            "gauss_features_dc": feat_dc,
        }
        if self.cfg.model.predict_offset:
            offset = torch.cat(offset_list, dim=1)
            out["gauss_offset"] = offset
        if self.cfg.model.max_sh_degree != 0:
            features_rest = torch.cat(feat_rest_list, dim=1)
            out["gauss_features_rest"] = features_rest
        return out


class ResnetDecoder(nn.Module):
    def __init__(self, cfg, num_ch_enc, use_skips=True):
        super().__init__()
        self.cfg = cfg
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(cfg.model.backbone.num_ch_dec)
        self.split_dimensions, scales, biases = get_splits_and_inits(cfg)
        self.num_output_channels = sum(self.split_dimensions)
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = (
                self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.out = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, 1)
        start_channel = 0
        for out_channel, scale, bias in zip(
            self.split_dimensions, scales, biases
        ):
            nn.init.xavier_uniform_(
                self.out.weight[
                    start_channel : start_channel + out_channel, :, :, :
                ],
                scale,
            )
            nn.init.constant_(
                self.out.bias[start_channel : start_channel + out_channel], bias
            )
            start_channel += out_channel
        self.gaussian_decoder = GaussianDecoder(cfg)

    def forward(self, input_features):
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.cfg.model.backbone.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)
        x = self.out(x)
        out = self.gaussian_decoder(x, self.split_dimensions)
        return out


class ResnetDepthDecoder(nn.Module):
    def __init__(self, cfg, num_ch_enc, use_skips=True):
        super().__init__()
        self.cfg = cfg
        self.scales = cfg.model.scales
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_output_channels = (
            cfg.model.gaussians_per_pixel - 1
            if "unidepth" in cfg.model.name
            else cfg.model.gaussians_per_pixel
        )
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = (
                self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            )
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            out = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("outconv", s)] = out
            nn.init.xavier_uniform_(out.conv.weight, cfg.model.depth_scale)
            nn.init.constant_(out.conv.bias, cfg.model.depth_bias)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        if cfg.model.depth_type in ["disp", "disp_inc"]:
            self.activate = nn.Sigmoid()
        elif cfg.model.depth_type == "depth":
            self.activate = nn.Softplus()
        elif cfg.model.depth_type == "depth_inc":
            self.activate = torch.exp

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x, mode=self.cfg.model.backbone.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                output = self.convs[("outconv", i)](x)
                if self.cfg.model.depth_type == "depth_inc":
                    output = torch.clamp(output, min=-10.0, max=6.0)
                output = rearrange(
                    self.activate(output),
                    "b (n c) ... -> (b n) c ...",
                    n=self.num_output_channels,
                )
                if self.cfg.model.depth_type in ["disp", "disp_inc"]:
                    output = disp_to_depth(
                        output,
                        self.cfg.model.min_depth,
                        self.cfg.model.max_depth,
                    )
                outputs[("depth", i)] = output
        return outputs


class UniDepthExtended(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        unidepth_config = deepcopy(_UNIDEPTH_V1_VITL14_CONFIG)
        _ckpt_path = getattr(
            cfg.model.depth, "pretrained_checkpoint_path", None
        )
        if not _ckpt_path:
            unidepth_config["model"]["pixel_encoder"]["pretrained"] = None
            self.unidepth = UniDepthV1(unidepth_config)
            import huggingface_hub

            path = huggingface_hub.hf_hub_download(
                repo_id="lpiccinelli/unidepth-v1-vitl14",
                filename="pytorch_model.bin",
                repo_type="model",
                revision="main",
            )
            print(path)
            info = self.unidepth.load_state_dict(
                torch.load(path, weights_only=True), strict=False
            )
            print("UniDepth_v1_vitl14 is loaded with:")
            print(f"\t missing keys: {info.missing_keys}")
            print(f"\t additional keys: {info.unexpected_keys}")
        else:
            unidepth_config["model"]["pixel_encoder"]["pretrained"] = None
            self.unidepth = UniDepthV1(unidepth_config)
            import lgtm.storage as storage

            pretrained_ckpt_path = storage.resolve_path(_ckpt_path)
            print(f"Loading UniDepth weights from: {pretrained_ckpt_path}")
            ckpt_weights = torch.load(
                pretrained_ckpt_path, map_location="cpu", weights_only=True
            )
            info = self.unidepth.load_state_dict(ckpt_weights, strict=False)
            print(
                f"UniDepth loaded: missing={len(info.missing_keys)},"
                f" unexpected={len(info.unexpected_keys)}"
            )

        self.parameters_to_train = []
        if cfg.model.backbone.name == "resnet":
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,
            )
            if cfg.model.backbone.depth_cond:
                self.encoder.encoder.conv1 = nn.Conv2d(
                    4,
                    self.encoder.encoder.conv1.out_channels,
                    kernel_size=self.encoder.encoder.conv1.kernel_size,
                    padding=self.encoder.encoder.conv1.padding,
                    stride=self.encoder.encoder.conv1.stride,
                )
            self.parameters_to_train += [{"params": self.encoder.parameters()}]
            models_dict = {}
            if cfg.model.gaussians_per_pixel > 1:
                models_dict["depth"] = ResnetDepthDecoder(
                    cfg=cfg, num_ch_enc=self.encoder.num_ch_enc
                )
                self.parameters_to_train += [
                    {"params": models_dict["depth"].parameters()}
                ]
            for i in range(cfg.model.gaussians_per_pixel):
                models_dict["gauss_decoder_" + str(i)] = ResnetDecoder(
                    cfg=cfg, num_ch_enc=self.encoder.num_ch_enc
                )
                self.parameters_to_train += [
                    {
                        "params": (
                            models_dict["gauss_decoder_" + str(i)].parameters()
                        )
                    }
                ]
                if cfg.model.one_gauss_decoder:
                    break
            self.models = nn.ModuleDict(models_dict)

    def get_parameter_groups(self):
        return self.parameters_to_train

    def forward(self, inputs):
        if ("unidepth", 0, 0) in inputs.keys() and inputs[
            ("unidepth", 0, 0)
        ] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[("unidepth", 0, 0)]
        else:
            with torch.no_grad():
                intrinsics = (
                    inputs[("K_src", 0)]
                    if ("K_src", 0) in inputs.keys()
                    else None
                )
                depth_outs = self.unidepth.infer(
                    inputs["color_aug", 0, 0], intrinsics=intrinsics
                )
        outputs_gauss = {}
        outputs_gauss[("K_src", 0)] = (
            inputs[("K_src", 0)]
            if ("K_src", 0) in inputs.keys()
            else depth_outs["intrinsics"]
        )
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(
            outputs_gauss[("K_src", 0)]
        )
        if self.cfg.model.backbone.depth_cond:
            input = torch.cat(
                [inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1
            )
        else:
            input = inputs["color_aug", 0, 0]
        encoded_features = self.encoder(input)
        outputs_gauss["encoder_features"] = encoded_features
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features)
            depth[("depth", 0)] = rearrange(
                depth[("depth", 0)],
                "(b n) ... -> b n ...",
                n=self.cfg.model.gaussians_per_pixel - 1,
            )
            depth[("depth", 0)] = torch.cumsum(
                torch.cat(
                    (depth_outs["depth"][:, None, ...], depth[("depth", 0)]),
                    dim=1,
                ),
                dim=1,
            )
            outputs_gauss[("depth", 0)] = rearrange(
                depth[("depth", 0)],
                "b n c ... -> (b n) c ...",
                n=self.cfg.model.gaussians_per_pixel,
            )
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]
        gauss_outs = dict()
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models["gauss_decoder_" + str(i)](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    gauss_outs[key] = (
                        outs[key]
                        if i == 0
                        else torch.cat([gauss_outs[key], outs[key]], dim=1)
                    )
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(gauss_outs[key], "b n ... -> (b n) ...")
        outputs_gauss |= gauss_outs
        return outputs_gauss


class GaussianPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.dataset.width % 32 != 0 or cfg.dataset.height % 32 != 0:
            raise ValueError("'width' and 'height' must be a multiple of 32")
        models_dict = {}
        self.parameters_to_train = []
        if "unidepth" in cfg.model.name:
            models_dict["unidepth_extended"] = UniDepthExtended(cfg)
            self.parameters_to_train += models_dict[
                "unidepth_extended"
            ].get_parameter_groups()
        self.models = nn.ModuleDict(models_dict)
        self.set_backproject()

    def set_backproject(self):
        cfg = self.cfg
        backproject_depth = {}
        H = cfg.dataset.height
        W = cfg.dataset.width
        for scale in cfg.model.scales:
            h = H // (2**scale)
            w = W // (2**scale)
            if cfg.model.shift_rays_half_pixel == "zero":
                shift_rays_half_pixel = 0
            elif cfg.model.shift_rays_half_pixel == "forward":
                shift_rays_half_pixel = 0.5
            elif cfg.model.shift_rays_half_pixel == "backward":
                shift_rays_half_pixel = -0.5
            else:
                raise NotImplementedError
            backproject_depth[str(scale)] = BackprojectDepth(
                h + 2 * self.cfg.dataset.pad_border_aug,
                w + 2 * self.cfg.dataset.pad_border_aug,
                shift_rays_half_pixel=shift_rays_half_pixel,
            )
        self.backproject_depth = nn.ModuleDict(backproject_depth)

    def train(self, mode=True):
        for m in self.models.values():
            m.train(mode)
        self._is_train = True

    def eval(self):
        for m in self.models.values():
            m.eval()
        self._is_train = False

    def forward(self, inputs):
        cfg = self.cfg
        if "unidepth" in cfg.model.name:
            outputs = self.models["unidepth_extended"](inputs)
        self.compute_gauss_means(inputs, outputs)
        return outputs

    def compute_gauss_means(self, inputs, outputs):
        cfg = self.cfg
        scale = self.cfg.model.scales[0]
        depth = outputs[("depth", scale)]
        B, _, H, W = depth.shape
        inv_K = outputs[("inv_K_src", scale)]
        if self.cfg.model.gaussians_per_pixel > 1:
            inv_K = rearrange(
                inv_K[:, None, ...].repeat(
                    1, self.cfg.model.gaussians_per_pixel, 1, 1
                ),
                "b n ... -> (b n) ...",
            )
        xyz = self.backproject_depth[str(scale)](depth, inv_K)
        if cfg.model.predict_offset:
            offset = outputs["gauss_offset"]
            if cfg.model.scaled_offset:
                offset = offset * depth.detach()
            offset = offset.view(B, 3, -1)
            zeros = torch.zeros(B, 1, H * W, device=depth.device)
            offset = torch.cat([offset, zeros], 1)
            xyz = xyz + offset
        inputs[("inv_K_src", scale)] = inv_K
        outputs["gauss_means"] = xyz

    def load_model(
        self, weights_path, optimiser=None, device="cpu", ckpt_ids=0
    ):
        """load model(s) from disk"""
        weights_path = Path(weights_path)

        if weights_path.is_dir():
            ckpts = sorted(list(weights_path.glob("model_*.pth")), reverse=True)
            if len(ckpts) > 0:
                weights_path = ckpts[ckpt_ids]
            else:
                ckpts = sorted(list(weights_path.glob("*.ckpt")), reverse=True)
                weights_path = ckpts[ckpt_ids]
        logging.info(f"Loading weights from {weights_path}...")
        state_dict = torch.load(
            weights_path, map_location=torch.device(device), weights_only=False
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            new_state_dict = {"model": {}}
            for key, value in state_dict.items():
                if key.startswith("encoder.gaussian_predictor."):
                    new_state_dict["model"][
                        key[len("encoder.gaussian_predictor.") :]
                    ] = value
            state_dict = new_state_dict
        new_dict = {}
        for k, v in state_dict["model"].items():
            if "backproject_depth" in k:
                new_dict[k] = self.state_dict()[k].clone()
            else:
                new_dict[k] = v.clone()
        missing_keys, unexpected_keys = self.load_state_dict(
            new_dict, strict=False
        )
        print(f"- len(missing_keys): {len(missing_keys)}")
        print(f"- len(unexpected_keys): {len(unexpected_keys)}")
