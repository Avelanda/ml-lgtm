"""
Test-time camera pose optimization for evaluation.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

Generic pose optimization interface for camera pose alignment.

This module provides a unified interface for optimizing camera poses
during testing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from lgtm.dataset.data_types import BatchedExample
from lgtm.model.gaussian.gaussian_rasterizer import GaussianRasterizer
from lgtm.model.gaussian.gaussians import Gaussians
from lgtm.utils.camera_utils import CameraOptModule
from lgtm.utils.logging_utils import get_exp_output_dir


@dataclass
class PoseOptConfig:
    """
    Configuration for pose optimization.
    """

    # Core optimization parameters.
    num_steps: int = 100
    learning_rate: float = 0.005

    # Backend override for optimization (if different from rasterizer.backend)
    backend_override: Optional[str] = None

    # Debugging/logging options.
    print_steps: bool = False
    save_losses: bool = False

    # Loss selection for pose optimization.
    # Must be a subset of the root-level "loss" configuration.
    enabled_losses: list[str] = field(default_factory=lambda: ["mse", "lpips"])


def optimize_camera_poses(
    gaussians: Gaussians,
    batch: BatchedExample,
    rasterizer: GaussianRasterizer,
    losses: nn.ModuleList,
    pose_opt_cfg: PoseOptConfig,
    global_step: int,
) -> Float[Tensor, "batch view 4 4"]:
    """
    Generic camera pose optimization function.

    Optimizes camera poses by minimizing the rendering loss between
    predicted and target images.

    Args:
        gaussians: 3D Gaussian representation to render.
        batch: BatchedExample containing target images, poses
            (c2w 4x4), intrinsics (normalized 3x3), near/far.
        rasterizer: GaussianRasterizer for rendering.
        losses: Loss functions (e.g., MSE + LPIPS).
        pose_opt_cfg: Optimization configuration.
        global_step: Current training step (for loss scheduling).

    Returns:
        Optimized c2w 4x4 poses [batch, view, 4, 4].

    Note:
        - Uses batch["target"]["poses"] as initial c2w poses
        - Optimizes at context image resolution
        - Caller is responsible for final rendering at desired
          resolution
    """
    optimized_poses = _optimize_poses_gsplat(
        gaussians, batch, rasterizer, losses, pose_opt_cfg, global_step
    )

    # Handle pose alignment step saving (if enabled)
    if pose_opt_cfg.save_losses:

        # Dump inputs for reproducibility.
        bs = batch["target"]["poses"].shape[0]
        dump_input_path = (
            Path("play")
            / "data"
            / f"dump_test_step_align_{rasterizer.backend}_bs{bs}.pt"
        )
        if not dump_input_path.exists():
            input_dump = {
                "batch": {
                    "scene": batch["scene"],
                    "target": {
                        "image": batch["target"]["image"].detach().cpu(),
                        "poses": batch["target"]["poses"].detach().cpu(),
                        "intrinsics": (
                            batch["target"]["intrinsics"].detach().cpu()
                        ),
                        "near": batch["target"]["near"].detach().cpu(),
                        "far": batch["target"]["far"].detach().cpu(),
                        "index": (
                            batch["target"]["index"].detach().cpu()
                            if isinstance(
                                batch["target"]["index"], torch.Tensor
                            )
                            else batch["target"]["index"]
                        ),
                    },
                },
                "gaussians": {
                    "means": gaussians.means.detach().cpu(),
                    "covariances": gaussians.covariances.detach().cpu(),
                    "harmonics": gaussians.harmonics.detach().cpu(),
                    "opacities": gaussians.opacities.detach().cpu(),
                    "scales": (
                        gaussians.scales.detach().cpu()
                        if gaussians.scales is not None
                        else None
                    ),
                    "xyzws": (
                        gaussians.xyzws.detach().cpu()
                        if gaussians.xyzws is not None
                        else None
                    ),
                    "texture_colors": (
                        gaussians.texture_colors.detach().cpu()
                        if gaussians.texture_colors is not None
                        else None
                    ),
                    "texture_alphas": (
                        gaussians.texture_alphas.detach().cpu()
                        if gaussians.texture_alphas is not None
                        else None
                    ),
                },
                "config": {
                    "num_steps": pose_opt_cfg.num_steps,
                    "learning_rate": pose_opt_cfg.learning_rate,
                    "backend_override": pose_opt_cfg.backend_override,
                    "print_steps": pose_opt_cfg.print_steps,
                    "save_losses": pose_opt_cfg.save_losses,
                },
                "rasterizer_cfg": {
                    "backend": rasterizer.backend,
                },
            }
            dump_input_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(input_dump, dump_input_path)
            print(
                f"Saved test_step_align_gsplat input dump to {dump_input_path}"
            )

        pose_align_dict = {
            "scene": batch["scene"][0],
            "pose_align_steps": pose_opt_cfg.num_steps,
            "learning_rate": pose_opt_cfg.learning_rate,
            "backend": rasterizer.backend,
            "backend_override": pose_opt_cfg.backend_override,
            "unified_interface": True,  # Mark that this used the new interface
        }
        scene_name = batch["scene"][0]
        pose_align_path = (
            get_exp_output_dir() / "pose_align_steps" / f"{scene_name}.json"
        )
        pose_align_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_align_path, "w") as f:
            json.dump(pose_align_dict, f, indent=2)

    return optimized_poses


def _optimize_poses_gsplat(
    gaussians: Gaussians,
    batch: BatchedExample,
    rasterizer: GaussianRasterizer,
    losses: nn.ModuleList,
    pose_opt_config: PoseOptConfig,
    global_step: int,
) -> Float[Tensor, "batch view 4 4"]:
    """
    GSplat backend: CameraOptModule optimization.
    """

    b, v, _, target_h, target_w = batch["target"]["image"].shape
    _, _, _, context_h, context_w = batch["context"]["image"].shape
    device = batch["target"]["poses"].device

    if not target_h == context_h or not target_w == context_w:
        raise ValueError(
            "Target and context image shapes must match: "
            f"{target_h}x{target_w} vs {context_h}x{context_w}"
        )

    with torch.set_grad_enabled(True):
        # Gsplat with pose_opt.
        pose_adjust = CameraOptModule(n=v).to(device)
        pose_adjust.zero_init()
        pose_optimizer = torch.optim.Adam(
            pose_adjust.parameters(),
            lr=pose_opt_config.learning_rate,
        )

        # Poses: c2w.
        # shape: (b, v, 4, 4), e.g., (1, 3, 4, 4)
        poses = batch["target"]["poses"].clone()

        # image_ids are constant indices [0, 1, .., v-1] for gsplat-based camera optimization.
        image_ids = torch.arange(v, device=device)

        total_losses = []
        for i in range(pose_opt_config.num_steps):
            # Do not modify the value of poses.
            new_poses = rearrange(poses, "b v i j -> (b v) i j")
            new_poses = pose_adjust(
                camtoworlds=new_poses,
                embed_ids=image_ids,
            )
            new_poses = rearrange(new_poses, "(b v) i j -> b v i j", b=b, v=v)

            # Set backend override.
            original_backend = None
            if pose_opt_config.backend_override is not None:
                original_backend = rasterizer.backend
                rasterizer.backend = pose_opt_config.backend_override

            output = rasterizer.forward(
                gaussians,
                new_poses,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (context_h, context_w),
                cam_rot_delta=None,
                cam_trans_delta=None,
            )

            # Restore backend.
            if original_backend is not None:
                rasterizer.backend = original_backend

            # Compute and log loss.
            total_loss = torch.tensor(0.0, device=device)
            for loss_fn in losses:
                loss = loss_fn.forward(output, batch, gaussians, global_step)
                total_loss = total_loss + loss

            total_loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)

            total_losses.append(total_loss.item())

            if pose_opt_config.print_steps:
                print(f"pose_opt step {i}: loss = {total_loss.item():.6f}")

        # Finally, apply the pose adjustment to the poses.
        new_poses = rearrange(poses, "b v i j -> (b v) i j")
        new_poses = pose_adjust(
            camtoworlds=new_poses,
            embed_ids=image_ids,
        )
        new_poses = rearrange(new_poses, "(b v) i j -> b v i j", b=b, v=v)

    return new_poses
