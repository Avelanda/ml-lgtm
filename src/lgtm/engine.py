"""
PyTorch Lightning module for LGTM training, validation, and inference.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import datetime
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rootutils
import torch
import wandb
from einops import pack, rearrange, repeat
from hydra import compose, initialize_config_dir
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import nn

import lgtm.storage as storage
from lgtm.config import (
    OptimizerCfg,
    TestCfg,
    TrainCfg,
    load_typed_root_config,
    set_cfg,
)
from lgtm.dataset.data_types import BatchedViews
from lgtm.dataset.dataset import DatasetCfg
from lgtm.loss.loss import Loss
from lgtm.model.encoder import get_encoder
from lgtm.model.encoder.encoder import Encoder
from lgtm.model.gaussian.gaussian_rasterizer import (
    GaussianRasterizer,
    GaussianRasterizerOutput,
    get_rasterizer,
)
from lgtm.model.gaussian.gaussians import Gaussians
from lgtm.utils.logging_utils import (
    LocalLogger,
    get_exp_output_dir,
    safe_log_to_file,
)
from lgtm.utils.metrics import (
    aggregate_test_metrics,
    compute_lpips,
    compute_psnr,
    compute_ssim,
)
from lgtm.utils.pose_opt import PoseOptConfig, optimize_camera_poses
from lgtm.utils.training_utils import StepTracker
from lgtm.utils.vis_utils import (
    build_comparison_image,
    build_interpolation_trajectory,
    render_video_frames,
)


@dataclass
class ForwardOutput:
    """
    Structured output from LGTMEngine.forward().
    """

    rasterizer_output: GaussianRasterizerOutput
    gaussians: Gaussians
    encoder_info: dict


class LGTMEngine(LightningModule):
    """
    Main LightningModule for LGTM training, testing, and inference.

    Wraps the encoder (which produces Gaussians from context views) and
    the rasterizer (which renders Gaussians at target camera poses).
    """

    logger: Optional[WandbLogger]
    encoder: nn.Module
    rasterizer: GaussianRasterizer
    losses: nn.ModuleList
    pose_opt_losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    train_cfg: TrainCfg
    test_cfg: TestCfg
    pose_opt_cfg: PoseOptConfig
    step_tracker: StepTracker | None

    def __init__(
        self,
        # Config objects.
        optimizer_cfg: OptimizerCfg,
        train_cfg: TrainCfg,
        test_cfg: TestCfg,
        pose_opt_cfg: PoseOptConfig,
        # Model components.
        encoder: Encoder,
        rasterizer: GaussianRasterizer,
        losses: list[Loss],
        pose_opt_losses: list[Loss],
        # Runtime.
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pose_opt_cfg = pose_opt_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.rasterizer = rasterizer
        self.losses = nn.ModuleList(losses)
        self.pose_opt_losses = nn.ModuleList(pose_opt_losses)

        # Timing tracking for training progress.
        self._last_print_time = None
        self._last_print_step = None

    @classmethod
    def from_config(
        cls,
        experiment_name: str,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> tuple["LGTMEngine", DatasetCfg | None]:
        """
        Load a model from an experiment config and checkpoint.

        Args:
            experiment_name: Experiment config name (without .yaml)
            checkpoint_path: Path to checkpoint file. Supports:
                - Local path: "/path/to/checkpoint.ckpt"
                - Remote path (requires lgtmext)
            device: Device to load model on (default: "cuda")

        Returns:
            Tuple of (engine, dataset_cfg) where dataset_cfg carries the
            preprocessing parameters (image shape, near/far, pose flags)
            needed by prepare_inference_batch().
        """
        _project_root = rootutils.find_root(
            search_from=__file__, indicator="pyproject.toml"
        )
        with initialize_config_dir(
            version_base=None,
            config_dir=str(_project_root / "configs"),
        ):
            cfg_dict = compose(
                config_name="main",
                overrides=[f"+experiment={experiment_name}"],
            )
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)

        encoder = get_encoder(cfg.encoder)

        # Propagate shared texture settings from encoder to rasterizer.
        rasterizer_cfg = cfg.rasterizer
        rasterizer_cfg.texture_alpha_mode = (
            cfg.encoder.gaussian.texture_alpha_mode
        )
        rasterizer_cfg.texture_color_sigma = (
            cfg.encoder.gaussian.texture_color_sigma
        )
        rasterizer = get_rasterizer(rasterizer_cfg)

        model = cls(
            optimizer_cfg=cfg.optimizer,
            train_cfg=cfg.train,
            test_cfg=cfg.test,
            pose_opt_cfg=cfg.pose_opt,
            encoder=encoder,
            rasterizer=rasterizer,
            losses=[],
            pose_opt_losses=[],
            step_tracker=None,
        )

        # Extract dataset config for preprocessing at inference time.
        dataset_cfg: DatasetCfg | None = None
        if cfg.dataset:
            dataset_wrapper = cfg.dataset[0]
            candidate = next(iter(vars(dataset_wrapper).values()))
            if isinstance(candidate, DatasetCfg):
                dataset_cfg = candidate

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt_local_path = storage.resolve_path(checkpoint_path)
        if not isinstance(ckpt_local_path, Path):
            raise FileNotFoundError(
                f"Could not resolve checkpoint path: {checkpoint_path}"
            )
        print(f"  Resolved to: {ckpt_local_path}")

        ckpt = torch.load(
            ckpt_local_path, map_location="cpu", weights_only=False
        )
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            raise ValueError(
                "Checkpoint does not contain 'state_dict' key: "
                f"{ckpt_local_path}"
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "  Loaded checkpoint: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
        )

        model = model.to(device)
        model.eval()
        print(f"  Model ready on {device}")

        return model, dataset_cfg

    def predict_gaussians(self, context: BatchedViews) -> Gaussians:
        """
        Run the encoder only to produce Gaussians (no rasterization).

        Useful when you want Gaussians without immediately rendering
        at target views, e.g. for video generation from novel poses.

        Args:
            context: Batched context views dict with keys
                "image", "poses", "intrinsics", "near", "far".

        Returns:
            Gaussians object representing the 3D scene.
        """
        gaussians, _ = self.encoder(
            context=context,
            global_step=self.global_step,
        )
        return gaussians

    def forward(self, batch) -> ForwardOutput:
        """
        Standard PyTorch forward pass.

        Runs the encoder on context views to produce Gaussians, then
        rasterizes them at the target camera poses. Used by
        training_step, validation_step, test_step, and standalone
        inference scripts.

        Args:
            batch: Dict with keys:
                - "context": dict with "image" [B, V, 3, H, W],
                    "poses" [B, V, 4, 4],
                    "intrinsics" [B, V, 3, 3],
                    "near" [B, V], "far" [B, V]
                - "target": dict with "image" [B, V, 3, H, W],
                    "poses" [B, V, 4, 4],
                    "intrinsics" [B, V, 3, 3],
                    "near" [B, V], "far" [B, V]

        Returns:
            ForwardOutput with rasterizer_output, gaussians,
            and encoder_info.
        """
        gaussians, encoder_info = self.encoder(
            context=batch["context"],
            global_step=self.global_step,
        )
        _, _, _, h, w = batch["target"]["image"].shape
        rasterizer_output = self.rasterizer.forward(
            gaussians=gaussians,
            poses=batch["target"]["poses"],
            intrinsics=batch["target"]["intrinsics"],
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
        )
        return ForwardOutput(
            rasterizer_output=rasterizer_output,
            gaussians=gaussians,
            encoder_info=encoder_info,
        )

    def training_step(self, batch, batch_idx):
        # Combine batch from different dataloaders.
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat(
                                    [
                                        batch_combined[k][kk],
                                        batch_per_dl[k][kk],
                                    ],
                                    dim=0,
                                )
                        else:
                            raise NotImplementedError
            batch = batch_combined

        # Dry run for all ranks, print on all ranks.
        if self.train_cfg.enable_dry_run:
            max_steps = self.trainer.max_steps
            print(
                f"[dry_run {self.global_step}/{max_steps} rank"
                f" {self.global_rank}] scene = {batch['scene']}; context ="
                f" {batch['context']['index'].tolist()}; target ="
                f" {batch['target']['index'].tolist()}"
            )
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        result = self(batch)
        gaussians = result.gaussians
        output = result.rasterizer_output
        encoder_info = result.encoder_info

        # Log Gaussian stats.
        gs_stats = gaussians.get_stats()
        for key, value in gs_stats.items():
            self.log(key, value)

        target_gt = batch["target"]["image"]

        # Compute and log loss.
        psnr_probabilistic = compute_psnr(
            ground_truth=rearrange(target_gt, "b v c h w -> (b v) c h w"),
            predicted=rearrange(output.colors, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                prediction=output,
                batch=batch,
                gaussians=gaussians,
                global_step=self.global_step,
                encoder_info=encoder_info,
            )
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            # Calculate timing and ETA.
            current_time = time.time()
            max_steps = self.trainer.max_steps

            if (
                self._last_print_time is not None
                and self._last_print_step is not None
            ):
                # Calculate time per step since last print.
                time_diff = current_time - self._last_print_time
                step_diff = self.global_step - self._last_print_step
                time_per_step = time_diff / step_diff if step_diff > 0 else 0

                # Calculate ETA.
                remaining_steps = max_steps - self.global_step
                eta_seconds = (
                    remaining_steps * time_per_step if time_per_step > 0 else 0
                )

                # Format ETA as hours, minutes, seconds.
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_hours}h{eta_minutes:02d}m{eta_secs:02d}s"

                print(
                    f"[train {self.global_step}/{max_steps}] "
                    f"step_time: {time_per_step:.3f}s, "
                    f"eta: {eta_str}, "
                    f"loss: {total_loss:.6f}"
                )
            else:
                # First print - no timing data available yet.
                print(
                    f"[train {self.global_step}/{max_steps}] "
                    "step_time: calculating, "
                    "eta: calculating, "
                    f"loss: {total_loss:.6f}"
                )

            # Update timing tracking.
            self._last_print_time = current_time
            self._last_print_step = self.global_step

            # Print detailed view indices for debugging.
            print(
                f"[train {self.global_step}] "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def on_test_start(self) -> None:
        """
        Initialize test state tracking and clean up test metrics folder.
        """
        self.test_start_time = time.time()

        if self.global_rank == 0:
            # Clean up test_metrics folder.
            test_metrics_dir = get_exp_output_dir() / "test_metrics"
            if test_metrics_dir.exists():
                shutil.rmtree(test_metrics_dir)
                print(f"Cleaned up test_metrics directory: {test_metrics_dir}")
            test_metrics_dir.mkdir(parents=True, exist_ok=True)

            # Clean up test_metrics_aggregated.json.
            test_metrics_aggregated_path = (
                get_exp_output_dir() / "test_metrics_aggregated.json"
            )
            if test_metrics_aggregated_path.exists():
                test_metrics_aggregated_path.unlink()
                print(
                    "Cleaned up test_metrics_aggregated.json:"
                    f" {test_metrics_aggregated_path}"
                )

        # Synchronize all ranks before proceeding with test.
        if hasattr(self.trainer.strategy, "barrier"):
            self.trainer.strategy.barrier()

    def test_step(self, batch, batch_idx):

        b, v, _, h, w = batch["target"]["image"].shape
        if b != 1:
            raise ValueError(f"Expected batch size 1 for test_step, got {b}")

        has_test_target = (
            "test_target" in batch and batch["test_target"] is not None
        )

        # Always call encoder and rasterizer separately so test_step can.
        # Intercept between them (e.g. for pose optimization or test_target.
        # Resolution override).
        gaussians, _ = self.encoder(
            context=batch["context"],
            global_step=self.global_step,
        )

        if has_test_target:
            _, _, _, test_target_h, test_target_w = batch["test_target"][
                "image"
            ].shape
            render_h, render_w = test_target_h, test_target_w
            render_intrinsics = batch["test_target"]["intrinsics"]
        else:
            render_intrinsics = batch["target"]["intrinsics"]
            render_h, render_w = h, w

        if self.test_cfg.enable_pose_opt:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

            optimized_poses = optimize_camera_poses(
                gaussians=gaussians,
                batch=batch,
                rasterizer=self.rasterizer,
                losses=self.pose_opt_losses,
                pose_opt_cfg=self.pose_opt_cfg,
                global_step=self.global_step,
            )

            output = self.rasterizer.forward(
                gaussians,
                optimized_poses,
                render_intrinsics,
                batch["target"]["near"],
                batch["target"]["far"],
                (render_h, render_w),
            )
        else:
            output = self.rasterizer.forward(
                gaussians,
                batch["target"]["poses"],
                render_intrinsics,
                batch["target"]["near"],
                batch["target"]["far"],
                (render_h, render_w),
            )

        # Compute and log metrics.
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]

            # Compute metrics using test_target if available.
            rgb_pred = output.colors[0]
            if has_test_target:
                rgb_gt = batch["test_target"]["image"][0]
            else:
                rgb_gt = batch["target"]["image"][0]

            # Compute metrics once for all views.
            lpips_vals = compute_lpips(
                ground_truth=rgb_gt,
                predicted=rgb_pred,
                lpips_metric_max_dim=self.test_cfg.lpips_metric_max_dim,
            )
            ssim_vals = compute_ssim(ground_truth=rgb_gt, predicted=rgb_pred)
            psnr_vals = compute_psnr(ground_truth=rgb_gt, predicted=rgb_pred)

            # Aggregated metrics (mean across all views)
            all_metrics = {
                "lpips": lpips_vals.mean(),
                "ssim": ssim_vals.mean(),
                "psnr": psnr_vals.mean(),
            }

            # Log per-scene, per-view metrics.
            scene_id = batch["scene"][0]
            target_indices = batch["target"]["index"][0]
            target_metrics = {}
            for i, target_idx in enumerate(target_indices):
                target_metrics[str(target_idx.item())] = {
                    "lpips": lpips_vals[i].item(),
                    "ssim": ssim_vals[i].item(),
                    "psnr": psnr_vals[i].item(),
                    "overlap": overlap.item(),
                }
            scene_metrics = {
                "scene_id": scene_id,
                "rank": self.global_rank,
                "overlap": overlap.item(),
                "context_indices": [
                    idx.item() for idx in batch["context"]["index"][0]
                ],
                "target_indices": [idx.item() for idx in target_indices],
                "avg_target_metrics": {
                    "lpips": all_metrics["lpips"].item(),
                    "ssim": all_metrics["ssim"].item(),
                    "psnr": all_metrics["psnr"].item(),
                },
                "target_metrics": target_metrics,
            }
            scene_metrics_json = json.dumps(scene_metrics, indent=2)
            safe_log_to_file(
                message=scene_metrics_json,
                rel_log_path=f"test_metrics/{scene_id}.json",
                mode="w",
            )

        # Calculate and print timing at the end of the step.
        elapsed_time = time.time() - self.test_start_time
        avg_time_per_step = elapsed_time / (batch_idx + 1)  # Global average
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        print(
            f"[eval] rank: {self.global_rank}, step: {batch_idx}, elapsed:"
            f" {elapsed_str}, avg speed: {avg_time_per_step:.2f} sec/scene"
        )

        # Print preview metrics.
        if self.global_rank == 0 and batch_idx % 10 == 0:
            partial_metrics = aggregate_test_metrics()
            if partial_metrics:  # Only print if we have metrics
                print(
                    f"[partial metrics] ({partial_metrics['num_scenes']}"
                    f" scenes) lpips: {partial_metrics['lpips']:.3f}, ssim:"
                    f" {partial_metrics['ssim']:.3f}, psnr:"
                    f" {partial_metrics['psnr']:.3f}"
                )

    def on_test_end(self) -> None:
        # Synchronize all ranks before proceeding with aggregation.
        if hasattr(self.trainer.strategy, "barrier"):
            self.trainer.strategy.barrier()

        if self.global_rank != 0:
            print(
                f"Skipping test logging on non-rank 0: rank={self.global_rank}"
            )
            return

        # Print configs (rank 0 only)
        config_dict_to_print = {
            "checkpointing.load": self.trainer.ckpt_path,
            "model.rasterizer.backend": self.rasterizer.backend,
            "test.enable_pose_opt": self.test_cfg.enable_pose_opt,
            "pose_opt.num_steps": self.pose_opt_cfg.num_steps,
            "pose_opt.learning_rate": self.pose_opt_cfg.learning_rate,
            "pose_opt.backend_override": self.pose_opt_cfg.backend_override,
        }
        print(tabulate(config_dict_to_print.items(), headers=["Key", "Value"]))

        # Aggregate and print test metrics (rank 0 only)
        test_metrics_aggregated = aggregate_test_metrics()
        if test_metrics_aggregated:
            # Save aggregated metrics.
            test_metrics_aggregated_path = (
                get_exp_output_dir() / "test_metrics_aggregated.json"
            )
            with open(test_metrics_aggregated_path, "w") as f:
                json.dump(test_metrics_aggregated, f, indent=2)

            # Print summary.
            title = "Aggregated Test Metrics"
            print("\n" + "=" * len(title))
            print(title)
            print("=" * len(title))
            for key, value in test_metrics_aggregated.items():
                if isinstance(value, float):
                    print(f"- {key}: {value}")
                else:
                    print(f"- {key}: {value}")
            print(f"\nMetrics saved to: {test_metrics_aggregated_path}\n")
        else:
            print("No valid target metrics found for aggregation")

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        if self.train_cfg.enable_dry_run:
            max_steps = self.trainer.max_steps
            prefix = f"[dry_run validation {self.global_step}/{max_steps}]"
        else:
            prefix = f"[validation {self.global_step}]"
        print(
            f"{prefix} "
            f"scene = {batch['scene']}; "
            f"context = {batch['context']['index'].tolist()}; "
            f"target = {batch['target']['index'].tolist()}"
        )

        if self.train_cfg.enable_dry_run:
            return

        b, _, _, h, w = batch["target"]["image"].shape
        if b != 1:
            raise ValueError(
                f"Expected batch size 1 for visualization, got {b}"
            )

        result = self(batch)
        gaussians = result.gaussians
        im_target_rgb_pred = result.rasterizer_output.colors[0]
        im_target_depth_pred = result.rasterizer_output.depths[0]

        # Render context views to get depth for the comparison image.
        context_output = self.rasterizer.forward(
            gaussians,
            batch["context"]["poses"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
        )
        im_context_depth_pred = context_output.depths[0]

        # Compute and log validation metrics.
        im_target_rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(
            ground_truth=im_target_rgb_gt,
            predicted=im_target_rgb_pred,
        ).mean()
        self.log("val/psnr", psnr)
        lpips = compute_lpips(
            ground_truth=im_target_rgb_gt,
            predicted=im_target_rgb_pred,
            lpips_metric_max_dim=self.train_cfg.validation_lpips_metric_max_dim,
        ).mean()
        self.log("val/lpips", lpips)
        ssim = compute_ssim(
            ground_truth=im_target_rgb_gt,
            predicted=im_target_rgb_pred,
        ).mean()
        self.log("val/ssim", ssim)

        # Build and log comparison image.
        im_comparison = build_comparison_image(
            context_images=batch["context"]["image"][0],
            context_depth_pred=im_context_depth_pred,
            target_rgb_gt=im_target_rgb_gt,
            target_rgb_pred=im_target_rgb_pred,
            target_depth_pred=im_target_depth_pred,
        )
        self.logger.log_image(
            "comparison",
            [im_comparison],
            step=self.global_step,
            caption=batch["scene"],
            file_type=["jpg"],
        )

        # Render interpolation video and log to wandb.
        _, v, _, _ = batch["context"]["poses"].shape
        end_poses = (
            batch["context"]["poses"][0, 1]
            if v == 2
            else batch["target"]["poses"][0, 0]
        )
        end_intrinsics = (
            batch["context"]["intrinsics"][0, 1]
            if v == 2
            else batch["target"]["intrinsics"][0, 0]
        )
        trajectory_fn = build_interpolation_trajectory(
            start_poses=batch["context"]["poses"][0, 0],
            start_intrinsics=batch["context"]["intrinsics"][0, 0],
            end_poses=end_poses,
            end_intrinsics=end_intrinsics,
        )

        # Render interpolation video and log to wandb.
        num_frames = 30
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        print(f"Rendering {num_frames} frames")
        cpu_images = render_video_frames(
            rasterizer=self.rasterizer,
            gaussians=gaussians,
            trajectory_fn=trajectory_fn,
            image_shape=(h, w),
            near=near,
            far=far,
            num_frames=num_frames,
            smooth=True,
            include_depth=True,
            device=self.device,
        )
        video = torch.stack(cpu_images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).numpy()
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            "video/rgb": wandb.Video(video[None], fps=30, format="mp4")
        }

        # PyTorch Lightning doesn't support video logging; use wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            if not isinstance(self.logger, LocalLogger):
                raise
            for key, video in visualizations.items():
                self.logger.log_video(key, video, step=self.global_step)

        del near, far
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        """
        Lightning optimizer configuration.

        Parameters are split into two LR groups: full-lr and scaled-lr:
        - full-lr  : lr (default)
        - scaled-lr: lr * multiplier (0.0 = frozen entirely)

        Exactly one multiplier may be set; setting more than one raises.
        When none is set, all parameters share the same lr.

        Encoder parameter layout:

          NoPoSplat:
            backbone.*                  ← CroCo backbone
              intrinsic_encoder.*       ← inside backbone
            downstream_head[12].*       ← pts3d heads
            gaussian_param_head[2].*    ← GS heads (one per context view)
              dpt.direct_2dgs_*.*       ← texture layers inside GS head

          DepthSplat:
            model.*                     ← UniMatch backbone + depth UNet
            gaussian_head.*             ← small GS MLP
            gaussian_param_head.*       ← GS+texture head
              direct_2dgs_*.*           ← texture layers inside GS head

          Flash3D:
            gaussian_predictor.*        ← UniDepth backbone + decoder
            gaussian_param_head.*       ← GS+texture head
              direct_2dgs_*.*           ← texture layers inside GS head

        Group containment (direct_2dgs ⊂ gaussian_param_head ⊂ all):

        - backbone_lr_mult:
            - full-lr   : gaussian_param_head + intrinsic_encoder
            - scaled-lr : everything else
        - non_head_lr_mult:
            - full-lr   : gaussian_param_head
            - scaled-lr : everything else
        - non_2dgs_lr_mult:
            - full-lr   : direct_2dgs
            - scaled-lr : everything else
        """
        cfg = self.optimizer_cfg
        lr = cfg.lr

        # At most one multiplier can be set.
        num_set = sum(
            x is not None
            for x in [
                cfg.backbone_lr_mult,
                cfg.non_head_lr_mult,
                cfg.non_2dgs_lr_mult,
            ]
        )
        if num_set > 1:
            raise ValueError(
                "Set at most one of: backbone_lr_mult, "
                "non_head_lr_mult, non_2dgs_lr_mult."
            )

        # Which params get full lr? Everything else gets lr * mult.
        def is_full_lr(n: str) -> bool:
            if cfg.non_2dgs_lr_mult is not None:
                return "direct_2dgs" in n
            if cfg.non_head_lr_mult is not None:
                return "gaussian_param_head" in n
            return "gaussian_param_head" in n or "intrinsic_encoder" in n

        if cfg.non_2dgs_lr_mult is not None:
            mult = cfg.non_2dgs_lr_mult
        elif cfg.non_head_lr_mult is not None:
            mult = cfg.non_head_lr_mult
        elif cfg.backbone_lr_mult is not None:
            mult = cfg.backbone_lr_mult
        else:
            mult = 1.0

        # Build param groups. Freeze scaled-lr params if mult == 0.
        full_lr_params, scaled_lr_params = [], []
        num_frozen = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if is_full_lr(name):
                full_lr_params.append(param)
            elif mult == 0.0:
                param.requires_grad = False
                num_frozen += 1
            else:
                scaled_lr_params.append(param)
        if num_frozen > 0:
            print(f"LR multiplier 0.0: frozen {num_frozen} scaled-lr params")

        param_dicts = [{"params": full_lr_params, "lr": lr}]
        if scaled_lr_params:
            param_dicts.append({"params": scaled_lr_params, "lr": lr * mult})

        # Log parameter group information.
        print("Optimizer configuration:")
        for i, param_dict in enumerate(param_dicts):
            num_params = len(param_dict["params"])
            group_lr = param_dict["lr"]
            print(f"  Group {i}: {num_params} parameters, lr={group_lr:.2e}")

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        warm_up_steps = cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=lr * 0.1,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_up, lr_scheduler],
            milestones=[warm_up_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
