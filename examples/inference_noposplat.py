"""
Minimal LGTM inference example using the NoPoSplat model.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

"""
Usage: python examples/inference_noposplat.py

Overview:
  Input: 2 RGB images + intrinsics (no poses needed for the encoder).
  Output: Textured 2D Gaussians representing the 3D scene.

  NoPoSplat uses cross-view attention (CroCo) to predict Gaussians directly
  from image pairs — camera poses are not required by the encoder. Intrinsics
  are embedded as conditioning tokens. The Gaussians are predicted in a
  normalized frame where context camera 0 = identity and the inter-camera
  baseline = 1 (controlled by normalize_scene_scale and relative_pose flags).

  Rendering and evaluation: poses ARE needed to render novel views. Target
  and context poses are normalized into the same frame as the Gaussians, so
  the rasterizer can render at any target pose and compare with ground truth.

  Video interpolation: smoothly interpolates between the two known context
  poses (which bound the scene) and renders the Gaussians at each step.

  Test-time pose optimization: available (test_step only) to refine target
  poses via gradient descent when computing formal evaluation metrics. Not
  used in this inference script.
"""

from pathlib import Path

import rootutils
import torch
from example_utils import download_dl3dv_test_data, load_dl3dv_test_data

from lgtm.dataset.data_utils import prepare_inference_batch
from lgtm.engine import LGTMEngine
from lgtm.utils.vis_utils import render_video_interpolation, save_image


def main():
    output_dir = Path(__file__).parent / "outputs_lgtm_noposplat"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _project_root = rootutils.find_root(
        search_from=__file__, indicator="pyproject.toml"
    )
    experiment = "dl3dv_noposplat_lgtm_288x512_2304x4096"
    checkpoint_path = str(
        _project_root / "weights/dl3dv_noposplat_lgtm_288x512_2304x4096.ckpt"
    )
    context_indices = [0, 9]
    target_indices = [1, 3, 5, 7]
    video_num_frames = 60
    video_fps = 30

    # Download DL3DV test data (cached after first run).
    print("Downloading DL3DV test data...")
    scene_dir = download_dl3dv_test_data()

    # Load data (converts cameras from OpenGL to OpenCV convention).
    print("Loading DL3DV test data...")
    poses, intrinsics, images = load_dl3dv_test_data(scene_dir)
    context_images = images[context_indices]
    target_images = images[target_indices]

    # Load model.
    print("Loading model...")
    engine, dataset_cfg = LGTMEngine.from_config(
        experiment_name=experiment,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    batch = prepare_inference_batch(
        context_images=list(context_images),
        context_poses=list(poses[context_indices]),
        context_intrinsics=list(intrinsics[context_indices]),
        target_images=list(target_images),
        target_poses=list(poses[target_indices]),
        target_intrinsics=list(intrinsics[target_indices]),
        dataset_cfg=dataset_cfg,
        device=device,
    )

    # Run inference (encoder + rasterizer).
    print("\nRunning inference...")
    with torch.no_grad():
        result = engine(batch)
    gaussians = result.gaussians
    print(f"  Produced {gaussians.means.shape[1]} Gaussians")

    rendered_targets = result.rasterizer_output.colors[0]
    gt_targets = batch["target"]["image"][0]

    # Save context images.
    for i, idx in enumerate(context_indices):
        save_image(
            batch["context"]["image"][0][i], output_dir / f"context_{idx}.png"
        )

    # Save target images.
    for i, idx in enumerate(target_indices):
        save_image(
            rendered_targets[i].clamp(0, 1),
            output_dir / f"target_pd_{idx}.png",
        )
        save_image(
            gt_targets[i].clamp(0, 1),
            output_dir / f"target_gt_{idx}.png",
        )

    # Save interpolation video.
    render_video_interpolation(
        rasterizer=engine.rasterizer,
        gaussians=gaussians,
        context=batch["context"],
        output_path=output_dir / "interpolation.mp4",
        num_frames=video_num_frames,
        fps=video_fps,
    )

    print(f"\n{'=' * 60}")
    print(f"Done! Outputs saved to: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
