"""
Image and depth visualization helpers.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

Visualization utility functions: image layout, annotation, color mapping,
depth visualization, image/video I/O, and stateless video rendering.

Images are assumed to be float32 tensors with shape (channel, height, width).
"""

from pathlib import Path
from string import ascii_letters, digits, punctuation
from typing import (
    Any,
    Generator,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import matplotlib
import numpy as np
import skvideo.io
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]


def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(
    iterable: Iterable, delimiter: Any
) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    if base < overlay:
        raise ValueError(f"base ({base}) must be >= overlay ({overlay})")
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def _overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay_img: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay_img.shape
    if base_height < overlay_height or base_width < overlay_width:
        raise ValueError(
            f"Base ({base_height}, {base_width}) must be >= "
            f"overlay ({overlay_height}, {overlay_width})"
        )

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay_img.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim],
        overlay_img.shape[cross_dim],
        cross_axis_alignment,
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[tuple(selector)] = overlay_img
    return result


def cat(
    main_axis: Axis,
    *images: Float[Tensor, "channel _ _"],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """
    Arrange images in a line. The interface resembles a CSS div with flexbox.
    """
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(_overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones(
            (c, *separator_size), dtype=torch.float32, device=device
        )
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Float[Tensor, "channel _ _"],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """
    Shorthand for a horizontal linear concatenation.
    """
    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Float[Tensor, "channel _ _"],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """
    Shorthand for a vertical linear concatenation.
    """
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def pad_border(
    image: Float[Tensor, "channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[Tensor, "channel new_height new_width"]:
    """
    Pad the image with a solid-color border on all sides.

    Args:
        image: Input image tensor [C, H, W].
        border: Border width in pixels (default: 8).
        color: Border color as scalar or per-channel value (default: 1 = white).

    Returns:
        Image with border added [C, H+2*border, W+2*border].
    """
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border),
        dtype=torch.float32,
        device=image.device,
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result


EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


def draw_label(
    text: str,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """
    Draw a black label on a white background using the built-in default font.
    """
    font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(
        np.array(image) / 255, dtype=torch.float32, device=device
    )
    return rearrange(image, "h w c -> c h w")


def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, image.device),
        image,
        align="left",
        gap=4,
    )


def vis_depth_map(
    depth: Float[Tensor, "... height width"],
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
    percentile: float = 2,
    cmap: str = "Spectral",
) -> Float[Tensor, "... 3 height width"]:
    """
    Visualize a depth map using a colormap.

    This function follows vis_depth_map's interface but keeps the original
    visualization logic: inverts depth (1/depth), uses percentile-based
    normalization, and applies a Spectral colormap by default.

    Args:
        depth: Input depth map tensor with shape (..., height, width)
        depth_min: Minimum depth value for normalization. If None, uses
            percentile
        depth_max: Maximum depth value for normalization. If None, uses
            percentile
        percentile: Percentile for min/max computation if not provided
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as tensor with shape (..., 3, height, width)
        and values in [0, 1]
    """
    # Store original shape and flatten batch dimensions.
    original_shape = depth.shape
    if depth.ndim == 2:
        # Single image (h, w) -> (1, h, w)
        depth = depth.unsqueeze(0)
        squeeze_output = True
    else:
        # Batch of images (..., h, w) -> (batch, h, w)
        depth = depth.reshape(-1, original_shape[-2], original_shape[-1])
        squeeze_output = False

    batch_size, h, w = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Work with a copy and invert depth.
    depth = depth.clone()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]

    # Compute min/max using percentile if not provided.
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0.0
        else:
            # Limit to 16M elements to avoid quantile() size limitations.
            valid_depth = depth[valid_mask].float()
            if valid_depth.numel() > 16_000_000:
                valid_depth = valid_depth[:16_000_000]
            depth_min = torch.quantile(valid_depth, percentile / 100.0).item()

    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0.0
        else:
            # Limit to 16M elements to avoid quantile() size limitations.
            valid_depth = depth[valid_mask].float()
            if valid_depth.numel() > 16_000_000:
                valid_depth = valid_depth[:16_000_000]
            depth_max = torch.quantile(
                valid_depth, 1.0 - percentile / 100.0
            ).item()

    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6

    # Normalize and invert.
    depth = ((depth - depth_min) / (depth_max - depth_min)).clamp(0, 1)
    depth = 1 - depth

    # Apply colormap.
    cm_func = matplotlib.colormaps[cmap]

    # Process each image in the batch.
    colored_images = []
    for i in range(batch_size):
        depth_np = depth[i].cpu().numpy()  # (h, w)
        img_colored_np = cm_func(depth_np, bytes=False)[:, :, 0:3]  # (h, w, 3)
        img_colored = torch.from_numpy(img_colored_np).to(device).to(dtype)
        img_colored = img_colored.permute(2, 0, 1)  # (3, h, w)
        colored_images.append(img_colored)

    result = torch.stack(colored_images, dim=0)  # (batch, 3, h, w)

    # Reshape back to original batch dimensions.
    if squeeze_output:
        result = result.squeeze(0)  # (3, h, w)
    else:
        result = result.reshape(
            *original_shape[:-2], 3, original_shape[-2], original_shape[-1]
        )

    return result


def build_comparison_image(
    context_images: Float[Tensor, "num_context_views 3 ctx_height ctx_width"],
    context_depth_pred: Float[Tensor, "num_context_views height width"],
    target_rgb_gt: Float[Tensor, "num_target_views 3 height width"],
    target_rgb_pred: Float[Tensor, "num_target_views 3 height width"],
    target_depth_pred: Float[Tensor, "num_target_views height width"],
) -> UInt8[np.ndarray, "comparison_height comparison_width channel"]:
    """
    Build a side-by-side comparison image for validation visualization.

    Produces 4 columns laid out horizontally:
    1. Context: alternating RGB + depth images stacked vertically
    2. Target Ground Truth: all target RGB images stacked vertically
    3. Target Prediction: predicted target RGB images stacked vertically
    4. Depth Prediction: predicted target depth visualizations

    Context images may have a different (typically smaller) resolution
    than the rendering output. The depth maps and target images share
    the same rendering resolution.

    Args:
        context_images: Context RGB images [V_ctx, 3, ctx_H, ctx_W]
        context_depth_pred: Predicted context depth maps [V_ctx, H, W]
        target_rgb_gt: Ground truth target RGB images [V_tgt, 3, H, W]
        target_rgb_pred: Predicted target RGB images [V_tgt, 3, H, W]
        target_depth_pred: Predicted target depth maps [V_tgt, H, W]

    Returns:
        uint8 HWC numpy array ready for logging (with white border applied).
    """
    target_depth_vis = vis_depth_map(target_depth_pred)
    context_depth_vis = vis_depth_map(context_depth_pred)

    context_rgbs_depths = []
    for i in range(context_images.shape[0]):
        context_rgbs_depths.append(context_images[i])
        context_rgbs_depths.append(context_depth_vis[i])

    image = hcat(
        add_label(vcat(*context_rgbs_depths), "Context"),
        add_label(vcat(*target_rgb_gt), "Target (Ground Truth)"),
        add_label(vcat(*target_rgb_pred), "Target (Prediction)"),
        add_label(vcat(*target_depth_vis), "Depth (Prediction)"),
    )
    return to_hwc_uint8(pad_border(image))


def to_hwc_uint8(
    image: FloatImage,
) -> UInt8[np.ndarray, "height width channel"]:
    """
    Convert a float image tensor to a uint8 HWC numpy array.

    Handles batched (B, C, H, W), single (C, H, W), and grayscale (H, W)
    inputs. Clips values to [0, 1] before scaling to [0, 255].

    Args:
        image: Float image tensor in range [0, 1].

    Returns:
        uint8 numpy array of shape [H, W, C].
    """
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    if image.shape[0] not in (3, 4):
        raise ValueError(f"Expected 3 or 4 channels, got {image.shape[0]}")

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
    verbose: bool = False,
) -> None:
    """
    Save an image. Assumed to be in range 0-1.
    """

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(to_hwc_uint8(image)).save(path)

    if verbose:
        dtype_str = str(image.dtype).replace("torch.", "")
        shape_str = str(tuple(image.shape))
        print(f"Saved image ({dtype_str}, {shape_str}) to {path}")


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
    fps: int = 30,
) -> None:
    """
    Save a video from a list of images.

    Args:
        images: List of images in range 0-1
        path: Output path for the video file
        fps: Frame rate for the video (default: 30)
    """

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(to_hwc_uint8(image)).save(path)
    frames = []
    for image in images:
        frames.append(to_hwc_uint8(image))

    writer = skvideo.io.FFmpegWriter(
        path,
        inputdict={
            "-r": str(fps),
        },
        outputdict={
            "-pix_fmt": "yuv420p",
            "-crf": "21",
            "-r": str(fps),
        },
    )
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()


@runtime_checkable
class TrajectoryFn(Protocol):
    """
    Protocol for camera trajectory functions used in video rendering.
    """

    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # poses: c2w 4x4
        Float[Tensor, "batch view 3 3"],  # intrinsics: normalized 3x3
    ]: ...


def build_interpolation_trajectory(
    start_poses: Float[Tensor, "4 4"],
    start_intrinsics: Float[Tensor, "3 3"],
    end_poses: Float[Tensor, "4 4"],
    end_intrinsics: Float[Tensor, "3 3"],
) -> TrajectoryFn:
    """
    Build a smooth interpolation TrajectoryFn between two camera poses.

    The returned function takes a time vector t in [0, 1] and returns
    interpolated poses and intrinsics with a leading batch dimension.

    Args:
        start_poses: Start c2w pose [4, 4].
        start_intrinsics: Start normalized intrinsics [3, 3].
        end_poses: End c2w pose [4, 4].
        end_intrinsics: End normalized intrinsics [3, 3].

    Returns:
        TrajectoryFn that maps t [T] to
        (poses [1, T, 4, 4], intrinsics [1, T, 3, 3]).
    """
    from lgtm.utils.camera_utils import (
        interpolate_intrinsics,
        interpolate_poses,
    )

    def trajectory_fn(t: Float[Tensor, " t"]) -> tuple[
        Float[Tensor, "1 t 4 4"],
        Float[Tensor, "1 t 3 3"],
    ]:
        poses = interpolate_poses(start_poses, end_poses, t)
        intrinsics = interpolate_intrinsics(start_intrinsics, end_intrinsics, t)
        return poses[None], intrinsics[None]

    return trajectory_fn


def render_video_frames(
    rasterizer,
    gaussians,
    trajectory_fn: TrajectoryFn,
    image_shape: tuple[int, int],
    near: Float[Tensor, "1 num_frames"],
    far: Float[Tensor, "1 num_frames"],
    num_frames: int = 30,
    smooth: bool = True,
    include_depth: bool = False,
    device: Optional[torch.device] = None,
) -> list[Float[Tensor, "3 height width"]]:
    """
    Render a sequence of frames along a camera trajectory.

    Stateless: takes rasterizer and gaussians as explicit arguments.

    Args:
        rasterizer: GaussianRasterizer instance.
        gaussians: Gaussians object to render.
        trajectory_fn: Callable mapping time t [0,1] to
            (poses [B, T, 4, 4], intrinsics [B, T, 3, 3]).
        image_shape: Output (height, width).
        near: Near plane values [1, num_frames].
        far: Far plane values [1, num_frames].
        num_frames: Number of frames to render.
        smooth: If True, apply cosine smoothing to time values.
        include_depth: If True, append a depth visualization below
            each RGB frame (vcat of rgb + depth colormap).
        device: Torch device; defaults to gaussians.means.device.

    Returns:
        List of rendered CPU tensors [3, H, W] (or [3, 2*H, W]
        when include_depth=True).
    """
    if device is None:
        device = gaussians.means.device

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    poses, intrinsics = trajectory_fn(t)
    h, w = image_shape

    cpu_frames = []
    for frame_idx in range(num_frames):
        with torch.no_grad():
            output = rasterizer.forward(
                gaussians,
                poses[:, frame_idx : frame_idx + 1],
                intrinsics[:, frame_idx : frame_idx + 1],
                near[:, frame_idx : frame_idx + 1],
                far[:, frame_idx : frame_idx + 1],
                (h, w),
            )
        rgb_frame = output.colors[0, 0]
        if include_depth:
            depth_frame = vis_depth_map(output.depths[0])[0]
            cpu_frames.append(vcat(rgb_frame, depth_frame).cpu())
        else:
            cpu_frames.append(rgb_frame.cpu())
        del output
        torch.cuda.empty_cache()

        if ((frame_idx + 1) % 30 == 0) or (frame_idx + 1 == num_frames):
            print(f"  Rendered frame {frame_idx + 1}/{num_frames}")

    return cpu_frames


def render_video_interpolation(
    rasterizer,
    gaussians,
    context: dict,
    output_path: Union[str, Path],
    num_frames: int = 30,
    fps: int = 30,
    loop_playback: bool = True,
) -> None:
    """
    Render an interpolation video between the first two context views.

    Interpolates smoothly between context view 0 and view 1 (or falls
    back to a single view if only one exists), renders all frames, and
    saves to an .mp4 file.

    Args:
        rasterizer: GaussianRasterizer instance.
        gaussians: Gaussians object from encoder output.
        context: Context views dict with keys "poses", "intrinsics",
            "image", "near", "far". Expected shapes:
            poses [1, V, 4, 4], intrinsics [1, V, 3, 3],
            image [1, V, 3, H, W], near [1, V], far [1, V].
        output_path: Path to save the output .mp4 file.
        num_frames: Number of interpolation frames (default: 30).
        fps: Video frame rate (default: 30).
        loop_playback: If True, append reversed frames for a
            ping-pong loop (default: True).
    """
    output_path = Path(output_path)
    device = gaussians.means.device

    _, v, _, _ = context["poses"].shape
    end_idx = min(1, v - 1)
    trajectory_fn = build_interpolation_trajectory(
        start_poses=context["poses"][0, 0],
        start_intrinsics=context["intrinsics"][0, 0],
        end_poses=context["poses"][0, end_idx],
        end_intrinsics=context["intrinsics"][0, end_idx],
    )

    _, _, _, h, w = context["image"].shape
    near = repeat(context["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(context["far"][:, 0], "b -> b v", v=num_frames)

    cpu_frames = render_video_frames(
        rasterizer=rasterizer,
        gaussians=gaussians,
        trajectory_fn=trajectory_fn,
        image_shape=(h, w),
        near=near,
        far=far,
        num_frames=num_frames,
        smooth=True,
        device=device,
    )

    if loop_playback:
        cpu_frames = cpu_frames + cpu_frames[::-1]

    save_video(cpu_frames, output_path, fps=fps)
    print(f"  Saved video: {output_path} ({len(cpu_frames)} frames)")
