"""
Logging, authentication, and experiment tracking utilities.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import fcntl
import netrc
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

try:
    from moviepy import ImageSequenceClip  # moviepy >= 2.0
except ImportError:
    from moviepy.editor import ImageSequenceClip  # moviepy < 2.0
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image


def parse_netrc_password(machine: str) -> Optional[str]:
    """
    Parse the ~/.netrc file to retrieve the password for a specific machine.

    Args:
        machine: The name of the machine entry in the .netrc file.

    Returns:
        The password (API key) for the specified machine, or None if not found
        or if the file cannot be parsed.
    """
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        print(f"Warning: .netrc file not found at {netrc_path}")
        return None

    try:
        # Copy .netrc to temporary directory to avoid unexpected modifications.
        # Python's netrc.netrc seems to be buggy sometimes.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_netrc_path = Path(temp_dir) / ".netrc"
            shutil.copy2(netrc_path, temp_netrc_path)

            netrc_data = netrc.netrc(str(temp_netrc_path))
            auth = netrc_data.authenticators(machine)
            if auth:
                _login, _account, password = auth
                return password
            else:
                print(
                    f"Warning: Could not find '{machine}' machine in .netrc"
                    " file."
                )
                return None

    except netrc.NetrcParseError as e:
        print(f"Warning: Could not parse .netrc file: {e}")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred reading .netrc: {e}")
        return None


def get_wandb_api_key() -> Optional[str]:
    """
    Try get W&B API key first from env var WANDB_API_KEY, then from ~/.netrc.

    If not found, return None.
    """
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    else:
        return parse_netrc_password("api.wandb.ai")


def get_exp_output_dir() -> Path | None:
    """
    Get the experiment output directory.

    This typically points to "outputs/<exp_name>".
    """
    try:
        return Path(HydraConfig.get()["runtime"]["output_dir"])
    except:
        return None


def safe_log_to_file(
    message: str,
    rel_log_path: str,
    mode: Literal["w", "a"] = "w",
    end: str = "\n",
) -> None:
    """
    Safely write a message to a log file in the experiment output directory.

    Args:
        message: The message to log
        rel_log_path: Relative path for the log file within the output
            directory. Must be a relative path, not absolute.
        mode: Mode to open the file in, "w" for write, "a" for append.
        end: String appended after the message, default is a newline.

    Note: this function will attempt to resolve the output directory from Hydra
    config if available. If not, it will use the current working directory.
    """
    # Validate that the path is relative.
    log_path = Path(rel_log_path)
    if log_path.is_absolute():
        raise ValueError(
            f"log_file_path must be a relative path, got: {rel_log_path}"
        )

    if mode not in ["w", "a"]:
        raise ValueError(f"Invalid mode: {mode}, must be 'w' or 'a'")

    # Get the Hydra output directory if available.
    output_dir = get_exp_output_dir()
    if output_dir is None:
        raise ValueError("Output directory not found")
    full_log_path = output_dir / log_path

    # Ensure the parent directory exists.
    full_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_log_path, mode=mode) as f:
        # Acquire an exclusive lock (blocks if another process has the lock).
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(f"{message}{end}")
        finally:
            # Release the lock.
            fcntl.flock(f, fcntl.LOCK_UN)


def save_wandb_run_id(run_id: str) -> None:
    """
    Save the Wandb run ID to a file in the experiment output directory.

    Args:
        run_id: The Wandb run ID to save
    """
    if run_id is None:
        return

    wandb_id_path = get_exp_output_dir() / "wandb_run_id.txt"
    with open(wandb_id_path, "w") as f:
        f.write(run_id)
    print(f"Saved Wandb run ID {run_id} to {wandb_id_path}")


def load_wandb_run_id() -> str | None:
    """
    Load the Wandb run ID from a file in the experiment output directory.

    Returns:
        The saved Wandb run ID, or None if not found
    """
    wandb_id_path = get_exp_output_dir() / "wandb_run_id.txt"
    if not wandb_id_path.exists():
        print(f"No Wandb run ID file found at {wandb_id_path}")
        return None

    with open(wandb_id_path, "r") as f:
        run_id = f.read().strip()

    if run_id:
        print(f"Loaded Wandb run ID {run_id} from {wandb_id_path}")
        return run_id
    else:
        print(f"Empty Wandb run ID file at {wandb_id_path}")
        return None


LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        shutil.rmtree(LOG_PATH, ignore_errors=True)

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        file_type: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Log images to local directory.

        Args:
            key: The key/name for the image
            images: List of images to log
            step: The step number (required)
            file_type: List of file formats ("jpg" or "png") for each image,
                defaults to ["png"] for all
            **kwargs: Additional arguments (for compatibility with wandb logger)
        """
        # The function signature is the same as the wandb logger's, but.
        # The step is actually required.
        if step is None:
            raise ValueError("step must be provided for LocalLogger")

        # Handle file_type parameter.
        n = len(images)
        if file_type is None:
            file_types = ["png"] * n
        elif isinstance(file_type, list):
            if len(file_type) != n:
                raise ValueError(
                    f"Expected {n} items but only found {len(file_type)} for"
                    " file_type"
                )
            file_types = file_type
        else:
            # Backward compatibility: if file_type is a single string,
            # Use it for all images.
            file_types = [file_type] * n

        for index, (image, img_file_type) in enumerate(zip(images, file_types)):
            # Determine file extension.
            if img_file_type == "jpg":
                ext = "jpg"
            else:
                ext = "png"

            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.{ext}"
            path.parent.mkdir(exist_ok=True, parents=True)

            pil_image = Image.fromarray(image)
            if img_file_type == "jpg":
                # Convert to RGB if saving as JPG (JPG doesn't support.
                # Alpha channel)
                if pil_image.mode in ("RGBA", "LA"):
                    pil_image = pil_image.convert("RGB")
                pil_image.save(path, quality=95)
            else:
                pil_image.save(path)

    @rank_zero_only
    def log_video(
        self,
        key: str,
        video: Any,
        step: int,
        fps: int = 30,
    ) -> None:
        """
        Save a wandb.Video object to a local .mp4 file.

        Args:
            key: The key/name for the video (used as subdirectory).
            video: A wandb.Video object whose frames will be extracted.
            step: The global training step (used in the filename).
            fps: Frame rate for the output video (default: 30).
        """
        tensor = video._prepare_video(video.data)
        clip = ImageSequenceClip(list(tensor), fps=fps)
        video_dir = LOG_PATH / key
        video_dir.mkdir(exist_ok=True, parents=True)
        clip.write_videofile(
            str(video_dir / f"{step:0>6}.mp4"),
            logger=None,
        )
