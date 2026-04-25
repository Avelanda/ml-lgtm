"""
Checkpoint path resolution and load/save helpers.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.
"""

import re
from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

import lgtm.storage as storage


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the checkpoint with the highest step number.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Extract step numbers and find the maximum.
    max_step = -1
    latest_checkpoint = None

    for checkpoint in checkpoints:
        step_match = re.search(r"step[_-](\d+)", checkpoint.name)
        if step_match:
            step = int(step_match.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint = checkpoint

    return latest_checkpoint


def search_same_step_ckpt_path(
    original_ckpt_path: Union[str, Path],
) -> Optional[str]:
    """
    Search for checkpoint with same step number but different epoch number.

    Only works for exact format: epoch_{N}-step_{N}.ckpt
    - If format doesn't match: returns None
    - If multiple matches found: prints them and returns None
    - If exactly one match found: returns the path
    """
    path_str = str(original_ckpt_path)

    # Remote paths: delegate to lgtmext.
    if storage._is_remote_scheme(path_str):
        return storage.search_same_step_remote_checkpoint(path_str)

    # Local path.
    path = Path(path_str)
    filename = path.name
    parent_dir = path.parent

    # Must match epoch_{N}-step_{N}.ckpt format.
    match = re.match(r"^epoch_(\d+)-step_(\d+)\.ckpt$", filename)
    if not match:
        return None

    step_num = match.group(2)

    # Find matching files.
    matches = []
    try:
        if not parent_dir.exists():
            return None
        for item in parent_dir.glob(f"epoch_*-step_{step_num}.ckpt"):
            if str(item) != path_str:
                matches.append(str(item))
    except Exception:
        return None

    # Return single match or None.
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Multiple checkpoints found with step {step_num}:")
        for f in sorted(matches):
            print(f"  {f}")

    return None


def get_remote_mirrored_ckpt_path(
    local_ckpt_path: Union[str, Path],
) -> Optional[str]:
    """
    Convert a local checkpoint path to its remote mirrored path.

    Returns None if the extension is unavailable or conversion fails.
    """
    remote_asset = storage.get_mirrored_remote_path(Path(local_ckpt_path))
    if remote_asset is None:
        return None
    try:
        return remote_asset.fully_qualified
    except Exception:
        return None


def resolve_test_checkpoint_path(
    test_ckpt_path: str,
    exp_output_dir: Path,
    maybe_use_remote_ckpt: bool = False,
) -> str:
    """
    Resolve checkpoint path with a->b->c->d fallback search strategy.

    Args:
        test_ckpt_path: Original checkpoint path from config
            (supports 'LATEST' keyword)
        exp_output_dir: Experiment output directory
        maybe_use_remote_ckpt: Whether to search remote mirrored paths

    Returns:
        Resolved checkpoint path

    Search strategy:
        0. If path ends with 'LATEST', find latest checkpoint in the directory
        a. For relative paths: try relative to CWD first, then exp output dir
        b. Search resolved path locally
        c. Search same step with different epoch locally
        d. Search original path on remote mirror (if enabled)
        e. Search same step with different epoch on remote mirror (if enabled)
    """

    # Helper function to check if checkpoint exists.
    def ckpt_exists(path_str: str) -> bool:
        try:
            if storage._is_remote_scheme(path_str):
                return storage.search_remote_checkpoint(path_str) is not None
            else:
                return Path(path_str).exists()
        except Exception:
            return False

    # Handle LATEST keyword before processing paths.
    if test_ckpt_path.endswith("LATEST"):
        checkpoint_dir_str = test_ckpt_path[:-6].rstrip("/")

        if checkpoint_dir_str.startswith("/") or storage._is_remote_scheme(
            checkpoint_dir_str
        ):
            checkpoint_dir_path = Path(checkpoint_dir_str)
        else:
            cwd_dir_path = Path(checkpoint_dir_str).resolve()
            if cwd_dir_path.exists():
                checkpoint_dir_path = cwd_dir_path
            else:
                checkpoint_dir_path = exp_output_dir / checkpoint_dir_str

        print(
            "Searching for latest checkpoint in directory:"
            f" {checkpoint_dir_path}"
        )
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir_path)

        if latest_checkpoint:
            print(f"Found latest checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)
        else:
            print(f"No checkpoints found in directory: {checkpoint_dir_path}")
            test_ckpt_path = checkpoint_dir_str

    # Absolute or remote path -- resolve via infra.
    if test_ckpt_path.startswith("/") or storage._is_remote_scheme(
        test_ckpt_path
    ):
        resolved = storage.resolve_path(test_ckpt_path)
        test_ckpt_path = str(resolved)
    # Relative path -- try CWD first, then experiment output directory.
    else:
        cwd_path = Path(test_ckpt_path).resolve()
        if cwd_path.exists():
            test_ckpt_path = str(cwd_path)
        else:
            test_ckpt_path = str(exp_output_dir / test_ckpt_path)

    final_ckpt_path = test_ckpt_path
    found = False

    # Search the original checkpoint path locally.
    if ckpt_exists(test_ckpt_path):
        print(f"Found checkpoint at original path: {test_ckpt_path}")
        found = True

    # Search for the same training step with a different epoch locally.
    if not found:
        alternative_path = search_same_step_ckpt_path(test_ckpt_path)
        if alternative_path and ckpt_exists(alternative_path):
            print(
                "Found local checkpoint with different epoch:"
                f" {alternative_path}"
            )
            final_ckpt_path = alternative_path
            found = True

    # Try the remote mirrored path when enabled and the file was not found locally.
    if not found and maybe_use_remote_ckpt:
        remote_ckpt_path = get_remote_mirrored_ckpt_path(test_ckpt_path)
        if remote_ckpt_path:
            # Search the original path on the remote mirror.
            if ckpt_exists(remote_ckpt_path):
                print(f"Found checkpoint on remote: {remote_ckpt_path}")
                final_ckpt_path = remote_ckpt_path
                found = True
            else:
                # Search for the same training step with a different epoch on the remote mirror.
                remote_alternative_path = search_same_step_ckpt_path(
                    remote_ckpt_path
                )
                if remote_alternative_path and ckpt_exists(
                    remote_alternative_path
                ):
                    print(
                        "Found remote checkpoint with different epoch: "
                        f"{remote_alternative_path}"
                    )
                    final_ckpt_path = remote_alternative_path
                    found = True

    # If we found a remote checkpoint, fetch it locally.
    if found and storage._is_remote_scheme(final_ckpt_path):
        try:
            resolved = storage.resolve_path(final_ckpt_path)
            if isinstance(resolved, Path):
                print(f"Fetched remote checkpoint to local: {resolved}")
                final_ckpt_path = str(resolved)
        except Exception as e:
            print(f"Warning: Failed to fetch remote checkpoint: {e}")

    if not found:
        print(f"Checkpoint not found at: {test_ckpt_path}")
        if maybe_use_remote_ckpt:
            print("Also checked remote mirrored paths")
        print("Proceeding with original path")

    print(f"Using test_ckpt_path: {final_ckpt_path}")
    return final_ckpt_path


class ModelCheckpointWithSync(ModelCheckpoint):
    """
    Custom ModelCheckpoint that automatically syncs the output directory to
    remote storage after each checkpoint is successfully saved to disk.

    This ensures that checkpoints are backed up immediately after being saved,
    providing protection against data loss from crashes or system failures.
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        output_dir: Optional[Path] = None,
        sync_enabled: bool = True,
        sync_update: bool = True,
        sync_remove_deleted: bool = False,
        sync_quiet: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the custom ModelCheckpoint with sync capabilities.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern
            output_dir: Full output directory to sync to remote storage
            sync_enabled: Whether to enable checkpoint syncing
            sync_update: Whether to update files that already exist on remote
            sync_remove_deleted: Whether to delete remote files that don't
                exist locally
            sync_quiet: Whether to suppress sync output messages
            **kwargs: Additional arguments passed to ModelCheckpoint
        """
        super().__init__(dirpath=dirpath, filename=filename, **kwargs)
        self.output_dir = output_dir
        self.sync_enabled = sync_enabled
        self.sync_update = sync_update
        self.sync_remove_deleted = sync_remove_deleted
        self.sync_quiet = sync_quiet

    def _save_checkpoint(self, trainer: Any, filepath: str) -> None:
        """
        Override the save method to add sync logic after successful save.

        Args:
            trainer: PyTorch Lightning trainer
            filepath: Path where checkpoint will be saved
        """
        super()._save_checkpoint(trainer, filepath)

        self._sync_to_remote_after_save(trainer, filepath)

    @rank_zero_only
    def _sync_to_remote_after_save(self, trainer: Any, filepath: str) -> None:
        """
        Sync output directory to remote storage after checkpoint save.

        Args:
            trainer: PyTorch Lightning trainer
            filepath: Path where checkpoint was saved
        """
        if not self.sync_enabled or self.output_dir is None:
            return

        try:
            current_step = trainer.global_step

            print(
                "[Checkpoint Sync] Syncing output directory to remote after"
                f" step {current_step} save"
            )

            storage.sync_output_dir(
                source_dir=self.output_dir,
                update=self.sync_update,
                remove_deleted=self.sync_remove_deleted,
                quiet=self.sync_quiet,
            )

            if not self.sync_quiet:
                print("[Checkpoint Sync] Successfully synced to remote")

        except Exception as e:
            print(f"[Checkpoint Sync] Warning: Failed to sync to remote: {e}")
            import traceback

            traceback.print_exc()
