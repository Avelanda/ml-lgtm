"""
Storage abstraction for local paths and optional cluster-backed I/O.

For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

For third-party code see ACKNOWLEDGMENTS file.

Storage abstraction layer.

When the ``cluster`` package is available on sys.path, extended features
(remote storage paths, cloud sync, cluster status) are enabled.
When it is not available, all functions gracefully degrade to local-only
behavior.

All code in ``src/lgtm/`` should go through this module instead of
importing storage/cluster libraries directly.
"""

from pathlib import Path
from typing import Optional

_ext = None

try:
    import cluster as _ext
except ImportError:
    pass


def _is_remote_scheme(path_str: str) -> bool:
    """
    True if *path_str* uses a remote URI scheme (anything with ``://``
    except ``file://``).
    """
    if "://" not in path_str:
        return False
    scheme = path_str.split("://", 1)[0]
    return scheme not in ("file",)


def is_available() -> bool:
    """
    True when the extension package is installed and its underlying
    libraries are both importable.
    """
    if _ext is None:
        return False
    return _ext.is_available()


def resolve_path(path) -> Path:
    """
    Resolve *path* to a local ``Path``.

    * Local paths are returned as-is (with ``~`` expanded).
    * Remote paths (custom URI schemes) are autofetched to a local cache
      when ``lgtmext`` is available.  If the remote path points to a
      directory, the returned object may be a remote asset that supports
      ``iterdir()`` and ``/`` operations.
    * Using a remote scheme without ``lgtmext`` raises ``RuntimeError``.
    """
    if _ext is not None:
        return _ext.resolve_path(path)

    path_str = str(path)
    if _is_remote_scheme(path_str):
        raise RuntimeError(
            "Remote paths require the `lgtmext` package. "
            "Use local file paths instead."
        )
    return Path(path_str).expanduser()


def is_remote_asset(obj) -> bool:
    """
    True when *obj* is a remote asset managed by the extension package.
    Always False when ``lgtmext`` is not available.
    """
    if _ext is None:
        return False
    return _ext.is_remote_asset(obj)


def list_children(path, suffix: Optional[str] = None):
    """
    List children of a directory *path*.
    Works for both local ``Path`` objects and remote assets.
    """
    if _ext is not None and is_remote_asset(path):
        return _ext.list_children(path, suffix=suffix)
    path = Path(str(path))
    children = sorted(path.iterdir())
    if suffix:
        children = [c for c in children if c.suffix == suffix]
    return children


def fetch_to_local(path, quiet: bool = True) -> Path:
    """
    Ensure *path* is available as a local file.  No-op for local paths.
    """
    if _ext is not None and is_remote_asset(path):
        return _ext.fetch_to_local(path, quiet=quiet)
    return Path(str(path))


def sync_output_dir(
    source_dir,
    update: bool = True,
    remove_deleted: bool = False,
    quiet: bool = False,
) -> None:
    """
    Sync a local output directory to its remote mirror.
    No-op when ``lgtmext`` is not available.
    """
    if _ext is None:
        if not quiet:
            print("Warning: lgtmext not available, skipping remote sync")
        return
    _ext.sync_output_dir(
        source_dir=source_dir,
        update=update,
        remove_deleted=remove_deleted,
        quiet=quiet,
    )


def get_task_id() -> Optional[str]:
    """
    Return the current cluster task ID, or ``None``.
    """
    if _ext is None:
        return None
    return _ext.get_task_id()


def set_status_message(message: str) -> None:
    """
    Set the cluster status message.  No-op outside of a cluster environment.
    """
    if _ext is None:
        return
    _ext.set_status_message(message)


def get_mirrored_remote_path(local_path):
    """
    Convert a local path to its mirrored remote path.
    Returns ``None`` when ``lgtmext`` is not available or conversion fails.
    """
    if _ext is None:
        return None
    return _ext.get_mirrored_remote_path(local_path)


def search_remote_checkpoint(path_str: str) -> Optional[str]:
    """
    Check if a checkpoint exists on the remote mirror.
    Returns the fully-qualified remote path, or ``None``.
    """
    if _ext is None:
        return None
    return _ext.search_remote_checkpoint(path_str)


def search_same_step_remote_checkpoint(path_str: str) -> Optional[str]:
    """
    Search for a checkpoint with the same step but different epoch on
    the remote mirror.  Returns ``None`` when not found or ``lgtmext``
    is unavailable.
    """
    if _ext is None:
        return None
    return _ext.search_same_step_remote_checkpoint(path_str)


def setup_environment() -> None:
    """
    Configure environment variables needed by the infrastructure.
    No-op when ``lgtmext`` is not available.
    """
    if _ext is None:
        return
    _ext.setup_environment()
