"""
Unified model-hub utility.

Downloads model weights preferentially from ModelScope (a drop-in replacement
for Hugging Face), and falls back to Hugging Face when a repository is not
available on ModelScope.

This lets downstream code keep using the familiar Hugging Face-style repo ids
(e.g. ``microsoft/TRELLIS.2-4B``, ``facebook/dinov3-vitl16-pretrain-lvd1689m``)
and transparently fetch them from ModelScope instead.
"""
import os
from typing import *


# ModelScope reads these env vars once, when ``modelscope`` is imported. Set
# sensible defaults *before* that import so that multi-GB weight files download
# with several parallel connections instead of a single slow stream.
#   MODELSCOPE_DOWNLOAD_PARALLELS     -> parallel connections per large file
#   MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB -> use parallel path above this size
os.environ.setdefault('MODELSCOPE_DOWNLOAD_PARALLELS', '16')
os.environ.setdefault('MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB', '128')


# Hugging Face repo id -> ModelScope repo id.
# Repos that share the same id on both hubs (the common case, e.g.
# ``microsoft/...`` and ``facebook/...``) are omitted.
HF_TO_MODELSCOPE = {
    'ZhengPeng7/BiRefNet': 'modelscope/BiRefNet',
}


def resolve_repo_id(repo_id: str) -> str:
    """Return the ModelScope repo id for a (possibly Hugging Face) repo id."""
    return HF_TO_MODELSCOPE.get(repo_id, repo_id)


# Number of parallel download workers. Override via env when needed.
_DEFAULT_MAX_WORKERS = int(os.environ.get('MODELSCOPE_MAX_WORKERS', '8'))

# Cache of already-resolved repository directories, keyed by (resolved_id, revision).
_repo_dir_cache: dict = {}


def _modelscope_snapshot(model_id: str, revision, allow_file_pattern=None, ignore_file_pattern=None) -> str:
    """Wrapper around ``modelscope.snapshot_download`` that uses more parallel
    workers, and falls back to default args on older versions that do not accept
    ``max_workers``."""
    from modelscope import snapshot_download as _ms_snapshot
    try:
        return _ms_snapshot(
            model_id,
            revision=revision,
            allow_file_pattern=allow_file_pattern,
            ignore_file_pattern=ignore_file_pattern,
            max_workers=_DEFAULT_MAX_WORKERS,
        )
    except TypeError:
        return _ms_snapshot(
            model_id,
            revision=revision,
            allow_file_pattern=allow_file_pattern,
            ignore_file_pattern=ignore_file_pattern,
        )


def snapshot_download(
    repo_id: str,
    allow_file_pattern: Optional[List[str]] = None,
    ignore_file_pattern: Optional[List[str]] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Download a model repository (or only the files matching ``allow_file_pattern``).

    Prefers ModelScope and falls back to Hugging Face.

    Args:
        repo_id: The repository id, e.g. ``microsoft/TRELLIS.2-4B``.
        allow_file_pattern: If given, only download files matching these patterns.
        ignore_file_pattern: If given, skip files matching these patterns.
        revision: Git revision/branch to download.

    Returns:
        The local directory containing the downloaded files.
    """
    try:
        path = _modelscope_snapshot(
            resolve_repo_id(repo_id),
            revision=revision,
            allow_file_pattern=allow_file_pattern,
            ignore_file_pattern=ignore_file_pattern,
        )
        return os.path.abspath(path)
    except Exception:
        from huggingface_hub import snapshot_download as _hf_snapshot
        return os.path.abspath(_hf_snapshot(
            repo_id,
            revision=revision,
            allow_patterns=allow_file_pattern,
            ignore_patterns=ignore_file_pattern,
        ))


def get_repo_dir(repo_id: str, revision: Optional[str] = None) -> str:
    """
    Download the *entire* repository once (cached) and return its local directory.

    This is far more efficient than repeatedly pulling single files for large
    repos: the repo is fetched a single time and every subsequent file access
    is served from the local cache on disk.
    """
    key = (resolve_repo_id(repo_id), revision)
    if key in _repo_dir_cache:
        return _repo_dir_cache[key]

    resolved = resolve_repo_id(repo_id)
    try:
        path = _modelscope_snapshot(resolved, revision=revision)
    except Exception:
        from huggingface_hub import snapshot_download as _hf_snapshot
        path = _hf_snapshot(repo_id, revision=revision)

    path = os.path.abspath(path)
    _repo_dir_cache[key] = path
    return path


def hf_hub_download(repo_id: str, filename: str, revision: Optional[str] = None, **kwargs) -> str:
    """
    Download a single file from a repository, preferring ModelScope.

    Args:
        repo_id: The repository id.
        filename: The file name/path within the repository (e.g. ``pipeline.json``).

    Returns:
        The absolute local path to the downloaded file.
    """
    model_file = None
    try:
        path = _modelscope_snapshot(
            resolve_repo_id(repo_id),
            revision=revision,
            allow_file_pattern=[filename],
        )
        candidate = os.path.join(path, filename)
        if not os.path.exists(candidate):
            # If the pattern-filtered download did not produce the file, fall back
            # to the (cached) full-repo directory and look it up there.
            full_path = get_repo_dir(repo_id, revision=revision)
            candidate = os.path.join(full_path, filename)
        if os.path.exists(candidate):
            model_file = os.path.abspath(candidate)
    except Exception:
        model_file = None

    if model_file is None:
        # Fall back to Hugging Face.
        from huggingface_hub import hf_hub_download as _hf_download
        model_file = os.path.abspath(_hf_download(repo_id, filename, revision=revision, **kwargs))
    return model_file
