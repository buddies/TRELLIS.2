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


# Hugging Face repo id -> ModelScope repo id.
# Repos that share the same id on both hubs (the common case, e.g.
# ``microsoft/...`` and ``facebook/...``) are omitted.
HF_TO_MODELSCOPE = {
    'ZhengPeng7/BiRefNet': 'modelscope/BiRefNet',
}


def resolve_repo_id(repo_id: str) -> str:
    """Return the ModelScope repo id for a (possibly Hugging Face) repo id."""
    return HF_TO_MODELSCOPE.get(repo_id, repo_id)


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
        from modelscope import snapshot_download as _ms_snapshot
        resolved = resolve_repo_id(repo_id)
        path = _ms_snapshot(
            resolved,
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
        from modelscope import snapshot_download as _ms_snapshot
        resolved = resolve_repo_id(repo_id)
        path = _ms_snapshot(resolved, allow_file_pattern=[filename], revision=revision)
        candidate = os.path.join(path, filename)
        if not os.path.exists(candidate):
            # If the pattern-filtered download did not produce the file, fall back
            # to a full download and look the file up.
            full_path = _ms_snapshot(resolved, revision=revision)
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
