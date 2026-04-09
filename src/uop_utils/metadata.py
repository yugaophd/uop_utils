"""Metadata and provenance helpers."""

from __future__ import annotations

import os
import subprocess
from typing import Optional


def get_git_governance_info(repo_path: Optional[str] = None, script_path: Optional[str] = None, status_limit: int = 20) -> dict:
    """Collect git provenance metadata for processing-governance records."""
    repo_path = os.path.abspath(repo_path or os.getcwd())
    info = {
        'git_metadata_available': 0,
        'git_repo_path_requested': repo_path,
    }

    script_abs = None
    if script_path is not None:
        script_abs = os.path.abspath(script_path)
        info['processing_script'] = os.path.basename(script_abs)
        info['processing_script_path'] = script_abs

    def _run_git(args, cwd):
        result = subprocess.run(
            ['git', *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        repo_root = _run_git(['rev-parse', '--show-toplevel'], repo_path)
        info['git_repo_root'] = repo_root
        info['git_branch'] = _run_git(['branch', '--show-current'], repo_root)
        info['git_commit'] = _run_git(['rev-parse', 'HEAD'], repo_root)
        info['git_commit_short'] = _run_git(['rev-parse', '--short', 'HEAD'], repo_root)

        try:
            info['git_remote_origin_url'] = _run_git(['config', '--get', 'remote.origin.url'], repo_root)
        except Exception:
            info['git_remote_origin_url'] = ''

        try:
            info['git_describe'] = _run_git(['describe', '--always', '--dirty', '--tags'], repo_root)
        except Exception:
            info['git_describe'] = info['git_commit_short']

        status_text = _run_git(['status', '--short'], repo_root)
        status_lines = [line for line in status_text.splitlines() if line.strip()]
        info['git_is_dirty'] = int(len(status_lines) > 0)
        info['git_status'] = '; '.join(status_lines[:status_limit])
        info['git_metadata_available'] = 1

        if script_abs is not None:
            try:
                info['processing_script_relative_path'] = os.path.relpath(script_abs, repo_root)
            except Exception:
                info['processing_script_relative_path'] = script_abs

    except Exception as e:
        info['git_error'] = str(e)

    return info
