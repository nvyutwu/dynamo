# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM release gates that survive untagged source builds.

A vLLM wheel built from a checkout without a reachable release tag reports
setuptools-scm's fallback version (``0.1.dev0+g<sha>``), or ``dev`` when its
``_version`` module is missing. Comparing that against a release number would
treat a current main build as an ancient release. Such builds come from recent
main, so they pass every release gate. Tagged releases compare as before.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version


def is_untagged_source_build(version: str) -> bool:
    """Whether ``version`` is setuptools-scm's no-tag fallback (or vLLM's ``dev``)."""
    try:
        parsed = Version(version)
    except InvalidVersion:
        return version == "dev"
    return parsed.dev is not None and parsed.release[:2] <= (0, 1)


def vllm_release_at_least(major: int, minor: int) -> bool:
    """``vllm.version.__version_tuple__[:2] >= (major, minor)``, or an untagged build."""
    from vllm.version import __version__, __version_tuple__

    if tuple(__version_tuple__[:2]) >= (major, minor):
        return True
    return is_untagged_source_build(str(__version__))
