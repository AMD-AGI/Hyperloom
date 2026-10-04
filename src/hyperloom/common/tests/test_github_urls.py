# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.common.github_urls."""

from __future__ import annotations

import pytest

from hyperloom.common.github_urls import repo_slug


def test_repo_slug_parses_https_ssh_and_git_suffix() -> None:
    """repo_slug handles https/.git/ssh URLs uniformly."""
    assert repo_slug("https://github.com/sgl-project/sglang.git") == "sgl-project/sglang"
    assert repo_slug("https://github.com/sgl-project/sglang") == "sgl-project/sglang"
    assert repo_slug("git@github.com:sgl-project/sglang.git") == "sgl-project/sglang"


def test_repo_slug_rejects_malformed() -> None:
    """Non-GitHub-shaped URLs raise ValueError."""
    with pytest.raises(ValueError):
        repo_slug("not-a-url")


def test_repo_slug_rejects_github_substring_in_path() -> None:
    """URLs that embed github.com in the path must not be accepted."""
    with pytest.raises(ValueError):
        repo_slug("https://evil.com/github.com/owner/repo.git")
    with pytest.raises(ValueError):
        repo_slug("https://github.com.evil.com/owner/repo.git")
