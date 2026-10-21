# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for tests that need a sandbox to get a private view of a framework directory."""

from __future__ import annotations

from pathlib import Path

from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.socket import SandboxUnavailable

__all__ = ["make_framework", "overlay_unavailable_reason"]


def make_framework(parent: Path, name: str = "sgl-workspace") -> Path:
    """A small stand-in for the image's framework install: a tuned table, files to delete and one to change the mode of."""
    root = parent / name
    for relative, text in (
        ("aiter/configs/tuned.csv", "M,N,K\n64,5120,5120\n"),
        ("aiter/configs/framework_victim", "present\n"),
        ("sglang/framework_victim", "present\n"),
        ("sglang/framework_other", "other\n"),
    ):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    return root


def overlay_unavailable_reason(parent: Path) -> str | None:
    """Why this machine cannot give a sandbox a private view of a directory, or ``None`` when it can.

    The probe is a real sandbox create with the real launcher on an empty directory, so a machine that passes it is
    one where the tests below mean something.
    """
    root = parent / "overlay-probe-root"
    root.mkdir(parents=True)
    template = LocalTemplate(
        "probe", "measurement", command=("/bin/true",), sealed=True, framework_roots=(str(root),), framework_upper_mib=1
    )
    with LocalSandboxProvider([template], base_dir=parent / "overlay-probe-sandboxes") as provider:
        try:
            provider.create("probe", role="measurement", timeout_sec=30, metadata={})
        except SandboxUnavailable as exc:
            return str(exc)
    return None
