# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The second tracer's hook: which installed tracer a profile test asks, and what is kept of a failed traced attempt."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.measure import second_tracer
from hyperloom.measure.second_tracer import ATTEMPT_LOG_FILE, ENTRY_POINT_GROUP, OUTPUT_DIR, SecondTracer


def _declared(monkeypatch: pytest.MonkeyPatch, **prepares: object) -> list[str]:
    """Declare ``prepares`` as other packages' entry points do; the groups asked for."""

    asked: list[str] = []

    def entry_points(*, group: str) -> list[SimpleNamespace]:
        asked.append(group)
        return [SimpleNamespace(name=name, load=lambda made=made: made) for name, made in prepares.items()]

    monkeypatch.setattr(second_tracer.metadata, "entry_points", entry_points)
    return asked


def test_with_no_tracer_installed_a_profile_is_given_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asked = _declared(monkeypatch)

    assert second_tracer.prepare((tmp_path / "point.yaml",), {}) is None
    assert asked == [ENTRY_POINT_GROUP]


def test_the_first_tracer_by_name_is_asked_with_the_points_and_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ready = SecondTracer("alpha", lambda points, environ: tuple(points), lambda output, ranks: (), lambda: None)
    seen: list[tuple[tuple[Path, ...], dict[str, str]]] = []

    def alpha(points, environ):
        seen.append((tuple(points), dict(environ)))
        return ready

    def zeta(points, environ):
        raise AssertionError("the second tracer by name was asked")

    _declared(monkeypatch, zeta=zeta, alpha=alpha)
    point = tmp_path / "point.yaml"

    assert second_tracer.prepare((point,), {"ROCM_PATH": "/opt/rocm"}) is ready
    assert seen == [((point,), {"ROCM_PATH": "/opt/rocm"})]


def test_a_tracer_that_says_why_not_gives_its_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _declared(monkeypatch, alpha=lambda points, environ: "alpha: skipped (not built for this image)")

    assert second_tracer.prepare((tmp_path / "point.yaml",), {}) == "alpha: skipped (not built for this image)"


def test_the_tail_of_a_failed_attempt_s_server_log_is_kept_beside_the_output(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_bytes(b"x" * 70_000 + b"Fatal Python error: Segmentation fault\n")

    second_tracer.keep_attempt(str(log), tmp_path / "out")
    second_tracer.keep_attempt(None, tmp_path / "nothing")
    second_tracer.keep_attempt(str(tmp_path / "missing.log"), tmp_path / "out")

    kept = (tmp_path / "out" / OUTPUT_DIR / ATTEMPT_LOG_FILE).read_bytes()
    assert len(kept) == 64 * 1024 and kept.endswith(b"Fatal Python error: Segmentation fault\n")
    assert not (tmp_path / "nothing").exists()
