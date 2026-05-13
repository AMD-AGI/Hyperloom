"""Tests for `_inferencex_patcher.ensure_benchmark_lib_patched`
(Hyperloom issue #194 §2).

The patcher's contract:

* Apply a minimal, backward-compatible patch to
  ``<INFERENCEX_PATH>/benchmarks/benchmark_lib.sh`` so it honours
  ``$NUM_PROMPTS`` when set, but behaves identically to upstream when
  it is unset.
* Idempotent — repeated calls are no-ops once the sentinel substring
  is present.
* Concurrency-safe — two callers racing the same checkout end up with
  one patched file, not double-patched garbage.
* Fail-soft — missing ``INFERENCEX_PATH``, missing file, or an
  already-mutated file all return ``False`` instead of raising, so
  unrelated unit tests and dry-runs don't blow up.

The fixtures here synthesize a fake ``InferenceX/benchmarks/`` tree
inside ``tmp_path`` so we never touch the real InferenceX checkout
pointed at by ``$INFERENCEX_PATH``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors._inferencex_patcher import (
    ensure_benchmark_lib_patched,
)


# A reduced fixture that captures the exact upstream shape the patcher
# targets. Keeping this verbatim (including the leading 8-space indent)
# is load-bearing — the patcher matches on that exact prefix to avoid
# false-positive matches against unrelated `num_prompts` references.
_UPSTREAM_FIXTURE = """\
#!/usr/bin/env bash
run_benchmark_serving() {
    local num_prompts=""
    local max_concurrency=""
    # ... arg parsing elided for fixture brevity ...
    if [[ "${PROFILE:-}" == "1" ]]; then
        profile_flag+=(--profile)
        num_prompts="$max_concurrency"
    fi
    invoke_benchmark --num-prompts "$num_prompts"
}
"""

_PATCHED_LINE = '        num_prompts="${NUM_PROMPTS:-$max_concurrency}"'
_LEGACY_LINE = '        num_prompts="$max_concurrency"'


@pytest.fixture
def fake_inferencex(tmp_path: Path) -> Path:
    """Build a minimal `<root>/benchmarks/benchmark_lib.sh` tree."""
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    lib = bench_dir / "benchmark_lib.sh"
    lib.write_text(_UPSTREAM_FIXTURE, encoding="utf-8")
    return tmp_path


# ===========================================================================
# Happy path: fresh checkout → patch lands; second call is a no-op.
# ===========================================================================
def test_first_call_patches_the_legacy_line(fake_inferencex):
    rc = ensure_benchmark_lib_patched(fake_inferencex)
    assert rc is True
    text = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    assert _PATCHED_LINE in text
    assert _LEGACY_LINE not in text


def test_second_call_is_a_noop(fake_inferencex):
    """Idempotency: re-applying must not double-patch or change bytes."""
    ensure_benchmark_lib_patched(fake_inferencex)
    after_first = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    rc = ensure_benchmark_lib_patched(fake_inferencex)
    assert rc is True
    after_second = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    assert after_first == after_second, (
        "Second call mutated the file — patch is not idempotent"
    )


def test_patched_line_appears_exactly_once(fake_inferencex):
    """Belt-and-braces: even with multiple invocations, the sentinel
    must appear exactly once. A double-patch (sentinel-in-sentinel)
    would corrupt bash parsing."""
    for _ in range(5):
        ensure_benchmark_lib_patched(fake_inferencex)
    text = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    assert text.count(_PATCHED_LINE) == 1


# ===========================================================================
# Backward-compatibility: the patched shell expression must reduce to
# the original behavior when NUM_PROMPTS is unset. We verify this by
# running the snippet through bash and inspecting the resulting value.
# ===========================================================================
def test_patch_is_backward_compatible_when_num_prompts_unset(
    fake_inferencex, tmp_path, monkeypatch,
):
    """Sanity-check that the patched line evaluates identically to the
    original when NUM_PROMPTS is not in the environment."""
    import shutil
    import subprocess
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable in this environment")
    ensure_benchmark_lib_patched(fake_inferencex)
    # Eval a minimal harness that mirrors the patched line.
    snippet = (
        'max_concurrency=42\n'
        'unset NUM_PROMPTS\n'
        'num_prompts="${NUM_PROMPTS:-$max_concurrency}"\n'
        'echo "$num_prompts"\n'
    )
    out = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "42"


def test_patched_line_uses_num_prompts_when_set(fake_inferencex):
    """The patch's whole point: when NUM_PROMPTS env IS set, it wins
    over the hard-coded reset."""
    import shutil
    import subprocess
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable in this environment")
    ensure_benchmark_lib_patched(fake_inferencex)
    snippet = (
        'max_concurrency=42\n'
        'NUM_PROMPTS=999\n'
        'num_prompts="${NUM_PROMPTS:-$max_concurrency}"\n'
        'echo "$num_prompts"\n'
    )
    out = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "999"


# ===========================================================================
# Fail-soft: missing config / file / legacy line must NOT raise.
# ===========================================================================
def test_returns_false_when_inferencex_path_unset(tmp_path, monkeypatch):
    """No INFERENCEX_PATH and no explicit arg → returns False, no
    crash. Lets dry-runs and CI-without-real-checkout proceed."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert ensure_benchmark_lib_patched(None) is False


def test_returns_false_when_benchmark_lib_missing(tmp_path):
    """A valid root with no benchmarks/ subtree must not raise."""
    assert ensure_benchmark_lib_patched(tmp_path) is False


def test_returns_false_when_legacy_line_missing(tmp_path):
    """If the file has been hand-patched to an unexpected shape or
    upstream layout has changed, the patcher refuses to guess. Caller
    sees False and can fall back to logging / failing-loud."""
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    lib = bench_dir / "benchmark_lib.sh"
    # Deliberately omit the legacy line.
    lib.write_text(
        "#!/usr/bin/env bash\n"
        "# This file was hand-patched to use a different shape.\n"
        "run_benchmark_serving() { echo 'something else'; }\n",
        encoding="utf-8",
    )
    rc = ensure_benchmark_lib_patched(tmp_path)
    assert rc is False
    # And the file must NOT have been mutated.
    assert "something else" in lib.read_text()


def test_already_patched_file_short_circuits(fake_inferencex):
    """If the sentinel is already present (e.g. left over from a
    previous run, possibly even by hand), the patcher returns True
    without touching the file."""
    lib = fake_inferencex / "benchmarks" / "benchmark_lib.sh"
    # Pre-apply the patch.
    lib.write_text(
        lib.read_text().replace(_LEGACY_LINE, _PATCHED_LINE),
        encoding="utf-8",
    )
    before = lib.read_text()
    rc = ensure_benchmark_lib_patched(fake_inferencex)
    assert rc is True
    assert lib.read_text() == before


# ===========================================================================
# INFERENCEX_PATH env fallback (when no explicit arg is provided).
# ===========================================================================
def test_env_var_is_used_when_no_explicit_path(fake_inferencex, monkeypatch):
    monkeypatch.setenv("INFERENCEX_PATH", str(fake_inferencex))
    rc = ensure_benchmark_lib_patched(None)
    assert rc is True
    lib = fake_inferencex / "benchmarks" / "benchmark_lib.sh"
    assert _PATCHED_LINE in lib.read_text()


# ===========================================================================
# Concurrency: multiple threads racing the same checkout must converge
# on a singly-patched file, never a double-patched or torn one.
# (Threads inside a single process is a slightly weaker test than
# multi-process flock, but it does exercise the under-lock re-check
# and the atomic-rename write path.)
# ===========================================================================
def test_concurrent_patchers_converge_to_single_patch(fake_inferencex):
    results: list[bool] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(ensure_benchmark_lib_patched(fake_inferencex))
        except BaseException as exc:  # noqa: BLE001 - test-only
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors
    assert all(results), results
    text = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    # Exactly one patched line — no doubles, no remaining legacy line.
    assert text.count(_PATCHED_LINE) == 1
    assert _LEGACY_LINE not in text
