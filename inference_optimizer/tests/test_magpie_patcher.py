# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_magpie_patcher.ensure_magpie_atomic_scripts_patch`` (bugs.md §C #1)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors._magpie_patcher import (
    _LEGACY_BLOCK,
    _PATCH_SENTINEL,
    _PATCHED_BLOCK,
    _REMOTE_TRUST_SENTINEL,
    _UPSTREAM_ATOMIC_HELPER,
    _extract_prepare_region,
    _upstream_is_already_atomic,
    ensure_magpie_atomic_scripts_patch,
    magpie_scripts_patch_status,
)


# Reduced fixture capturing the upstream shape the patcher targets.
# Indentation is load-bearing (patcher matches the exact 12-space prefix).
_UPSTREAM_BENCHMARKER_PY = """\
\"\"\"Reduced benchmarker.py — only the surface the patcher cares about.\"\"\"
import shutil
from pathlib import Path


class _FakeBenchmarker:
    def __init__(self, magpie_scripts_dir, target_dir):
        self.magpie_scripts_dir = Path(magpie_scripts_dir)
        self.target_dir = Path(target_dir)

    def _prepare_benchmark_scripts(self):
        magpie_scripts = self.magpie_scripts_dir
        target_dir = self.target_dir
        if not magpie_scripts.exists():
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for script in magpie_scripts.glob("*.sh"):
            target_file = target_dir / script.name
            shutil.copy2(script, target_file)
            target_file.chmod(0o755)
"""


# Newer upstream Magpie: copy loop delegates to ``_copy_benchmark_script_atomic``
# (mkstemp + copy2 + chmod + os.replace), so the #C1 patch is a redundant no-op.
_UPSTREAM_ATOMIC_BENCHMARKER_PY = """\
\"\"\"Reduced benchmarker.py — upstream refactored to an atomic copy helper.\"\"\"
import os
import shutil
import tempfile
from pathlib import Path


class _FakeBenchmarker:
    def __init__(self, magpie_scripts_dir, target_dir):
        self.magpie_scripts_dir = Path(magpie_scripts_dir)
        self.target_dir = Path(target_dir)

    def _prepare_benchmark_scripts(self):
        magpie_scripts = self.magpie_scripts_dir
        target_dir = self.target_dir
        if not magpie_scripts.exists():
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for script in magpie_scripts.glob("*.sh"):
            target_file = target_dir / script.name
            self._copy_benchmark_script_atomic(script, target_file)

    @staticmethod
    def _copy_benchmark_script_atomic(script, target_file):
        target_file = Path(target_file)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target_file.name}.", dir=str(target_file.parent),
        )
        os.close(fd)
        shutil.copy2(script, tmp_name)
        os.chmod(tmp_name, 0o755)
        os.replace(tmp_name, target_file)
"""


# Same race-safe outcome inlined: temp-file + rename directly in
# ``_prepare_benchmark_scripts`` (region-scoped detection signal).
_INLINE_ATOMIC_BENCHMARKER_PY = """\
\"\"\"Reduced benchmarker.py — inline atomic copy, no named helper.\"\"\"
import os
import shutil
import tempfile
from pathlib import Path


class _FakeBenchmarker:
    def __init__(self, magpie_scripts_dir, target_dir):
        self.magpie_scripts_dir = Path(magpie_scripts_dir)
        self.target_dir = Path(target_dir)

    def _prepare_benchmark_scripts(self):
        magpie_scripts = self.magpie_scripts_dir
        target_dir = self.target_dir
        if not magpie_scripts.exists():
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for script in magpie_scripts.glob("*.sh"):
            target_file = target_dir / script.name
            fd, tmp_name = tempfile.mkstemp(dir=str(target_dir))
            os.close(fd)
            shutil.copy2(script, tmp_name)
            os.chmod(tmp_name, 0o755)
            os.replace(tmp_name, target_file)
"""


_UPSTREAM_SGLANG_MI300X_SH = """\
#!/usr/bin/env bash
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    # Remote server: call Python benchmark_serving.py directly.
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct || exit $?
  else
    run_benchmark_serving --model "$MODEL" || exit $?
  fi
fi
"""


# Genuine layout drift: unrecognisable prepare body + atomic ops only in an
# unrelated method. Region scoping must keep this out of "already atomic".
_GARBAGE_WITH_UNRELATED_ATOMIC_PY = """\
\"\"\"Reduced benchmarker.py — drifted prepare; atomic ops live elsewhere.\"\"\"
import os
import tempfile


class _FakeBenchmarker:
    def _prepare_benchmark_scripts(self):
        # body refactored to a shape the patcher does not recognise
        raise NotImplementedError("hand-edited")

    def _unrelated_helper(self, target_file):
        fd, tmp_name = tempfile.mkstemp()
        os.close(fd)
        os.replace(tmp_name, target_file)
"""


@pytest.fixture
def fake_magpie(tmp_path: Path) -> Path:
    """Build ``<root>/Magpie/modes/benchmark/benchmarker.py`` tree."""
    bench_dir = tmp_path / "Magpie" / "modes" / "benchmark"
    bench_dir.mkdir(parents=True)
    (bench_dir / "__init__.py").write_text("", encoding="utf-8")
    (bench_dir / "benchmarker.py").write_text(
        _UPSTREAM_BENCHMARKER_PY, encoding="utf-8",
    )
    return tmp_path


def _write_magpie_tree(root: Path, benchmarker_src: str) -> Path:
    """Materialise a minimal Magpie tree under ``root`` and return the
    ``benchmarker.py`` path."""
    bench_dir = root / "Magpie" / "modes" / "benchmark"
    bench_dir.mkdir(parents=True)
    (bench_dir / "__init__.py").write_text("", encoding="utf-8")
    bench_py = bench_dir / "benchmarker.py"
    bench_py.write_text(benchmarker_src, encoding="utf-8")
    return bench_py


def _write_sglang_script(root: Path, src: str = _UPSTREAM_SGLANG_MI300X_SH) -> Path:
    script_dir = root / "Magpie" / "scripts" / "benchmark"
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / "sglang_mi300x.sh"
    script.write_text(src, encoding="utf-8")
    script.chmod(0o755)
    return script


# Basic shape / sanity
def test_legacy_block_is_present_in_fixture():
    """Sanity: the fixture must contain the exact block the patcher matches."""
    assert _LEGACY_BLOCK in _UPSTREAM_BENCHMARKER_PY
    assert _PATCH_SENTINEL not in _UPSTREAM_BENCHMARKER_PY


def test_missing_magpie_dir_returns_false(tmp_path: Path):
    """No MAGPIE_DIR / no benchmarker.py → soft return False."""
    empty = tmp_path / "no_magpie_here"
    empty.mkdir()
    assert ensure_magpie_atomic_scripts_patch(empty) is False


def test_no_arg_and_no_env_returns_false(monkeypatch):
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert ensure_magpie_atomic_scripts_patch(None) is False


def test_env_fallback_resolves_path(monkeypatch, fake_magpie: Path):
    monkeypatch.setenv("MAGPIE_DIR", str(fake_magpie))
    assert ensure_magpie_atomic_scripts_patch(None) is True


# Patch application + idempotency
def test_patch_applied_replaces_legacy_block(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    text = bench_py.read_text(encoding="utf-8")
    assert _PATCH_SENTINEL in text
    assert _LEGACY_BLOCK not in text
    assert text.count(_PATCH_SENTINEL) == 1


def test_patch_is_idempotent(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    after_first = bench_py.read_text(encoding="utf-8")
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    after_second = bench_py.read_text(encoding="utf-8")
    assert after_first == after_second
    assert after_second.count(_PATCH_SENTINEL) == 1


def test_sglang_remote_client_trust_patch_is_env_gated(fake_magpie: Path):
    script = _write_sglang_script(fake_magpie)

    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    text = script.read_text(encoding="utf-8")

    assert _REMOTE_TRUST_SENTINEL in text
    assert '[[ "${MAGPIE_TRUST_REMOTE_CODE:-0}" == "1" ]]' in text
    assert "magpie_run_benchmark_serving_remote_direct trust" in text
    assert "magpie_run_benchmark_serving_remote_direct || exit $?" in text

    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    assert script.read_text(encoding="utf-8") == text


def test_remote_trust_drift_is_reported_separately(
    tmp_path: Path,
    caplog,
):
    """Atomic copy can be fixed while the SGLang trust patch drifts.

    The status API must expose those as separate bits so install.sh can warn
    about remote trust specifically instead of blaming the atomic-copy patch.
    """
    _write_magpie_tree(tmp_path, _UPSTREAM_ATOMIC_BENCHMARKER_PY)
    script = _write_sglang_script(
        tmp_path,
        _UPSTREAM_SGLANG_MI300X_SH.replace(
            "magpie_run_benchmark_serving_remote_direct || exit $?",
            "magpie_run_benchmark_serving_remote_direct \"$@\" || exit $?",
        ),
    )

    with caplog.at_level(logging.WARNING):
        status = magpie_scripts_patch_status(tmp_path)

    assert status.atomic_ok is True
    assert status.remote_trust_ok is False
    assert status.ok is False
    assert _REMOTE_TRUST_SENTINEL not in script.read_text(encoding="utf-8")
    assert any(
        "remote trust patch did not apply" in r.getMessage()
        for r in caplog.records
    )

    # The bool compat wrapper reflects the atomic-copy race only (its name /
    # docstring), so an optional remote-trust drift must NOT flip it to False.
    assert ensure_magpie_atomic_scripts_patch(tmp_path) is True


def test_patch_preserves_file_mode(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    bench_py.chmod(0o644)
    pre_mode = bench_py.stat().st_mode
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    post_mode = bench_py.stat().st_mode
    assert pre_mode == post_mode, (
        f"file mode changed: {oct(pre_mode)} -> {oct(post_mode)}"
    )


# Layout-drift fail-soft (install script escalates to fail-loud)
def test_fail_soft_when_legacy_block_missing(tmp_path: Path):
    """Missing legacy block → patcher returns False, leaves the file alone."""
    bench_dir = tmp_path / "Magpie" / "modes" / "benchmark"
    bench_dir.mkdir(parents=True)
    drifted = (
        "class _FakeBenchmarker:\n"
        "    def _prepare_benchmark_scripts(self):\n"
        "        # someone replaced the body, no shutil.copy2 here\n"
        "        pass\n"
    )
    (bench_dir / "benchmarker.py").write_text(drifted, encoding="utf-8")
    assert ensure_magpie_atomic_scripts_patch(tmp_path) is False
    assert (bench_dir / "benchmarker.py").read_text(encoding="utf-8") == drifted


def test_already_patched_returns_true_without_rewriting(fake_magpie: Path):
    """An already-present sentinel is accepted as patched without rewriting."""
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    bench_py.write_text(
        f"# top of file\n# {_PATCH_SENTINEL} present here\npass\n",
        encoding="utf-8",
    )
    pre = bench_py.read_text(encoding="utf-8")
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    assert bench_py.read_text(encoding="utf-8") == pre


# Upstream-aware: an already-atomic Magpie is "already fixed", not drifted.
#   (1) atomic upstream -> no-op True; (2) legacy -> patched; (3) already-patched
#   -> no-op; (4) neither -> False + warning.
def _patcher_warnings(caplog) -> list:
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "_magpie_patcher" in r.name
    ]


def test_atomic_helper_upstream_is_noop_true(tmp_path: Path, caplog):
    """(1) Atomic upstream: no-op True, file unchanged, no warning, info no-op line."""
    bench_py = _write_magpie_tree(tmp_path, _UPSTREAM_ATOMIC_BENCHMARKER_PY)
    pre = bench_py.read_text(encoding="utf-8")

    with caplog.at_level(logging.INFO):
        result = ensure_magpie_atomic_scripts_patch(tmp_path)

    assert result is True
    post = bench_py.read_text(encoding="utf-8")
    assert post == pre, "already-atomic upstream must not be rewritten"
    assert _PATCH_SENTINEL not in post
    assert not _patcher_warnings(caplog), "no-op must not warn"
    assert any(
        "already performs atomic script copy" in r.getMessage()
        and "_magpie_patcher" in r.name
        for r in caplog.records
    ), "expected the explicit already-atomic info line"


def test_inline_atomic_upstream_is_noop_true(tmp_path: Path, caplog):
    """(1b) Inline atomic outcome (no named helper) is still a no-op True, file untouched."""
    bench_py = _write_magpie_tree(tmp_path, _INLINE_ATOMIC_BENCHMARKER_PY)
    pre = bench_py.read_text(encoding="utf-8")

    with caplog.at_level(logging.INFO):
        assert ensure_magpie_atomic_scripts_patch(tmp_path) is True

    assert bench_py.read_text(encoding="utf-8") == pre
    assert not _patcher_warnings(caplog)


def test_atomic_upstream_fixture_still_copies_scripts(tmp_path: Path):
    """End-to-end sanity: the no-op path leaves a working Magpie that still copies scripts."""
    bench_py = _write_magpie_tree(tmp_path, _UPSTREAM_ATOMIC_BENCHMARKER_PY)
    assert ensure_magpie_atomic_scripts_patch(tmp_path) is True

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    (src_dir / "vllm_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho hi\n", encoding="utf-8",
    )
    dst_dir = tmp_path / "InferenceX_benchmarks"

    namespace: dict = {"__name__": "_atomic_benchmarker_smoke"}
    exec(compile(bench_py.read_text("utf-8"), str(bench_py), "exec"), namespace)
    inst = namespace["_FakeBenchmarker"](src_dir, dst_dir)
    inst._prepare_benchmark_scripts()

    out = dst_dir / "vllm_mi300x.sh"
    assert out.is_file()
    assert out.read_text("utf-8") == (src_dir / "vllm_mi300x.sh").read_text("utf-8")
    assert out.stat().st_mode & 0o111


def test_legacy_block_still_patched_true(tmp_path: Path):
    """(2) Old Magpie (legacy block) is still patched in place: sentinel present, block gone."""
    bench_py = _write_magpie_tree(tmp_path, _UPSTREAM_BENCHMARKER_PY)
    assert ensure_magpie_atomic_scripts_patch(tmp_path) is True
    text = bench_py.read_text(encoding="utf-8")
    assert _PATCH_SENTINEL in text
    assert _LEGACY_BLOCK not in text


def test_already_patched_sentinel_is_noop_true(tmp_path: Path):
    """(3) A previously-patched checkout (sentinel present) is a fast-path no-op."""
    patched_src = _UPSTREAM_BENCHMARKER_PY.replace(_LEGACY_BLOCK, _PATCHED_BLOCK, 1)
    assert _PATCH_SENTINEL in patched_src
    bench_py = _write_magpie_tree(tmp_path, patched_src)
    pre = bench_py.read_text(encoding="utf-8")
    assert ensure_magpie_atomic_scripts_patch(tmp_path) is True
    assert bench_py.read_text(encoding="utf-8") == pre


def test_neither_legacy_nor_atomic_warns_false(tmp_path: Path, caplog):
    """(4) Neither legacy nor atomic is a genuine anomaly: False, untouched, warns."""
    drifted = (
        "class _FakeBenchmarker:\n"
        "    def _prepare_benchmark_scripts(self):\n"
        "        # someone replaced the body with no copy logic at all\n"
        "        return None\n"
    )
    bench_py = _write_magpie_tree(tmp_path, drifted)

    with caplog.at_level(logging.INFO):
        result = ensure_magpie_atomic_scripts_patch(tmp_path)

    assert result is False
    assert bench_py.read_text(encoding="utf-8") == drifted
    warnings = _patcher_warnings(caplog)
    assert warnings, "genuine anomaly must warn"
    assert any("manual review" in r.getMessage().lower() for r in warnings)


def test_unrelated_atomic_ops_do_not_count_as_fixed(tmp_path: Path, caplog):
    """Region scoping guard: atomic ops in an unrelated method don't count as fixed → False + warning."""
    bench_py = _write_magpie_tree(tmp_path, _GARBAGE_WITH_UNRELATED_ATOMIC_PY)

    with caplog.at_level(logging.INFO):
        result = ensure_magpie_atomic_scripts_patch(tmp_path)

    assert result is False
    assert bench_py.read_text(encoding="utf-8") == _GARBAGE_WITH_UNRELATED_ATOMIC_PY
    assert _patcher_warnings(caplog)


# Concurrency — N patchers racing the same file
def test_concurrent_patchers_produce_one_patch(fake_magpie: Path):
    """Eight concurrent ``ensure_*`` calls must end with exactly one applied patch."""
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    barrier = threading.Barrier(8)
    results: list[bool] = [False] * 8

    def _run(idx: int) -> None:
        barrier.wait()
        results[idx] = ensure_magpie_atomic_scripts_patch(fake_magpie)

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results), f"some patcher threads returned False: {results}"
    text = bench_py.read_text(encoding="utf-8")
    assert text.count(_PATCH_SENTINEL) == 1
    assert _LEGACY_BLOCK not in text


def test_reader_never_sees_torn_file(fake_magpie: Path):
    """bugs.md §C #1: a concurrent reader must never see a torn file —
    every snapshot is either the verbatim original or contains the sentinel."""
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    original = bench_py.read_text(encoding="utf-8")

    stop = threading.Event()
    torn_snapshots: list[str] = []

    def _reader() -> None:
        while not stop.is_set():
            try:
                snapshot = bench_py.read_text(encoding="utf-8")
            except OSError:
                continue
            if snapshot == original:
                continue
            if _PATCH_SENTINEL in snapshot:
                continue
            torn_snapshots.append(snapshot)

    readers = [threading.Thread(target=_reader) for _ in range(4)]
    for r in readers:
        r.start()
    try:
        time.sleep(0.05)
        assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
        # Cover both the during-rename and post-rename window.
        time.sleep(0.1)
    finally:
        stop.set()
        for r in readers:
            r.join()

    assert not torn_snapshots, (
        f"observed {len(torn_snapshots)} torn snapshot(s); first 200 chars: "
        f"{torn_snapshots[0][:200]!r}"
    )


# Smoke — patched benchmarker.py is valid Python with correct semantics
def test_patched_file_is_valid_python(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    res = subprocess.run(
        [sys.executable, "-c",
         "import py_compile, sys; py_compile.compile(sys.argv[1], doraise=True)",
         str(bench_py)],
        check=False, capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"patched benchmarker.py failed py_compile:\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}\n"
        f"--- file content ---\n{bench_py.read_text(encoding='utf-8')}"
    )


def test_patched_benchmarker_copies_scripts_atomically(
    fake_magpie: Path, tmp_path: Path,
):
    """The patched ``_prepare_benchmark_scripts`` still copies scripts correctly, atomically."""
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    (src_dir / "vllm_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho hello\n", encoding="utf-8",
    )
    (src_dir / "sglang_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho world\n", encoding="utf-8",
    )

    dst_dir = tmp_path / "InferenceX_benchmarks"

    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    namespace: dict = {"__name__": "_patched_benchmarker_smoke"}
    exec(compile(bench_py.read_text("utf-8"), str(bench_py), "exec"), namespace)
    cls = namespace["_FakeBenchmarker"]
    inst = cls(src_dir, dst_dir)
    inst._prepare_benchmark_scripts()

    for name in ("vllm_mi300x.sh", "sglang_mi300x.sh"):
        out = dst_dir / name
        assert out.is_file(), f"missing {out}"
        assert out.read_text(encoding="utf-8") == (src_dir / name).read_text("utf-8")
        assert out.stat().st_mode & 0o111, f"exec bit not set on {out}"


# Patched-block shape sanity
def test_patched_block_calls_os_replace():
    """The replacement must end in ``os.replace`` (the atomic rename)."""
    assert "_hyperloom_os.replace(_tmp_name, target_file)" in _PATCHED_BLOCK


def test_patched_block_uses_unique_aliases():
    """Aliases must be ``_hyperloom_*`` so they can't shadow upstream names."""
    assert "_hyperloom_os" in _PATCHED_BLOCK
    assert "_hyperloom_tempfile" in _PATCHED_BLOCK
    first_line = _PATCHED_BLOCK.splitlines()[0]
    assert first_line.startswith("            #") or first_line.strip() == ""


# Helper-method unit tests (formerly test_magpie_patcher_units.py)
from inference_optimizer.orchestrator.action_executors import _magpie_patcher as mp


# Legacy block wrapped in a tiny function body so the file tokenises as Python.
_LEGACY_FILE = (
    "def stub():\n"
    "    pass\n"
    "    # block\n"
    + mp._LEGACY_BLOCK
)


@pytest.fixture
def magpie_dir(tmp_path: Path) -> Path:
    target = tmp_path / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    target.parent.mkdir(parents=True)
    target.write_text(_LEGACY_FILE)
    return tmp_path


# _resolve_benchmarker_path
class TestResolveBenchmarkerPath:
    def test_explicit_dir_returns_existing_file(self, magpie_dir):
        out = mp._resolve_benchmarker_path(magpie_dir)
        assert out is not None
        assert out.name == "benchmarker.py"

    def test_returns_none_when_dir_missing(self, tmp_path):
        assert mp._resolve_benchmarker_path(tmp_path) is None

    def test_returns_none_when_no_input_or_env(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mp._resolve_benchmarker_path(None) is None

    def test_env_fallback(self, monkeypatch, magpie_dir):
        monkeypatch.setenv("MAGPIE_DIR", str(magpie_dir))
        out = mp._resolve_benchmarker_path(None)
        assert out is not None


# _is_patched
class TestIsPatched:
    def test_false_when_legacy(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._is_patched(target) is False

    def test_true_after_apply(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._apply_patch_atomic(target) is True
        assert mp._is_patched(target) is True

    def test_returns_false_when_unreadable(self, tmp_path, monkeypatch):
        ghost = tmp_path / "no.py"

        def boom(self, **kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(Path, "read_text", boom)
        assert mp._is_patched(ghost) is False


# _apply_patch_atomic
class TestApplyPatchAtomic:
    def test_returns_false_when_legacy_block_missing(self, tmp_path):
        target = tmp_path / "benchmarker.py"
        target.write_text("def foo():\n    pass\n")
        assert mp._apply_patch_atomic(target) is False
        assert "Hyperloom #C1 patch" not in target.read_text()

    def test_returns_false_when_read_fails(self, tmp_path, monkeypatch):
        target = tmp_path / "benchmarker.py"

        def boom(self, **kwargs):
            raise OSError("io")

        monkeypatch.setattr(Path, "read_text", boom)
        assert mp._apply_patch_atomic(target) is False

    def test_applies_patch_and_returns_true(self, magpie_dir):
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert mp._apply_patch_atomic(target) is True
        assert mp._PATCH_SENTINEL in target.read_text()
        assert "def stub" in target.read_text()


# ensure_magpie_atomic_scripts_patch
class TestEnsurePatch:
    def test_returns_false_without_magpie_tree(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mp.ensure_magpie_atomic_scripts_patch(None) is False

    def test_full_roundtrip_idempotent(self, magpie_dir):
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        assert target.read_text().count(mp._PATCH_SENTINEL) == 1


# Read-only / shared InferenceX/benchmarks deployment (bugs.md §C #1 follow-up).
# Contract: identical target -> no-op; writable -> atomic os.replace; read-only +
# stale/missing -> a clear error naming the script + dir, not a bare [Errno 30].
def _exec_patched_benchmarker(fake_magpie: Path):
    """Apply the patch, exec the patched benchmarker.py, return ``_FakeBenchmarker``."""
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    namespace: dict = {"__name__": "_patched_benchmarker_ro"}
    exec(compile(bench_py.read_text("utf-8"), str(bench_py), "exec"), namespace)
    return namespace["_FakeBenchmarker"]


def _simulate_readonly_dir(monkeypatch, readonly_dir: Path) -> None:
    """Make ``tempfile.mkstemp`` raise EROFS when targeting ``readonly_dir``."""
    import errno
    import tempfile as _tempfile

    real_mkstemp = _tempfile.mkstemp

    def _ro_mkstemp(*args, **kwargs):
        if str(kwargs.get("dir", "")) == str(readonly_dir):
            raise OSError(errno.EROFS, "Read-only file system")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", _ro_mkstemp)


def test_readonly_target_with_uptodate_scripts_is_noop(
    fake_magpie: Path, tmp_path: Path, monkeypatch,
):
    """Read-only target with already-identical scripts is a no-op, not an OSError."""
    import shutil as _shutil

    cls = _exec_patched_benchmarker(fake_magpie)

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    body = "#!/usr/bin/env bash\necho hi\n"
    (src_dir / "sglang_mi300x.sh").write_text(body, encoding="utf-8")

    dst_dir = tmp_path / "InferenceX_benchmarks"
    dst_dir.mkdir()
    _shutil.copy2(src_dir / "sglang_mi300x.sh", dst_dir / "sglang_mi300x.sh")

    _simulate_readonly_dir(monkeypatch, dst_dir)

    inst = cls(src_dir, dst_dir)
    # Must NOT raise — script already up to date, so no temp write.
    inst._prepare_benchmark_scripts()

    assert (dst_dir / "sglang_mi300x.sh").read_text(encoding="utf-8") == body


def test_readonly_target_with_stale_script_raises_clear_error(
    fake_magpie: Path, tmp_path: Path, monkeypatch,
):
    """Read-only target with a stale script raises a clear error, not a bare [Errno 30]."""
    cls = _exec_patched_benchmarker(fake_magpie)

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    (src_dir / "sglang_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho new\n", encoding="utf-8",
    )

    dst_dir = tmp_path / "InferenceX_benchmarks"
    dst_dir.mkdir()
    # Stale content present -> needs a rewrite.
    (dst_dir / "sglang_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho stale\n", encoding="utf-8",
    )

    _simulate_readonly_dir(monkeypatch, dst_dir)

    inst = cls(src_dir, dst_dir)
    with pytest.raises(OSError) as exc:
        inst._prepare_benchmark_scripts()
    msg = str(exc.value)
    assert "sglang_mi300x.sh" in msg
    assert "read-only" in msg.lower()


def test_writable_target_rewrites_stale_script(fake_magpie: Path, tmp_path: Path):
    """A writable target with a stale staged script is atomically rewritten."""
    cls = _exec_patched_benchmarker(fake_magpie)

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    new_body = "#!/usr/bin/env bash\necho new\n"
    (src_dir / "vllm_mi300x.sh").write_text(new_body, encoding="utf-8")

    dst_dir = tmp_path / "InferenceX_benchmarks"
    dst_dir.mkdir()
    (dst_dir / "vllm_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho old\n", encoding="utf-8",
    )

    inst = cls(src_dir, dst_dir)
    inst._prepare_benchmark_scripts()

    assert (dst_dir / "vllm_mi300x.sh").read_text(encoding="utf-8") == new_body
    assert (dst_dir / "vllm_mi300x.sh").stat().st_mode & 0o111


def test_patched_block_skips_identical_target():
    """The patched block contains the idempotent content check (filecmp)."""
    assert "_hyperloom_filecmp" in _PATCHED_BLOCK


# _extract_prepare_region
class TestExtractPrepareRegion:
    def test_returns_empty_when_marker_absent(self):
        assert _extract_prepare_region("def other():\n    pass\n") == ""

    def test_bounds_a_single_method_body(self):
        region = _extract_prepare_region(_GARBAGE_WITH_UNRELATED_ATOMIC_PY)
        assert "_prepare_benchmark_scripts" in region
        assert "_unrelated_helper" not in region
        assert "os.replace(" not in region

    def test_includes_inline_atomic_body(self):
        region = _extract_prepare_region(_INLINE_ATOMIC_BENCHMARKER_PY)
        assert "tempfile.mkstemp(" in region
        assert "os.replace(" in region


# _upstream_is_already_atomic
class TestUpstreamIsAlreadyAtomic:
    def test_named_helper_detected(self):
        assert _UPSTREAM_ATOMIC_HELPER in _UPSTREAM_ATOMIC_BENCHMARKER_PY
        assert _upstream_is_already_atomic(_UPSTREAM_ATOMIC_BENCHMARKER_PY) is True

    def test_inline_mkstemp_replace_detected(self):
        assert _upstream_is_already_atomic(_INLINE_ATOMIC_BENCHMARKER_PY) is True

    def test_legacy_block_is_not_atomic(self):
        assert _upstream_is_already_atomic(_UPSTREAM_BENCHMARKER_PY) is False

    def test_unrelated_atomic_ops_not_detected(self):
        assert _upstream_is_already_atomic(_GARBAGE_WITH_UNRELATED_ATOMIC_PY) is False

    def test_hyperloom_patched_output_is_atomic(self):
        # Defence in depth: the patcher's own output also reads as atomic.
        patched = _UPSTREAM_BENCHMARKER_PY.replace(
            _LEGACY_BLOCK, _PATCHED_BLOCK, 1,
        )
        assert _upstream_is_already_atomic(patched) is True
