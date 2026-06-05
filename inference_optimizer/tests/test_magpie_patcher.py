"""Tests for ``_magpie_patcher.ensure_magpie_atomic_scripts_patch``
(Hyperloom ``bugs.md`` §C #1 root-cause fix).

The patcher's contract:

* Replace the non-atomic ``shutil.copy2 + chmod`` block inside
  ``<MAGPIE_DIR>/Magpie/modes/benchmark/benchmarker.py``
  ``_prepare_benchmark_scripts`` with a temp-file + ``os.replace`` form
  so concurrent ``bash source`` readers never see a half-truncated
  ``<InferenceX>/benchmarks/*.sh``.
* Idempotent — repeated calls are no-ops once the sentinel substring
  ``Hyperloom #C1 patch`` is present.
* Concurrency-safe — two installers racing the same checkout end up
  with one patched file, not double-patched garbage.
* Fail-soft on layout drift — missing ``MAGPIE_DIR``, missing file, or
  an already-mutated file all return ``False`` instead of raising. The
  install script is expected to escalate ``False`` to ``die`` (this is
  a known RCA fix; the unit-test contract is the soft return only).

The fixtures synthesise a minimal Magpie tree inside ``tmp_path`` so
we never touch the real ``$MAGPIE_DIR`` checkout. The fixture
``benchmarker.py`` is reduced to just the ``_prepare_benchmark_scripts``
method body — that is the entire surface the patcher matches against.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors._magpie_patcher import (
    _LEGACY_BLOCK,
    _PATCHED_BLOCK,
    _PATCH_SENTINEL,
    ensure_magpie_atomic_scripts_patch,
)


# Reduced fixture capturing the exact upstream shape the patcher targets.
# Indentation is load-bearing — the patcher matches on the exact 12-space
# prefix to avoid false-positive matches against unrelated `shutil.copy2`
# references elsewhere in benchmarker.py.
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


# ---------------------------------------------------------------------------
# Basic shape / sanity
# ---------------------------------------------------------------------------
def test_legacy_block_is_present_in_fixture():
    """Sanity: the fixture must contain the exact block the patcher
    matches against, otherwise the tests are meaningless."""
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


# ---------------------------------------------------------------------------
# Patch application + idempotency
# ---------------------------------------------------------------------------
def test_patch_applied_replaces_legacy_block(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    text = bench_py.read_text(encoding="utf-8")
    assert _PATCH_SENTINEL in text
    assert _LEGACY_BLOCK not in text
    # The patched block must appear exactly once — replace(_, _, 1) shape.
    assert text.count(_PATCH_SENTINEL) == 1


def test_patch_is_idempotent(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    after_first = bench_py.read_text(encoding="utf-8")
    # Second invocation should hit fast path; bytes must be identical.
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    after_second = bench_py.read_text(encoding="utf-8")
    assert after_first == after_second
    assert after_second.count(_PATCH_SENTINEL) == 1


def test_patch_preserves_file_mode(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    bench_py.chmod(0o644)
    pre_mode = bench_py.stat().st_mode
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    post_mode = bench_py.stat().st_mode
    assert pre_mode == post_mode, (
        f"file mode changed: {oct(pre_mode)} -> {oct(post_mode)}"
    )


# ---------------------------------------------------------------------------
# Layout-drift fail-soft (install script escalates to fail-loud)
# ---------------------------------------------------------------------------
def test_fail_soft_when_legacy_block_missing(tmp_path: Path):
    """Upstream refactor / hand-edit removes the legacy two-line block →
    patcher returns False, leaves the file alone."""
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
    """If someone (e.g. an upstream PR landing) already inserted the
    sentinel, we must accept the file as patched and not rewrite."""
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    bench_py.write_text(
        f"# top of file\n# {_PATCH_SENTINEL} present here\npass\n",
        encoding="utf-8",
    )
    pre = bench_py.read_text(encoding="utf-8")
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    assert bench_py.read_text(encoding="utf-8") == pre


# ---------------------------------------------------------------------------
# Concurrency — N patchers racing the same file
# ---------------------------------------------------------------------------
def test_concurrent_patchers_produce_one_patch(fake_magpie: Path):
    """Eight threads invoking ``ensure_*`` simultaneously must end with
    exactly one applied patch (sentinel appears once) and no torn
    file content."""
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
    """One thread patches the file while N reader threads continuously
    read its bytes. Every read must observe either the pre-patch or
    post-patch content, never a torn intermediate state.

    This is the load-bearing test: it exercises exactly the failure
    mode bugs.md §C #1 describes (a concurrent reader catching a
    half-written file). The pre-patch fixture is a self-contained
    Python module, so a "torn" view would fail to compile or be missing
    the sentinel that the post-patch view must contain. We assert that
    every snapshot is either the verbatim original or contains the
    sentinel — there is no third state."""
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
        # Give readers a moment to start spinning, then patch.
        time.sleep(0.05)
        assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
        # Let readers continue for a bit after the patch so we cover
        # both the during-rename and post-rename window.
        time.sleep(0.1)
    finally:
        stop.set()
        for r in readers:
            r.join()

    assert not torn_snapshots, (
        f"observed {len(torn_snapshots)} torn snapshot(s); first 200 chars: "
        f"{torn_snapshots[0][:200]!r}"
    )


# ---------------------------------------------------------------------------
# Smoke — the patched benchmarker.py is still valid Python and produces
# correct semantics for _prepare_benchmark_scripts
# ---------------------------------------------------------------------------
def test_patched_file_is_valid_python(fake_magpie: Path):
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    # Use py_compile via subprocess so we don't have to manage import paths.
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
    """Run the patched ``_prepare_benchmark_scripts`` against a real
    ``<src>/*.sh`` set and confirm the output is correct + the rename
    target ends up with the right perms. This is the end-to-end
    behaviour check: even after patching, the function must still do
    what upstream did, only atomically."""
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

    # Load the patched module out of the fixture and exercise it.
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    namespace: dict = {"__name__": "_patched_benchmarker_smoke"}
    exec(compile(bench_py.read_text("utf-8"), str(bench_py), "exec"), namespace)
    cls = namespace["_FakeBenchmarker"]
    inst = cls(src_dir, dst_dir)
    inst._prepare_benchmark_scripts()

    # Both scripts copied verbatim, both with exec bit set.
    for name in ("vllm_mi300x.sh", "sglang_mi300x.sh"):
        out = dst_dir / name
        assert out.is_file(), f"missing {out}"
        assert out.read_text(encoding="utf-8") == (src_dir / name).read_text("utf-8")
        assert out.stat().st_mode & 0o111, f"exec bit not set on {out}"


# ---------------------------------------------------------------------------
# Patched-block shape sanity (changes here force the test author to
# re-read _magpie_patcher.py and acknowledge the change).
# ---------------------------------------------------------------------------
def test_patched_block_calls_os_replace():
    """The replacement must end in ``os.replace`` (the atomic rename) —
    if a future edit drops this line we break the whole contract."""
    assert "_hyperloom_os.replace(_tmp_name, target_file)" in _PATCHED_BLOCK


def test_patched_block_uses_unique_aliases():
    """Aliases must be ``_hyperloom_*`` so we can't shadow upstream
    names. Plain ``os`` / ``tempfile`` would only collide harmlessly
    today but are landmines under future Magpie refactors."""
    assert "_hyperloom_os" in _PATCHED_BLOCK
    assert "_hyperloom_tempfile" in _PATCHED_BLOCK
    # The replacement is a single textual block — no other code can
    # leak into it accidentally.
    assert re.match(r"^( {12}#|\s*$)", _PATCHED_BLOCK.splitlines()[0])


# ===========================================================================
# Helper-method unit tests (formerly test_magpie_patcher_units.py)
# ===========================================================================

from inference_optimizer.orchestrator.action_executors import _magpie_patcher as mp


# Minimal stand-in for the upstream legacy block, byte-for-byte equal to
# the production ``_LEGACY_BLOCK``. Wrapped in a tiny function body so the
# patched file still tokenises as valid Python and we can re-import to
# verify roundtrip.
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


# ---------------------------------------------------------------------------
# _resolve_benchmarker_path
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _is_patched
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _apply_patch_atomic
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ensure_magpie_atomic_scripts_patch
# ---------------------------------------------------------------------------

class TestEnsurePatch:
    def test_returns_false_without_magpie_tree(self, monkeypatch):
        monkeypatch.delenv("MAGPIE_DIR", raising=False)
        assert mp.ensure_magpie_atomic_scripts_patch(None) is False

    def test_full_roundtrip_idempotent(self, magpie_dir):
        # First call: applies the patch.
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        # Second call: hits the fast path (no flock).
        assert mp.ensure_magpie_atomic_scripts_patch(magpie_dir) is True
        target = magpie_dir / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
        # Only one copy of the sentinel — the patch ran exactly once.
        assert target.read_text().count(mp._PATCH_SENTINEL) == 1


# ===========================================================================
# Read-only / shared InferenceX/benchmarks deployment
# (bugs.md §C #1 follow-up).
#
# The atomic-write patch put its temp file in the TARGET dir
# (mkstemp(dir=target_dir)). When InferenceX/benchmarks is a shared
# read-only mount (e.g. /wekafs/hyperloom/InferenceX), that mkstemp raised
# `OSError: [Errno 30] Read-only file system` and took down EVERY model's
# first benchmark session — regardless of model or framework.
#
# Contract after the fix:
#   * target script already byte-identical -> no-op (pre-staged on a
#     read-only shared deployment is fine);
#   * target dir writable -> atomic os.replace (unchanged: still closes the
#     §C #1 torn-read race);
#   * target dir read-only AND script missing/stale -> a clear, actionable
#     error naming the script + the read-only dir, NOT a bare [Errno 30].
# ===========================================================================
def _exec_patched_benchmarker(fake_magpie: Path):
    """Apply the patch, then exec the patched benchmarker.py and return its
    ``_FakeBenchmarker`` class (same technique as the atomic-copy smoke
    test above)."""
    assert ensure_magpie_atomic_scripts_patch(fake_magpie) is True
    bench_py = fake_magpie / "Magpie" / "modes" / "benchmark" / "benchmarker.py"
    namespace: dict = {"__name__": "_patched_benchmarker_ro"}
    exec(compile(bench_py.read_text("utf-8"), str(bench_py), "exec"), namespace)
    return namespace["_FakeBenchmarker"]


def _simulate_readonly_dir(monkeypatch, readonly_dir: Path) -> None:
    """Make ``tempfile.mkstemp`` raise EROFS when targeting ``readonly_dir``.

    Tests run as root, where a ``chmod 0o555`` is bypassed by DAC, so we
    cannot model a read-only mount with permissions. Patching ``mkstemp``
    for the one target directory reproduces the exact production failure
    (mkstemp into a read-only InferenceX/benchmarks) deterministically.
    """
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
    """Read-only target whose scripts are ALREADY identical to Magpie
    source must be a no-op, not an OSError. This is the exact failure that
    killed all 8 models' first sessions."""
    import shutil as _shutil

    cls = _exec_patched_benchmarker(fake_magpie)

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    body = "#!/usr/bin/env bash\necho hi\n"
    (src_dir / "sglang_mi300x.sh").write_text(body, encoding="utf-8")

    # Shared deployment already has the identical script staged.
    dst_dir = tmp_path / "InferenceX_benchmarks"
    dst_dir.mkdir()
    _shutil.copy2(src_dir / "sglang_mi300x.sh", dst_dir / "sglang_mi300x.sh")

    # Now any temp write into dst_dir behaves read-only.
    _simulate_readonly_dir(monkeypatch, dst_dir)

    inst = cls(src_dir, dst_dir)
    # Must NOT raise — the script is already up to date, so no temp write.
    inst._prepare_benchmark_scripts()

    assert (dst_dir / "sglang_mi300x.sh").read_text(encoding="utf-8") == body


def test_readonly_target_with_stale_script_raises_clear_error(
    fake_magpie: Path, tmp_path: Path, monkeypatch,
):
    """Read-only target whose script is stale/missing must raise a clear,
    actionable error (naming the script + read-only dir), not a bare
    [Errno 30]."""
    cls = _exec_patched_benchmarker(fake_magpie)

    src_dir = tmp_path / "magpie_scripts"
    src_dir.mkdir()
    (src_dir / "sglang_mi300x.sh").write_text(
        "#!/usr/bin/env bash\necho new\n", encoding="utf-8",
    )

    dst_dir = tmp_path / "InferenceX_benchmarks"
    dst_dir.mkdir()
    # Stale (different) content already present -> needs a rewrite.
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
    """Regression: a writable target with a stale staged script is still
    atomically rewritten to match source (exec bit preserved)."""
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
    """The patched block must contain the idempotent content check so a
    read-only shared deployment with pre-staged scripts is a no-op rather
    than a mkstemp OSError."""
    assert "_hyperloom_filecmp" in _PATCHED_BLOCK
