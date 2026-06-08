# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_inferencex_patcher.ensure_benchmark_lib_patched`` (Hyperloom issue #194 §2).

Pins the patcher contract: a backward-compatible, idempotent, concurrency-safe,
fail-soft patch to ``benchmark_lib.sh`` that honours ``$NUM_PROMPTS``. Fixtures
synthesize a fake InferenceX tree in ``tmp_path``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _inferencex_patcher
from inference_optimizer.orchestrator.action_executors._inferencex_patcher import (
    ensure_benchmark_lib_patched,
)


# Verbatim upstream shape (incl. the 8-space indent the patcher matches on).
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


# Happy path: fresh checkout → patch lands; second call is a no-op.
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
    """Belt-and-braces: the sentinel must appear exactly once even with multiple invocations."""
    for _ in range(5):
        ensure_benchmark_lib_patched(fake_inferencex)
    text = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    assert text.count(_PATCHED_LINE) == 1


# Backward-compatibility: patched line reduces to original when NUM_PROMPTS unset.
def test_patch_is_backward_compatible_when_num_prompts_unset(
    fake_inferencex, tmp_path, monkeypatch,
):
    """The patched line evaluates identically to the original when NUM_PROMPTS is unset."""
    import shutil
    import subprocess
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable in this environment")
    ensure_benchmark_lib_patched(fake_inferencex)
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
    """When NUM_PROMPTS env IS set, it wins over the hard-coded reset."""
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


# Fail-soft: missing config / file / legacy line must NOT raise.
def test_returns_false_when_inferencex_path_unset(tmp_path, monkeypatch):
    """No INFERENCEX_PATH and no explicit arg → returns False, no crash."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert ensure_benchmark_lib_patched(None) is False


def test_returns_false_when_benchmark_lib_missing(tmp_path):
    """A valid root with no benchmarks/ subtree must not raise."""
    assert ensure_benchmark_lib_patched(tmp_path) is False


def test_returns_false_when_legacy_line_missing(tmp_path):
    """If the legacy line is absent the patcher refuses to guess and returns False."""
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    lib = bench_dir / "benchmark_lib.sh"
    lib.write_text(
        "#!/usr/bin/env bash\n"
        "# This file was hand-patched to use a different shape.\n"
        "run_benchmark_serving() { echo 'something else'; }\n",
        encoding="utf-8",
    )
    rc = ensure_benchmark_lib_patched(tmp_path)
    assert rc is False
    assert "something else" in lib.read_text()


def test_already_patched_file_short_circuits(fake_inferencex):
    """If the sentinel is already present, the patcher returns True without touching the file."""
    lib = fake_inferencex / "benchmarks" / "benchmark_lib.sh"
    lib.write_text(
        lib.read_text().replace(_LEGACY_LINE, _PATCHED_LINE),
        encoding="utf-8",
    )
    before = lib.read_text()
    rc = ensure_benchmark_lib_patched(fake_inferencex)
    assert rc is True
    assert lib.read_text() == before


# INFERENCEX_PATH env fallback (when no explicit arg is provided).
def test_env_var_is_used_when_no_explicit_path(fake_inferencex, monkeypatch):
    monkeypatch.setenv("INFERENCEX_PATH", str(fake_inferencex))
    rc = ensure_benchmark_lib_patched(None)
    assert rc is True
    lib = fake_inferencex / "benchmarks" / "benchmark_lib.sh"
    assert _PATCHED_LINE in lib.read_text()


# Concurrency: multiple threads racing the same checkout must converge on a
# singly-patched file, exercising the under-lock re-check and atomic-rename path.
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
    assert text.count(_PATCHED_LINE) == 1
    assert _LEGACY_LINE not in text


# PR-D §2: benchmark_serving.py PROFILE_EXTRA_BODY consumer patch
from inference_optimizer.orchestrator.action_executors._inferencex_patcher import (  # noqa: E402
    ensure_benchmark_serving_patched,
)


# Verbatim copy of the PR-D §2 line (41-space indent the patcher matches on).
_BS_LEGACY_LINE = (
    '                                         extra_body={"num_steps": 1, '
    '"merge_profiles": True, "profile_by_stage": True},'
)
_BS_UPSTREAM_FIXTURE = f"""\
async def benchmark_serving_main(...):
    # Pretend we're inside the function-call site at line 541.
    await async_request_openai_completions(
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
{_BS_LEGACY_LINE}
                                         logprobs=logprobs,
                                         best_of=best_of,
    )
"""


@pytest.fixture
def fake_inferencex_with_benchmark_serving(tmp_path: Path) -> Path:
    bench_dir = tmp_path / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    lib = bench_dir / "benchmark_serving.py"
    lib.write_text(_BS_UPSTREAM_FIXTURE, encoding="utf-8")
    lib.chmod(0o644)
    return tmp_path


def test_benchmark_serving_patch_adds_profile_extra_body_lookup(
    fake_inferencex_with_benchmark_serving,
):
    """The patched line reads ``PROFILE_EXTRA_BODY`` from env, keeping the literal dict as the JSON fallback."""
    src = (
        fake_inferencex_with_benchmark_serving / "utils" / "bench_serving"
        / "benchmark_serving.py"
    )
    rc = ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving)
    assert rc is True
    text = src.read_text(encoding="utf-8")
    assert "PROFILE_EXTRA_BODY" in text, (
        "patched file must reference PROFILE_EXTRA_BODY env var"
    )
    assert _BS_LEGACY_LINE not in text, (
        "legacy hardcoded extra_body line must be replaced, not retained"
    )
    # JSON-form default (lowercase ``true``) survives as the unset-env fallback.
    assert (
        '{"num_steps": 1, "merge_profiles": true, "profile_by_stage": true}'
        in text
    )


def test_benchmark_serving_patch_is_idempotent(
    fake_inferencex_with_benchmark_serving,
):
    """Second call short-circuits on the sentinel — file content stable."""
    src = (
        fake_inferencex_with_benchmark_serving / "utils" / "bench_serving"
        / "benchmark_serving.py"
    )
    assert ensure_benchmark_serving_patched(
        fake_inferencex_with_benchmark_serving
    ) is True
    snapshot = src.read_text(encoding="utf-8")
    assert ensure_benchmark_serving_patched(
        fake_inferencex_with_benchmark_serving
    ) is True
    assert src.read_text(encoding="utf-8") == snapshot
    assert snapshot.count("PROFILE_EXTRA_BODY") == 1


def test_benchmark_serving_patch_returns_false_when_path_missing(
    tmp_path, monkeypatch,
):
    """A tmpdir without the benchmark_serving.py file must fail-soft."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    assert ensure_benchmark_serving_patched(tmp_path) is False


def test_benchmark_serving_patch_returns_false_when_legacy_line_missing(
    tmp_path,
):
    """If upstream refactored the extra_body= line, the patcher refuses to guess and returns False."""
    bench_dir = tmp_path / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    src = bench_dir / "benchmark_serving.py"
    src.write_text(
        "async def benchmark_serving_main(...): pass\n"
        "# This is a hand-patched / refactored file with no legacy line.\n",
        encoding="utf-8",
    )
    rc = ensure_benchmark_serving_patched(tmp_path)
    assert rc is False
    assert "PROFILE_EXTRA_BODY" not in src.read_text(encoding="utf-8")


def test_benchmark_serving_patch_uses_env_var_when_no_explicit_path(
    fake_inferencex_with_benchmark_serving, monkeypatch,
):
    """Honours ``$INFERENCEX_PATH`` when no explicit path is passed."""
    monkeypatch.setenv(
        "INFERENCEX_PATH", str(fake_inferencex_with_benchmark_serving),
    )
    assert ensure_benchmark_serving_patched(None) is True
    src = (
        fake_inferencex_with_benchmark_serving / "utils" / "bench_serving"
        / "benchmark_serving.py"
    )
    assert "PROFILE_EXTRA_BODY" in src.read_text(encoding="utf-8")


def test_benchmark_serving_patched_line_is_executable_python(
    fake_inferencex_with_benchmark_serving,
):
    """The patched line must be syntactically valid Python and honour PROFILE_EXTRA_BODY."""
    ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving)
    src = (
        fake_inferencex_with_benchmark_serving / "utils" / "bench_serving"
        / "benchmark_serving.py"
    )
    text = src.read_text(encoding="utf-8")
    patched_line = next(
        ln for ln in text.splitlines() if "PROFILE_EXTRA_BODY" in ln
    )
    expr = patched_line.strip()
    assert expr.startswith("extra_body="), expr
    expr = expr[len("extra_body="):].rstrip(",")
    compile(expr, "<patched>", "eval")
    import os
    os.environ.pop("PROFILE_EXTRA_BODY", None)
    result = eval(expr, {"__builtins__": __builtins__})  # noqa: PGH001
    assert result == {
        "num_steps": 1, "merge_profiles": True, "profile_by_stage": True,
    }, f"unexpected default extra_body: {result!r}"
    os.environ["PROFILE_EXTRA_BODY"] = (
        '{"num_steps": 10, "shape_discovery": true, "roofline_annotations": true}'
    )
    try:
        result_env = eval(expr, {"__builtins__": __builtins__})  # noqa: PGH001
        assert result_env == {
            "num_steps": 10,
            "shape_discovery": True,
            "roofline_annotations": True,
        }
    finally:
        os.environ.pop("PROFILE_EXTRA_BODY", None)


# #210 fix (Deval comments 4 + 6): patch every InferenceX root, not just
# $INFERENCEX_PATH — Magpie loads its bundled $MAGPIE_DIR/InferenceX at runtime.
def _make_inferencex_tree_with_serving(
    root: Path, profile_by_stage: bool = True,
) -> Path:
    """Build a minimal ``<root>/utils/bench_serving/benchmark_serving.py`` fixture."""
    bench_dir = root / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True, exist_ok=True)
    f = bench_dir / "benchmark_serving.py"
    f.write_text(
        "async def x():\n"
        "    await async_request(\n"
        "                                         api_url=base + \"/start_profile\",\n"
        "                                         prompt_len=test_prompt_len,\n"
        "                                         output_len=test_output_len,\n"
        '                                         extra_body={"num_steps": 1, '
        '"merge_profiles": True, "profile_by_stage": True},\n'
        "    )\n",
        encoding="utf-8",
    )
    f.chmod(0o644)
    return f


def test_discover_inferencex_roots_dedupes_when_paths_resolve_same(
    tmp_path, monkeypatch,
):
    """When both paths resolve to the SAME directory, discovery returns one entry, not two."""
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    magpie = tmp_path / "Magpie"
    magpie.mkdir()
    (magpie / "InferenceX").symlink_to(inferencex)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 1, f"expected dedup to one root, got {roots}"
    assert roots[0] == inferencex.resolve()


def test_discover_inferencex_roots_includes_both_when_paths_differ(
    tmp_path, monkeypatch,
):
    """The #210 case: distinct ``$INFERENCEX_PATH`` and ``$MAGPIE_DIR/InferenceX`` both show up so both get patched."""
    inferencex_external = tmp_path / "external" / "InferenceX"
    inferencex_external.mkdir(parents=True)
    magpie_dir = tmp_path / "workspace" / "Magpie"
    (magpie_dir / "InferenceX").mkdir(parents=True)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex_external))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie_dir))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 2, f"expected 2 distinct roots, got {roots}"
    resolved = {p.resolve() for p in roots}
    assert inferencex_external.resolve() in resolved
    assert (magpie_dir / "InferenceX").resolve() in resolved


def test_discover_inferencex_roots_when_only_magpie_dir_set(
    tmp_path, monkeypatch,
):
    """With only ``$MAGPIE_DIR`` set, Magpie's bundled InferenceX is still discovered."""
    magpie_dir = tmp_path / "Magpie"
    (magpie_dir / "InferenceX").mkdir(parents=True)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setenv("MAGPIE_DIR", str(magpie_dir))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 1
    assert roots[0] == (magpie_dir / "InferenceX").resolve()


def test_discover_inferencex_roots_returns_empty_when_nothing_set(monkeypatch):
    """No env, no caller arg → empty list (caller fail-softs)."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert _inferencex_patcher._discover_inferencex_roots(None) == []


def test_ensure_benchmark_serving_patched_patches_both_roots_when_they_differ(
    tmp_path, monkeypatch,
):
    """End-to-end #210 invariant: distinct roots → both ``benchmark_serving.py`` files get patched."""
    external = tmp_path / "external" / "InferenceX"
    external.mkdir(parents=True)
    magpie = tmp_path / "Magpie"
    (magpie / "InferenceX").mkdir(parents=True)
    f_external = _make_inferencex_tree_with_serving(external)
    f_magpie = _make_inferencex_tree_with_serving(magpie / "InferenceX")
    monkeypatch.setenv("INFERENCEX_PATH", str(external))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie))

    rc = ensure_benchmark_serving_patched()
    assert rc is True
    text_external = f_external.read_text(encoding="utf-8")
    text_magpie = f_magpie.read_text(encoding="utf-8")
    assert "PROFILE_EXTRA_BODY" in text_external, (
        "$INFERENCEX_PATH file must be patched"
    )
    assert "PROFILE_EXTRA_BODY" in text_magpie, (
        "$MAGPIE_DIR/InferenceX file must ALSO be patched (the #210 fix)"
    )


def test_ensure_benchmark_serving_patched_returns_true_when_only_magpie_path_present(
    tmp_path, monkeypatch,
):
    """When ``$INFERENCEX_PATH`` lacks ``benchmark_serving.py`` but Magpie's copy has it, the patcher still succeeds."""
    external = tmp_path / "external" / "InferenceX"
    external.mkdir(parents=True)
    magpie = tmp_path / "Magpie"
    (magpie / "InferenceX").mkdir(parents=True)
    f_magpie = _make_inferencex_tree_with_serving(magpie / "InferenceX")
    monkeypatch.setenv("INFERENCEX_PATH", str(external))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie))

    rc = ensure_benchmark_serving_patched()
    assert rc is True
    assert "PROFILE_EXTRA_BODY" in f_magpie.read_text(encoding="utf-8")


def test_ensure_benchmark_lib_patched_patches_both_roots_when_they_differ(
    tmp_path, monkeypatch,
):
    """Same multi-root contract for ``benchmark_lib.sh``."""
    legacy = (
        "#!/usr/bin/env bash\n"
        "run_benchmark_serving() {\n"
        "    local num_prompts=\"\"\n"
        "    local max_concurrency=\"\"\n"
        "    if [[ \"${PROFILE:-}\" == \"1\" ]]; then\n"
        "        num_prompts=\"$max_concurrency\"\n"
        "    fi\n"
        "}\n"
    )
    external = tmp_path / "external" / "InferenceX"
    (external / "benchmarks").mkdir(parents=True)
    f_external = external / "benchmarks" / "benchmark_lib.sh"
    f_external.write_text(legacy, encoding="utf-8")
    magpie = tmp_path / "Magpie"
    (magpie / "InferenceX" / "benchmarks").mkdir(parents=True)
    f_magpie = magpie / "InferenceX" / "benchmarks" / "benchmark_lib.sh"
    f_magpie.write_text(legacy, encoding="utf-8")
    monkeypatch.setenv("INFERENCEX_PATH", str(external))
    monkeypatch.setenv("MAGPIE_DIR", str(magpie))

    rc = ensure_benchmark_lib_patched()
    assert rc is True
    assert "${NUM_PROMPTS:-$max_concurrency}" in f_external.read_text(encoding="utf-8")
    assert "${NUM_PROMPTS:-$max_concurrency}" in f_magpie.read_text(encoding="utf-8"), (
        "Magpie's bundled InferenceX benchmark_lib.sh must ALSO be patched"
    )
