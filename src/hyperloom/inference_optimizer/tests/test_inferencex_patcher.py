# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_inferencex_patcher.ensure_benchmark_lib_patched``.

Pins the patcher contract: a backward-compatible, idempotent, concurrency-safe,
fail-soft patch to ``benchmark_lib.sh`` that honours ``$NUM_PROMPTS``. Fixtures
synthesize a fake InferenceX tree in ``tmp_path``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _inferencex_patcher
from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
    ensure_benchmark_lib_patched,
)


@pytest.fixture(autouse=True)
def _isolate_inferencex_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test hermetic w.r.t. the discovery env: clear ``$INFERENCEX_PATH`` / ``$MAGPIE_PATH`` so a synthetic ``tmp_path`` test never discovers a real on-pod checkout (tests that exercise the fallback re-set them)."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)


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
    assert after_first == after_second, "Second call mutated the file — patch is not idempotent"


def test_patched_line_appears_exactly_once(fake_inferencex):
    """Belt-and-braces: the sentinel must appear exactly once even with multiple invocations."""
    for _ in range(5):
        ensure_benchmark_lib_patched(fake_inferencex)
    text = (fake_inferencex / "benchmarks" / "benchmark_lib.sh").read_text()
    assert text.count(_PATCHED_LINE) == 1


# Backward-compatibility: patched line reduces to original when NUM_PROMPTS unset.
def test_patch_is_backward_compatible_when_num_prompts_unset(
    fake_inferencex,
    tmp_path,
    monkeypatch,
):
    """The patched line evaluates identically to the original when NUM_PROMPTS is unset."""
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("bash unavailable in this environment")
    ensure_benchmark_lib_patched(fake_inferencex)
    snippet = (
        'max_concurrency=42\nunset NUM_PROMPTS\nnum_prompts="${NUM_PROMPTS:-$max_concurrency}"\necho "$num_prompts"\n'
    )
    out = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
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
        'max_concurrency=42\nNUM_PROMPTS=999\nnum_prompts="${NUM_PROMPTS:-$max_concurrency}"\necho "$num_prompts"\n'
    )
    out = subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
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
        except Exception as exc:  # noqa: BLE001 - test-only
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


# benchmark_serving.py PROFILE_EXTRA_BODY consumer patch.
from hyperloom.orchestrator.actions.executors._inferencex_patcher import (  # noqa: E402
    ensure_benchmark_serving_patched,
)


# Verbatim copy of the legacy line (41-space indent the patcher matches on).
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
    src = fake_inferencex_with_benchmark_serving / "utils" / "bench_serving" / "benchmark_serving.py"
    rc = ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving)
    assert rc is True
    text = src.read_text(encoding="utf-8")
    assert "PROFILE_EXTRA_BODY" in text, "patched file must reference PROFILE_EXTRA_BODY env var"
    assert _BS_LEGACY_LINE not in text, "legacy hardcoded extra_body line must be replaced, not retained"
    # JSON-form default (lowercase ``true``) survives as the unset-env fallback.
    assert '{"num_steps": 1, "merge_profiles": true, "profile_by_stage": true}' in text


def test_benchmark_serving_patch_is_idempotent(
    fake_inferencex_with_benchmark_serving,
):
    """Second call short-circuits on the sentinel — file content stable."""
    src = fake_inferencex_with_benchmark_serving / "utils" / "bench_serving" / "benchmark_serving.py"
    assert ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving) is True
    snapshot = src.read_text(encoding="utf-8")
    assert ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving) is True
    assert src.read_text(encoding="utf-8") == snapshot
    assert snapshot.count("PROFILE_EXTRA_BODY") == 1


def test_benchmark_serving_patch_returns_false_when_path_missing(
    tmp_path,
    monkeypatch,
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
    fake_inferencex_with_benchmark_serving,
    monkeypatch,
):
    """Honours ``$INFERENCEX_PATH`` when no explicit path is passed."""
    monkeypatch.setenv(
        "INFERENCEX_PATH",
        str(fake_inferencex_with_benchmark_serving),
    )
    assert ensure_benchmark_serving_patched(None) is True
    src = fake_inferencex_with_benchmark_serving / "utils" / "bench_serving" / "benchmark_serving.py"
    assert "PROFILE_EXTRA_BODY" in src.read_text(encoding="utf-8")


def test_benchmark_serving_patched_line_is_executable_python(
    fake_inferencex_with_benchmark_serving,
):
    """The patched line must be syntactically valid Python and honour PROFILE_EXTRA_BODY."""
    ensure_benchmark_serving_patched(fake_inferencex_with_benchmark_serving)
    src = fake_inferencex_with_benchmark_serving / "utils" / "bench_serving" / "benchmark_serving.py"
    text = src.read_text(encoding="utf-8")
    patched_line = next(ln for ln in text.splitlines() if "PROFILE_EXTRA_BODY" in ln)
    expr = patched_line.strip()
    assert expr.startswith("extra_body="), expr
    expr = expr[len("extra_body=") :].rstrip(",")
    compile(expr, "<patched>", "eval")
    import os

    os.environ.pop("PROFILE_EXTRA_BODY", None)
    result = eval(expr, {"__builtins__": __builtins__})  # noqa: PGH001
    assert result == {
        "num_steps": 1,
        "merge_profiles": True,
        "profile_by_stage": True,
    }, f"unexpected default extra_body: {result!r}"
    os.environ["PROFILE_EXTRA_BODY"] = '{"num_steps": 10, "shape_discovery": true, "detailed_annotations": true}'
    try:
        result_env = eval(expr, {"__builtins__": __builtins__})  # noqa: PGH001
        assert result_env == {
            "num_steps": 10,
            "shape_discovery": True,
            "detailed_annotations": True,
        }
    finally:
        os.environ.pop("PROFILE_EXTRA_BODY", None)


# #210 fix (Deval comments 4 + 6): patch every InferenceX root, not just
# $INFERENCEX_PATH — Magpie loads its bundled $MAGPIE_PATH/InferenceX at runtime.
def _make_inferencex_tree_with_serving(
    root: Path,
    profile_by_stage: bool = True,
) -> Path:
    """Build a minimal ``<root>/utils/bench_serving/benchmark_serving.py`` fixture."""
    bench_dir = root / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True, exist_ok=True)
    f = bench_dir / "benchmark_serving.py"
    f.write_text(
        "async def x():\n"
        "    await async_request(\n"
        '                                         api_url=base + "/start_profile",\n'
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
    tmp_path,
    monkeypatch,
):
    """When both paths resolve to the SAME directory, discovery returns one entry, not two."""
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    magpie = tmp_path / "Magpie"
    magpie.mkdir()
    (magpie / "InferenceX").symlink_to(inferencex)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 1, f"expected dedup to one root, got {roots}"
    assert roots[0] == inferencex.resolve()


def test_discover_inferencex_roots_includes_both_when_paths_differ(
    tmp_path,
    monkeypatch,
):
    """Distinct ``$INFERENCEX_PATH`` and ``$MAGPIE_PATH/InferenceX`` both show up so both get patched."""
    inferencex_external = tmp_path / "external" / "InferenceX"
    inferencex_external.mkdir(parents=True)
    magpie_dir = tmp_path / "workspace" / "Magpie"
    (magpie_dir / "InferenceX").mkdir(parents=True)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex_external))
    monkeypatch.setenv("MAGPIE_PATH", str(magpie_dir))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 2, f"expected 2 distinct roots, got {roots}"
    resolved = {p.resolve() for p in roots}
    assert inferencex_external.resolve() in resolved
    assert (magpie_dir / "InferenceX").resolve() in resolved


def test_discover_inferencex_roots_when_only_magpie_dir_set(
    tmp_path,
    monkeypatch,
):
    """With only ``$MAGPIE_PATH`` set, Magpie's bundled InferenceX is still discovered."""
    magpie_dir = tmp_path / "Magpie"
    (magpie_dir / "InferenceX").mkdir(parents=True)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setenv("MAGPIE_PATH", str(magpie_dir))
    roots = _inferencex_patcher._discover_inferencex_roots(None)
    assert len(roots) == 1
    assert roots[0] == (magpie_dir / "InferenceX").resolve()


def test_discover_inferencex_roots_returns_empty_when_nothing_set(monkeypatch):
    """No env, no caller arg → empty list (caller fail-softs)."""
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert _inferencex_patcher._discover_inferencex_roots(None) == []


def test_ensure_benchmark_serving_patched_patches_both_roots_when_they_differ(
    tmp_path,
    monkeypatch,
):
    """Distinct roots → both ``benchmark_serving.py`` files get patched."""
    external = tmp_path / "external" / "InferenceX"
    external.mkdir(parents=True)
    magpie = tmp_path / "Magpie"
    (magpie / "InferenceX").mkdir(parents=True)
    f_external = _make_inferencex_tree_with_serving(external)
    f_magpie = _make_inferencex_tree_with_serving(magpie / "InferenceX")
    monkeypatch.setenv("INFERENCEX_PATH", str(external))
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))

    rc = ensure_benchmark_serving_patched()
    assert rc is True
    text_external = f_external.read_text(encoding="utf-8")
    text_magpie = f_magpie.read_text(encoding="utf-8")
    assert "PROFILE_EXTRA_BODY" in text_external, "$INFERENCEX_PATH file must be patched"
    assert "PROFILE_EXTRA_BODY" in text_magpie, "$MAGPIE_PATH/InferenceX file must ALSO be patched (the #210 fix)"


def test_ensure_benchmark_serving_patched_returns_true_when_only_magpie_path_present(
    tmp_path,
    monkeypatch,
):
    """When ``$INFERENCEX_PATH`` lacks ``benchmark_serving.py`` but Magpie's copy has it, the patcher still succeeds."""
    external = tmp_path / "external" / "InferenceX"
    external.mkdir(parents=True)
    magpie = tmp_path / "Magpie"
    (magpie / "InferenceX").mkdir(parents=True)
    f_magpie = _make_inferencex_tree_with_serving(magpie / "InferenceX")
    monkeypatch.setenv("INFERENCEX_PATH", str(external))
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))

    rc = ensure_benchmark_serving_patched()
    assert rc is True
    assert "PROFILE_EXTRA_BODY" in f_magpie.read_text(encoding="utf-8")


def test_ensure_benchmark_lib_patched_patches_both_roots_when_they_differ(
    tmp_path,
    monkeypatch,
):
    """Same multi-root contract for ``benchmark_lib.sh``."""
    legacy = (
        "#!/usr/bin/env bash\n"
        "run_benchmark_serving() {\n"
        '    local num_prompts=""\n'
        '    local max_concurrency=""\n'
        '    if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '        num_prompts="$max_concurrency"\n'
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
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))

    rc = ensure_benchmark_lib_patched()
    assert rc is True
    assert "${NUM_PROMPTS:-$max_concurrency}" in f_external.read_text(encoding="utf-8")
    assert "${NUM_PROMPTS:-$max_concurrency}" in f_magpie.read_text(encoding="utf-8"), (
        "Magpie's bundled InferenceX benchmark_lib.sh must ALSO be patched"
    )


# --- eval-dest patch: append_lm_eval_summary mv ./ -> $RESULT_DIR ---
_EVAL_DEST_FIXTURE = """\
#!/usr/bin/env bash
append_lm_eval_summary() {
    if [ -d "${out_dir}" ]; then
        while IFS= read -r -d '' jf; do
            base=$(basename "$jf")
            if [ "$base" != "meta_env.json" ]; then
                mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2
            fi
        done < <(find "${out_dir}" -type f -name "*.json*" -print0 2>/dev/null)
    fi
}
"""


def _write_eval_dest_lib(root: Path) -> Path:
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True)
    lib = bench_dir / "benchmark_lib.sh"
    lib.write_text(_EVAL_DEST_FIXTURE, encoding="utf-8")
    return lib


def test_eval_dest_patch_redirects_mv_to_result_dir(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_benchmark_lib_eval_dest_patched,
    )

    lib = _write_eval_dest_lib(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    rc = ensure_benchmark_lib_eval_dest_patched(tmp_path)
    assert rc is True
    text = lib.read_text(encoding="utf-8")
    assert 'mv -f "$jf" "${RESULT_DIR:-.}/"' in text
    assert 'mv -f "$jf" ./ ' not in text


def test_eval_dest_patch_is_idempotent(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_benchmark_lib_eval_dest_patched,
    )

    lib = _write_eval_dest_lib(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    ensure_benchmark_lib_eval_dest_patched(tmp_path)
    after_first = lib.read_text(encoding="utf-8")
    ensure_benchmark_lib_eval_dest_patched(tmp_path)
    assert lib.read_text(encoding="utf-8") == after_first
    assert after_first.count('"${RESULT_DIR:-.}/"') == 1


def test_baseline_after_materialize_applies_eval_dest_patch(tmp_path, monkeypatch):
    """BaselineExecutor's base hook (not just ProfileExecutor) must apply the
    eval-dest redirect, so a pure baseline run's ``mv ./`` writes results into
    $RESULT_DIR instead of the process cwd (the local-disk InferenceX mirror)."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_eval_dest_lib(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out")

    assert out is None  # baseline is never short-circuited
    text = lib.read_text(encoding="utf-8")
    assert 'mv -f "$jf" "${RESULT_DIR:-.}/"' in text
    assert 'mv -f "$jf" ./ ' not in text


_EVAL_START_FIXTURE = """#!/usr/bin/env bash
run_eval() {
    local results_dir="$1"

    # Export for append_lm_eval_summary to pick up
    export EVAL_RESULT_DIR="$results_dir"
    set -x
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \\
      --tasks "${tasks_dir}"
    local eval_exit=$?
    set +x
    return $eval_exit
}
"""


def _write_eval_start_lib(root: Path) -> Path:
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True)
    lib = bench_dir / "benchmark_lib.sh"
    lib.write_text(_EVAL_START_FIXTURE, encoding="utf-8")
    return lib


def _write_full_lib(root: Path) -> Path:
    """Write a ``benchmark_lib.sh`` carrying every anchor, as a checkout does.

    The single-purpose fixtures above are the right scope for testing one patcher
    directly. The executor hook is different: it verifies the contract as a whole
    before launch, and against a one-anchor stub it would correctly report the
    other patches as no longer appliable.
    """
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True)
    lib = bench_dir / "benchmark_lib.sh"
    bodies = [
        fixture.split("\n", 1)[1] if fixture.startswith("#!") else fixture
        for fixture in (_UPSTREAM_FIXTURE, _EVAL_DEST_FIXTURE, _EVAL_START_FIXTURE)
    ]
    lib.write_text("#!/usr/bin/env bash\n" + "\n".join(bodies), encoding="utf-8")
    return lib


def test_eval_start_patch_emits_marker_before_lm_eval(tmp_path, monkeypatch):
    """The marker must land after the EVAL_RESULT_DIR export and before lm_eval,
    so the soft-deadline watcher sees it exactly when the eval begins."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_benchmark_lib_eval_start_patched,
    )

    lib = _write_eval_start_lib(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    rc = ensure_benchmark_lib_eval_start_patched(tmp_path)
    assert rc is True
    text = lib.read_text(encoding="utf-8")
    assert 'echo "HYPERLOOM_EVAL_START" >&2' in text
    marker_at = text.index("HYPERLOOM_EVAL_START")
    assert text.index('export EVAL_RESULT_DIR="$results_dir"') < marker_at
    assert marker_at < text.index("python3 -m lm_eval")


def test_eval_start_patch_is_idempotent(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_benchmark_lib_eval_start_patched,
    )

    lib = _write_eval_start_lib(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    ensure_benchmark_lib_eval_start_patched(tmp_path)
    after_first = lib.read_text(encoding="utf-8")
    ensure_benchmark_lib_eval_start_patched(tmp_path)
    assert lib.read_text(encoding="utf-8") == after_first
    assert after_first.count("HYPERLOOM_EVAL_START") == 1


def test_baseline_after_materialize_applies_eval_start_patch(tmp_path, monkeypatch):
    """The eval-start marker is what keeps the explore overtime kill scoped to
    the throughput phase, so the baseline hook must install it too."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_full_lib(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out")

    assert out is None
    assert 'echo "HYPERLOOM_EVAL_START" >&2' in lib.read_text(encoding="utf-8")


# Upstream lm_eval_sitecustomize.py reduced to what the probe contract needs:
# valid Python that ends by installing its own ``parse_generations``.
_EVAL_PROBE_FIXTURE = '''"""Runtime compatibility hooks for lm-eval."""

from lm_eval.models.openai_completions import LocalChatCompletion


def _parse_generations(outputs, **kwargs):
    return [""]


LocalChatCompletion.parse_generations = staticmethod(_parse_generations)
'''


def _write_eval_probe_target(root: Path) -> Path:
    """Write a lm_eval_sitecustomize.py at the real upstream path."""
    patches_dir = root / "utils" / "evals" / "patches"
    patches_dir.mkdir(parents=True)
    target = patches_dir / "lm_eval_sitecustomize.py"
    target.write_text(_EVAL_PROBE_FIXTURE, encoding="utf-8")
    return target


def test_eval_probe_appends_to_sitecustomize_py(tmp_path, monkeypatch):
    """Probe must be appended AFTER InferenceX's own patches so _hl_prev_parse
    captures the upstream _parse_generations, not the stock lm_eval default."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_eval_probe_patched,
    )

    target = _write_eval_probe_target(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    rc = ensure_eval_probe_patched(tmp_path)
    assert rc is True
    text = target.read_text(encoding="utf-8")
    assert "HYPERLOOM_EVAL_PROBE" in text
    upstream_at = text.index("LocalChatCompletion.parse_generations = staticmethod")
    probe_at = text.index("_hl_eval_probe_install")
    assert upstream_at < probe_at


def test_eval_probe_emits_valid_python(tmp_path, monkeypatch):
    """_EVAL_PROBE_PY is a string constant never seen by the linter — compile it
    here and verify the result is syntactically valid Python."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        _EVAL_PROBE_PY,
        ensure_eval_probe_patched,
    )

    target = _write_eval_probe_target(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    ensure_eval_probe_patched(tmp_path)

    compile(_EVAL_PROBE_PY, "<probe>", "exec")
    compile(target.read_text(encoding="utf-8"), "<sitecustomize-with-probe>", "exec")


def test_eval_probe_is_idempotent(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_eval_probe_patched,
    )

    target = _write_eval_probe_target(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    ensure_eval_probe_patched(tmp_path)
    after_first = target.read_text(encoding="utf-8")
    ensure_eval_probe_patched(tmp_path)
    assert target.read_text(encoding="utf-8") == after_first
    # one def, one call
    assert after_first.count("_hl_eval_probe_install()") == 2


def test_eval_probe_returns_false_when_target_missing(tmp_path, monkeypatch):
    """An InferenceX tree without the target file is reported, not raised."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_eval_probe_patched,
    )

    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    assert ensure_eval_probe_patched(tmp_path) is False


def test_eval_probe_unreadable_target_returns_false(tmp_path, monkeypatch):
    """An unreadable target degrades to False rather than raising into the caller."""
    from hyperloom.orchestrator.actions.executors import _inferencex_patcher as patcher

    target = _write_eval_probe_target(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    def _boom(*_args, **_kwargs):
        raise OSError("read-only mount")

    monkeypatch.setattr(Path, "read_text", _boom)
    assert patcher.ensure_eval_probe_patched(tmp_path) is False
    assert target.exists()


def test_baseline_fails_loud_when_probe_target_present_but_unpatchable(tmp_path, monkeypatch):
    """A checkout that HAS the target and still cannot be patched is a hard stop."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    _write_eval_dest_lib(ix_root)
    _write_eval_probe_target(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.setattr(
        _inferencex_patcher,
        "_apply_eval_probe_atomic",
        lambda _src: False,
    )

    out = BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out")

    assert isinstance(out, dict)
    assert out["error_class"] == "eval_probe_unpatchable"


def test_baseline_warns_when_probe_target_absent(tmp_path, monkeypatch):
    """An unrecognized layout warns; it must not fail every eval run on a guess."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    _write_eval_dest_lib(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    assert BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out") is None


def test_baseline_skips_probe_gate_when_eval_disabled(tmp_path, monkeypatch):
    """RUN_EVAL off: a missing probe target is irrelevant and must not block."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    _write_eval_dest_lib(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root), "envs": {"RUN_EVAL": "false"}}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    assert BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out") is None


def test_eval_probe_is_concurrency_safe(tmp_path, monkeypatch):
    """Several executors can patch one shared checkout at once."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_eval_probe_patched,
    )

    target = _write_eval_probe_target(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(ensure_eval_probe_patched(tmp_path))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results)
    assert target.read_text(encoding="utf-8").count("_hl_eval_probe_install()") == 2


def test_baseline_after_materialize_applies_eval_probe(tmp_path, monkeypatch):
    """baseline._after_materialize_config must apply the probe patch before
    launching: a non-terminating model there stops the whole run."""
    import yaml

    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    target = _write_eval_probe_target(ix_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        yaml.safe_dump({"benchmark": {"inferencex_path": str(ix_root)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(config_path, tmp_path / "out")

    assert out is None
    assert "HYPERLOOM_EVAL_PROBE" in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Anchor contract
# ---------------------------------------------------------------------------
# The probe is absent here by design: it is appended to a real file and has no
# anchor left to rot. These cover the four patches that still locate exact
# upstream text, where a False return is indistinguishable from "nothing to do".


def _write_baseline_config(tmp_path: Path, ix_root: Path, *, run_eval: bool = True) -> Path:
    """Write the materialized Magpie YAML the hook reads its root from."""
    import yaml

    benchmark: dict[str, Any] = {"inferencex_path": str(ix_root)}
    if not run_eval:
        benchmark["envs"] = {"RUN_EVAL": "false"}
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    return config_path


def test_verify_patch_anchors_finds_every_anchor_on_a_pristine_checkout(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import verify_patch_anchors

    _write_full_lib(tmp_path)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    statuses = verify_patch_anchors(tmp_path)

    assert {s.name for s in statuses} == {"num_prompts", "eval_dest", "eval_start"}
    assert all(s.ok and s.hits == 1 and not s.patched for s in statuses)


def test_verify_patch_anchors_omits_files_that_do_not_exist(tmp_path, monkeypatch):
    """A tree with no benchmark_serving.py has nothing to patch there, which the
    ensure_* functions already treat as a skip rather than a failure."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import verify_patch_anchors

    _write_full_lib(tmp_path)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    assert "profile_extra_body" not in {s.name for s in verify_patch_anchors(tmp_path)}


def test_failed_patch_anchors_flags_text_upstream_rewrote(tmp_path, monkeypatch):
    """The regression this exists to catch: upstream rewrites the line without
    changing its meaning, so the patch stops applying and nothing says so."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import failed_patch_anchors

    lib = _write_full_lib(tmp_path)
    lib.write_text(
        lib.read_text(encoding="utf-8").replace('mv -f "$jf" ./ ', 'mv --force "$jf" ./ '),
        encoding="utf-8",
    )
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    broken = failed_patch_anchors(tmp_path)

    assert [s.name for s in broken] == ["eval_dest"]
    assert broken[0].hits == 0
    assert "ANCHOR MISSING" in broken[0].describe()


def test_failed_patch_anchors_flags_an_anchor_that_matches_twice(tmp_path, monkeypatch):
    """Every patch here rewrites one site, so an ambiguous anchor means the file
    drifted into a shape the patcher was never written for."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import failed_patch_anchors

    lib = _write_full_lib(tmp_path)
    lib.write_text(lib.read_text(encoding="utf-8") + _EVAL_START_FIXTURE, encoding="utf-8")
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    broken = failed_patch_anchors(tmp_path)

    assert [s.name for s in broken] == ["eval_start"]
    assert broken[0].hits == 2
    assert "AMBIGUOUS" in broken[0].describe()


def test_verify_patch_anchors_accepts_an_already_patched_file(tmp_path, monkeypatch):
    """Patching consumes the anchor, so a second call must not read as rot."""
    from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
        ensure_benchmark_lib_eval_dest_patched,
        failed_patch_anchors,
        verify_patch_anchors,
    )

    _write_full_lib(tmp_path)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert ensure_benchmark_lib_eval_dest_patched(tmp_path) is True

    dest = next(s for s in verify_patch_anchors(tmp_path) if s.name == "eval_dest")
    assert dest.patched and dest.hits == 0 and dest.ok
    assert failed_patch_anchors(tmp_path) == []


def test_baseline_hook_fails_loudly_when_an_eval_critical_anchor_rots(tmp_path, monkeypatch):
    """Without eval_dest the results file lands in the benchmark's cwd, where the
    accuracy parser never looks, so the gate would see no score at all."""
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_full_lib(ix_root)
    _write_eval_probe_target(ix_root)
    lib.write_text(
        lib.read_text(encoding="utf-8").replace('mv -f "$jf" ./ ', 'mv --force "$jf" ./ '),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(_write_baseline_config(tmp_path, ix_root), tmp_path / "out")

    assert isinstance(out, dict)
    assert out["error_class"] == "inferencex_patch_anchor_broken"
    assert "eval_dest" in out["error"]


def test_baseline_hook_proceeds_when_only_a_non_critical_anchor_rots(tmp_path, monkeypatch):
    """eval_start is a log breadcrumb for the soft-deadline watcher: worth
    reporting, never worth aborting a run for."""
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_full_lib(ix_root)
    _write_eval_probe_target(ix_root)
    lib.write_text(
        lib.read_text(encoding="utf-8").replace('export EVAL_RESULT_DIR="$results_dir"', "true"),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(_write_baseline_config(tmp_path, ix_root), tmp_path / "out")

    assert out is None


def test_baseline_hook_ignores_anchors_it_does_not_own(tmp_path, monkeypatch):
    """ProfileExecutor replaces this hook entirely and validates NUM_PROMPTS
    itself, so a rotted num_prompts anchor must not fail a baseline."""
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_full_lib(ix_root)
    _write_eval_probe_target(ix_root)
    lib.write_text(
        lib.read_text(encoding="utf-8").replace('        num_prompts="$max_concurrency"', "        true"),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(_write_baseline_config(tmp_path, ix_root), tmp_path / "out")

    assert out is None


def test_baseline_hook_skips_the_anchor_check_when_eval_is_off(tmp_path, monkeypatch):
    """A throughput-only run does not care where lm-eval would have written."""
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    ix_root = tmp_path / "InferenceX@deadbeef"
    lib = _write_full_lib(ix_root)
    lib.write_text(
        lib.read_text(encoding="utf-8").replace('mv -f "$jf" ./ ', 'mv --force "$jf" ./ '),
        encoding="utf-8",
    )
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    monkeypatch.delenv("MAGPIE_PATH", raising=False)

    out = BaselineExecutor()._after_materialize_config(
        _write_baseline_config(tmp_path, ix_root, run_eval=False), tmp_path / "out"
    )

    assert out is None
