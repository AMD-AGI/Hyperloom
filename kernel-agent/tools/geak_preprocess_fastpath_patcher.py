#!/usr/bin/env python3
"""Patch GEAK preprocess_v3 adapter with a deterministic Path-A fast-path.

Background
==========

GEAK v3 preprocess is LLM-orchestrator driven. When Hyperloom (or mini.py)
already supplies a pre-validated harness or explicit ``eval_command``, the
orchestrator should take Case-A (``commandment_from_user_command`` +
``collect_baseline``) without discovery/harness-generator subagents.

Observed failure (run7, 20260608T215834Z): both parallel GEAK jobs sat in
``harness-init`` for 900s with **zero** tool calls — no ``harness_sanitizer.log``,
no GPU jobs, no ``benchmark_baseline.txt``. The preprocess soft-cap watchdog
hard-failed both runs and the coordinator died.

Root cause: the v3 orchestrator blocked on its first ``model.query()`` and
never invoked tools; ``PreprocessState.current_stage`` stayed at the default
``harness-init`` the whole time, so the stage-aware soft-stop policy treated
the stall as a broken harness setup.

Fix: when ``harness`` or ``eval_command`` is supplied by the call site, bypass
the LLM orchestrator and run deterministic baseline + COMMANDMENT steps
directly. Write ``benchmark_baseline.txt`` as soon as baseline collection
succeeds so the watchdog sees progress.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SENTINEL = "# HYPERLOOM_PATH_A_FASTPATH"

_INSERT_AFTER = "logger = logging.getLogger(__name__)\n"

_FASTPATH_FUNC = '''

# HYPERLOOM_PATH_A_FASTPATH
def _run_path_a_fastpath(
    *,
    harness: str | None,
    eval_command: str | None,
    output_dir: Path,
    kernel_path: Path,
    repo_root: str,
    detected_language: KernelLanguage,
    gpu_id: int,
    state: Any,
) -> PreprocessResult | None:
    """Deterministic preprocess when the call site already supplied a runnable command."""
    if os.environ.get("GEAK_SKIP_PATH_A_FASTPATH", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    from minisweagent.run.preprocess_v3.baseline import (
        BaselineMetrics,
        capture_full_benchmark_stdout,
        collect_baseline_from_eval_command,
        collect_baseline_metrics,
        collect_profile,
    )
    from minisweagent.run.preprocess_v3.commandment import CommandmentContext, render_commandment
    from minisweagent.run.state import PreprocessStage

    harness_path: Path | None = None
    if harness:
        candidate = Path(harness).expanduser()
        if not candidate.is_file():
            return None
        try:
            from minisweagent.run.preprocess.harness_utils import validate_harness

            valid, _messages = validate_harness(str(candidate.resolve()))
        except Exception:
            valid = True
        if not valid:
            return None
        harness_path = candidate.resolve()

    eval_cmd = (eval_command or "").strip() or None
    if harness_path is None and not eval_cmd:
        return None

    logger.info(
        "v3 preprocess: deterministic Path-A fast-path (harness=%s, eval_command=%s)",
        harness_path,
        eval_cmd,
    )

    if state is not None:
        try:
            state.set_stage(PreprocessStage.HARNESS_BENCHMARK)
        except Exception as exc:
            logger.debug("Path-A fast-path: state.set_stage failed (non-fatal): %s", exc)

    t0 = time.monotonic()
    baseline: BaselineMetrics | None = None
    profile = None
    full_benchmark_stdout: str | None = None

    skip_profile = os.environ.get("GEAK_SKIP_PREPROCESS_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    bench_repeats = 5
    if harness_path is not None and "bench_moe_tile" in harness_path.name:
        # fused_moe tile harness: baseline benchmark is the verified gate;
        # PROFILE + FULL_BENCHMARK duplicate work and burned 40+ min in run11.
        skip_profile = True
        bench_repeats = 1
        # NOTE: do NOT wrap the harness in a .sh here. A sanitized .sh wrapper
        # (exec python <harness>) breaks every downstream caller that invokes
        # the harness with python/python3 — both _benchmark_command's
        # correctness gate AND the COMMANDMENT's run.sh do `python3 "$@"`, so a
        # .sh harness fails with `SyntaxError: invalid syntax` on
        # `set -euo pipefail`. The .py harness is self-sufficient: it derives
        # PYTHONPATH from GEAK_WORK_DIR (worktree-aware import), and the
        # COMMANDMENT run.sh prepends GEAK_WORK_DIR/python anyway. Keep
        # harness_path as the .py so it stays python-invocable end to end.

    if harness_path is not None:
        baseline = collect_baseline_metrics(
            harness_path, gpu_id=gpu_id, repeats=bench_repeats,
        )
        if not skip_profile:
            try:
                profile = collect_profile(harness_path, gpu_id=gpu_id)
            except Exception as exc:
                logger.warning("Path-A fast-path profile failed (non-fatal): %s", exc)
            full_benchmark_stdout = capture_full_benchmark_stdout(harness_path, gpu_id=gpu_id)
        else:
            logger.info(
                "Path-A fast-path: skipping PROFILE/FULL_BENCHMARK "
                "(GEAK_SKIP_PREPROCESS_PROFILE or bench_moe_tile harness)"
            )
    else:
        baseline = collect_baseline_from_eval_command(eval_cmd, gpu_id=gpu_id, repeats=5)

    if baseline is not None and baseline.success:
        for raw in baseline.raw_outputs:
            if raw.get("returncode") == 0 and str(raw.get("stdout") or "").strip():
                text = str(raw["stdout"])
                (output_dir / "benchmark_baseline.txt").write_text(text, encoding="utf-8")
                (output_dir / "full_benchmark_baseline.txt").write_text(text, encoding="utf-8")
                break

    commandment_path = output_dir / "COMMANDMENT.md"
    if harness_path is not None:
        baseline_metrics: dict[str, Any] = {}
        if baseline is not None and baseline.median_ms is not None:
            baseline_metrics = {
                "median_ms": baseline.median_ms,
                "samples_ms": list(baseline.samples_ms),
                "stdev_ms": baseline.stdev_ms,
                "repeats": baseline.repeats,
                "command": baseline.command,
            }
        ctx = CommandmentContext(
            kernel_path=kernel_path,
            harness_path=harness_path,
            repo_root=Path(repo_root),
            baseline_metrics=baseline_metrics or None,
        )
        render_commandment(detected_language, ctx, out_path=commandment_path)
    elif eval_cmd:
        _nl = chr(10)
        commandment_path.write_text(
            _nl.join(
                [
                    "# COMMANDMENT (Path-A fast-path)",
                    "",
                    "## Setup",
                    eval_cmd,
                    "",
                    "## Benchmark",
                    eval_cmd,
                ]
            ),
            encoding="utf-8",
        )

    elapsed_s = round(time.monotonic() - t0, 3)
    success = (
        commandment_path.is_file()
        and baseline is not None
        and baseline.success
    )
    errors: list[str] = []
    if baseline is None or not baseline.success:
        errors.append("Path-A fast-path: baseline collection failed")

    return PreprocessResult(
        success=success,
        kernel_language=detected_language,
        kernel_path=kernel_path,
        harness_path=harness_path,
        baseline=baseline,
        full_benchmark_stdout=full_benchmark_stdout,
        profile=profile,
        commandment_path=commandment_path if commandment_path.is_file() else None,
        path_taken="A",
        elapsed_s=elapsed_s,
        errors=errors,
    )
'''

_CALL_SITE_OLD = """    task = _build_orchestrator_task(
        user_task=user_task,
        harness=harness,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
        benchmark_timeout=benchmark_timeout,
        translate_only=translate_only,
    )

    t0 = time.monotonic()
    result: PreprocessResult = agent.run("""

_CALL_SITE_NEW = """    task = _build_orchestrator_task(
        user_task=user_task,
        harness=harness,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
        benchmark_timeout=benchmark_timeout,
        translate_only=translate_only,
    )

    fast = _run_path_a_fastpath(
        harness=harness,
        eval_command=eval_command,
        output_dir=output_dir,
        kernel_path=kernel_path,
        repo_root=repo_root,
        detected_language=detected_language,
        gpu_id=gpu_id,
        state=state,
    )
    if fast is not None:
        logger.info(
            "v3 preprocess Path-A fast-path completed in %.1fs (success=%s, errors=%d)",
            fast.elapsed_s,
            fast.success,
            len(fast.errors),
        )
        if not fast.success:
            raise RuntimeError(
                "v3 preprocess failed: "
                + ("; ".join(fast.errors) if fast.errors else "Path-A fast-path produced no baseline")
            )
        return _preprocess_result_to_legacy_context(
            result=fast,
            repo_root=repo_root,
            output_dir=output_dir,
            kernel_path_input=kernel_path,
            harness=harness,
            eval_command=eval_command,
            correctness_command=correctness_command,
            performance_command=performance_command,
        )

    t0 = time.monotonic()
    result: PreprocessResult = agent.run("""


def _adapter_paths() -> list[Path]:
    paths: list[Path] = []
    mirror = Path(os.environ.get("HYPERLOOM_ROOT", "")) / "geak" / "src" / "minisweagent" / "run" / "preprocess_v3" / "adapter.py"
    if mirror.is_file():
        paths.append(mirror)
    try:
        import minisweagent.run.preprocess_v3.adapter as mod

        paths.append(Path(mod.__file__).resolve())
    except Exception:
        pass
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def patch_file(path: Path) -> tuple[bool, str]:
    text = path.read_text(encoding="utf-8")
    if _SENTINEL in text:
        # Re-apply if a prior broken patch left a syntax error (unterminated string).
        if 'commandment_path.write_text(\n            "\n".join(' not in text and "_nl.join(" not in text:
            return True, "already patched"
        text = _strip_broken_fastpath(text)

    if _SENTINEL in text:
        return True, "already patched"

    if _INSERT_AFTER not in text:
        return False, f"anchor not found in {path}"

    if _CALL_SITE_OLD not in text:
        return False, f"call-site anchor not found in {path}"

    text = text.replace(_INSERT_AFTER, _INSERT_AFTER + _FASTPATH_FUNC, 1)
    text = text.replace(_CALL_SITE_OLD, _CALL_SITE_NEW, 1)
    path.write_text(text, encoding="utf-8")
    return True, "patched"


def _strip_broken_fastpath(text: str) -> str:
    """Remove a broken fast-path insertion so we can re-patch cleanly."""
    start = text.find(_SENTINEL)
    if start == -1:
        return text
    end = text.find("\n\n# ---------------------------------------------------------------------------\n# Public entry point", start)
    if end == -1:
        end = text.find("\ndef run_preprocess_v3(", start)
    if end == -1:
        return text
    return text[:start] + text[end + 2 :]


def main() -> int:
    patched_any = False
    for path in _adapter_paths():
        ok, msg = patch_file(path)
        print(f"[geak-preprocess-fastpath] {path}: {msg}")
        if ok and msg == "patched":
            patched_any = True
    if not _adapter_paths():
        print("[geak-preprocess-fastpath] WARN: no adapter.py targets found", file=sys.stderr)
        return 1
    return 0 if patched_any or any(
        _SENTINEL in p.read_text(encoding="utf-8") for p in _adapter_paths() if p.is_file()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
