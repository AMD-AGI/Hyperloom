"""Real ``profile`` ActionRunner — Magpie SGLang run with torch profiler on.

 profile action.

Reuses the BaselineExecutor shell-out machinery; the only meaningful
difference is the YAML config — the profile config has
``profiler.torch_profiler.enabled: true`` so Magpie writes trace files under
``torch_trace/`` or, for TraceLens-patched vLLM graph capture,
``capture_traces/``.

Result schema (delivered on the bus as ``delegated_result``)::

    status:        "succeeded" | "failed"
    framework:     "sglang"
    model:         path
    request/output/total_token_throughput, latency stats (same as baseline)
    workspace:     absolute path of the Magpie workspace
    trace_dir:     absolute path of the torch_trace dir (or None)
    report_path:   absolute path of benchmark_report.json

Downstream consumers (Kernel agent → tracelens_analysis.py) only need
``trace_dir``; we surface the rest so the same SharedState promotion
logic the baseline path uses (current_best, output_throughput, etc.)
works unchanged.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ...paths import asset_root, mn_profile_trace_root
from ._inferencex_patcher import (
    ensure_benchmark_lib_patched,
    ensure_benchmark_serving_patched,
)
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# Bytes of a trace file to sample when scanning for sentinel
# substrings. Reading the entire decompressed trace would burn tens of
# seconds on a 100 MB+ rank-0 trace; the leading 2 MB is enough to
# catch presence/absence of the markers we care about (per Deval's
# check_torch_trace.py reference output: thousands of events for each
# sentinel in a properly-patched run).
_TRACE_INSPECT_BYTES = 2_000_000

# Minimum fraction of ``cpu_op`` events that should carry an
# ``Input Dims`` field for a ``capture_traces/`` file to be considered
# healthy (Deval's reference run reports 99.97%; we set the gate
# generously low so trivial misses don't false-positive).
_INPUT_DIMS_FRACTION_FLOOR = 0.90


def _sample_trace_text(path: Path) -> str | None:
    """Read up to ``_TRACE_INSPECT_BYTES`` of decompressed text from a
    gzipped trace file. Returns ``None`` (and debug-logs) on any IO /
    decode error so callers can skip the check rather than fail the
    profile path."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read(_TRACE_INSPECT_BYTES)
    except (OSError, EOFError, UnicodeDecodeError) as e:
        # gzip raises ``EOFError`` on truncated streams, ``OSError``
        # on read failures, ``UnicodeDecodeError`` on malformed
        # UTF-8. Validator is best-effort — never let a malformed
        # sample fail the profile post-execution path.
        log.debug(
            "_validate_trace_structure: cannot sample %s: %s", path, e,
        )
        return None


def _count_substring_occurrences(text: str, substring: str) -> int:
    """Count non-overlapping ``substring`` occurrences in ``text``.
    Used as a cheap proxy for "this kind of event appears N times in
    the trace JSON" without paying for full JSON parsing — JSON event
    names appear as ``"name": "<value>"`` so a substring count is a
    reasonable lower-bound."""
    if not substring:
        return 0
    return text.count(substring)


def _validate_trace_structure(
    trace_dir: Path, framework: str, expected_pieces: int = 1,
) -> None:
    """Post-profile sanity check (#210 / Deval's ``check_torch_trace.py``).

    Logs *warnings* — never raises — when the trace folder structure
    or contents suggest TraceLens-only profiling features didn't
    reach the live framework. Mirrors the 5 checks in
    ``check_torch_trace.py`` (Deval, #210 comment 1) plus one
    Hyperloom-specific check for the #210 smoking-gun symptom:

    1. ``capture_traces/`` exists with files (graph capture fired)
    2. capture files contain ``cpu_op`` events with ``Input Dims``
       (shape-discovery instrumentation recording per-event shapes)
    3. main-directory trace has ``user_annotation`` events including
       ``execute_*`` annotations (InferenceX per-step instrumentation
       fired — relates to ``roofline_annotations`` flag landing)
    4. ``trace_split/`` per-file ``execute_*`` user_annotations
       counted (splitter ran AND each split is non-empty)
    5. (sglang only) main trace contains ``kernel_shape_profiler``
       substring (server-side patch from PR #207 landed)
    6. (Hyperloom-specific) ``trace_split/`` has ``_steady_state_*``
       files, NOT ``_extend_*`` / ``_decode_*`` (the #210 smoking-
       gun: profile_by_stage leaked through PROFILE_EXTRA_BODY)

    Read-only; safe to call after every profile execution. Each
    check's failure produces an independent warning so partial
    signals stay actionable.
    """
    issues: list[str] = []

    # --- Check 1: capture_traces/ presence ---
    capture = trace_dir / "capture_traces"
    capture_files: list[Path] = []
    if not capture.is_dir():
        issues.append(
            "[1] capture_traces/ subdirectory missing — graph capture "
            "didn't fire. Verify EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS "
            "include the TraceLens flag and the server-side patch landed."
        )
    else:
        capture_files = sorted(p for p in capture.iterdir() if p.is_file())
        if not capture_files:
            issues.append(
                "[1] capture_traces/ exists but is empty — graph capture "
                "path fired but produced no files."
            )

    # --- Check 2 (Deval): capture file has cpu_op + Input Dims ---
    # Sample the heaviest capture file and count cpu_op events vs
    # cpu_op events with an "Input Dims" field. A healthy run has
    # >= 99% with Input Dims (Deval ref: 12441/12445 = 99.97%); we
    # gate at _INPUT_DIMS_FRACTION_FLOOR to avoid false positives on
    # tiny traces with rounding noise.
    if capture_files:
        target = max(capture_files, key=lambda p: p.stat().st_size)
        text = _sample_trace_text(target)
        if text is not None:
            cpu_op_count = _count_substring_occurrences(text, '"name": "cpu_op"')
            input_dims_count = _count_substring_occurrences(text, '"Input Dims"')
            if cpu_op_count == 0:
                issues.append(
                    f"[2] capture file {target.name} has no cpu_op events "
                    f"in the first {_TRACE_INSPECT_BYTES//1_000_000} MB — "
                    "graph capture wrote files but they don't contain "
                    "kernel-level events. Possible upstream profiler "
                    "regression."
                )
            elif input_dims_count / max(cpu_op_count, 1) < _INPUT_DIMS_FRACTION_FLOOR:
                pct = 100.0 * input_dims_count / cpu_op_count
                issues.append(
                    f"[2] capture file {target.name}: only {pct:.1f}% of "
                    f"cpu_op events carry 'Input Dims' (expected ≥ "
                    f"{int(_INPUT_DIMS_FRACTION_FLOOR * 100)}%). Shape-"
                    "discovery instrumentation may not be fully active — "
                    "verify TraceLens server patch and capture flag."
                )

    # --- Check 3 (Deval): main trace has user_annotation + execute_* ---
    # The execute_* annotations are what InferenceX writes per-step
    # when roofline_annotations=True is honoured by the server. Their
    # absence is a separate failure mode from kernel_shape_profiler
    # absence (check 5) — a partially-patched run can have one
    # without the other.
    main_traces = sorted(
        (p for p in trace_dir.glob("*.trace.json.gz") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    main_text: str | None = None
    if main_traces:
        main_text = _sample_trace_text(main_traces[0])
        if main_text is not None:
            user_ann_count = _count_substring_occurrences(
                main_text, '"name": "user_annotation"',
            )
            execute_count = _count_substring_occurrences(main_text, '"execute_')
            if user_ann_count == 0:
                issues.append(
                    f"[3] main trace {main_traces[0].name} has no "
                    "user_annotation events — InferenceX per-step "
                    "annotations didn't fire. Verify roofline_annotations "
                    "reached the framework (PROFILE_EXTRA_BODY consumed; "
                    "see #210)."
                )
            elif execute_count == 0:
                issues.append(
                    f"[3] main trace {main_traces[0].name} has "
                    f"user_annotation events but no execute_* annotations "
                    "— InferenceX is annotating but the per-step "
                    "execute_* labels are missing. Check the InferenceX "
                    "version actually loaded by Magpie."
                )

    # --- Check 4 (Deval): per-file execute_* in trace_split/ ---
    # Each splitter output should contain its own slice of execute_*
    # annotations; an empty split means the splitter ran but received
    # no usable events for that window.
    split = trace_dir / "trace_split"
    split_files: list[Path] = []
    if split.is_dir():
        split_files = sorted(p for p in split.iterdir() if p.is_file())
        empty_splits: list[str] = []
        for sp in split_files:
            if not sp.name.endswith(".json.gz"):
                continue
            text = _sample_trace_text(sp)
            if text is None:
                continue
            if _count_substring_occurrences(text, '"execute_') == 0:
                empty_splits.append(sp.name)
        if split_files and empty_splits:
            issues.append(
                f"[4] {len(empty_splits)} trace_split/ file(s) have NO "
                "execute_* user_annotations: "
                f"{', '.join(empty_splits[:3])}"
                + (f" (and {len(empty_splits) - 3} more)" if len(empty_splits) > 3 else "")
                + " — splitter ran but the chunks are empty. Likely the "
                "trace lacks the per-step annotations needed for splitting "
                "(see check [3])."
            )

    # --- Check 6 (Hyperloom-specific #210 smoking-gun): _extend_* / ---
    # _decode_* without _steady_state_* in trace_split/.
    if split.is_dir():
        names = [p.name for p in split_files]
        has_extend = any("_extend_" in n or "extend_only_" in n for n in names)
        has_decode = any("_decode_" in n or "decode_only_" in n for n in names)
        has_steady_state = any("steady_state" in n for n in names)
        # Single ``decode_only_steady_state_*`` is fine (steady-state
        # decode chunk per Deval's example layout); the failure mode is
        # ``_extend_<step>_*`` / ``_decode_<step>_*`` per-step files
        # without any ``steady_state`` marker.
        if (has_extend or has_decode) and not has_steady_state:
            issues.append(
                "[6] trace_split/ has _extend_* / _decode_* files but NO "
                "_steady_state_* — profile_by_stage=True leaked through, "
                "PROFILE_EXTRA_BODY env was not consumed by the framework. "
                "Confirm _inferencex_patcher patched Magpie's bundled "
                "InferenceX (#210; check $MAGPIE_DIR/InferenceX/utils/"
                "bench_serving/benchmark_serving.py)."
            )

    # --- Check 5 (Deval): sglang kernel_shape_profiler presence ---
    if framework.lower() == "sglang" and main_text is not None:
        if "kernel_shape_profiler" not in main_text:
            issues.append(
                f"[5] sglang main trace ({main_traces[0].name}, sampled "
                f"first {_TRACE_INSPECT_BYTES//1_000_000} MB) lacks "
                "kernel_shape_profiler events — shape-discovery "
                "patch didn't reach the live SGLang. Verify "
                "_server_patcher (PR #207) succeeded for the "
                "deployed SGLang version (check log warnings)."
            )

    if issues:
        for issue in issues:
            log.warning("trace structure check: %s", issue)
        log.warning(
            "trace structure check: %d issue(s) detected — TraceLens "
            "downstream analysis may be degraded. See per-issue messages "
            "above for the actionable check.", len(issues),
        )


# Legacy constant kept pointing at the sglang profile yaml for fixture use.
# Runtime sglang/vllm selection goes through `_default_profile_config()`.
PROFILE_DEFAULT_CONFIG = (
    asset_root() / "scripts" / "configs" / "profile_sglang.yaml"
)
PROFILE_DEFAULT_TIMEOUT_SEC = 14400    # 4 h wall cap; Qwen3-32B TP=1 profile needs ~3 h with steady-state window


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return trace files under ``trace_dir`` in a stable order.

    Magpie's classic torch-profiler path writes under
    ``<benchmark_workspace>/torch_trace``. TraceLens-patched vLLM writes graph
    capture traces under ``<profile_task>/capture_traces``. Both use
    ``*.trace.json.gz`` names, and nested layouts are possible as the profiler
    evolves.
    """
    return sorted(trace_dir.rglob("*.trace.json.gz"))


def _preferred_main_trace_path(trace_dir: Path, trace_files: list[Path]) -> Path:
    """Trace path to pass downstream to TraceLens.

    The old `sorted(...)[0]` behaviour picked `TP-0-DECODE.trace.json.gz`
    before `merged-*.trace.json.gz`, which hands TraceLens a tiny single-rank
    decode slice instead of the large annotated trace its splitter expects.
    Prefer the merged trace when Magpie produced one; otherwise pass the trace
    directory so kernel-agent can apply its own ordering instead of pinning a
    single staged file.
    """
    merged = sorted(p for p in trace_files if p.name.startswith("merged-"))
    return merged[0] if merged else trace_dir


def _candidate_trace_dirs(workspace: Path) -> list[Path]:
    """Trace directories to probe for a Magpie profile workspace."""
    return [
        workspace / "torch_trace",
        workspace / "capture_traces",
        workspace.parent / "capture_traces",
    ]


def _safe_mtime(p: Path) -> float:
    """Return st_mtime, or 0 on stat() failure (e.g. NFS stale handle)."""
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _default_profile_config() -> Path:
    """Resolve default profile YAML based on $FRAMEWORK env (sglang/vllm)."""
    fw = os.environ.get("FRAMEWORK", "sglang").strip().lower()
    name = "profile_vllm.yaml" if fw == "vllm" else "profile_sglang.yaml"
    return asset_root() / "scripts" / "configs" / name


class ProfileExecutor(BaselineExecutor):
    """Subclass that swaps the default config + extracts trace_dir."""

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_timeout_sec: int = PROFILE_DEFAULT_TIMEOUT_SEC,
        cwd: Path | str = "/tmp",
    ):
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            session_dir=session_dir,
            default_timeout_sec=default_timeout_sec,
            cwd=cwd,
        )

    def _resolve_default_config(self) -> Path:
        """Override BaselineExecutor's resolver to pick the profile yaml."""
        return _default_profile_config()

    def _resolve_mn_round_trace_root(self, ctx) -> str:
        """Return the shared torch-trace base dir for multi-node, or ''.

        Returns the SAME base dir for every profile round in the session.
        ``cli.py`` exports it as
        ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` =
        ``<mn_profile_trace_root>/<rayjob>/torch_trace`` at provisioning
        time, where ``mn_profile_trace_root`` is anchored on
        ``$USER_DATA_PATH`` (see :func:`inference_optimizer.paths.
        mn_profile_trace_root`). Sglang server's
        ``SGLANG_TORCH_PROFILER_DIR`` is pinned to this base on first
        launch (see ``multi_node/scripts/launch_multinode.py``), so all
        profile rounds write trace.json.gz files into a single shared dir.

        The ``__call__`` mtime gate (records ``task_started_unix`` and
        filters trace files newer than that) is what isolates the
        current round's traces from earlier rounds' leftovers.

        Why not round-scoped subdirs anymore: the multi_node restart
        path now resumes a running launch (cli.py's
        ``MULTI_NODE_RESTART_RESUME_RUNNING``), so SGLANG_TORCH_PROFILER_DIR
        never gets re-injected per round — only the first sglang launch
        sees that env. A single shared base + mtime gate is the simplest
        fix that keeps resume's 14-min cold-start saving intact.

        Three-tier resolution (each non-empty result short-circuits):

        1. ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` env (in-process provision).
        2. State-file ``rayjob_id`` →
           ``<mn_profile_trace_root>/<rayjob>/torch_trace``. Mirrors
           ``multi_node/cli.py::cmd_restart_server`` so out-of-band
           launches (agent invokes ``multi_node create-rayjob`` directly,
           without going through ``inference_optimizer.cli._run_optimize``)
           still get a per-RayJob unique dir.
        3. ``<mn_profile_trace_root>/default-<pid>/torch_trace`` —
           last-resort guard against concurrent sandbox processes
           silently writing into a shared ``default/`` dir when both
           env and state-file are missing. The pid uniquely partitions
           per Python process inside the sandbox; collisions across
           sandboxes would require BOTH a state-file gap AND the same
           pid recycled, which is functionally never.

        The resolved dir is mkdir'd best-effort so a sandbox-side reader
        doesn't immediately FileNotFoundError on probe.
        """
        from ._multi_node_env import is_multi_node, rayjob_id_from_state
        if not is_multi_node():
            return ""
        provisioned = os.environ.get(
            "HYPERLOOM_MN_PROFILE_TRACE_DIR", ""
        ).strip()
        if provisioned:
            return provisioned
        # Tier 2: derive from state-file rayjob_id (out-of-band launches).
        rid = rayjob_id_from_state()
        if rid:
            scoped = mn_profile_trace_root() / rid / "torch_trace"
        else:
            # Tier 3: pid-scoped last-resort so concurrent sandboxes
            # never share a dir.
            scoped = mn_profile_trace_root() / f"default-{os.getpid()}" / "torch_trace"
        try:
            scoped.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning(
                "cannot mkdir multi-node profile fallback dir %s: %s; "
                "downstream readers may FileNotFoundError",
                scoped, exc,
            )
        return str(scoped)

    def _after_materialize_config(
        self, config_path: Path, output_dir: Path,
    ) -> dict[str, Any] | None:
        """Patch the exact InferenceX checkout Magpie will execute.

        `$INFERENCEX_PATH` alone is not enough: Magpie resolves an empty
        `benchmark.inferencex_path` to its own sibling checkout. Read the
        materialized YAML and patch that resolved path, so NUM_PROMPTS and
        PROFILE_EXTRA_BODY cannot be applied to one checkout while Magpie runs
        another.
        """
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "error_class": "profile_config_unreadable",
                "error": f"cannot read materialized profile config {config_path}: {exc}",
            }
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        inferencex_path = ""
        if isinstance(bench, dict):
            inferencex_path = str(bench.get("inferencex_path") or "").strip()
        if not inferencex_path:
            inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
        if not inferencex_path:
            log.warning(
                "profile_executor: no benchmark.inferencex_path / "
                "INFERENCEX_PATH configured; skipping InferenceX profile "
                "patch validation"
            )
            return None

        ix_root = Path(inferencex_path)
        lib_ok = ensure_benchmark_lib_patched(ix_root)
        serving_ok = ensure_benchmark_serving_patched(ix_root)
        lib_path = ix_root / "benchmarks" / "benchmark_lib.sh"
        serving_path = ix_root / "utils" / "bench_serving" / "benchmark_serving.py"

        def _contains(path: Path, needle: str) -> bool:
            try:
                return needle in path.read_text(encoding="utf-8")
            except OSError:
                return False

        lib_valid = _contains(lib_path, '${NUM_PROMPTS:-$max_concurrency}')
        serving_valid = _contains(serving_path, "PROFILE_EXTRA_BODY")
        if not (lib_ok and serving_ok and lib_valid and serving_valid):
            return {
                "status": "failed",
                "error_class": "profile_inferencex_patch_failed",
                "error": (
                    "profile requires InferenceX to honour NUM_PROMPTS and "
                    "PROFILE_EXTRA_BODY, but the checkout Magpie will use is "
                    f"not patched: inferencex_path={ix_root}, "
                    f"benchmark_lib_ok={lib_ok}/{lib_valid}, "
                    f"benchmark_serving_ok={serving_ok}/{serving_valid}"
                ),
                "inferencex_path": str(ix_root),
                "benchmark_lib": str(lib_path),
                "benchmark_serving": str(serving_path),
            }
        return None

    async def __call__(self, ctx) -> dict[str, Any]:
        # B2: atom (Magpie v1) has no torch_profiler wiring. atom_mi*x.sh
        # accepts PROFILE=1 but silently no-ops; injecting sglang/vllm
        # TraceLens flags into EXTRA_ATOM_ARGS would crash atom's argparse.
        # Short-circuit here so the EXPLORE specialist's occasional profile
        # proposal degrades to a skipped delegated_result instead of a
        # spurious failed run. Coordinator already treats skipped as
        # non-fatal (no RCA escalation, no current_best mutation).
        if os.environ.get("FRAMEWORK", "").strip().lower() == "atom":
            return {
                "status": "skipped",
                "framework": "atom",
                "error_class": "atom_no_profiler",
                "error": (
                    "atom framework has no torch_profiler integration in "
                    "Magpie v1; profile/roofline are no-ops for this run. "
                    "See atom_boost_tutorials.md §6."
                ),
            }
        # Override action label so per-task output lands under runs/profile/
        # rather than runs/baseline/ when the runner derives the path.
        params = ctx.task.params or {}
        extra = getattr(ctx, "extra", None) or {}
        if not (params.get("output_dir") or extra.get("workspace")):
            output_dir = self._resolve_workspace(ctx, "profile")
            output_dir.mkdir(parents=True, exist_ok=True)
            # Stash so BaselineExecutor.__call__ picks it up via extra.
            if extra is None:
                ctx.extra = {"workspace": str(output_dir)}
                extra = ctx.extra
            else:
                extra["workspace"] = str(output_dir)

        # Mtime gate for the multi-node shared-trace-dir layout. Captured
        # BEFORE super().__call__ kicks off the magpie subprocess so any
        # trace.json.gz written by this round's /start_profile request is
        # newer than this timestamp. Previous rounds' traces stay below
        # this watermark and get filtered out below.
        import time as _time
        task_started_unix = _time.time()

        # Multi-node banner: silent for single-node. Surfaces the round's
        # trace dir so an operator can tell apart a multi-node profile
        # round (uses shared wekafs trace base) from a single-node one
        # (uses workspace-local torch_trace/).
        from ._multi_node_env import log_mn_banner
        log_mn_banner(
            "profile_executor", log,
            trace_dir=self._resolve_mn_round_trace_root(ctx),
        )

        # Multi-node only: pre-restart the inference server with this
        # round's profiler dir so sglang launches with
        # ``--torch-profiler-dir <round_path>``. Mark
        # ``ctx.extra['mn_round_restarted']`` so BaselineExecutor.__call__
        # skips a second restart. No-op in single-node.
        round_trace_root = self._resolve_mn_round_trace_root(ctx)
        if round_trace_root:
            from ._multi_node_server_lifecycle import (
                ServerRestartFailed,
                restart_server_for_round,
            )
            try:
                # PD knobs auto-resolved by the helper from $PD_* env
                # (cli.py exported them). See baseline.py for rationale.
                await restart_server_for_round(
                    extra_sglang_args=str(params.get("extra_sglang_args") or ""),
                    torch_profiler_dir=round_trace_root,
                    framework=os.environ.get("FRAMEWORK") or None,
                    model_path=(
                        str(params.get("model_path") or "").strip()
                        or os.environ.get("MODEL_PATH") or None
                    ),
                    tp=int(os.environ.get("TP") or 0) or None,
                    ep=int(os.environ.get("EP") or 0) or None,
                )
            except ServerRestartFailed as exc:
                return {
                    "status": "failed",
                    "error_class": "mn_server_restart_failed",
                    "error": str(exc),
                    "trace_dir": round_trace_root,
                }
            if isinstance(extra, dict):
                extra["mn_round_restarted"] = True

        # NOTE: InferenceX patches (``ensure_benchmark_lib_patched`` and
        # ``ensure_benchmark_serving_patched``) used to live here on the
        # multi-node feature branch. Main moved them into the
        # ``_after_materialize_config`` hook (run earlier in the
        # materialize step) so the patch always covers the exact
        # InferenceX checkout Magpie will execute, regardless of which
        # subdirectory benchmark.inferencex_path resolves to. Removing
        # the duplicate calls here keeps the patch idempotent (same
        # behaviour) while letting the single-source-of-truth in
        # ``_after_materialize_config`` carry the resolved-path
        # validation.
        result = await super().__call__(ctx)

        # Augment with trace_dir if the workspace produced one.
        # Multi-node: trace files live at the round-scoped wekafs dir we
        # just restarted with. We do NOT read $HYPERLOOM_MN_PROFILE_TRACE_DIR
        # here because the helper restored it back to the rayjob-root
        # default after restart so a subsequent non-profile round won't
        # leak this round's path. Single-node falls through to the
        # workspace/torch_trace branch below.
        workspace_str = result.get("workspace")
        if round_trace_root:
            # Multi-node branch: torch traces land at the SHARED wekafs
            # base dir that sglang's ``SGLANG_TORCH_PROFILER_DIR`` was
            # pinned to on first launch (see
            # ``_resolve_mn_round_trace_root`` for design rationale).
            # ``_candidate_trace_dirs`` is workspace-local, so it never
            # matches in multi-node — handle it explicitly here.
            #
            # Mtime gate: every profile round writes into the same base
            # dir under the resume path, so we filter to files created
            # at-or-after this round's ``task_started_unix``. Without
            # this, the executor would always pick up the FIRST round's
            # stale trace.
            trace_dir = Path(round_trace_root)
            if trace_dir.is_dir():
                all_files = sorted(trace_dir.glob("*.trace.json.gz"))
                trace_files = [
                    p for p in all_files
                    if _safe_mtime(p) >= task_started_unix
                ]
                result["trace_dir"] = str(trace_dir)
                result["trace_files"] = [str(p) for p in trace_files]
                if trace_files:
                    result["main_trace_path"] = str(trace_files[0])
                elif all_files:
                    log.warning(
                        "profile_executor: multi-node trace dir %s has "
                        "%d historical trace(s) but none with mtime >= "
                        "%.0f (this round's start); sglang may have "
                        "skipped /start_profile or the trace flush is "
                        "lagging", trace_dir, len(all_files),
                        task_started_unix,
                    )
                else:
                    log.warning(
                        "profile_executor: multi-node trace dir %s exists "
                        "but no .trace.json.gz files found yet (server "
                        "pods may still be flushing)", trace_dir,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                log.warning(
                    "profile_executor: round trace dir %s does not exist "
                    "after magpie completed; check sglang server logs for "
                    "torch profiler errors",
                    round_trace_root,
                )
        elif workspace_str:
            # Single-node branch: multi-candidate trace discovery.
            workspace = Path(workspace_str)
            selected_trace_dir: Path | None = None
            selected_trace_files: list[Path] = []
            existing_empty_dirs: list[Path] = []
            for trace_dir in _candidate_trace_dirs(workspace):
                if not trace_dir.is_dir():
                    continue
                trace_files = _trace_files_for_dir(trace_dir)
                if trace_files:
                    selected_trace_dir = trace_dir
                    selected_trace_files = trace_files
                    break
                existing_empty_dirs.append(trace_dir)

            if selected_trace_dir is not None:
                result["trace_dir"] = str(selected_trace_dir)
                result["trace_files"] = [str(p) for p in selected_trace_files]
                main_trace = _preferred_main_trace_path(
                    selected_trace_dir, selected_trace_files,
                )
                result["main_trace_path"] = str(main_trace)
                result["profile_trace_selection_reason"] = (
                    "merged_trace_preferred"
                    if main_trace.name.startswith("merged-")
                    else "trace_dir_preferred"
                )
                # #210 / Deval's check_torch_trace.py guidance: warn
                # if the trace folder shape suggests PROFILE_EXTRA_BODY
                # leaked / shape-discovery missing. Read-only inspect;
                # never blocks the run.
                try:
                    framework = str(
                        getattr(ctx, "framework", "")
                        or (extra.get("framework") if isinstance(extra, dict) else "")
                        or ""
                    )
                    _validate_trace_structure(selected_trace_dir, framework)
                except Exception as e:  # noqa: BLE001 - validator is best-effort
                    log.debug(
                        "profile_executor: trace structure validator failed: %s", e,
                    )
            else:
                result["trace_dir"] = None
                result["trace_files"] = []
                result["status"] = "failed"
                result["error_class"] = "no_trace_files"
                probed = ", ".join(
                    str(p) for p in _candidate_trace_dirs(workspace)
                )
                result["error"] = (
                    f"no .trace.json.gz under {workspace_str} (probed: {probed})"
                )
                if existing_empty_dirs:
                    log.warning(
                        "profile_executor: trace dirs exist but no "
                        ".trace.json.gz files in %s",
                        ", ".join(str(p) for p in existing_empty_dirs),
                    )
                else:
                    log.warning(
                        "profile_executor: workspace=%s has no trace dir "
                        "(checked: %s)",
                        workspace_str,
                        ", ".join(str(p) for p in _candidate_trace_dirs(workspace)),
                    )
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_CONFIG",
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]
