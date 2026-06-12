# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared helper for the ``explore`` executor's grid runs.

Takes a base Magpie YAML + a list of (name, extra_server_args, extra_envs)
variants, runs Magpie once per variant, parses ``benchmark_report.json``,
returns the winners.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ._robustness_pulse import pulse as _robustness_pulse
from ._subprocess_kill import (
    OVERTIME_KILL_RETURNCODE,
    SERVER_DEAD_RETURNCODE,
    run_with_session_kill,
)
from .benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)


log = logging.getLogger(__name__)


# Content-based variant fingerprint (cross-action dedup ledger key). Hashes the
# content that changes Magpie behavior (server args + env overrides) so a rename
# maps to the same key in ``SharedState.explore_search.tested``. Normalization:
# args ``shlex.split`` → sorted tokens (order-insensitive); envs ``(str(k),
# str(v))`` sorted by key (so ``"1"`` and ``1`` collide); 16-char SHA-1 prefix.
def variant_fingerprint(
    extra_server_args: str | None,
    extra_envs: dict[str, Any] | None,
) -> str:
    """Stable content fingerprint for a (extra_server_args, extra_envs) pair.

    Name and note are NOT inputs — variants with identical content but
    different names collapse to the same fingerprint.
    """
    args_text = str(extra_server_args or "")
    try:
        args_tokens = sorted(shlex.split(args_text))
    except ValueError:
        # Shell-parse failure: whitespace split so we still produce a
        # fingerprint (identical bad strings still collide).
        args_tokens = sorted(args_text.split())
    env_pairs = sorted(
        (str(k), str(v)) for k, v in (extra_envs or {}).items()
    )
    payload = json.dumps(
        [args_tokens, [list(p) for p in env_pairs]],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_magpie_python() -> str:
    """Resolve the Python interpreter for Magpie subprocesses.

    Order: $MAGPIE_PYTHON (only when it can ``import Magpie``) > first PATH
    ``python3`` that can ``import Magpie`` > /opt/venv/bin/python (canonical
    Magpie venv) as unconditional last resort. A stale ``$MAGPIE_PYTHON``
    resolved before Magpie was pip-installed (e.g. ``/usr/bin/python3``) is
    validated and skipped to avoid ``ModuleNotFoundError`` at benchmark time.
    """
    def _can_import_magpie(py: str) -> bool:
        """Whether an interpreter can import Magpie and its ``yaml`` dep.

        Probes both ``Magpie`` and ``yaml`` so interpreters that resolve
        Magpie via a ``.pth`` but lack PyYAML are rejected.

        Args:
            py: Path to the candidate Python interpreter.

        Returns:
            ``True`` if both imports succeed in the interpreter.
        """
        # Probe Magpie AND its top-level runtime dep ``yaml``: an editable
        # install puts Magpie on sys.path via a .pth regardless of whether
        # the interpreter has PyYAML, so ``import Magpie`` alone can succeed
        # on an interpreter that then dies at Magpie startup with
        # ``ModuleNotFoundError: No module named 'yaml'`` (surfaced as
        # subprocess_nonzero / baseline_failed). Requiring yaml here makes
        # the resolver skip such interpreters and fall through to the
        # canonical /opt/venv that has the full dependency set.
        try:
            # ``run_with_session_kill`` captures stdout/stderr internally and
            # rejects ``capture_output`` (would raise TypeError).
            proc = run_with_session_kill(
                [py, "-c", "import Magpie, yaml"],
                timeout=10,
            )
            return getattr(proc, "returncode", 1) == 0
        except Exception:
            return False

    env_val = os.environ.get("MAGPIE_PYTHON", "").strip()
    if env_val:
        if _can_import_magpie(env_val):
            return env_val
        log.warning(
            "MAGPIE_PYTHON=%s cannot import Magpie; ignoring it and "
            "auto-detecting an interpreter that can. (A stale value is often "
            "baked into kernel-agent.env.sh when install.sh resolved it "
            "before Magpie was pip-installed.)",
            env_val,
        )

    candidate = shutil.which("python3")
    if candidate and _can_import_magpie(candidate):
        return candidate

    return "/opt/venv/bin/python"


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root.

    Reads :func:`inference_optimizer.paths.session_dir` (honors
    ``$USER_DATA_PATH``, else ``/workspace/hyperloom``). Used by fallback
    paths when ``ctx.extra["workspace"]`` was not pre-mkdir'd.
    """
    from ...paths import session_dir as _sd
    return _sd()


_MAGPIE_CWD_DEFAULT = "/tmp"
_VARIANT_TIMEOUT_SEC_DEFAULT = 7800  # 130 min; matches BASELINE_DEFAULT_TIMEOUT_SEC for Qwen3-32B TP=1 CONC=64 ISL/OSL=1024 NUM_PROMPTS=320 workload


# User-declared variant skip list: SKIP_VARIANTS is a comma/whitespace list of
# patterns matched (exact or fnmatch glob) against ``GridVariant.name``.
# Resolution order: params["skip_variants"] > $SKIP_VARIANTS > "". Name-based
# only; model/TP predicates live in each executor's filter.
import fnmatch as _fnmatch  # noqa: E402  (kept near callers for grep-ability)


def resolve_skip_spec(params: dict | None) -> str:
    """Resolve the active skip spec from task params + process env.

    ``params["skip_variants"]`` may be a list[str] or a single str; both are
    flattened to comma-joined form before pattern parsing. Resolution order
    is ``params["skip_variants"]`` > ``$SKIP_VARIANTS`` > ``""``.

    Args:
        params (dict | None): Task params; ``skip_variants`` (list/tuple/str)
            takes precedence over the environment when present and non-empty.

    Returns:
        str: The stripped skip spec string, or ``""`` when neither source
        supplies a value.
    """
    val = ""
    if params and "skip_variants" in params:
        raw = params.get("skip_variants")
        if isinstance(raw, (list, tuple)):
            val = ",".join(str(x) for x in raw if x is not None)
        elif raw is not None:
            val = str(raw)
    if not val.strip():
        val = os.environ.get("SKIP_VARIANTS", "")
    return (val or "").strip()


def _parse_skip_spec(spec: str) -> list[str]:
    """Split ``spec`` on commas and whitespace; drop empties.

    Newlines are treated as commas, then each comma-separated token is
    further split on whitespace so mixed separators all flatten into one
    list of patterns.

    Args:
        spec (str): Raw skip spec (e.g. ``"attn_*, sched_dfs\nvllm_aiter"``).

    Returns:
        list[str]: Non-empty, stripped pattern tokens in source order.
    """
    if not spec:
        return []
    out: list[str] = []
    for token in spec.replace("\n", ",").split(","):
        for sub in token.split():
            t = sub.strip()
            if t:
                out.append(t)
    return out


# Matches ``--cuda-graph-max-bs 64`` and ``--cuda_graph_max_bs=64``; captures
# the integer value.
_RE_CUDA_GRAPH_MAX_BS = re.compile(
    r"--cuda[-_]graph[-_]max[-_]bs[= ]+(\d+)"
)


def annotate_multi_node_cuda_graph_max_bs(
    grid: list["GridVariant"],
) -> list[dict]:
    """Return advisory notes for ``--cuda-graph-max-bs N < $CONC`` variants.

    These regress ~50% in multi-node mode (cuda graph cache misses every
    cross-node decode tick), but the variant is kept in the grid and surfaced
    as an advisory rather than auto-dropped. Returns ``[]`` outside multi-node
    mode, when ``$CONC`` is unset/non-positive, or when no variant matches.
    """
    from ._multi_node_env import is_multi_node
    if not is_multi_node():
        return []
    try:
        conc = int(os.environ.get("CONC", "64") or 64)
    except ValueError:
        conc = 64
    if conc <= 0:
        return []
    notes: list[dict] = []
    for v in grid:
        m = _RE_CUDA_GRAPH_MAX_BS.search(v.extra_server_args or "")
        if m and int(m.group(1)) < conc:
            notes.append({
                "name": v.name,
                "source": "multi_node_advisory",
                "reason": (
                    f"cuda_graph_max_bs={m.group(1)} < CONC={conc} "
                    "(multi-node graph-cache miss is a known regression; "
                    "advisory only, not auto-skipped)"
                ),
            })
    return notes


# Framework / hardware compatibility filter: drops variants whose flag literals
# are unsupported by the live framework + model class (real incompatibility,
# not a strategy gate). Each entry maps an ``extra_server_args`` substring to a
# required model class.
_COMPATIBILITY_FLAG_RULES: tuple[tuple[str, str], ...] = (
    ("--enable-flashinfer-mla", "mla"),
    ("--enable-deepep-moe",      "moe"),
    ("--enable-ep-moe",          "moe"),
)


# Per-framework cache for ``_probe_server_help_text`` (avoids a subprocess per
# variant). Empty results are NOT cached so a transient failure re-probes.
_HELP_TEXT_CACHE: dict[str, str] = {}

# Per-framework ``--help`` extraction commands. Each is a single-shot
# ``python3 -c <inline>`` so the probe's 10s timeout covers the import cost.
_HELP_PROBE_COMMANDS: dict[str, tuple[str, ...]] = {
    "sglang": (
        "python3", "-c",
        "from sglang.launch_server import parser; parser.print_help()",
    ),
    "vllm": (
        "python3", "-c",
        "from vllm.entrypoints.openai.api_server import make_arg_parser; "
        "make_arg_parser(None).print_help()",
    ),
    # atom exposes EngineArgs.add_cli_args on ``atom.model_engine.arg_utils``
    # (mirrors vLLM); populate a throwaway parser and print its help.
    "atom": (
        "python3", "-c",
        "import argparse; from atom.model_engine.arg_utils import EngineArgs; "
        "p = argparse.ArgumentParser(); EngineArgs.add_cli_args(p); "
        "p.print_help()",
    ),
}


def _probe_server_help_text(framework: str) -> str:
    """Best-effort fetch of ``<framework> --help`` text for flag validation.

    Supported: ``sglang``, ``vllm``, ``atom``; unknown values return ``""``.
    Returns ``""`` on ANY failure — callers MUST treat empty as "unknown" and
    fall through to NOT filtering. Empty results are NOT cached. The broad
    ``except`` is deliberate: this probe is a perf optimisation only and must
    never crash the optimizer.
    """
    fw = (framework or "").strip().lower()
    if fw in _HELP_TEXT_CACHE:
        return _HELP_TEXT_CACHE[fw]
    cmd = _HELP_PROBE_COMMANDS.get(fw)
    if cmd is None:
        return ""
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True, text=True, timeout=10,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        out = ""
    if out:
        _HELP_TEXT_CACHE[fw] = out
    return out


def _probe_sglang_help_text() -> str:
    """Back-compat shim — defer to the framework-keyed probe.

    Kept so tests that monkey-patch this exact name still work; new call
    sites should use ``_probe_server_help_text("sglang")``.
    """
    return _probe_server_help_text("sglang")


def _detect_model_class(model_path: str) -> tuple[bool, bool]:
    """Heuristic detect of (is_mla_model, is_moe_model) from model path.

    Lowercased substring match — a cheap check to skip an obviously-wrong
    variant before a 10-min doomed sglang restart. Misclassifications cost at
    most one restart. MLA: DeepSeek (V2/V3/R1), GLM-5, Kimi-K2; MoE: MLA set +
    Qwen3-MoE.
    """
    p = model_path.lower()
    mla_keys = ("glm-5", "glm5", "deepseek", "kimi-k2", "kimi_k2", "kimi")
    moe_keys = (
        "glm-5", "glm5", "deepseek-v2", "deepseek-v3", "deepseek-r1",
        "kimi", "qwen3-moe", "qwen3_moe", "mixtral",
    )
    is_mla = any(k in p for k in mla_keys)
    is_moe = any(k in p for k in moe_keys)
    return is_mla, is_moe


def apply_compatibility_filter(
    grid: list["GridVariant"],
) -> tuple[list["GridVariant"], list[dict]]:
    """Skip variants known to be incompatible with current model/sglang.

    Two dimensions, each conservative (assume compatible) on probe failure:
    model class (MLA / MoE flags dropped when ``$MODEL_PATH`` lacks the family
    keyword), and sglang version (flags absent from ``launch_server --help``
    dropped). Returns the ``(kept, dropped)`` shape of ``apply_user_skip_list``.
    """
    model_path = os.environ.get("MODEL_PATH", "")
    if model_path:
        is_mla, is_moe = _detect_model_class(model_path)
    else:
        # No MODEL_PATH set -> can't detect -> assume compatible.
        is_mla, is_moe = True, True

    # Live framework's --help text; defaults to sglang for fixtures/callers
    # that don't thread ``benchmark.framework``.
    fw = (os.environ.get("FRAMEWORK", "") or "sglang").strip().lower()
    help_text = _probe_server_help_text(fw)
    help_available = bool(help_text)

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        args = v.extra_server_args or ""
        skip_reason: str | None = None
        for flag, required_class in _COMPATIBILITY_FLAG_RULES:
            if flag not in args:
                continue
            # Model-class predicate
            class_ok = (
                (required_class == "mla" and is_mla)
                or (required_class == "moe" and is_moe)
            )
            if not class_ok:
                skip_reason = (
                    f"{flag} requires {required_class.upper()} model; "
                    f"MODEL_PATH={model_path!r} not recognised as "
                    f"{required_class.upper()}-class"
                )
                break
            # Framework flag-support predicate (only when help is readable).
            if help_available and flag not in help_text:
                skip_reason = (
                    f"{flag} not present in `{fw} --help` output; "
                    f"current {fw} version likely too old"
                )
                break
        if skip_reason:
            dropped.append({
                "name": v.name,
                "source": "compatibility_filter",
                "reason": skip_reason,
            })
        else:
            kept.append(v)
    return kept, dropped


def apply_user_skip_list(
    grid: list["GridVariant"],
    *,
    skip_spec: str,
) -> tuple[list["GridVariant"], list[dict]]:
    """Drop variants whose name matches any pattern in ``skip_spec``.

    Returns ``(kept, dropped)`` where each dropped entry is
    ``{"name", "reason", "source"}`` with source=``"user_skip"``.
    """
    patterns = _parse_skip_spec(skip_spec)
    if not patterns:
        return list(grid), []

    kept: list[GridVariant] = []
    dropped: list[dict] = []
    for v in grid:
        matched_pat: str | None = None
        for pat in patterns:
            # Exact name first (cheaper), then fnmatch for globs.
            if pat == v.name or _fnmatch.fnmatchcase(v.name, pat):
                matched_pat = pat
                break
        if matched_pat is None:
            kept.append(v)
            continue
        dropped.append({
            "name": v.name,
            "source": "user_skip",
            "reason": f"matched SKIP_VARIANTS pattern '{matched_pat}'",
        })
    return kept, dropped


@dataclass(init=False)
class GridVariant:
    """One row of the grid we're going to test.

    Describes a single server-config candidate: the flags/env overrides to
    apply on top of the base Magpie config for one benchmark run.

    Attributes:
        name (str): Human-readable label for the variant.
        extra_server_args (str): Backend server args appended via
            ``EXTRA_{SGLANG,VLLM,ATOM}_ARGS``. Defaults to ``""``.
        extra_envs (dict[str, str]): Per-variant environment overrides.
            Defaults to an empty dict.
        note (str): Optional reason/category tag (e.g. ``multi_node_only_*``).
            Defaults to ``""``.
    """

    name: str                                    # human-readable label
    extra_server_args: str = ""                  # appended via EXTRA_{SGLANG,VLLM,ATOM}_ARGS env
    extra_envs: dict[str, str] = field(default_factory=dict)
    note: str = ""                                # optional reason / category

    def __init__(
        self,
        name: str,
        extra_server_args: str = "",
        extra_envs: dict[str, str] | None = None,
        note: str = "",
        *,
        extra_sglang_args: str | None = None,
    ) -> None:
        """Initialize a grid variant descriptor.

        Args:
            name: Variant name.
            extra_server_args: Extra server CLI args for this variant.
            extra_envs: Extra environment variables for this variant.
            note: Optional reason/category note.
            extra_sglang_args: Deprecated alias for ``extra_server_args``;
                routed into the canonical attribute with a warning.
        """
        # Back-compat alias for the historical ``extra_sglang_args`` kwarg;
        # routed into the canonical attribute with a DeprecationWarning.
        if extra_sglang_args is not None:
            import warnings as _warnings
            _warnings.warn(
                "GridVariant(extra_sglang_args=...) is a deprecation "
                "alias for GridVariant(extra_server_args=...) and will "
                "be removed in the next Hyperloom release.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not extra_server_args:
                extra_server_args = extra_sglang_args
        self.name = name
        self.extra_server_args = extra_server_args
        self.extra_envs = dict(extra_envs) if extra_envs is not None else {}
        self.note = note

    @property
    def fingerprint(self) -> str:
        """Content fingerprint used as dedup-ledger key. See module doc.

        Returns:
            str: :func:`variant_fingerprint` of this variant's
            ``extra_server_args`` and ``extra_envs``.
        """
        return variant_fingerprint(self.extra_server_args, self.extra_envs)


def coerce_extra_envs(value: Any) -> dict[str, str]:
    """Normalize Orchestration-supplied ``extra_envs`` to ``dict[str,str]``.

    Accepts the three shapes the LLM emits — canonical dict, shell-style
    ``"FOO=1 BAR=2"`` string, and ``["FOO=1", "BAR=2"]`` token list — so
    downstream ``.items()`` callers never crash on a non-dict. Unknown shapes
    coerce to an empty dict.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if k is not None}
    if isinstance(value, str):
        out: dict[str, str] = {}
        # Split on the first ``=`` only to preserve URL-style assignments
        # like ``HF_ENDPOINT=https://...``.
        tokens = re.split(r"[\s;]+", value.strip())
        for tok in tokens:
            if not tok:
                continue
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out[k] = v.strip()
        return out
    if isinstance(value, (list, tuple)):
        out_l: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                # ``[{"FOO": "1"}, {"BAR": "2"}]`` — later entries win.
                for k, v in item.items():
                    if k is None:
                        continue
                    out_l[str(k)] = str(v)
                continue
            if not isinstance(item, str) or "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out_l[k] = v.strip()
        return out_l
    return {}


@dataclass
class VariantResult:
    """One bench run's parsed result.

    Captures the parsed outcome of a single variant's Magpie run: identity,
    status, the headline throughput/latency metrics, artifact paths, and
    failure-classification metadata.

    Attributes:
        name (str): Variant label (mirrors :attr:`GridVariant.name`).
        extra_server_args (str): Server args used for this run.
        extra_envs (dict[str, str]): Env overrides used for this run.
        status (str): ``"succeeded"`` or ``"failed"``.
        output_throughput (float | None): Output tokens/sec, if measured.
        request_throughput (float | None): Requests/sec, if measured.
        total_token_throughput (float | None): Total tokens/sec, if measured.
        completed_requests (int | None): Number of completed requests.
        duration_seconds (float | None): Benchmark duration in seconds.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms).
        e2el_mean_ms (float | None): Mean end-to-end latency (ms).
        workspace (str | None): Path to the located ``benchmark_*`` workspace.
        report_path (str | None): Path to ``benchmark_report.json`` if present.
        raw_result_path (str | None): Path to the raw result JSON, if salvaged.
        reported_success (bool | None): Magpie's own success flag, if known.
        returncode (int | None): Magpie subprocess return code.
        nonfatal_warnings (list[str]): Non-fatal warning tags (e.g. harvested
            leaked artifacts).
        error (str | None): Error summary for failed/nonzero runs.
        error_class (str): Short failure-classification tag (empty on success).
        note (str): Optional reason/category tag carried from the variant.
        runtime_sec (float | None): Wall-clock seconds the subprocess consumed.
        killed_overtime (bool): ``True`` iff reaped by the soft overtime
            deadline rather than crashing/timing-out/succeeding.
    """

    name: str
    extra_server_args: str
    extra_envs: dict[str, str]
    status: str
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    tpot_mean_ms: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Short failure-classification tag matching ``_write_variant_abort_marker``
    # (e.g. ``magpie_timeout``, ``yaml_build_error``); empty for successes.
    # Surfaced in the LLM critic prompt as ``failed_variants[*].error_class``.
    error_class: str = ""
    note: str = ""
    # Fix-E: wall-clock seconds the Magpie subprocess consumed; populated on
    # success AND on the ``killed_overtime`` path.
    runtime_sec: float | None = None
    # Fix-E: True iff reaped by the overtime soft deadline; caller demotes to
    # the synthetic ``KILLED_OVERTIME`` outcome (no tput / fingerprint).
    killed_overtime: bool = False

    @property
    def fingerprint(self) -> str:
        """Same fingerprint scheme as :class:`GridVariant`.

        Returns:
            str: :func:`variant_fingerprint` of this result's
            ``extra_server_args`` and ``extra_envs``.
        """
        return variant_fingerprint(self.extra_server_args, self.extra_envs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain JSON-friendly dict.

        Returns:
            dict[str, Any]: All result fields plus the computed
            ``fingerprint``, keyed by attribute name.
        """
        return {
            "name":               self.name,
            "extra_server_args":  self.extra_server_args,
            "extra_envs":         self.extra_envs,
            "fingerprint":        self.fingerprint,
            "status":             self.status,
            "output_throughput":  self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds":   self.duration_seconds,
            "ttft_mean_ms":       self.ttft_mean_ms,
            "e2el_mean_ms":       self.e2el_mean_ms,
            "tpot_mean_ms":       self.tpot_mean_ms,
            "workspace":          self.workspace,
            "report_path":        self.report_path,
            "raw_result_path":    self.raw_result_path,
            "reported_success":   self.reported_success,
            "returncode":         self.returncode,
            "nonfatal_warnings":  self.nonfatal_warnings,
            "error":              self.error,
            "error_class":        self.error_class,
            "note":               self.note,
            "runtime_sec":        self.runtime_sec,
            "killed_overtime":    self.killed_overtime,
        }


# Shared sanitization for Orchestration-supplied overrides (benchmark_script /
# result_dir). Both originate from LLM proposals so we reject path separators /
# shell metacharacters; the helpers raise ``ValueError`` (Coordinator surfaces
# ``error_class=bad_param``) instead of running an unsafe subprocess.
_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.sh$")
_RESULT_DIR_FORBID_RE = re.compile(r"[\s\"'`$;&|<>(){}\[\]\\*?!]")


def sanitize_script_name(value: Any) -> str | None:
    """Return ``value`` if it's a safe Magpie benchmark script file name.

    Must be a bare ``*.sh`` name (no slashes / ``..``). Empty/``None`` →
    ``None``; anything resembling shell injection raises ``ValueError``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SCRIPT_NAME_RE.match(text):
        raise ValueError(
            f"benchmark_script={text!r} rejected: must be a bare *.sh "
            "file name (no path separators, no shell metacharacters)"
        )
    return text


def sanitize_result_dir(value: Any) -> str | None:
    """Return ``value`` if it's a safe absolute (or workspace-relative) dir.

    Lands in a shell ``cd`` / ``mkdir`` via ``$RESULT_DIR``, so reject any
    character that could escape into a different shell word. Empty/``None`` →
    ``None``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _RESULT_DIR_FORBID_RE.search(text):
        raise ValueError(
            f"result_dir={text!r} rejected: contains whitespace or shell "
            "metacharacters; pass an absolute or workspace-relative path"
        )
    return text


def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"EXTRA_ATOM_ARGS"`` for atom, ``"EXTRA_VLLM_ARGS"`` for vLLM,
        otherwise ``"EXTRA_SGLANG_ARGS"`` (the sglang default).
    """
    name = str(framework or "").strip().lower()
    # atom checked first so a future overlapping-substring framework name
    # cannot match the wrong branch.
    if "atom" in name:
        return "EXTRA_ATOM_ARGS"
    if "vllm" in name:
        return "EXTRA_VLLM_ARGS"
    return "EXTRA_SGLANG_ARGS"


def merge_server_args(*parts: str | None) -> str:
    """Merge server arg strings preserving left-to-right override semantics.

    Only removes empty chunks; does NOT de-duplicate option names, because
    repeated flags are how later args override base args (e.g. ``--block-size
    1`` then ``--block-size 256``).
    """
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


# sglang scheduler watchdog timeout injection: on MI300X with aiter, the first
# request's ``mha_batch_prefill`` JIT compile can exceed sglang's 300s default
# watchdog, firing SIGQUIT mid-warmup -> baseline_failed. Inject a longer
# timeout via ``EXTRA_SGLANG_ARGS`` unless the user already pinned one.
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC = 1800
SGLANG_WATCHDOG_TIMEOUT_ENV = "SGLANG_WATCHDOG_TIMEOUT"
_SGLANG_WATCHDOG_FLAG = "--watchdog-timeout"
# Matches space- or equals-separated form so a user-pinned value suppresses
# injection without false-matching a longer flag.
_SGLANG_WATCHDOG_RE = re.compile(r"--watchdog-timeout(?:[=\s]|$)")


def resolve_sglang_watchdog_timeout() -> int:
    """Resolve the sglang scheduler watchdog timeout in seconds.

    Reads ``$SGLANG_WATCHDOG_TIMEOUT`` (integer seconds) and falls back to
    :data:`DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC` when the env var is unset,
    empty, non-integer, or non-positive. A malformed value logs a warning
    and uses the default rather than crashing the YAML materialization.
    """
    raw = os.environ.get(SGLANG_WATCHDOG_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not an integer; using default %ds.",
            SGLANG_WATCHDOG_TIMEOUT_ENV, raw,
            DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
        )
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    if val <= 0:
        log.warning(
            "%s=%d is not positive; using default %ds.",
            SGLANG_WATCHDOG_TIMEOUT_ENV, val,
            DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
        )
        return DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC
    return val


def inject_sglang_watchdog_timeout(
    server_args: str | None, framework: str | None,
) -> str:
    """Append ``--watchdog-timeout <N>`` to ``server_args`` for sglang runs.

    Returns ``server_args`` unchanged when the framework is not sglang
    (empty/unknown is treated as sglang) or the flag is already present.
    Otherwise appends the value from :func:`resolve_sglang_watchdog_timeout`;
    no other flag is touched.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_WATCHDOG_RE.search(args):
        return args
    timeout = resolve_sglang_watchdog_timeout()
    return merge_server_args(args, f"{_SGLANG_WATCHDOG_FLAG} {timeout}")


# sglang ``--context-length`` cap injection: sglang sizes ``max_total_tokens``
# off the model's ``max_position_embeddings``, so a huge native window (e.g.
# Mistral-Nemo's 1024000) balloons the aiter workspace_buffer past GPU memory
# -> HIP OOM -> baseline_failed. vllm already caps via ``--max-model-len``, so
# this fixes the sglang-only asymmetry: cap to ISL+OSL+headroom (floored,
# clamped to the native window) unless the flag is already pinned.
DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS = 2048
DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS = 8192
SGLANG_CONTEXT_HEADROOM_ENV = "SGLANG_CONTEXT_HEADROOM_TOKENS"
SGLANG_CONTEXT_FLOOR_ENV = "SGLANG_CONTEXT_FLOOR_TOKENS"
_SGLANG_CONTEXT_LENGTH_FLAG = "--context-length"
# Matches space- or equals-separated form so an operator-pinned value
# suppresses injection without false-matching a longer flag.
_SGLANG_CONTEXT_LENGTH_RE = re.compile(r"--context-length(?:[=\s]|$)")
_SGLANG_ATTN_BACKEND_FLAG = "--attention-backend"
_SGLANG_ATTN_BACKEND_RE = re.compile(r"--attention-backend(?:[=\s]|$)")
_SGLANG_DUAL_CHUNK_BACKEND = "dual_chunk_flash_attn"


def _resolve_nonneg_int_env(name: str, default: int) -> int:
    """Read a non-negative integer env override, else return ``default``.

    A blank/non-integer/negative value logs a warning and falls back to the
    default rather than crashing the YAML materialization.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not an integer; using default %d.", name, raw, default,
        )
        return default
    if val < 0:
        log.warning(
            "%s=%d is negative; using default %d.", name, val, default,
        )
        return default
    return val


def resolve_sglang_context_cap(isl: int, osl: int) -> int:
    """Resolve the sglang ``--context-length`` cap for an ISL+OSL workload.

    Returns ``max(isl + osl + headroom, floor)`` (headroom / floor are
    operator-tunable via ``$SGLANG_CONTEXT_HEADROOM_TOKENS`` /
    ``$SGLANG_CONTEXT_FLOOR_TOKENS``). Caller clamps to the model's native
    window before injecting.
    """
    headroom = _resolve_nonneg_int_env(
        SGLANG_CONTEXT_HEADROOM_ENV, DEFAULT_SGLANG_CONTEXT_HEADROOM_TOKENS,
    )
    floor = _resolve_nonneg_int_env(
        SGLANG_CONTEXT_FLOOR_ENV, DEFAULT_SGLANG_CONTEXT_FLOOR_TOKENS,
    )
    return max(int(isl) + int(osl) + headroom, floor)


def inject_sglang_context_length(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    isl: int,
    osl: int,
) -> str:
    """Append ``--context-length <N>`` to ``server_args`` for sglang runs.

    Returns ``server_args`` unchanged when the framework is not sglang
    (empty/unknown treated as sglang), the flag is already present, or the
    model's ``max_position_embeddings`` cannot be read. Otherwise appends
    ``min(max_pos, cap)`` from :func:`resolve_sglang_context_cap`; only this
    flag is added.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_CONTEXT_LENGTH_RE.search(args):
        return args
    # Lazy import to avoid a module-level cycle through the heavy cli.py.
    from ...cli import _load_model_max_position_embeddings
    max_pos = _load_model_max_position_embeddings(str(model_path or ""))
    if not max_pos:
        return args
    cap = resolve_sglang_context_cap(isl, osl)
    context_length = min(int(max_pos), cap)
    return merge_server_args(
        args, f"{_SGLANG_CONTEXT_LENGTH_FLAG} {context_length}",
    )


def _resolve_dual_chunk_backend(gpu_type: str | None = None) -> str:
    """Pick the dual-chunk attention backend for the current hardware.

    ``dual_chunk_flash_attn`` is the only backend sglang accepts when the
    model declares ``dual_chunk_attention_config``. It requires sm90+
    (NVIDIA Hopper); on AMD/ROCm the preflight gate
    (``_detect_incompatible_model_config``) blocks these models before
    they reach this point. If a session somehow arrives here on AMD
    (e.g. operator override), return the canonical backend and let sglang
    raise the clear error rather than silently injecting triton which
    sglang also rejects.  Override via ``$HYPERLOOM_DUAL_CHUNK_BACKEND``.
    """
    override = os.environ.get("HYPERLOOM_DUAL_CHUNK_BACKEND", "").strip()
    if override:
        return override
    return _SGLANG_DUAL_CHUNK_BACKEND


def inject_sglang_attention_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    gpu_type: str | None = None,
) -> str:
    """Append an ``--attention-backend`` for dual-chunk sglang models.

    Models that declare ``dual_chunk_attention_config`` (Qwen 1M) make
    sglang hard-reject its default aiter backend with ``ValueError: Dual
    chunk attention is enabled, but attention backend is set to aiter.``.
    On NVIDIA sm90+ the fix is ``dual_chunk_flash_attn``; on AMD/ROCm that
    kernel is unsupported (``sm90 and above``), so we inject ``triton``
    instead (see :func:`_resolve_dual_chunk_backend`). ``gpu_type`` (when
    known by the caller) takes precedence over runtime autodetect.

    Returns ``server_args`` unchanged when: framework is not sglang, an
    ``--attention-backend`` is already pinned (operator wins), or the model
    config has no dual-chunk block (fail-safe: inject nothing).
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_ATTN_BACKEND_RE.search(args):
        return args
    from ...cli import _model_has_dual_chunk_attention
    if not _model_has_dual_chunk_attention(str(model_path or "")):
        return args
    backend = _resolve_dual_chunk_backend(gpu_type)
    if backend != _SGLANG_DUAL_CHUNK_BACKEND:
        log.info(
            "dual-chunk model on AMD/ROCm: injecting "
            "--attention-backend %s (dual_chunk_flash_attn needs sm90+).",
            backend,
        )
    return merge_server_args(
        args, f"{_SGLANG_ATTN_BACKEND_FLAG} {backend}",
    )


# sglang MoE runner backend injection: on MI300X/MI355X with aiter, sglang's
# default ``--moe-runner-backend auto`` routes Mixture-of-Experts models
# through aiter's CK 2-stage fused-MoE kernel. Its first-request JIT build
# (``module_moe_ck2stages_*``) is broken in some ROCm images — ``thrust`` pulls
# in a missing ``<cub/detail/detect_cuda_runtime.cuh>`` so hipcc fails to
# compile, and the killed build leaves a stale lock that makes the next
# attempts hang on "waiting for baton release" until sglang's 600s warmup
# read-timeout fires -> baseline_failed. ``triton`` is the ROCm-capable
# fused-MoE backend sglang itself falls back to (same "aiter CK kernel doesn't
# support all GEMM dimensions" reason), so inject it for MoE models on AMD
# unless the operator already pinned a backend. Override via
# ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND``.
HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV = "HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND"
DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND = "triton"
_SGLANG_MOE_RUNNER_BACKEND_FLAG = "--moe-runner-backend"
# Matches space- or equals-separated form so a user-pinned value suppresses
# injection without false-matching a longer flag.
_SGLANG_MOE_RUNNER_BACKEND_RE = re.compile(r"--moe-runner-backend(?:[=\s]|$)")


def inject_sglang_moe_runner_backend(
    server_args: str | None,
    framework: str | None,
    model_path: str | None,
    gpu_type: str | None = None,
) -> str:
    """Append a ``--moe-runner-backend`` for MoE sglang models on AMD/ROCm.

    Returns ``server_args`` unchanged when: framework is not sglang, a
    ``--moe-runner-backend`` is already pinned (operator wins), the GPU is not
    an AMD/ROCm runner, or the model is not Mixture-of-Experts (fail-safe:
    inject nothing). Otherwise appends the backend from
    ``$HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND`` (default ``triton``); only this
    flag is added.
    """
    args = str(server_args or "").strip()
    if server_args_env_name(framework) != "EXTRA_SGLANG_ARGS":
        return args
    if _SGLANG_MOE_RUNNER_BACKEND_RE.search(args):
        return args
    from ...cli import _model_is_moe, _resolve_amd_gpu_type
    if not _resolve_amd_gpu_type(gpu_type):
        return args
    if not _model_is_moe(str(model_path or "")):
        return args
    backend = (
        os.environ.get(HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV, "").strip()
        or DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND
    )
    log.info(
        "MoE model on AMD/ROCm: injecting --moe-runner-backend %s (aiter CK "
        "2-stage fused-MoE JIT build is broken in this image).", backend,
    )
    return merge_server_args(
        args, f"{_SGLANG_MOE_RUNNER_BACKEND_FLAG} {backend}",
    )


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a Magpie benchmark YAML.

    Single shared path for baseline/profile and grid executors so
    backends/params/sweep no longer fall back to shipped YAML defaults.
    ``benchmark_script`` (must be pre-sanitized via :func:`sanitize_script_name`)
    force-selects a specific Magpie script, applied AFTER the
    ``gpu_type``-derived generic script so the operator pick wins.
    """
    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        # Force-pin the generic ``{framework}_{gpu_type}.sh`` so Magpie's
        # resolver doesn't fall through to InferenceX native scripts that
        # hardcode ``--result-dir /workspace/`` and ignore ``EXTRA_*_ARGS``.
        # See ``design/magpie-generic-script-and-user-data-path.md``.
        framework = str(bench.get("framework") or "").lower()
        if framework:
            bench["benchmark_script"] = f"{framework}_{gpu_type}.sh"
        else:
            bench.pop("benchmark_script", None)

    if benchmark_script:
        bench["benchmark_script"] = str(benchmark_script)

    envs = bench.setdefault("envs", {})
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if not val:
            continue
        # TP yaml-explicit wins: a stale state.tp re-exported on resume must
        # not downgrade a YAML-pinned TP.
        if env_key == "TP":
            yaml_tp = envs.get("TP")
            if yaml_tp not in (None, 0, "", "0"):
                continue
        envs[env_key] = int(val)

    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_val = int(envs.get("TP", 1) or 1)
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = (
            len([x for x in existing_rocr.split(",") if x.strip()])
            if existing_rocr else 0
        )
        if tp_val > 1 and existing_count < tp_val:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_val))

    return envs


def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk.

    Injects the variant's flags via ``EXTRA_SGLANG_ARGS``. ``model_path``
    overrides the legacy hardcoded ``benchmark.model``; ``gpu_type`` pins the
    generic ``{framework}_{gpu_type}.sh``; ``benchmark_script`` (pre-sanitized)
    force-pins a script, applied last so the operator pick wins.
    """
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = apply_runtime_benchmark_overrides(
        bench, model_path=model_path, gpu_type=gpu_type,
        benchmark_script=benchmark_script,
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    combined = merge_server_args(
        str(envs.get(extra_args_env, "")),
        base_extra_args,
        variant.extra_server_args,
    )
    if combined:
        from ..coordinator_helpers import _dedupe_extra_server_args
        envs[extra_args_env] = _dedupe_extra_server_args(combined)
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)

    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    """Load ``benchmark_report.json`` from a benchmark workspace.

    Args:
        workspace (Path): Directory expected to contain
            ``benchmark_report.json``.

    Returns:
        dict[str, Any] | None: The parsed report dict, or ``None`` if the
        file is missing, unreadable, invalid JSON, or not a JSON object.
    """
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _kill_stale_servers() -> None:
    """Deep-clean any lingering inference server processes + shared memory.

    Magpie's server_cleanup.sh only reaps the pgrp leader, but vLLM::Worker /
    EngineCore children escape it; without this pre-clean the next startup
    hangs ~5 min on zmq / shared-mem conflicts. Called before every Magpie
    invocation. Uses /proc scan (not pgrep) to avoid clashing with test
    subprocess mocks. No-op in multi-node mode (servers live in RayJob pods).
    """
    from ._multi_node_env import is_multi_node
    if is_multi_node():
        return

    import signal
    import glob
    import time

    _KILL_PATTERNS = ("VLLM::Worker", "VLLM::EngineCore", "vllm.entrypoints",
                      "vllm serve", "sglang.srt", "sglang.launch_server",
                      "atom.entrypoints", "atom.entrypoints.openai_server")

    # atom's ModelRunner workers spawn via ``multiprocessing.spawn`` (generic
    # ``spawn_main ... --multiprocessing-fork`` cmdline, unmatchable by
    # _KILL_PATTERNS) and orphan to init holding their full HIP/VRAM, OOM-ing
    # the next atom server. Identify survivors by the atom / aiter JIT mmaps in
    # their address space — a signature our own children never carry.
    _FORK_MARKERS = (b"--multiprocessing-fork", b"spawn_main")
    _ATOM_MAP_SIGNATURES = ("/ATOM/atom/", "/aiter/jit/", "/aiter-test/aiter/")

    my_pid = os.getpid()
    try:
        my_pgid = os.getpgrp()
    except OSError:
        my_pgid = -1

    def _is_orphaned_atom_worker(pid: int, cmdline: bytes) -> bool:
        """Detect an orphaned atom ModelRunner worker by its memory maps.

        A spawned atom worker has a generic ``--multiprocessing-fork`` cmdline,
        so it is identified instead by atom/aiter signatures mmap'd into its
        address space. Workers belonging to this process group are excluded.

        Args:
            pid (int): Candidate process id.
            cmdline (bytes): The process's raw ``/proc/<pid>/cmdline``.

        Returns:
            bool: ``True`` iff ``cmdline`` carries a fork marker, the process
            is outside our process group, and its ``/proc/<pid>/maps`` shows
            an atom/aiter signature; ``False`` otherwise (including on any
            read/permission error).
        """
        if not any(m in cmdline for m in _FORK_MARKERS):
            return False
        # Never touch a worker that belongs to *our* process group.
        try:
            if my_pgid != -1 and os.getpgid(pid) == my_pgid:
                return False
        except (OSError, ProcessLookupError):
            return False
        try:
            with open(f"/proc/{pid}/maps", "r", errors="replace") as fh:
                maps = fh.read()
        except (OSError, PermissionError):
            return False
        return any(sig in maps for sig in _ATOM_MAP_SIGNATURES)

    killed_atom = False
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read()
        except (OSError, PermissionError):
            continue
        text = cmdline.replace(b"\0", b" ").decode("utf-8", "replace")
        is_atom_server = "atom.entrypoints" in text
        if any(pat in text for pat in _KILL_PATTERNS) or _is_orphaned_atom_worker(pid, cmdline):
            killed_atom = killed_atom or is_atom_server or b"--multiprocessing-fork" in cmdline
            # Kill the whole pgrp — atom ModelRunner children must die with the
            # leader.
            try:
                pgid = os.getpgid(pid)
                if pgid not in (my_pgid, 0):
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                # Group already gone or not ours to signal; fall through to per-pid kill.
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                # Process already exited or owned by another user; nothing to kill.
                pass

    # Clear /dev/shm segments that prevent re-binding.
    for pattern in ("/dev/shm/vllm*", "/dev/shm/nccl*", "/dev/shm/cuda*",
                    "/dev/shm/torch*", "/dev/shm/atom*"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                # Segment already removed or held by another process; safe to skip.
                pass

    # Pause for KFD async VRAM release; atom workers' teardown lags past 2s.
    time.sleep(8 if killed_atom else 2)


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    timeout_sec: int,
    cwd: str,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr).

    ``result_dir`` (pre-sanitized via :func:`sanitize_result_dir`) overrides
    ``$RESULT_DIR``, which is always set (default ``output_dir``) so results
    land in the per-task workspace, not ``/workspace/``. ``soft_deadline_sec``
    is the Fix-E overtime cap: the tree is reaped and a sentinel
    ``OVERTIME_KILL_RETURNCODE`` returned instead of raising ``TimeoutExpired``.
    """
    # Pre-clean lingering servers + shared memory (skip under pytest).
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_DIR", "")
    if magpie_dir:
        env["PYTHONPATH"] = f"{magpie_dir}:{env.get('PYTHONPATH', '')}"

    # Multi-node: tell Magpie to skip its local-server launch and point
    # benchmark_serving at the head pod's ClusterIP ({} in single-node).
    from ._multi_node_env import magpie_remote_env
    env.update(magpie_remote_env())

    # #210: pin Magpie's InferenceX resolution to ``$INFERENCEX_PATH`` so it
    # loads the SAME checkout ``_inferencex_patcher`` patched, not a stale
    # ``./InferenceX`` / cache copy. ``MAGPIE_INFERENCEX_PATH`` is Magpie's
    # highest-precedence resolution rung.
    inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
    if inferencex_path:
        env["MAGPIE_INFERENCEX_PATH"] = inferencex_path
    # Always-on RESULT_DIR default; scripts that ignore it still leak and are
    # picked up by the ``extract_benchmark_measurement`` salvage path.
    env["RESULT_DIR"] = result_dir or str(output_dir)
    # Pin SERVER_LOG / GPU_METRICS_CSV per-task so logs land alongside
    # ``benchmark_report.json`` instead of leaking to ``/workspace/``. Always
    # overwrite so a stale parent value can't redirect into a prior run's slot;
    # ``harvest_leaked_artifacts`` covers wrappers that ignore these vars.
    env["SERVER_LOG"] = str(output_dir / "server.log")
    env["GPU_METRICS_CSV"] = str(output_dir / "gpu_metrics.csv")
    cmd = [
        magpie_python, "-m", "Magpie", "-v", "benchmark",
        "--benchmark-config", str(config_path),
        "--output-dir", str(output_dir),
        "--run-mode", "local",
    ]
    # run_with_session_kill launches Magpie in its own POSIX session and tears
    # down the whole descendant tree on every exit path (bugs.md §B). See
    # ``_subprocess_kill.py``.
    proc = run_with_session_kill(
        cmd, env=env, cwd=cwd, timeout=timeout_sec,
        soft_deadline_sec=soft_deadline_sec,
        server_log_path=str(output_dir / "server.log"),
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


async def run_grid(
    *,
    base_yaml_path: Path,
    base_extra_args: str,
    grid: list[GridVariant],
    output_root: Path,
    magpie_python: str | None = None,
    cwd: str = _MAGPIE_CWD_DEFAULT,
    variant_timeout_sec: int = _VARIANT_TIMEOUT_SEC_DEFAULT,
    keep_going_on_failure: bool = True,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    result_dir: str | None = None,
    soft_deadline_sec: float | None = None,
) -> list[VariantResult]:
    """Execute every variant in ``grid`` once, in order.

    Returns the per-variant :class:`VariantResult` list (all attempts); the
    caller picks winners. Subprocess calls run in ``asyncio.to_thread`` so the
    Coordinator reactor isn't blocked.

    ``model_path`` / ``gpu_type`` are forwarded to every variant's YAML render.
    ``benchmark_script`` / ``result_dir`` (pre-sanitized) route around scripts
    that hardcode ``--result-dir /workspace/`` (see SKILL.md "Magpie leak-path
    salvage"). ``soft_deadline_sec`` (Fix E): reap a variant once wall-clock
    exceeds it, marking it ``killed_overtime=True``; None/0 disables (legacy).
    """
    if not magpie_python:
        magpie_python = _resolve_magpie_python()
    results: list[VariantResult] = []
    # Variant-boundary robustness pulse: a bounded deterministic tick after
    # every variant so a mid-grid leak/crash surfaces between variants instead
    # of after the whole grid. Best-effort; see ``_robustness_pulse.py``.
    async def _pulse_after_variant(idx: int) -> None:
        """Run a best-effort robustness pulse after a variant completes.

        Exceptions from the pulse are swallowed (logged at debug) so a pulse
        failure never aborts the grid.

        Args:
            idx (int): Zero-based index of the just-finished variant, passed
                through as the pulse ``tick_index``.
        """
        try:
            await _robustness_pulse(tick_index=idx)
        except Exception as exc:  # noqa: BLE001
            log.debug("robustness pulse swallowed: %r", exc)

    for i, variant in enumerate(grid):
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        try:
            cfg_path = _build_variant_yaml(
                base_yaml_path, base_extra_args, variant, output_subdir=slot,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: yaml_build_error: %r",
                i + 1, len(grid), variant.name, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="yaml_build_error",
                error_summary=repr(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"yaml_build_error: {exc!r}",
                error_class="yaml_build_error",
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        from ._multi_node_env import log_mn_banner
        log_mn_banner(
            "grid_runner", log,
            variant=f"{i+1}/{len(grid)}:{variant.name}",
        )
        log.info(
            "grid_runner: variant %d/%d name=%s args=%s",
            i + 1, len(grid), variant.name, variant.extra_server_args,
        )

        # Multi-node only: restart sglang/vllm with this variant's flags so
        # each row runs against a fresh server (parity with single-node
        # PHASE=all). No-op in single-node mode.
        from ._multi_node_server_lifecycle import (
            ServerRestartFailed,
            restart_server_for_round,
        )
        try:
            # PD knobs auto-resolved from $PD_* env; PD config stays constant
            # across variants within one run.
            await restart_server_for_round(
                extra_server_args=merge_server_args(
                    base_extra_args, variant.extra_server_args,
                ),
                model_path=model_path,
                ep=int(os.environ.get("EP") or 0) or None,
            )
        except ServerRestartFailed as exc:
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "mn_server_restart_failed: %s",
                i + 1, len(grid), variant.name, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="mn_server_restart_failed",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                error=f"mn_server_restart_failed: {exc}",
                error_class="mn_server_restart_failed",
                note=variant.note,
            ))
            if not keep_going_on_failure:
                break
            continue

        # Snapshot wall-clock before launch so the salvage path can mtime-gate
        # leak destinations per-variant (else a stale prior-run artifact
        # masquerades as this variant's result).
        variant_started_unix = time.time()
        try:
            rc, stdout, stderr = await asyncio.to_thread(
                _run_magpie,
                magpie_python=magpie_python,
                config_path=cfg_path,
                output_dir=slot,
                timeout_sec=variant_timeout_sec,
                cwd=cwd,
                result_dir=result_dir,
                soft_deadline_sec=soft_deadline_sec,
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest pre-timeout leaks so the variant slot captures whatever
            # the wrapper wrote before the timer fired.
            to_candidates = sorted(slot.glob("benchmark_*"))
            to_destination = to_candidates[-1] if to_candidates else slot
            to_harvested = harvest_leaked_artifacts(
                to_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "magpie timeout (timeout_sec=%d): %s",
                i + 1, len(grid), variant.name, variant_timeout_sec, exc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="magpie_timeout",
                error_summary=str(exc),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"timeout: {exc}",
                error_class="magpie_timeout",
                note=variant.note,
                runtime_sec=round(
                    max(0.0, time.time() - variant_started_unix), 2,
                ),
                nonfatal_warnings=[
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in to_harvested
                ],
            ))
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Server-liveness watchdog fired: the variant's server engine/worker
        # bootstrap died but the parent process hung. Record a fast failure
        # (instead of a ~2h hard-timeout stall) so the ExploreExecutor drops
        # this variant and the round proceeds. Harvest the crash server.log.
        if rc == SERVER_DEAD_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix), 2,
            )
            sd_candidates = sorted(slot.glob("benchmark_*"))
            sd_destination = sd_candidates[-1] if sd_candidates else slot
            sd_harvested = harvest_leaked_artifacts(
                sd_destination,
                subprocess_started_unix=variant_started_unix,
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "server_init_dead (engine/worker bootstrap failed; parent "
                "hung) after %.1fs",
                i + 1, len(grid), variant.name, variant_runtime_sec,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="server_init_dead",
                error_summary=(
                    "server engine/worker init failed; parent process hung "
                    "and was reaped by the liveness watchdog"
                ),
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                runtime_sec=variant_runtime_sec,
                error="server_init_dead: engine/worker bootstrap failed",
                error_class="server_init_dead",
                note=variant.note,
                nonfatal_warnings=[
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in sd_harvested
                ],
            ))
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Soft overtime gate fired: record a ``killed_overtime=True`` result
        # with no tput so the ExploreExecutor demotes it to ``KILLED_OVERTIME``.
        # Still harvest leaks for post-mortem.
        if rc == OVERTIME_KILL_RETURNCODE:
            variant_runtime_sec = round(
                max(0.0, time.time() - variant_started_unix), 2,
            )
            ok_candidates = sorted(slot.glob("benchmark_*"))
            ok_destination = ok_candidates[-1] if ok_candidates else slot
            ok_harvested = harvest_leaked_artifacts(
                ok_destination,
                subprocess_started_unix=variant_started_unix,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                killed_overtime=True,
                runtime_sec=variant_runtime_sec,
                error=(
                    f"killed_overtime: wall-clock {variant_runtime_sec:.1f}s "
                    f"exceeded soft_deadline_sec={float(soft_deadline_sec or 0.0):.1f}s"
                ),
                note=variant.note,
                nonfatal_warnings=[
                    f"harvested_leaked_artifact:{src}"
                    for src, _ in ok_harvested
                ],
            ))
            log.info(
                "_grid_runner: variant %s killed_overtime "
                "(runtime=%.1fs deadline=%.1fs)",
                variant.name, variant_runtime_sec,
                float(soft_deadline_sec or 0.0),
            )
            await _pulse_after_variant(i)
            if not keep_going_on_failure:
                break
            continue

        # Locate workspace inside slot.
        candidates = sorted(slot.glob("benchmark_*"))
        # Always-on artifact harvest (parity with BaselineExecutor) so each
        # variant slot keeps its server.log / gpu_metrics / profile relay for
        # Robustness RCA.
        harvest_destination = candidates[-1] if candidates else slot
        harvested = harvest_leaked_artifacts(
            harvest_destination,
            subprocess_started_unix=variant_started_unix,
        )
        if harvested:
            log.info(
                "_grid_runner: variant=%s harvested %d leaked artifact(s): %s",
                variant.name,
                len(harvested),
                ", ".join(src.name for src, _ in harvested),
            )
        if not candidates:
            harvest_tags = [f"harvested_leaked_artifact:{src}" for src, _ in harvested]
            no_ws_error_summary = (
                (stderr or stdout)[-2000:]
                if rc != 0 else "no benchmark_* workspace produced"
            )
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: "
                "no_benchmark_workspace (rc=%s)",
                i + 1, len(grid), variant.name, rc,
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class="no_benchmark_workspace",
                error_summary=no_ws_error_summary,
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                error=no_ws_error_summary,
                error_class="no_benchmark_workspace",
                nonfatal_warnings=harvest_tags,
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        workspace = candidates[-1]
        report = _parse_report(workspace)
        report_path = workspace / "benchmark_report.json"
        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=variant_started_unix,
        )
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if rc != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")
        for leak_src, _ in harvested:
            warnings.append(f"harvested_leaked_artifact:{leak_src}")

        if not measurement.get("valid_measurement"):
            if rc != 0:
                error = (stderr or stdout)[-2000:]
                invalid_class = "magpie_nonzero_invalid_measurement"
            elif not report:
                error = "benchmark_report missing"
                invalid_class = "benchmark_report_missing"
            else:
                error = "benchmark_report missing valid throughput/completed requests"
                invalid_class = "benchmark_report_invalid_metric"
            log.warning(
                "grid_runner: variant %d/%d name=%s aborted: %s (rc=%s): %s",
                i + 1, len(grid), variant.name, invalid_class, rc, error[:200],
            )
            _write_variant_abort_marker(
                slot,
                variant_name=variant.name,
                error_class=invalid_class,
                error_summary=error,
                extra_args=variant.extra_server_args,
            )
            results.append(VariantResult(
                name=variant.name, extra_server_args=variant.extra_server_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                workspace=str(workspace),
                report_path=str(report_path) if report_path.exists() else None,
                raw_result_path=measurement.get("raw_result_path"),
                reported_success=measurement.get("reported_success"),
                returncode=rc,
                nonfatal_warnings=warnings,
                error=error,
                error_class=invalid_class,
                note=variant.note,
            ))
            await _pulse_after_variant(i)
            if rc != 0 and not keep_going_on_failure:
                break
            continue

        results.append(VariantResult(
            name=variant.name, extra_server_args=variant.extra_server_args,
            extra_envs=dict(variant.extra_envs),
            status="succeeded",
            output_throughput=measurement.get("output_throughput"),
            request_throughput=measurement.get("request_throughput"),
            total_token_throughput=measurement.get("total_token_throughput"),
            completed_requests=measurement.get("completed_requests"),
            duration_seconds=measurement.get("duration_seconds"),
            ttft_mean_ms=measurement.get("ttft_mean_ms"),
            e2el_mean_ms=measurement.get("e2el_mean_ms"),
            tpot_mean_ms=measurement.get("tpot_mean_ms"),
            workspace=str(workspace),
            report_path=str(report_path) if report_path.exists() else None,
            raw_result_path=measurement.get("raw_result_path"),
            reported_success=measurement.get("reported_success"),
            returncode=rc,
            nonfatal_warnings=warnings,
            error=(stderr or stdout)[-2000:] if rc != 0 else None,
            note=variant.note,
            runtime_sec=round(
                max(0.0, time.time() - variant_started_unix), 2,
            ),
        ))
        log.info(
            "grid_runner: variant %s tput=%.1f tok/s",
            variant.name, results[-1].output_throughput or 0.0,
        )
        await _pulse_after_variant(i)
    return results


SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 1.0
MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT = 2.0


def pick_winners(
    results: list[VariantResult],
    baseline_tput: float,
    *,
    keep_threshold_pct: float | None = None,
) -> list[VariantResult]:
    """Filter variants whose throughput beats ``baseline_tput`` by
    ``keep_threshold_pct`` percent (> 1% = KEEP).

    Resolution of ``keep_threshold_pct``: an explicit caller value wins; ``None``
    falls back to ``MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT`` (2.0%, the empirical
    cross-node noise floor) in multi-node mode, else
    ``SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT`` (1.0%).
    """
    if keep_threshold_pct is None:
        from ._multi_node_env import is_multi_node
        keep_threshold_pct = (
            MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT
            if is_multi_node()
            else SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT
        )
    cutoff = baseline_tput * (1.0 + keep_threshold_pct / 100.0)
    return [
        r for r in results
        if r.status == "succeeded"
        and isinstance(r.output_throughput, (int, float))
        and r.output_throughput > cutoff
    ]


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names.

    Args:
        name (str): The variant name to slugify.

    Returns:
        str: ``name`` with every character that is not alphanumeric or in
        ``-_.`` replaced by ``_``, truncated to 60 characters.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


def _write_variant_abort_marker(
    slot: Path,
    *,
    variant_name: str,
    error_class: str,
    error_summary: str,
    extra_args: str = "",
) -> None:
    """Write ``abort_reason.json`` into the variant slot directory.

    When a variant aborts before benchmark_report.json exists, the slot has
    only ``config.yaml`` and a reader can't tell "tested-but-failed" from
    "untested". This marker lets final-report / post-mortem tools count failed
    variants and find an explicit reason even after the log rotated. Failure
    to write it is non-fatal (log and continue).
    """
    try:
        slot.mkdir(parents=True, exist_ok=True)
        marker = {
            "variant": variant_name,
            "error_class": error_class,
            "error": (error_summary or "")[:2000],
            "extra_args": extra_args,
            "aborted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            ),
        }
        (slot / "abort_reason.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning(
            "_grid_runner: failed to write abort_reason.json at %s: %s",
            slot, exc,
        )


__all__ = [
    "DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC",
    "GridVariant",
    "MULTI_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "SGLANG_WATCHDOG_TIMEOUT_ENV",
    "SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT",
    "VariantResult",
    "apply_runtime_benchmark_overrides",
    "inject_sglang_attention_backend",
    "inject_sglang_context_length",
    "inject_sglang_watchdog_timeout",
    "merge_server_args",
    "pick_winners",
    "resolve_sglang_watchdog_timeout",
    "run_grid",
    "sanitize_result_dir",
    "sanitize_script_name",
    "server_args_env_name",
    "variant_fingerprint",
]
