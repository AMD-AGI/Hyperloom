# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Single source of truth for inference-framework capabilities.

A *framework* is one serving/execution backend a session optimizes
(``sglang`` / ``vllm`` / ``atom`` / ``xdit``). Historically the allowed set
and per-framework behavior (extra-args env name, repo URL, server-reuse
eligibility) were hardcoded across ``cli.py`` / ``_grid_runner.py`` /
``_server_lifecycle.py`` / ``framework-agent``. This module centralizes them
so adding a framework is a single-table edit.

It also introduces ``kind`` to split the two execution models the loop must
support:

* ``serving``    — a long-lived OpenAI-compatible server benchmarked by a
  client (``benchmark_serving.py``); throughput is tokens/sec; accuracy is a
  numeric eval (GSM8K). This is sglang/vllm/atom.
* ``scriptable`` — a server-less single-command workload (e.g. xDiT diffusion)
  whose bench script writes a ``benchmark_report.json`` directly; throughput is
  images/sec; "accuracy" is an image-quality gate (LPIPS/SSIM/MSE). Server
  reuse, the ``benchmark_serving`` client, and the GSM8K gate do not apply.

The module has no third-party imports so it is safe to import from the CLI,
the executors, and standalone tools alike.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameworkSpec:
    """Static capabilities of one inference framework.

    Attributes:
        name: Canonical lowercase framework name.
        kind: Execution model — ``"serving"`` or ``"scriptable"``.
        extra_args_env: Env var Magpie scripts expand to append backend args.
        repo_url: Canonical upstream git URL (FRAMEWORK), or ``None``.
        supports_server_reuse: Whether the Magpie ``server_lifecycle`` reuse
            protocol applies (always ``False`` for ``scriptable``).
        throughput_unit: Human-readable throughput unit for reports.
        magpie_script_fmt: Format string for the Magpie benchmark script name,
            resolved with ``framework`` and ``runner_type``.
    """

    name: str
    kind: str
    extra_args_env: str
    repo_url: str | None
    supports_server_reuse: bool
    throughput_unit: str
    magpie_script_fmt: str = "{framework}_{runner_type}.sh"


SERVING = "serving"
SCRIPTABLE = "scriptable"


# The single registry. Adding a framework = one entry here (+ a Magpie script
# and a benchmark YAML). Order is the canonical order surfaced in CLI help.
FRAMEWORKS: dict[str, FrameworkSpec] = {
    "sglang": FrameworkSpec(
        name="sglang",
        kind=SERVING,
        extra_args_env="EXTRA_SGLANG_ARGS",
        repo_url="https://github.com/sgl-project/sglang.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "vllm": FrameworkSpec(
        name="vllm",
        kind=SERVING,
        extra_args_env="EXTRA_VLLM_ARGS",
        repo_url="https://github.com/ROCm/vllm.git",
        supports_server_reuse=True,
        throughput_unit="tok/s",
    ),
    "atom": FrameworkSpec(
        name="atom",
        kind=SERVING,
        extra_args_env="EXTRA_ATOM_ARGS",
        repo_url="https://github.com/ROCm/ATOM.git",
        supports_server_reuse=False,
        throughput_unit="tok/s",
    ),
    "xdit": FrameworkSpec(
        name="xdit",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_XDIT_ARGS",
        repo_url="https://github.com/xdit-project/xDiT.git",
        supports_server_reuse=False,
        throughput_unit="img/s",
    ),
    # HunyuanImage-3.0: an 80B unified autoregressive multimodal MoE text-to-
    # image model. Unlike the xfuser/diffusers pipelines that back ``xdit``, it
    # is a transformers ``AutoModelForCausalLM`` (trust_remote_code) driven via
    # its own ``generate_image`` method, so it does NOT use the xfuser
    # sequence-parallel runner registry. It is still a SCRIPTABLE image workload
    # (server-less single command; throughput img/s; LPIPS/SSIM/MSE quality
    # gate), so it reuses the scriptable env plumbing in _workload_envs
    # (XDIT_QUALITY_* injection) and the img/s -> e2el_mean_ms metric mapping.
    "hunyuan_image3": FrameworkSpec(
        name="hunyuan_image3",
        kind=SCRIPTABLE,
        extra_args_env="EXTRA_HUNYUAN_IMAGE3_ARGS",
        # No framework-agent source repo: HunyuanImage-3.0 is a model we run
        # as-is, not a framework whose source the perf-PR agent scouts/patches
        # (unlike xdit). Kept out of framework_agent's _FRAMEWORK_TO_REPO_URL,
        # so repo_url stays None to satisfy the registry<->repo_map consistency
        # guard (test_registry_urls_match_repo_map skips None entries).
        repo_url=None,
        supports_server_reuse=False,
        throughput_unit="img/s",
    ),
}

DEFAULT_FRAMEWORK = "sglang"


def names() -> tuple[str, ...]:
    """Return the canonical tuple of supported framework names.

    Returns:
        tuple[str, ...]: All registered framework names, in registry order.
    """
    return tuple(FRAMEWORKS)


def is_supported(framework: str | None) -> bool:
    """Return whether ``framework`` is a registered framework.

    Args:
        framework (str | None): Candidate name; matched case-insensitively.

    Returns:
        bool: ``True`` when the name is registered.
    """
    return str(framework or "").strip().lower() in FRAMEWORKS


def get(framework: str | None) -> FrameworkSpec:
    """Return the :class:`FrameworkSpec` for ``framework``.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        FrameworkSpec: The matching spec.

    Raises:
        KeyError: When the name is not registered.
    """
    key = str(framework or "").strip().lower()
    return FRAMEWORKS[key]


def _spec_or_default(framework: str | None) -> FrameworkSpec:
    """Return the spec for ``framework`` or the default's spec when unknown.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        FrameworkSpec: The matching spec, or the default framework's spec.
    """
    key = str(framework or "").strip().lower()
    return FRAMEWORKS.get(key, FRAMEWORKS[DEFAULT_FRAMEWORK])


def kind(framework: str | None) -> str:
    """Return the execution ``kind`` for ``framework``.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"serving"`` or ``"scriptable"`` (default framework's kind for
        unknown names).
    """
    return _spec_or_default(framework).kind


def is_scriptable(framework: str | None) -> bool:
    """Return whether ``framework`` is a server-less scriptable workload.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        bool: ``True`` for ``scriptable`` frameworks (e.g. xDiT).
    """
    return _spec_or_default(framework).kind == SCRIPTABLE


def extra_args_env(framework: str | None) -> str:
    """Return the Magpie env var used to append backend args.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: The ``EXTRA_*_ARGS`` env name (default framework's when unknown).
    """
    return _spec_or_default(framework).extra_args_env


def throughput_unit(framework: str | None) -> str:
    """Return the throughput unit string for ``framework``.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"tok/s"`` or ``"img/s"``.
    """
    return _spec_or_default(framework).throughput_unit


def primary_metric_unit(framework: str | None) -> str:
    """Return the human-readable unit for a session's primary display metric.

    Serving frameworks display token throughput (``tok/s/GPU``). Scriptable
    image frameworks (e.g. xDiT) display the per-image end-to-end latency
    ``e2el_mean_ms`` in milliseconds instead of a reciprocal-of-latency value.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"ms"`` for scriptable xDiT, else ``"tok/s/GPU"``.
    """
    return "ms" if is_scriptable(framework) else "tok/s/GPU"


def primary_metric_name(framework: str | None) -> str:
    """Return the state field name holding a session's primary result metric.

    Serving frameworks are ranked by token throughput
    (``throughput_tok_s_per_gpu``); scriptable image frameworks (e.g. xDiT) are
    ranked by per-image end-to-end latency (``e2el_mean_ms``). Consumers use
    this to pick the correct headline/result field per framework.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str: ``"e2el_mean_ms"`` for scriptable xDiT, else
        ``"throughput_tok_s_per_gpu"``.
    """
    return "e2el_mean_ms" if is_scriptable(framework) else "throughput_tok_s_per_gpu"


def primary_metric_value(
    framework: str | None, tput_per_gpu: float | int | None
) -> float | None:
    """Convert stored per-GPU throughput into the value shown for ``framework``.

    Scriptable image frameworks (xDiT) store throughput as ``img/s``
    (``1 / latency``); the displayed metric is the equivalent per-image latency
    ``e2el_mean_ms = 1000 / img_per_s``. Serving frameworks display the stored
    throughput unchanged.

    Args:
        framework (str | None): Framework name; matched case-insensitively.
        tput_per_gpu (float | int | None): Per-GPU throughput as stored in
            state (``tok/s`` for serving, ``img/s`` for scriptable xDiT).

    Returns:
        float | None: The value to display, or ``None`` when it is undefined
        (non-positive throughput for a scriptable framework).
    """
    tput = float(tput_per_gpu or 0.0)
    if is_scriptable(framework):
        return (1000.0 / tput) if tput > 0 else None
    return tput


def format_primary_metric(
    framework: str | None, tput_per_gpu: float | int | None, *, precision: int = 1
) -> str:
    """Format a session's primary performance metric for human-readable display.

    Serving frameworks report token throughput, so the value is shown as
    ``"<tput> tok/s/GPU"``. Scriptable image frameworks (e.g. xDiT) are
    server-less and measure a single-stream, per-image end-to-end latency; their
    ``output_throughput`` is merely ``1 / latency`` (img/s). Rendering that
    reciprocal as ``tok/s/GPU`` is misleading (there is no comparable token
    stream), so instead surface the equivalent per-image latency
    ``e2el_mean_ms`` — derived exactly as ``1000 / img_per_s`` — in
    milliseconds.

    Args:
        framework (str | None): Session framework name; matched
            case-insensitively.
        tput_per_gpu (float | int | None): Per-GPU throughput as stored in
            state (``tok/s`` for serving, ``img/s`` for scriptable xDiT).
        precision (int): Number of decimals to render (default 1).

    Returns:
        str: A display string such as ``"123.4 tok/s/GPU"`` (serving) or
        ``"6440.0 ms"`` (xDiT ``e2el_mean_ms``). Returns ``"n/a ms"`` /
        ``"0.0 tok/s/GPU"`` for non-positive inputs.
    """
    unit = primary_metric_unit(framework)
    value = primary_metric_value(framework, tput_per_gpu)
    if value is None:
        return f"n/a {unit}"
    return f"{value:.{precision}f} {unit}"


def supports_server_reuse(framework: str | None) -> bool:
    """Return whether ``framework`` supports the server_lifecycle reuse path.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        bool: ``True`` only for serving frameworks that ship a reusable server.
    """
    return _spec_or_default(framework).supports_server_reuse


def repo_url(framework: str | None) -> str | None:
    """Return the canonical upstream repo URL for ``framework``.

    Args:
        framework (str | None): Framework name; matched case-insensitively.

    Returns:
        str | None: The repo URL, or ``None`` when not registered / unset.
    """
    spec = FRAMEWORKS.get(str(framework or "").strip().lower())
    return spec.repo_url if spec is not None else None
