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
