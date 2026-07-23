#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side server-args preflight, run as the FIRST step of the restart kill job.

Validates a variant's extra server args (denied flags / unsupported flags /
missing optional backend) BEFORE ``kill_multinode.py`` tears down the running
server, so a rejected variant leaves the existing multi-node server intact
instead of a killed, non-serving cluster. This replaces the separate
``--preflight-only`` Ray job (one fewer submission per restart) while keeping the
"validate before kill" guarantee.

Contract:
* Positive reject (a concrete, actionable reason): print ``MN_PREFLIGHT_REJECT:``
  + reason and exit ``3``. The kill entrypoint then aborts before killing; the
  controller maps the sentinel to ``EXIT_CONFIG_ERROR``.
* Otherwise (clean, or the check itself cannot run): exit ``0`` — fail-open, so a
  preflight infra hiccup never blocks a launch (the launcher re-checks anyway).

The check helpers mirror ``launch_multinode.py`` / ``launch_infera_node.py``;
they are pure, side-effect-free, and kept in sync by design.
"""

from __future__ import annotations

import argparse
import shlex
import sys

# Machine-readable marker the controller greps for in the kill job logs.
REJECT_SENTINEL = "MN_PREFLIGHT_REJECT:"

# Keep in sync with multi_node/_internal/server_args_safety.py
_DENIED_SERVER_FLAGS = frozenset(
    {
        "--adapter-model-path",
        "--adapter-path",
        "--allowed-local-media-path",
        "--chat-template",
        "--code-revision",
        "--config",
        "--download-dir",
        "--hf-overrides",
        "--lora-dirs",
        "--lora-modules",
        "--lora-path",
        "--lora-paths",
        "--model",
        "--model-id",
        "--model-path",
        "--quantization-param-path",
        "--revision",
        "--tokenizer",
        "--tokenizer-path",
        "--tokenizer-revision",
    }
)
_DENIED_SERVER_FLAG_SUFFIXES = ("-dir", "-file", "-path")


def _is_denied_server_flag(flag: str) -> bool:
    """Return whether a single ``--flag`` token is denied at the pod boundary."""
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_SERVER_FLAGS:
        return True
    return any(name.endswith(suffix) for suffix in _DENIED_SERVER_FLAG_SUFFIXES)


def _denied_extra_args(raw: str) -> list[str]:
    """Return denied CLI flag tokens in a pod-side extra-args string.

    Args:
        raw: Whitespace-separated server flags.

    Returns:
        list[str]: Denied flag names (empty when clean).
    """
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return ["<unparseable>"]
    out: list[str] = []
    for tok in tokens:
        flag = tok.split("=", 1)[0]
        if _is_denied_server_flag(flag) and flag not in out:
            out.append(flag)
    return out


def _sglang_cli_parser():
    """Return an argparse parser exposing this build's sglang server CLI flags.

    Tries the legacy ``sglang.launch_server.parser`` export first, then falls
    back to ``ServerArgs.add_cli_args`` (current sglang releases). Returns
    ``None`` when neither path is available so callers can fail open.
    """
    try:
        from sglang.launch_server import parser as _legacy_parser

        if getattr(_legacy_parser, "_option_string_actions", None):
            return _legacy_parser
    except Exception:  # noqa: BLE001 - probe must not block launch
        pass
    try:
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser(add_help=False)
        ServerArgs.add_cli_args(parser)
        if getattr(parser, "_option_string_actions", None):
            return parser
    except Exception:  # noqa: BLE001 - probe must not block launch
        pass
    return None


def _unsupported_extra_arg_flags(framework: str, extra_args: list[str]) -> list[str]:
    """Return ``--`` flags not registered on the installed server build's parser.

    Best-effort: returns ``[]`` when the parser cannot be introspected (missing
    framework / import error) so a preflight infra problem never blocks a launch.

    Args:
        framework: Active server framework (only ``sglang`` is introspected).
        extra_args: Tokenized extra server args.

    Returns:
        list[str]: Flag tokens not registered on the build's parser, or ``[]``.
    """
    flags = [t for t in extra_args if t.startswith("--")]
    if not flags:
        return []
    if framework != "sglang":
        return []
    parser = _sglang_cli_parser()
    known = set(getattr(parser, "_option_string_actions", {}).keys()) if parser is not None else set()
    if not known:
        return []
    return [f for f in flags if f.split("=", 1)[0] not in known]


def _extra_arg_value(extra_args: list[str], flag: str) -> str | None:
    """Return the value for ``flag`` in ``extra_args`` (space- or ``=``-separated, last-wins)."""
    value: str | None = None
    for i, tok in enumerate(extra_args):
        if tok == flag and i + 1 < len(extra_args):
            value = extra_args[i + 1]
        elif tok.startswith(flag + "="):
            value = tok.split("=", 1)[1]
    return value


def _resolve_default_moe_a2a_backend(framework: str) -> str | None:
    """Best-effort default ``--moe-a2a-backend`` for the installed build, else ``None``."""
    if framework != "sglang":
        return None
    try:
        parser = _sglang_cli_parser()
        if parser is None:
            return None
        act = getattr(parser, "_option_string_actions", {}).get("--moe-a2a-backend")
        default = getattr(act, "default", None) if act is not None else None
        if default is not None:
            return str(default).strip().lower()
    except Exception:  # noqa: BLE001 - preflight must never block on its own failure
        return None
    return None


def _missing_capability_reason(framework: str, extra_args: list[str]) -> str | None:
    """Return a reason if a requested flag needs an optional backend package that is absent.

    DeepEP is required ONLY when the effective MoE a2a backend is ``deepep``
    (explicit flag wins; TBO with no explicit backend falls back to the build
    default). Only fail fast when it is positively ``deepep``; when in doubt, do
    not block.

    Args:
        framework: Active server framework (used to resolve the default backend).
        extra_args: Tokenized extra server args.

    Returns:
        str | None: A reason when a required backend package is missing, else ``None``.
    """
    try:
        import importlib.util

        backend = (_extra_arg_value(extra_args, "--moe-a2a-backend") or "").strip().lower()
        if not backend and "--enable-two-batch-overlap" in extra_args:
            backend = _resolve_default_moe_a2a_backend(framework) or ""
        if backend == "deepep" and importlib.util.find_spec("deep_ep") is None:
            return (
                "requires the DeepEP a2a backend, but the `deep_ep` package is "
                "not installed in this image (--moe-a2a-backend deepep)"
            )
    except Exception:  # noqa: BLE001 - preflight must never block on its own failure
        return None
    return None


def _reject(reason: str) -> int:
    """Emit the reject sentinel + reason and return the config-error exit code."""
    print(f"{REJECT_SENTINEL}{reason}")
    return 3


def main(argv: list[str] | None = None) -> int:
    """Validate the variant's server args; exit 3 on a positive reject, else 0.

    Returns:
        int: ``3`` (reject, kill must be skipped) or ``0`` (fail-open / clean).
    """
    p = argparse.ArgumentParser(prog="_server_args_preflight.py")
    p.add_argument("--framework", required=True, help="sglang / vllm")
    p.add_argument("--extra-args", default="", help="raw extra server-args string")
    args = p.parse_args(argv)

    framework = (args.framework or "").strip().lower()
    raw = args.extra_args or ""

    denied = _denied_extra_args(raw)
    if denied:
        return _reject(f"denied server flags: {denied}")

    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    unsupported = _unsupported_extra_arg_flags(framework, tokens)
    if unsupported:
        return _reject(
            f"unsupported server flags for {framework} build: {unsupported} "
            "(flag names vary by version)"
        )

    cap_reason = _missing_capability_reason(framework, tokens)
    if cap_reason:
        return _reject(f"server flags [{raw}] {cap_reason}")

    print(f"MN_PREFLIGHT_OK framework={framework}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
