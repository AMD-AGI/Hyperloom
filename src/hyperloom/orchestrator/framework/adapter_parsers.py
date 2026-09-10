# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-framework argv-parser source strings.

Each string is Python source defining ``_build_parser()``, run in the interpreter
that will serve so an argv is judged by the parser that would actually reject it.

Kept apart from the adapters so ``bringup`` can judge an argv without importing
runtime acquisition, which would invert the layering.
"""

from __future__ import annotations

_VLLM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    from vllm.entrypoints.openai.cli_args import make_arg_parser\n"
    "    try:\n"
    "        from vllm.utils.argparse_utils import FlexibleArgumentParser\n"
    "    except ImportError:\n"
    "        from vllm.utils import FlexibleArgumentParser\n"
    "    return make_arg_parser(FlexibleArgumentParser())\n"
)

_SGLANG_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import argparse\n"
    "    from sglang.srt.server_args import ServerArgs\n"
    "    parser = argparse.ArgumentParser()\n"
    "    ServerArgs.add_cli_args(parser)\n"
    "    return parser\n"
)

_ATOM_PARSER_SOURCE = (
    "def _build_parser():\n"
    "    import argparse\n"
    "    from atom.model_engine.arg_utils import EngineArgs\n"
    "    parser = argparse.ArgumentParser()\n"
    "    EngineArgs.add_cli_args(parser)\n"
    "    return parser\n"
)

_PARSER_SOURCES: dict[str, str] = {
    "vllm": _VLLM_PARSER_SOURCE,
    "sglang": _SGLANG_PARSER_SOURCE,
    "atom": _ATOM_PARSER_SOURCE,
}


def parser_source_for(framework: str) -> str:
    """Return the ``_build_parser()`` source for *framework*, or ``""`` when unknown.

    Args:
        framework: Lower-cased framework name (e.g. ``"vllm"``, ``"sglang"``).

    Returns:
        str: Python source string, or ``""`` for frameworks that expose no probed
        parser (``NullAdapter``/``XditAdapter`` behaviour — an unavailable verdict).
    """
    return _PARSER_SOURCES.get((framework or "").strip().lower(), "")


__all__ = ["parser_source_for"]
