# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bootstrap shell and ``dual_protocol_endpoint_pair`` must derive the same pair.

``install_baremetal.sh`` runs before any Python is importable, so it carries its
own copy of the derivation. A divergence between the two only shows up once a
gateway is configured through the installer rather than the library, so the two
are pinned here against one corpus.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from hyperloom.common.llm_config import dual_protocol_endpoint_pair

_INSTALL_SH = Path(__file__).resolve().parents[1] / "assets" / "install_baremetal.sh"

_URLS = [
    "",
    "https://gw.example",
    "https://gw.example/",
    "https://gw.example/v1",
    "https://gw.example/anthropic",
    "https://gw.example/Anthropic",
    "https://api.deepseek.com",
    "https://api.deepseek.com/",
    "https://api.deepseek.com/v1",
    "https://api.deepseek.com/anthropic",
    "http://api.deepseek.com",
    "https://a.b/c/d",
    "https://gw.example:8443",
]


def _shell_case_block() -> str:
    """Lift the ``case`` that derives the pair out of the installer function."""
    fn = subprocess.run(
        ["sed", "-n", "/^migrate_legacy_deepseek_env()/,/^}/p", str(_INSTALL_SH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    block = re.search(r"^\s*case \"\$lowered\" in$.*?^\s*esac$", fn, re.S | re.M)
    assert block, "endpoint-derivation case block not found in migrate_legacy_deepseek_env"
    return block.group(0)


def _shell_pair(url: str) -> tuple[str, str]:
    script = (
        "set -eu\n"
        f'url="{url}"\n'
        'base="${url%/}"\n'
        "lowered=\"$(printf '%s' \"$base\" | tr '[:upper:]' '[:lower:]')\"\n"
        f"{_shell_case_block()}\n"
        'printf "%s\\n%s\\n" "$anthropic_url" "$openai_url"\n'
    )
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout
    anthropic_url, openai_url = out.splitlines()
    return anthropic_url, openai_url


@pytest.mark.parametrize("url", _URLS)
def test_shell_and_python_derive_the_same_pair(url: str) -> None:
    assert _shell_pair(url) == dual_protocol_endpoint_pair(url)


def test_a_bare_unknown_host_gains_no_v1_on_either_side() -> None:
    """The branch the two implementations disagreed on, pinned on both sides."""
    assert dual_protocol_endpoint_pair("https://gw.example") == ("https://gw.example", "https://gw.example")
    assert _shell_pair("https://gw.example") == ("https://gw.example", "https://gw.example")
