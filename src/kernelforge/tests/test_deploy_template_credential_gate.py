# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Contract tests for the Claude credential gate in deploy/robust/kernelforge.yaml.

The gate is plain shell inside a Kubernetes template, so nothing in the Python
test suite would notice it regressing. These tests lift the snippet verbatim out
of the template and run it under bash, which needs no cluster and no GPU.

Two shapes are legal: a gateway pair (ANTHROPIC_BASE_URL with
ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY) or a subscription
CLAUDE_CODE_OAUTH_TOKEN whose endpoint the Claude CLI defaults on its own. The
CLI resolves the gateway variables ahead of the OAuth token, which is why the
subscription token must never reach primaryApiKey or be synthesized into any
ANTHROPIC_* variable -- that would silently move billing to API credits.

The same split decides TLS: only a custom endpoint gets verification relaxed,
since the OAuth shape carries a long-lived token to the public default one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "deploy" / "robust" / "kernelforge.yaml"

_GATE_FIRST_LINE = "_kf_has_gateway_pair() {"
_GATE_STOP_LINE = "# OPENAI_BASE_URL + OPENAI_API_KEY are the separate line"
_CONFIG_WRITER_MARKER = 'pathlib.Path("/root/.claude/config.json")'
_CONFIG_GUARD = "if _kf_has_gateway_pair; then"

_CREDENTIAL_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_")


def _template_lines() -> list[str]:
    return TEMPLATE.read_text(encoding="utf-8").splitlines()


def _index_of(lines: list[str], needle: str) -> int:
    for i, line in enumerate(lines):
        if needle in line:
            return i
    raise AssertionError(f"{TEMPLATE.name} no longer contains {needle!r}")


def _gate_script() -> str:
    """Return the gate block as a standalone bash script."""
    lines = _template_lines()
    start = _index_of(lines, _GATE_FIRST_LINE)
    stop = _index_of(lines, _GATE_STOP_LINE)
    assert start < stop, "gate block and the OpenAI comment swapped order"
    body = textwrap.dedent("\n".join(lines[start:stop]))
    assert not body.startswith(" "), "gate block is not uniformly indented"
    return "set -euo pipefail\n" + body + "\n"


def _run_gate(*, probe: str = "", **credentials: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CREDENTIAL_PREFIXES)}
    # Not ANTHROPIC_-prefixed, so the filter above misses it and a developer's
    # own value would leak into what the gate is observed to decide.
    env.pop("NODE_TLS_REJECT_UNAUTHORIZED", None)
    env.update(credentials)
    script = _gate_script() + (f"{probe}\n" if probe else "")
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ── accepted and rejected credential shapes ───────────────────────────────────


def test_oauth_token_alone_is_accepted():
    """The point of the change: a subscription token needs no endpoint, because
    the CLI supplies api.anthropic.com itself. This shape used to exit 5."""
    result = _run_gate(CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key_var", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_gateway_pair_is_still_accepted(key_var):
    result = _run_gate(**{"ANTHROPIC_BASE_URL": "https://gateway.example/v1", key_var: "secret"})
    assert result.returncode == 0, result.stderr


def test_endpoint_without_any_credential_is_rejected():
    result = _run_gate(ANTHROPIC_BASE_URL="https://gateway.example/v1")
    assert result.returncode == 5
    assert "CLAUDE_CODE_OAUTH_TOKEN" in result.stderr


def test_no_credential_at_all_is_rejected():
    result = _run_gate()
    assert result.returncode == 5


def test_credential_from_one_shape_does_not_complete_the_other():
    """A bare key with no endpoint is still half a pair, and the OAuth token does
    not stand in for the missing half."""
    result = _run_gate(ANTHROPIC_API_KEY="secret")
    assert result.returncode == 5


# ── warnings on ambiguous but legal combinations ───────────────────────────────


def test_gateway_key_beside_oauth_token_warns_about_billing():
    result = _run_gate(
        CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test",
        ANTHROPIC_BASE_URL="https://gateway.example/v1",
        ANTHROPIC_API_KEY="secret",
    )
    assert result.returncode == 0, result.stderr
    assert "bills to" in result.stderr


def test_oauth_token_against_custom_endpoint_warns_about_auth():
    result = _run_gate(
        CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test",
        ANTHROPIC_BASE_URL="https://gateway.example/v1",
    )
    assert result.returncode == 0, result.stderr
    assert "auth failures" in result.stderr


# ── TLS verification follows the endpoint, not the credential ─────────────────

_TLS_PROBE = 'printf "%s %s\\n" "$ANTHROPIC_SKIP_TLS_VERIFY" "$NODE_TLS_REJECT_UNAUTHORIZED"'


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}, "false 1"),
        (
            {"ANTHROPIC_BASE_URL": "https://gateway.example/v1", "ANTHROPIC_API_KEY": "secret"},
            "true 0",
        ),
    ],
    ids=["default endpoint", "custom endpoint"],
)
def test_tls_is_only_relaxed_for_a_custom_endpoint(credentials, expected):
    """A self-hosted gateway presents a chain the image does not carry. The
    public default endpoint does not, and relaxing there exposes the token."""
    result = _run_gate(probe=_TLS_PROBE, **credentials)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_the_secret_can_still_relax_tls_on_the_default_endpoint():
    """The stricter default is a default, not a policy: an operator who knows
    their egress is intercepted can set both back in the workspace secret."""
    result = _run_gate(
        probe=_TLS_PROBE,
        CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-test",
        ANTHROPIC_SKIP_TLS_VERIFY="true",
        NODE_TLS_REJECT_UNAUTHORIZED="0",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "true 0"


# ── ~/.claude/config.json must stay a gateway-only artifact ───────────────────


def test_config_json_write_is_guarded_by_the_gateway_shape():
    lines = _template_lines()
    writer = _index_of(lines, _CONFIG_WRITER_MARKER)
    preceding = lines[max(0, writer - 6) : writer]
    assert any(_CONFIG_GUARD in line for line in preceding), (
        "the config.json write must stay inside the gateway-shape branch; "
        "OAuth-only would otherwise get an empty primaryApiKey/customApiUrl"
    )


def test_oauth_token_never_reaches_config_json():
    lines = _template_lines()
    writer = lines[_index_of(lines, _CONFIG_WRITER_MARKER)]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in writer


def test_no_anthropic_variable_is_synthesized_from_the_oauth_token():
    synthesis = re.compile(r"ANTHROPIC_[A-Z_]+=[^\n]*CLAUDE_CODE_OAUTH_TOKEN")
    offenders = [line for line in _template_lines() if synthesis.search(line)]
    assert not offenders, offenders


def test_gateway_config_json_content_is_unchanged(tmp_path):
    """Pin the written bytes, so guarding the write cannot quietly reshape it."""
    lines = _template_lines()
    writer = lines[_index_of(lines, _CONFIG_WRITER_MARKER)].strip()
    snippet = writer.split("-c ", 1)[1].strip("'")
    target = tmp_path / "config.json"
    snippet = snippet.replace("/root/.claude/config.json", str(target))

    env = {k: v for k, v in os.environ.items() if not k.startswith(_CREDENTIAL_PREFIXES)}
    env["ANTHROPIC_API_KEY"] = "gateway-key"
    env["ANTHROPIC_BASE_URL"] = "https://gateway.example/v1"
    subprocess.run([sys.executable, "-c", snippet], env=env, check=True)

    assert target.read_text(encoding="utf-8") == json.dumps(
        {
            "theme": "dark",
            "hasCompletedOnboarding": True,
            "primaryApiKey": "gateway-key",
            "customApiUrl": "https://gateway.example/v1",
        }
    )
    assert target.stat().st_mode & 0o777 == 0o600
