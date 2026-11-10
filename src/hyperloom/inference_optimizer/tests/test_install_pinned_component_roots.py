# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A rerun of the kernel-agent install follows the pinned Magpie, InferenceX and TraceLens revisions and GEAK's ref.

Installer-managed tool paths stay out of ``.env``, so a later install cannot read them back as operator overrides.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from ._kernel_tools_harness import KERNEL_TOOLS_SH, install_sh_function, sourceable_kernel_tools


def _source_kernel_tools(tmp_path: Path, exports: dict[str, Path], then: str) -> str:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "geak").mkdir(exist_ok=True)
    lines = [
        "set -euo pipefail",
        "export ANTHROPIC_API_KEY=sk-test ANTHROPIC_BASE_URL=https://api.anthropic.com",
        f"export REPO_ROOT={repo_root} USER_DATA_PATH={tmp_path} HYPERLOOM_CACHE_DIR={tmp_path / 'cache'}",
        f"export HYPERLOOM_RUNTIME_DIR={tmp_path / 'runtime'} KERNEL_AGENT_ENV={tmp_path / 'runtime' / 'env.sh'}",
        f"export GEAK_ROOT={tmp_path / 'geak'}",
        *(f"export {key}={value}" for key, value in exports.items()),
        f"source {sourceable_kernel_tools(tmp_path)}",
        then,
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "GEAK_", "TRACELENS_"))}
    proc = subprocess.run(["bash", "-c", "\n".join(lines)], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_magpie_and_inferencex_paths_stay_out_of_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / "repo" / ".env"
    dotenv.parent.mkdir()
    dotenv.write_text(f"MAGPIE_PATH={tmp_path / 'old'}\nINFERENCEX_PATH={tmp_path / 'old'}\n", encoding="utf-8")

    _source_kernel_tools(
        tmp_path, {"MAGPIE_PATH": tmp_path / "Magpie", "INFERENCEX_PATH": tmp_path / "InferenceX"}, "write_env_file"
    )

    text = dotenv.read_text(encoding="utf-8")
    assert "MAGPIE_PATH=" not in text
    assert "INFERENCEX_PATH=" not in text


@pytest.mark.parametrize("managed", [True, False], ids=["installer-cache-checkouts", "operator-checkouts"])
def test_tool_roots_in_dotenv_are_left_to_the_operator(tmp_path: Path, managed: bool) -> None:
    """An installer-managed TraceLens/GEAK path leaves .env; one the operator wrote there stays as written.

    A managed path written back is read as operator input by the next install, and is kept as an override once it no
    longer canonicalises to the cache root.
    """
    if managed:
        tracelens, geak = tmp_path / "cache" / "TraceLens@0000000", tmp_path / "cache" / "GEAK@0000000"
    else:
        tracelens, geak = tmp_path / "opt" / "TraceLens", tmp_path / "geak"
        tracelens.mkdir(parents=True)
    runner = geak / "interface" / "run_e2e.py"
    entries = {"TRACELENS_ROOT": tracelens, "GEAK_ROOT": geak, "GEAK_E2E_RUNNER": runner}
    dotenv = tmp_path / "repo" / ".env"
    dotenv.parent.mkdir()
    dotenv.write_text("".join(f"{key}={value}\n" for key, value in entries.items()), encoding="utf-8")

    _source_kernel_tools(tmp_path, {**entries, "GEAK_REF": "1" * 40}, "write_env_file")

    lines = dotenv.read_text(encoding="utf-8").splitlines()
    for key, value in entries.items():
        if managed:
            assert not any(line.startswith(f"{key}=") for line in lines), key
        else:
            assert f"{key}={value}" in lines, key


def test_geak_cache_root_and_runner_follow_the_ref(tmp_path: Path) -> None:
    ref = "1" * 40
    old = tmp_path / "cache" / "GEAK@0000000"
    exports = {"GEAK_REF": ref, "GEAK_ROOT": old, "GEAK_E2E_RUNNER": old / "interface" / "run_e2e.py"}

    printed = _source_kernel_tools(tmp_path, exports, 'printf "%s\\n%s" "$GEAK_ROOT" "$GEAK_E2E_RUNNER"')
    root, runner = printed.split("\n")

    assert root == str(tmp_path / "cache" / f"GEAK@{ref}")
    assert runner == f"{root}/interface/run_e2e.py"


def test_a_missing_operator_tracelens_root_falls_back_to_the_pin(tmp_path: Path) -> None:
    printed = _source_kernel_tools(
        tmp_path, {"TRACELENS_ROOT": tmp_path / "gone"}, 'printf "%s\\n%s" "$TRACELENS_ROOT" "$_TRACELENS_SHA"'
    )
    resolved, sha = printed.split("\n")

    assert resolved == str(tmp_path / "cache" / f"TraceLens@{sha}")


@pytest.mark.parametrize("stale", [True, False], ids=["stale-cache-checkout", "operator-checkout"])
def test_tracelens_root_follows_the_pin_unless_the_operator_chose_it(tmp_path: Path, stale: bool) -> None:
    root = tmp_path / "cache" / "TraceLens@0000000" if stale else tmp_path / "my-tracelens"
    if not stale:
        root.mkdir()

    resolved, sha = _source_kernel_tools(
        tmp_path, {"TRACELENS_ROOT": root}, 'printf "%s\\n%s" "$TRACELENS_ROOT" "$_TRACELENS_SHA"'
    ).split("\n")

    assert resolved == str(tmp_path / "cache" / f"TraceLens@{sha}" if stale else root)


@pytest.mark.parametrize("stale", [True, False], ids=["checkouts-missing", "checkouts-present"])
def test_check_only_report_carries_the_kernel_tools_stale_keys(tmp_path: Path, stale: bool) -> None:
    """The kernel tools run in a subshell, yet the pins they find stale still reach the --check-only report."""
    ref = "1" * 40
    cache = tmp_path / "cache"
    script_dir = tmp_path / "assets"
    script_dir.mkdir()
    steps = sourceable_kernel_tools(script_dir)
    # Only the two steps that judge a pin; the rest install packages.
    steps.write_text(
        steps.read_text(encoding="utf-8") + "\ninstall_kernel_tools() { ensure_tracelens; ensure_geak; }\n",
        encoding="utf-8",
    )
    steps.rename(script_dir / "install_kernel_tools.sh")
    if not stale:
        tracelens_ref = re.search(r'^TRACELENS_REF="(\w+)"$', KERNEL_TOOLS_SH.read_text(encoding="utf-8"), re.M)
        (cache / f"TraceLens@{tracelens_ref.group(1)}" / ".git").mkdir(parents=True)
        (cache / f"GEAK@{ref}" / ".git").mkdir(parents=True)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"export REPO_ROOT={tmp_path} USER_DATA_PATH={tmp_path} HYPERLOOM_CACHE_DIR={cache} GEAK_REF={ref}",
            f"_script_dir={script_dir} CHECK_ONLY=1 DRY_RUN=0",
            "log() { :; }",
            '_STALE_KEYS=("MAGPIE_REF")',
            install_sh_function("install_kernel_agent_tools"),
            "install_kernel_agent_tools",
            'printf "stale: %s\\n" "${_STALE_KEYS[@]}"',
        ]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEAK_", "TRACELENS_"))}

    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)

    assert proc.returncode == 0, proc.stderr
    reported = [line.removeprefix("stale: ") for line in proc.stdout.splitlines() if line.startswith("stale: ")]
    assert reported == (["MAGPIE_REF", "TRACELENS_ROOT", "GEAK_ROOT"] if stale else ["MAGPIE_REF"])
