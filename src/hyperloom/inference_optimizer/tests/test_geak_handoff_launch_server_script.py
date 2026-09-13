# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``launch_server_script`` the GEAK handoff advertises.

GEAK infers its server launcher from the recipe's ``benchmark_script``. That is
a real launcher on every non-AgentX run, but the AgentX switch pins the aiperf
client there instead -- so the inference elects a CLIENT as the server launcher,
which boots a server, replays the corpus, then tears the server down, leaving
GEAK's bench with no pid and an empty ``bench_runs.jsonl``.

These cover the resolver that closes that gap: it must name the builtin the
client itself delegates to, must stay silent on every recipe where GEAK's own
derivation is already right, and must fail open to silence rather than pin a
path that cannot work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.phases.kernel import KernelPhase


def _RESOLVE(recipe_path: str) -> str:
    """Resolve from a recipe on disk, the way the handoff builder does.

    The resolver takes the parsed ``benchmark`` mapping so the handoff can read
    the recipe once for both of its resolvers; these cases still start from a
    file, so they go through the same parse step.
    """
    return KernelPhase._resolve_launch_server_script(KernelPhase._recipe_benchmark(recipe_path))


def _checkout(tmp_path: Path, *scripts: str, lib: bool = True) -> Path:
    """Build an InferenceX-shaped checkout containing ``scripts``."""
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    if lib:
        (benchmarks / "benchmark_lib.sh").write_text("# stub\n", encoding="utf-8")
    for name in scripts:
        (benchmarks / name).write_text("# stub\n", encoding="utf-8")
    return benchmarks.parent


def _recipe(
    tmp_path: Path,
    *,
    benchmark_script: str,
    inferencex_path: str,
    framework: str = "vllm",
    runner_type: str = "mi355x",
    envs: dict | None = None,
) -> str:
    recipe = tmp_path / "baseline_config.with_envs.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "runner_type": runner_type,
                    "envs": envs if envs is not None else {"FRAMEWORK": framework},
                    "benchmark_script": benchmark_script,
                    "inferencex_path": inferencex_path,
                }
            }
        ),
        encoding="utf-8",
    )
    return str(recipe)


def test_agentx_recipe_names_the_builtin_the_client_delegates_to(tmp_path: Path) -> None:
    """The whole point: aiperf_client.sh in the recipe -> advertise vllm_mi355x.sh.

    ``runner_type`` supplies the GPU, matching the value Magpie projects into
    ``RUNNER_TYPE`` for the client's own ``{framework}_{gpu}.sh`` derivation.
    """
    root = _checkout(tmp_path, "vllm_mi355x.sh", "aiperf_client.sh")
    resolved = _RESOLVE(_recipe(tmp_path, benchmark_script="aiperf_client.sh", inferencex_path=str(root)))
    assert resolved == str(root / "benchmarks" / "vllm_mi355x.sh")


def test_a_non_agentx_recipe_is_left_to_geaks_own_derivation(tmp_path: Path) -> None:
    """A recipe naming a real launcher must return "".

    GEAK derives that same script itself, and its version is the one that
    provably launched the baseline -- including any operator-pinned variant we
    would silently replace with ``{framework}_{gpu}.sh`` if we answered here.
    """
    root = _checkout(tmp_path, "vllm_mi355x.sh")
    assert _RESOLVE(_recipe(tmp_path, benchmark_script="vllm_mi355x.sh", inferencex_path=str(root))) == ""


def test_a_pinned_custom_launcher_is_never_rewritten(tmp_path: Path) -> None:
    """An operator-pinned launcher must survive untouched.

    This is the fidelity regression the AgentX-only trigger exists to prevent:
    answering for every recipe would swap a deliberate pin for the derived name.
    """
    root = _checkout(tmp_path, "vllm_mi355x.sh", "vllm_custom_tuned.sh")
    assert _RESOLVE(_recipe(tmp_path, benchmark_script="vllm_custom_tuned.sh", inferencex_path=str(root))) == ""


def test_a_missing_builtin_stays_silent_instead_of_pinning_a_dead_path(tmp_path: Path) -> None:
    """No vllm_mi355x.sh on disk -> "" so GEAK degrades as it does today."""
    root = _checkout(tmp_path, "aiperf_client.sh")
    assert _RESOLVE(_recipe(tmp_path, benchmark_script="aiperf_client.sh", inferencex_path=str(root))) == ""


def test_a_checkout_without_benchmark_lib_stays_silent(tmp_path: Path) -> None:
    """The builtin sources benchmark_lib.sh from its own dir and dies without it,
    so a half-populated checkout must not be advertised."""
    root = _checkout(tmp_path, "vllm_mi355x.sh", "aiperf_client.sh", lib=False)
    assert _RESOLVE(_recipe(tmp_path, benchmark_script="aiperf_client.sh", inferencex_path=str(root))) == ""


def test_agentx_server_script_override_is_honoured(tmp_path: Path) -> None:
    """AGENTX_SERVER_SCRIPT overrides the derived name for the client, so it has
    to override ours too or we would advertise a server it never booted."""
    root = _checkout(tmp_path, "vllm_special.sh", "aiperf_client.sh")
    resolved = _RESOLVE(
        _recipe(
            tmp_path,
            benchmark_script="aiperf_client.sh",
            inferencex_path=str(root),
            envs={"FRAMEWORK": "vllm", "AGENTX_SERVER_SCRIPT": "vllm_special.sh"},
        )
    )
    assert resolved == str(root / "benchmarks" / "vllm_special.sh")


def test_recipe_gpu_type_env_beats_runner_type(tmp_path: Path) -> None:
    """Mirrors the client's ``${GPU_TYPE:-${RUNNER_TYPE:-mi300x}}`` precedence."""
    root = _checkout(tmp_path, "vllm_mi300x.sh", "vllm_mi355x.sh", "aiperf_client.sh")
    resolved = _RESOLVE(
        _recipe(
            tmp_path,
            benchmark_script="aiperf_client.sh",
            inferencex_path=str(root),
            runner_type="mi355x",
            envs={"FRAMEWORK": "vllm", "GPU_TYPE": "mi300x"},
        )
    )
    assert resolved == str(root / "benchmarks" / "vllm_mi300x.sh")


def test_falls_back_to_the_inferencex_path_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe's checkout is container-local and content-hash addressed, so a
    post-mortem or rebuilt box resolves against $INFERENCEX_PATH instead."""
    root = _checkout(tmp_path, "vllm_mi355x.sh", "aiperf_client.sh")
    monkeypatch.setenv("INFERENCEX_PATH", str(root))
    resolved = _RESOLVE(
        _recipe(
            tmp_path,
            benchmark_script="aiperf_client.sh",
            inferencex_path="/gone/InferenceX@deadbeef",
        )
    )
    assert resolved == str(root / "benchmarks" / "vllm_mi355x.sh")


def test_a_framework_less_agentx_recipe_stays_silent(tmp_path: Path) -> None:
    """Without a framework there is no builtin name to derive, and guessing one
    would boot the wrong server."""
    root = _checkout(tmp_path, "vllm_mi355x.sh", "aiperf_client.sh")
    assert (
        _RESOLVE(
            _recipe(
                tmp_path,
                benchmark_script="aiperf_client.sh",
                inferencex_path=str(root),
                framework="",
                envs={},
            )
        )
        == ""
    )


@pytest.mark.parametrize("body", ["", "not: [a, yaml", "just-a-string"])
def test_an_unreadable_recipe_never_raises(tmp_path: Path, body: str) -> None:
    """The handoff must never fail to be written because of this field."""
    recipe = tmp_path / "broken.yaml"
    recipe.write_text(body, encoding="utf-8")
    assert _RESOLVE(str(recipe)) == ""


def test_a_missing_recipe_path_stays_silent(tmp_path: Path) -> None:
    assert _RESOLVE("") == ""
    assert _RESOLVE(str(tmp_path / "nope.yaml")) == ""
