# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The seam for an agent run as a component of its host: the registry of adapters, a session without one, and the
launch a component agent is served, as a launch file or a base launch record."""

from collections.abc import Callable, Iterator
from dataclasses import replace
import json
from pathlib import Path
import sys
import types

import pytest

from hyperloom_multibranch import cli_options
from hyperloom_multibranch.blackboard.local.agent_manifest import REDACTED
from hyperloom_multibranch.scheduling.session_base import served_launch
from hyperloom_multibranch.specialist import component_agent
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.component_agent import (
    ENTRY_POINT_GROUP,
    LAUNCH_KEYS,
    AgentCheckout,
    AgentRefused,
    ComponentAgent,
    ComponentRun,
    SplitFailed,
    adapter_for,
    agent_of,
    eval_envs,
    installed,
    read_launch,
    served_config,
)
from hyperloom_multibranch.tests.agent_rig import _spec
from hyperloom_multibranch.tests.base_records import TOKEN, base_record, frozen_credential_base, write_record
from inference_suite.contracts import Redacted
from inference_suite.contracts import Workload as LaunchWorkload

MARKER = "stand-in.agent"
MODULE = "stand_in_adapters"


def _manifest(spec: AgentSpec, checkout: AgentCheckout, run: ComponentRun) -> AgentManifest:
    raise NotImplementedError("the stand-in is never made ready")


async def _generate(checkout: Path, profile: Path, out: Path) -> Path:
    raise SplitFailed("the stand-in generates no map")


async def _split(checkout: Path, map_file: Path, shares: int, out: Path) -> tuple[Path, ...]:
    raise SplitFailed("the stand-in splits no map")


def stand_in(name: str = "stand-in", envs: tuple[str, ...] = ("STAND_IN_EVAL",)) -> ComponentAgent:
    """An adapter for an agent whose checkout holds :data:`MARKER`."""

    return ComponentAgent(
        name=name,
        recognises=lambda checkout: (checkout / MARKER).is_file(),
        manifest=_manifest,
        fit_keys=lambda config, checkout: (dict(config), ()),
        eval_envs=envs,
        map_file="map.json",
        generate_directions=_generate,
        split_directions=_split,
        dossier_index="INDEX.md",
        pass_dossier=lambda checkout, session: None,
        dossier_notes=lambda checkout, dossier: (),
    )


class _EntryPoint:
    """An entry point as ``importlib.metadata`` gives one: its name, and the ``module:attribute`` it names."""

    def __init__(self, name: str, value: str) -> None:
        self.name, self.value = name, value


Declare = Callable[..., None]


@pytest.fixture
def declare(monkeypatch: pytest.MonkeyPatch) -> Iterator[Declare]:
    """Declare adapters as other packages' entry points do."""

    def _declare(*agents: object) -> None:
        module = types.ModuleType(MODULE)
        points = []
        for number, agent in enumerate(agents):
            setattr(module, f"AGENT_{number}", agent)
            points.append(_EntryPoint(getattr(agent, "name", f"declared-{number}"), f"{MODULE}:AGENT_{number}"))
        monkeypatch.setitem(sys.modules, MODULE, module)
        monkeypatch.setattr(
            component_agent.metadata,
            "entry_points",
            lambda group: points if group == ENTRY_POINT_GROUP else [],
        )
        installed.cache_clear()

    yield _declare
    installed.cache_clear()


def _checkout(root: Path, name: str, marked: bool) -> Path:
    path = root / name
    path.mkdir()
    if marked:
        (path / MARKER).write_text("", encoding="utf-8")
    return path


def test_another_package_s_adapter_is_found_by_its_entry_point(declare: Declare, tmp_path: Path) -> None:
    agent = stand_in()
    declare(agent)
    its, other = _checkout(tmp_path, "its", True), _checkout(tmp_path, "other", False)

    assert installed() == (agent,)
    assert adapter_for(its) is agent and agent_of(its) is agent
    assert adapter_for(other) is None, "any other agent takes the onboarding model's path"
    with pytest.raises(LookupError, match="no installed adapter recognises"):
        agent_of(other)
    assert eval_envs() == frozenset({"STAND_IN_EVAL"})


def test_adapters_are_listed_by_name(declare: Declare) -> None:
    declare(stand_in("zeta", ("Z",)), stand_in("alpha", ("A",)))

    assert [agent.name for agent in installed()] == ["alpha", "zeta"]
    assert eval_envs() == frozenset({"A", "Z"})


def test_a_declaration_that_names_no_adapter_is_refused(declare: Declare) -> None:
    declare(object())

    with pytest.raises(TypeError, match=r"declared-0 \(stand_in_adapters:AGENT_0\) names no ComponentAgent"):
        installed()


def test_with_no_adapter_no_agent_is_run_as_a_component(declare: Declare, tmp_path: Path) -> None:
    declare()
    checkout = _checkout(tmp_path, "agent", True)

    assert installed() == () and adapter_for(checkout) is None and eval_envs() == frozenset()
    (tmp_path / "INDEX.md").write_text("# a dossier\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is no dossier: it has no index an installed component agent knows"):
        cli_options._agent_dossier(tmp_path, tmp_path / "map.json")


def test_a_dossier_is_one_an_installed_adapter_knows(declare: Declare, tmp_path: Path) -> None:
    declare(stand_in())

    with pytest.raises(ValueError, match="is no dossier: it has no INDEX.md"):
        cli_options._agent_dossier(tmp_path, tmp_path / "map.json")
    (tmp_path / "INDEX.md").write_text("# a dossier\n", encoding="utf-8")
    assert cli_options._agent_dossier(tmp_path, tmp_path / "map.json") == str(tmp_path.absolute())


# ---- the launch a component agent is served ------------------------------------------------------------------------

RESOLVED = {
    "server_flags": "--mem-fraction-static 0.8 --chunked-prefill-size 16384",
    "server_env": {"SGLANG_SIMULATE_ACC_LEN": "3", "VLLM_ROCM_USE_AITER": "0"},
    "model_name": "Qwen/Qwen3.8-Flash-Next",
}
#: The repository's template of a launch file.
TEMPLATE = Path(__file__).parents[3] / "examples" / "multibranch-agent-launch" / "launch.template.json"


def _launch_file(tmp_path: Path, document: object) -> Path:
    """``document`` as a launch file, a string written as it is."""

    path = tmp_path / "launch.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("document", "said"),
    [
        ("{", "cannot be read"),
        (["--tp 1"], "is not a JSON object"),
        ({"server_env": {}}, "no server_flags"),
        ({**RESOLVED, "server_flags": "--chat-template 'a b"}, "a quote is left open"),
        ({**RESOLVED, "server_env": {"N": 1}}, "server_env is not an object of strings"),
        ({**RESOLVED, "flags": "--tp 1", "model_name": 7}, "keys a launch does not: flags; its model_name is not"),
        ({**RESOLVED, "client": ""}, "keys a launch does not: client"),
    ],
    ids=[
        "unreadable",
        "a-list",
        "no-flags",
        "open-quote",
        "env-not-strings",
        "unknown-key-and-name",
        "a-client-the-gate-gives",
    ],
)
def test_a_launch_file_that_cannot_be_used_is_refused_with_why(tmp_path: Path, document: object, said: str) -> None:
    with pytest.raises(AgentRefused) as refused:
        read_launch(_launch_file(tmp_path, document))

    assert said.split("; ")[0] in str(refused.value) and said.split("; ")[-1] in str(refused.value)


def test_the_repository_s_launch_template_has_a_launch_file_s_keys_and_is_refused_until_filled(tmp_path: Path) -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    filled = {**RESOLVED, "server_flags": f"{RESOLVED['server_flags']} --reasoning-start-str <think>"}

    with pytest.raises(AgentRefused) as refused:
        read_launch(TEMPLATE)

    assert set(template) == set(LAUNCH_KEYS)
    assert "server_flags, server_env, model_name still hold the template's placeholders" in str(refused.value)
    launch = read_launch(_launch_file(tmp_path, filled))
    assert (launch.server_flags, launch.server_env, launch.model_name) == (
        filled["server_flags"],
        RESOLVED["server_env"],
        RESOLVED["model_name"],
    )
    assert (launch.record, launch.digest, launch.entrypoint) == ("", "", ())


def test_a_base_launch_record_is_served_as_the_gate_seeds_its_launch_with_its_golden_layer(tmp_path: Path) -> None:
    record = write_record(tmp_path / "record" / "base_launch.json")

    launch = read_launch(record)

    assert launch.server_flags == (
        "--tensor-parallel-size 2 --max-num-seqs 128 --chat-template 'a b'"
        """ --speculative-config '{"method":"mtp","num_speculative_tokens":3}'"""
    )
    assert launch.server_env == {"VLLM_ROCM_USE_AITER": "1", "VLLM_SIMULATE_ACC": "3"}
    assert (launch.record, launch.digest, launch.entrypoint) == (str(record), base_record().digest(), ("vllm", "serve"))
    assert launch.model_name == ""


def test_a_base_launch_record_that_cannot_be_used_is_refused_with_why(tmp_path: Path) -> None:
    tampered = write_record(tmp_path / "tampered.json", args=["--max-num-seqs", "64"])
    scriptable = base_record(
        query=replace(base_record().query, workload=LaunchWorkload.SCRIPTABLE, concurrencies=(8,), tp=None),
        entrypoint=("bash", "bench.sh"),
        source=replace(base_record().source, files={"bench.sh": "c" * 64}),
        args=(),
        env={},
        golden_al=None,
        client=None,
    )
    script = write_record(tmp_path / "script.json", scriptable)

    with pytest.raises(AgentRefused, match="base_digest_mismatch"):
        read_launch(tampered)
    with pytest.raises(AgentRefused, match="runs a script and launches no server"):
        read_launch(script)


def test_a_frozen_base_s_credential_is_served_from_the_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = frozen_credential_base(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    launch = read_launch(Path(served_launch(frozen, "")))

    assert isinstance(frozen.launch.env["HF_TOKEN"], Redacted)
    assert launch.server_env["HF_TOKEN"] == TOKEN and launch.record == str(frozen.path)


@pytest.mark.parametrize(
    ("held", "said"),
    [
        (None, "the host holds no value of it"),
        ("hf_another_token", "the host's value is not the one the base was resolved with"),
    ],
    ids=["unset", "another-value"],
)
def test_a_frozen_base_s_credential_the_host_does_not_hold_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: str | None, said: str
) -> None:
    frozen = frozen_credential_base(tmp_path, monkeypatch)
    if held is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", held)

    with pytest.raises(AgentRefused, match=f"HF_TOKEN is redacted in the base and {said}"):
        read_launch(Path(served_launch(frozen, "")))


def test_a_stored_config_s_masked_credential_is_served_again_from_the_host_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = frozen_credential_base(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    spec = _spec(tmp_path, launch=served_launch(frozen, ""))
    stored = {"server_flags": "--tp 1", "server_env": {"HF_TOKEN": REDACTED, "OTHER": "1"}}

    assert served_config(stored, spec) == {"server_flags": "--tp 1", "server_env": {"HF_TOKEN": TOKEN, "OTHER": "1"}}
    assert served_config(stored, replace(spec, launch=None)) == stored
    assert served_config({"server_flags": "--tp 1", "server_env": {"OTHER": "1"}}, spec)["server_env"] == {"OTHER": "1"}
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(AgentRefused, match="HF_TOKEN is redacted in the base and the host holds no value of it"):
        served_config(stored, spec)
