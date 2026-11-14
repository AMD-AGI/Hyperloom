# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The seam for an agent run as a component of its host: the registry of adapters, and a session without one."""

from collections.abc import Callable, Iterator
from pathlib import Path
import sys
import types

import pytest

from hyperloom_multibranch import cli
from hyperloom_multibranch.specialist import component_agent
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.component_agent import (
    ENTRY_POINT_GROUP,
    AgentCheckout,
    ComponentAgent,
    ComponentRun,
    SplitFailed,
    adapter_for,
    agent_of,
    eval_envs,
    installed,
)

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
        cli._agent_dossier(tmp_path, tmp_path / "map.json")


def test_a_dossier_is_one_an_installed_adapter_knows(declare: Declare, tmp_path: Path) -> None:
    declare(stand_in())

    with pytest.raises(ValueError, match="is no dossier: it has no INDEX.md"):
        cli._agent_dossier(tmp_path, tmp_path / "map.json")
    (tmp_path / "INDEX.md").write_text("# a dossier\n", encoding="utf-8")
    assert cli._agent_dossier(tmp_path, tmp_path / "map.json") == str(tmp_path.absolute())
