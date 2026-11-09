# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loading a brain: by module name, through the ``brain`` plug socket, refused when it breaks the contract."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

from hyperloom.brain_api import CONTRACT_VERSION, BrainContractError, compatible, validate_brain
from hyperloom.brain_api.loader import load_brain
from hyperloom.plugs.builtin import builtin_registry
from hyperloom.plugs.host import PlugHost
from hyperloom.plugs.manifest import ManifestSources, resolve_manifest

_SRC = Path(__file__).resolve().parents[3]

_GOOD = """
from hyperloom.brain_api import Proposal, ProposeAsk, Review, ReviewAsk, eval_point

class Brain:
    name = "fake"
    contract_version = {version!r}

    def __init__(self, config):
        self.config = config

    @eval_point("main.propose")
    async def propose(self, state, ask: ProposeAsk, ports) -> Proposal:
        return Proposal()

    @eval_point("main.review")
    async def review(self, state, ask: ReviewAsk, ports) -> Review:
        return Review()

def create_brain(config):
    return Brain(config)
"""


@pytest.fixture
def brain_module(tmp_path, monkeypatch):
    """Write a brain module under a fresh name and return that name."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(source: str) -> str:
        name = f"fake_brain_{uuid.uuid4().hex}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
        return name

    return write


def test_the_default_module_loads_brain_v0() -> None:
    brain = load_brain()
    assert (brain.name, brain.contract_version) == ("v0", CONTRACT_VERSION)


def test_a_module_brain_is_built_with_its_config(brain_module) -> None:
    brain = load_brain(brain_module(_GOOD.format(version=CONTRACT_VERSION)), {"temperature": 0})
    assert brain.name == "fake"
    assert brain.config == {"temperature": 0}


@pytest.mark.parametrize(
    ("brain", "body", "ok"),
    [
        ("0.1", "0.1", True),
        ("0.1", "0.2", False),
        ("0.2", "0.1", False),
        ("1.0", "1.3", True),
        ("1.4", "1.3", False),
        ("2.0", "1.3", False),
        ("1", "1.0", False),
    ],
)
def test_before_1_0_a_brain_matches_the_minor_and_after_it_the_major(brain, body, ok) -> None:
    assert compatible(brain, body) is ok


@pytest.mark.parametrize("version", ["0.99", "99.0", "1", "one.two", None])
def test_a_brain_for_another_contract_is_refused(brain_module, version) -> None:
    with pytest.raises(BrainContractError, match="contract"):
        load_brain(brain_module(_GOOD.format(version=version)))


def test_a_brain_missing_an_async_call_site_is_refused(brain_module) -> None:
    source = _GOOD.format(version=CONTRACT_VERSION).replace('    @eval_point("main.review")\n', "")
    source = source.replace("async def review", "def review")
    with pytest.raises(BrainContractError, match="review"):
        load_brain(brain_module(source))


def test_a_brain_whose_main_methods_are_not_their_eval_points_is_refused(brain_module) -> None:
    source = _GOOD.format(version=CONTRACT_VERSION).replace('@eval_point("main.propose")', '@eval_point("x.propose")')
    with pytest.raises(BrainContractError, match=r"propose must be marked @eval_point\('main.propose'\)"):
        load_brain(brain_module(source))


def test_a_module_without_a_factory_or_that_cannot_import_is_refused(brain_module) -> None:
    with pytest.raises(BrainContractError, match="create_brain"):
        load_brain(brain_module("x = 1\n"))
    with pytest.raises(BrainContractError, match="cannot import"):
        load_brain("hyperloom_no_such_brain")
    with pytest.raises(BrainContractError, match="cannot import"):
        load_brain(brain_module("import hyperloom_no_such_dependency\n"))


def test_an_object_without_a_name_is_not_a_brain() -> None:
    with pytest.raises(BrainContractError, match="no name"):
        validate_brain(object())


def _host(*plugs: str, configs: tuple[str, ...] = ()) -> PlugHost:
    registry = builtin_registry()
    manifest = resolve_manifest(registry, ManifestSources(cli_plugs=plugs, cli_configs=configs))
    return PlugHost(manifest, registry, session_dir=None, env={})


def test_the_builtin_profile_starts_no_brain() -> None:
    host = _host()
    host.start()
    try:
        assert not host.has("brain")
    finally:
        host.close()


def test_the_brain_socket_starts_v0_or_the_module_its_config_names(brain_module) -> None:
    host = _host("brain=v0")
    host.start()
    try:
        assert host.get("brain").name == "v0"
    finally:
        host.close()
    module = brain_module(_GOOD.format(version=CONTRACT_VERSION))
    host = _host("brain=v0", configs=(f"brain.module={module}", "brain.depth=2"))
    host.start()
    try:
        brain = host.get("brain")
        assert (brain.name, brain.config) == ("fake", {"depth": 2})
    finally:
        host.close()


def test_loading_a_brain_pulls_in_no_body_module() -> None:
    code = (
        "import sys\n"
        "from hyperloom.brain_api.loader import load_brain\n"
        "load_brain()\n"
        "allowed = ('hyperloom.brain_api', 'hyperloom.brain', 'hyperloom.common')\n"
        "body = sorted(m for m in sys.modules if m.startswith('hyperloom.')"
        " and not any(m == a or m.startswith(a + '.') for a in allowed))\n"
        "print(body)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": str(_SRC)}, check=False
    )
    assert (out.returncode, out.stdout.strip()) == (0, "[]"), out.stderr
