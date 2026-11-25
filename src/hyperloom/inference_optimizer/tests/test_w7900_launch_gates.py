# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A launch the visible board cannot serve stops with exit 2 before any install, model resolution or session dir."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hyperloom.common import provenance
from hyperloom_common.gpu_identity import GpuAgent
from hyperloom.inference_optimizer import cli, gpu_types
from hyperloom.inference_optimizer.cli import launch_gpu, launch_shape, preflight
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.gpu_types import gpu_type_conflict, product_launch_error
from hyperloom.inference_optimizer.session.paths import ENV_CURRENT_SESSION_DIR
from hyperloom.inference_optimizer.session.sbd_v6 import pending_install_event

from .conftest import cli_call_sequence

W7900_DS = GpuAgent("gfx1100", "AMD Radeon Pro W7900 Dual Slot", "0x744a", 96)
W7900 = GpuAgent("gfx1100", "AMD Radeon Pro W7900", "0x7448", 96)
#: rocminfo prints no Marketing Name for a card whose name libdrm cannot read; the chip ID still names it.
W7900_DS_CHIP_ONLY = GpuAgent("gfx1100", "", "0x744a", 96)
W7900_CHIP_ONLY = GpuAgent("gfx1100", "", "0x7448", 96)
W7900D = GpuAgent("gfx1100", "AMD Radeon PRO W7900D", "0x744b", 96)
W7800 = GpuAgent("gfx1100", "AMD Radeon PRO W7800", "", 70)
RX_7900_XTX = GpuAgent("gfx1100", "AMD Radeon RX 7900 XTX", "", 96)
RX_7900_XTX_CHIP_ONLY = GpuAgent("gfx1100", "", "0x744c", 96)
W7800_CHIP_ONLY = GpuAgent("gfx1100", "", "0x745e", 70)
RX_7800_XT = GpuAgent("gfx1101", "AMD Radeon RX 7800 XT", "", 60)
RX_7600 = GpuAgent("gfx1102", "AMD Radeon RX 7600", "", 32)
R9700 = GpuAgent("gfx1201", "AMD Radeon AI PRO R9700", "", 64)
MI300X = GpuAgent("gfx942", "AMD Instinct MI300X", "", 304)
MI355X = GpuAgent("gfx950", "AMD Instinct MI355X", "", 256)

#: Taken before the fixtures stub them, for the test that counts the launch's real rocminfo runs.
_REAL_PROBE_GPUS = provenance.probe_gpus
_REAL_AUTODETECT = gpu_types._autodetect_gpu_type
_ROCMINFO_4X_W7900 = Path(__file__).parent / "fixtures" / "rocminfo_4x_w7900_dual_slot.txt"

_W7900_IDS = "w7900 (PCI 1002:7448 or 1002:744a)"
_SESSION = "the session's GPU type"
_MASK_HINT = (
    "If the host has other GPUs, give the container only the /dev/dri render nodes of the GPUs this session should "
    "use, or export ROCR_VISIBLE_DEVICES with their indices in the shell that launches Hyperloom (a value in .env is "
    "ignored)."
)


def _error(gpu_type, agents, *, framework="vllm", resumed=False, remote=False):
    probed = None if remote else gpu_types.probed_gpu_type(agents)
    return product_launch_error(gpu_type, agents, framework=framework, probed=probed, resumed=resumed)


# --- the decision ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("agent", [GpuAgent("gfx1100"), RX_7900_XTX, W7800, W7800_CHIP_ONLY])
def test_a_bare_gfx1100_with_no_gpu_type_is_refused(agent):
    error = _error(None, [agent])
    assert error is not None
    assert "a bare gfx1100 is not product evidence" in error
    assert "Supported gfx1100 products: w7900, w7900d." in error


@pytest.mark.parametrize("agent", [RX_7800_XT, RX_7600])
def test_a_navi3_isa_with_no_validated_product_is_refused(agent):
    error = _error(None, [agent])
    assert error is not None
    assert f"a bare {agent.gfx} is not product evidence. No {agent.gfx} product is validated." in error


@pytest.mark.parametrize("agent", [RX_7900_XTX, W7800])
def test_w7900_on_another_gfx1100_board_is_refused(agent):
    assert _error("w7900", [agent]) == (
        f"--gpu-type w7900 is {_W7900_IDS}, but the visible GPU is {agent.marketing_name!r}. {_MASK_HINT}"
    )


@pytest.mark.parametrize("name", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"])
def test_a_device_mask_in_dotenv_is_ignored_as_the_refusals_say(name):
    from hyperloom.common.env_safety import is_allowed_dotenv_key

    assert not is_allowed_dotenv_key(name)


@pytest.mark.parametrize("agent", [RX_7800_XT, RX_7600])
def test_w7900_on_another_navi3_isa_is_refused(agent):
    error = _error("w7900", [agent])
    assert error is not None
    assert error.startswith(
        f"--gpu-type w7900 dispatches to gfx1100, but the visible GPU {agent.marketing_name!r} reports {agent.gfx}"
    )
    assert error.endswith(_MASK_HINT)


@pytest.mark.parametrize("agent", [GpuAgent("gfx1201"), GpuAgent("gfx942"), GpuAgent("gfx1151")])
def test_w7900_on_an_unnamed_board_of_another_isa_is_refused(agent):
    assert _error("w7900", [agent]) == (
        f"--gpu-type w7900 dispatches to gfx1100, but the visible GPU reports {agent.gfx}. {_MASK_HINT}"
    )


_W7900_PCI = "--gpu-type w7900 is PCI 1002:7448 or 1002:744a, but the visible GPU"


@pytest.mark.parametrize(
    ("gpu_type", "agents", "message"),
    [
        ("w7900", [RX_7900_XTX_CHIP_ONLY], f"{_W7900_PCI} has PCI ID 1002:744c."),
        ("w7900", [W7800_CHIP_ONLY], f"{_W7900_PCI} has PCI ID 1002:745e."),
        (
            "w7900",
            [GpuAgent("gfx1100", "AMD Radeon Pro W7900", "0x744c", 96)],
            f"{_W7900_PCI} 'AMD Radeon Pro W7900' has PCI ID 1002:744c.",
        ),
        (
            None,
            [GpuAgent("gfx1100", "AMD Radeon RX 7900 XTX", "0x744a", 96)],
            f"the probed board w7900 is {_W7900_IDS}, but the visible GPU is 'AMD Radeon RX 7900 XTX'.",
        ),
        (
            "w7900",
            [GpuAgent("gfx1100", "", "", 96)],
            "--gpu-type w7900 is identified by its nameplate or PCI ID, and rocminfo prints neither for the visible "
            "gfx1100 GPU.",
        ),
        ("w7900", [W7900_DS, RX_7900_XTX_CHIP_ONLY], f"{_W7900_PCI} has PCI ID 1002:744c."),
        (
            None,
            [RX_7900_XTX_CHIP_ONLY],
            "no supported product was identified; the visible GPU 1002:744c reports gfx1100, and a bare gfx1100 is "
            "not product evidence.",
        ),
    ],
    ids=[
        "no-nameplate-rx-7900-chip",
        "no-nameplate-w7800-chip",
        "w7900-nameplate-other-chip",
        "other-nameplate-w7900-chip",
        "neither",
        "mixed-host-chip-only",
        "probed-chip-only-rx-7900",
    ],
)
def test_w7900_needs_its_nameplate_or_pci_id_and_neither_may_disagree(gpu_type, agents, message):
    error = _error(gpu_type, agents)
    assert error is not None
    assert error.startswith(message)
    assert error.endswith(_MASK_HINT)


@pytest.mark.parametrize(
    ("agents", "message"),
    [
        (
            None,
            "but rocminfo could not run, so the card is unconfirmed. Put rocminfo on PATH and give the session the "
            "card's /dev/kfd and /dev/dri nodes.",
        ),
        (
            [],
            "but rocminfo lists no GPU, so the card is unconfirmed. Check that ROCR_VISIBLE_DEVICES in the launching "
            "shell does not hide the card, and give the session the card's /dev/kfd and /dev/dri nodes.",
        ),
    ],
    ids=["no-rocminfo", "no-gpu"],
)
def test_a_local_w7900_launch_rocminfo_cannot_show_is_refused(agents, message):
    error = _error("w7900", agents)
    assert error == f"--gpu-type w7900 is identified by its nameplate or PCI ID, {message}"
    assert _error("w7900", agents, remote=True) is None


@pytest.mark.parametrize(
    ("gpu_type", "agent"),
    [("w7900", R9700), ("w7900", MI300X), ("w7900", W7900D), ("r9700", W7900_DS), ("w7900d", W7900_DS)],
)
def test_a_gpu_type_the_nameplate_contradicts_is_refused(gpu_type, agent):
    error = _error(gpu_type, [agent])
    assert error is not None
    assert error.startswith(f"--gpu-type {gpu_type} disagrees with the probed ")
    assert _W7900_IDS in error


@pytest.mark.parametrize(
    ("gpu_type", "agent", "message"),
    [
        ("w7900", R9700, f"this session was created for w7900, but the probed board is r9700; a {_W7900_IDS} "),
        ("r9700", W7900_DS, f"this session was created for r9700, but the probed board is w7900; a {_W7900_IDS} "),
        ("w7900", RX_7900_XTX, f"{_SESSION} w7900 is {_W7900_IDS}, but the visible GPU is "),
        (
            "w7900",
            RX_7800_XT,
            f"{_SESSION} w7900 dispatches to gfx1100, but the visible GPU 'AMD Radeon RX 7800 XT' reports gfx1101",
        ),
    ],
)
def test_a_resumed_session_is_refused_for_its_own_gpu_type_not_a_flag(gpu_type, agent, message):
    error = _error(gpu_type, [agent], resumed=True)
    assert error is not None
    assert error.startswith(message)
    assert "--gpu-type" not in error


@pytest.mark.parametrize(
    ("gpu_type", "agents"),
    [
        ("mi300x", [GpuAgent("gfx1100")]),
        ("radeon8060s", [GpuAgent("gfx1100")]),
        ("mi300x", [MI300X, RX_7900_XTX]),
        (None, [MI300X, RX_7900_XTX]),
        (None, [R9700, RX_7600]),
        ("r9700", [R9700, RX_7800_XT]),
    ],
    ids=["mi300x-flag", "radeon8060s-flag", "mi300x-rx-7900-flag", "mi300x-rx-7900", "r9700-rx-7600", "r9700-rx-7800"],
)
def test_a_launch_on_a_board_off_gfx110x_does_not_judge_an_unregistered_gfx110x_card(gpu_type, agents):
    """As on ``main``: an MI, R9700 or Strix Halo launch next to a display card it will not use."""
    assert _error(gpu_type, agents) is None


def test_every_visible_card_must_be_the_board():
    assert _error("w7900", [W7900_DS, W7900]) is None
    for gpu_type in ("w7900", None):
        error = _error(gpu_type, [W7900_DS, RX_7900_XTX])
        assert error is not None
        assert "'AMD Radeon RX 7900 XTX'" in error
        assert error.endswith(_MASK_HINT)


@pytest.mark.parametrize(
    ("gpu_type", "agents", "message"),
    [
        (
            None,
            [GpuAgent("gfx942"), RX_7900_XTX],
            "no supported product was identified; the visible GPU 'AMD Radeon RX 7900 XTX' reports gfx1100",
        ),
        (
            "w7900d",
            [W7900D, RX_7800_XT],
            "--gpu-type w7900d dispatches to gfx1100, but the visible GPU 'AMD Radeon RX 7800 XT' reports gfx1101",
        ),
    ],
    ids=["no-board", "gfx110x-board-of-another-isa"],
)
def test_a_visible_gfx110x_card_on_a_mixed_host_points_to_rocr_visible_devices(gpu_type, agents, message):
    error = _error(gpu_type, agents)
    assert error is not None
    assert error.startswith(message)
    assert error.endswith(_MASK_HINT)


@pytest.mark.parametrize(
    ("gpu_type", "agents", "named"),
    [
        (None, [W7900D, W7900], "the probed board w7900d"),
        ("w7900d", [W7900D, W7900_DS], "--gpu-type w7900d"),
        (None, [MI355X, W7900_DS], "the probed board mi355x"),
        ("mi300x", [MI300X, W7900_CHIP_ONLY], "--gpu-type mi300x"),
    ],
    ids=["w7900d-first", "w7900d-flag", "mi355x-first", "mi300x-flag-chip-only"],
)
def test_a_w7900_next_to_the_board_a_launch_runs_as_is_refused(gpu_type, agents, named):
    """Whichever card rocminfo lists first, a W7900 is never driven as another board's sibling."""
    error = _error(gpu_type, agents)
    assert error is not None
    assert error.endswith(
        f" is {_W7900_IDS}, which a launch confirms on the card, but {named} is another board. {_MASK_HINT}"
    )


def test_a_w7900_listed_first_confirms_every_card_after_it():
    error = _error(None, [W7900, W7900D])
    assert error is not None
    assert error.startswith(f"the probed board w7900 is {_W7900_IDS}, but the visible GPU is 'AMD Radeon PRO W7900D'")


def test_the_board_the_launch_runs_as_is_the_one_whose_frameworks_are_checked():
    """Probe-wins: the flag only warns, so the probed board's framework list applies, not the flag's."""
    assert _error("r9700", [MI300X], framework="sglang") is None
    assert _error("w7900d", [MI355X], framework="sglang") is None
    assert _error("mi300x", [R9700], framework="sglang") == (
        "r9700 is currently validated only for llamacpp, vllm; framework 'sglang' has not been validated on it"
    )


@pytest.mark.parametrize("framework", ["sglang", "atom"])
def test_w7900_refuses_a_framework_it_is_not_validated_for(framework):
    assert _error("w7900", [W7900_DS], framework=framework) == (
        f"w7900 is currently validated only for vllm; framework {framework!r} has not been validated on it"
    )


@pytest.mark.parametrize(
    ("gpu_type", "agents"),
    [
        ("w7900", [W7900_DS]),
        ("W7900", [W7900]),
        ("w7900", [W7900_DS] * 4),
        (None, [W7900_DS]),
        ("", [W7900]),
        ("w7900", [W7900_DS_CHIP_ONLY]),
        ("w7900", [W7900_CHIP_ONLY]),
        (None, [W7900_DS_CHIP_ONLY, W7900_DS]),
    ],
    ids=["dual-slot", "blower", "four-cards", "probed", "probed-empty-flag", "chip-744a", "chip-7448", "probed-chip"],
)
def test_a_w7900_vllm_launch_is_accepted(gpu_type, agents):
    assert _error(gpu_type, agents) is None


def test_a_remote_w7900_launch_checks_only_the_flags():
    assert _error("w7900", None, remote=True) is None


@pytest.mark.parametrize(
    ("gpu_type", "agents"),
    [
        ("w7900d", [W7900D]),
        (None, [W7900D]),
        ("w7900d", None),
        ("w7900d", []),
        ("w7900d", [GpuAgent("gfx1100")]),
        ("w7900d", [R9700]),
        ("r9700", [W7900D]),
    ],
    ids=["named", "probed", "no-rocminfo", "no-gpu", "bare-gfx1100", "probe-wins-r9700", "probe-wins-w7900d"],
)
def test_w7900d_keeps_the_upstream_contract(gpu_type, agents):
    """The W7900D is not confirmed on the card: a flag the probe contradicts is resolved by the probe, not refused."""
    assert _error(gpu_type, agents) is None


def test_w7900d_refuses_a_framework_it_is_not_validated_for():
    assert _error("w7900d", [W7900D], framework="sglang") == (
        "w7900d is currently validated only for llamacpp, vllm; framework 'sglang' has not been validated on it"
    )


@pytest.mark.parametrize(
    ("gpu_type", "agents"),
    [
        ("r9700", [R9700]),
        ("mi300x", [MI300X]),
        ("mi300x", [MI355X]),
        ("r9700", [GpuAgent("gfx942")]),
        (None, [MI355X]),
        (None, [R9700]),
        (None, None),
        ("mi355x", None),
        ("r9700", None),
        ("r9700", []),
    ],
)
def test_other_boards_keep_their_contract(gpu_type, agents):
    assert _error(gpu_type, agents) is None


@pytest.mark.parametrize(
    ("user", "probed", "refused"),
    [
        ("w7900", "r9700", True),
        ("r9700", "w7900", True),
        ("mi300x", "w7900", True),
        ("w7900", "w7900d", True),
        ("w7900d", "w7900", True),
        ("w7900", "w7900", False),
        ("w7900", "", False),
        ("", "w7900", False),
        ("w7900d", "r9700", False),
        ("r9700", "w7900d", False),
        ("mi300x", "mi355x", False),
        ("r9700", "radeon8060s", False),
    ],
)
def test_only_a_w7900_on_either_side_turns_a_probe_disagreement_into_a_refusal(user, probed, refused):
    assert (gpu_type_conflict(user, probed) is not None) is refused


# --- preflight: exit 2 before anything runs -------------------------------------------------------------------------


class PreflightReachedCredentials(Exception):
    """The launch got past every GPU check; stop before any external operation."""


class Launch:
    """One ``optimize`` launch on a host whose rocminfo shows ``agents``; model resolution fails the test."""

    def __init__(self, monkeypatch, sessions: Path, models: Mock) -> None:
        self._monkeypatch = monkeypatch
        self.sessions = sessions
        self.models = models
        self.args = None

    def _parse(self, argv, agents, rocminfo=""):
        probe = None if agents is None else provenance.GpuProbe(rocminfo, list(agents))
        self._monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: probe)
        self.args = cli._build_parser().parse_args(["optimize", *argv])

    def run(self, argv, agents, rocminfo="") -> None:
        """``_run_optimize``: a refusal must leave no session, not even a preflight-failure one."""
        self._parse(argv, agents, rocminfo)
        with patch.dict(os.environ):
            try:
                asyncio.run(cli._run_optimize(self.args))
            finally:
                assert ENV_CURRENT_SESSION_DIR not in os.environ

    def preflight(self, argv, agents, rocminfo="") -> dict:
        """``_preflight`` up to the credentials check, and the check_gpu_product step it recorded."""
        self._parse(argv, agents, rocminfo)
        with patch.dict(os.environ), pytest.raises(PreflightReachedCredentials):
            preflight._preflight(self.args)
        return self.gpu_step()

    def step_ids(self) -> list[str]:
        return [step["step_id"] for step in pending_install_event(self.args)["ext"]["steps"]]

    def step(self, step_id: str) -> dict | None:
        steps = pending_install_event(self.args)["ext"]["steps"]
        return next((step for step in steps if step["step_id"] == step_id), None)

    def gpu_step(self) -> dict:
        step = self.step("check_gpu_product")
        assert step is not None
        return step

    def left_no_session(self) -> bool:
        return not self.sessions.exists() or not any(self.sessions.iterdir())


def _geak_root(parent: Path, identity_helper: str | None) -> Path:
    """A GEAK checkout as ``resolve_e2e_runner`` finds one, whose ``scripts/gpu_identity.py`` is ``identity_helper``."""
    root = parent / "GEAK"
    (root / "interface").mkdir(parents=True)
    (root / "interface" / "run_e2e.py").write_text("", encoding="utf-8")
    if identity_helper is not None:
        (root / "scripts").mkdir()
        (root / "scripts" / "gpu_identity.py").write_text(identity_helper, encoding="utf-8")
    return root


def _use_geak(monkeypatch, root: Path) -> Path:
    """Point the launch at GEAK checkout ``root`` as install.sh does: GEAK_ROOT and the runner inside it."""
    monkeypatch.setenv("GEAK_ROOT", str(root))
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(root / "interface" / "run_e2e.py"))
    return root


def _prints_identity(target: str) -> str:
    """An identity helper that keeps what it read on stdin next to itself and prints ``target`` on gfx1100."""
    return (
        "import json, pathlib, sys\n"
        "pathlib.Path(__file__).with_name('stdin.txt').write_text(sys.stdin.read())\n"
        f"print(json.dumps({{'gfx': 'gfx1100', 'target': {target!r}, 'visible_gpu_agents': 1}}))\n"
    )


@pytest.fixture
def launch(tmp_path, monkeypatch) -> Launch:
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("USER_DATA_PATH", str(sessions))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)
    for name in ("GPU_TYPE", "TARGET_GPU_TYPE", "FRAMEWORK", "MAX_MODEL_LEN"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HYPERLOOM_CACHE_DIR", "KERNEL_OPT_BACKEND_ORDER"):
        monkeypatch.delenv(name, raising=False)
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-w7900", _prints_identity("w7900")))
    # rocm-smi and torch would answer for the host's own GPU; the board comes from the stubbed rocminfo alone.
    monkeypatch.setattr(
        gpu_types, "_autodetect_gpu_type", lambda agents=None: gpu_types.probed_gpu_type(agents) or None
    )
    for name in ("_load_dotenv_fallback", "_load_kernel_agent_env_fallback", "_derive_runtime_paths"):
        monkeypatch.setattr(preflight, name, lambda: None)
    monkeypatch.setattr(preflight, "_normalize_legacy_deepseek_env", dict)
    monkeypatch.setattr(preflight, "_expand_custom_header_env_refs", lambda: None)
    monkeypatch.setattr(preflight, "_validate_credentials", Mock(side_effect=PreflightReachedCredentials))
    monkeypatch.setattr(
        cli, "clean_stale_aiter_locks", lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0}
    )
    models = Mock(side_effect=AssertionError("model resolution ran"))
    monkeypatch.setattr(cli, "_resolve_models_for_run", models)
    return Launch(monkeypatch, sessions, models)


_FRESH = ["--model", "/models/Qwen3.5-9B", "--framework", "vllm"]
_W7900_VLLM = [*_FRESH, "--gpu-type", "w7900"]


@pytest.mark.parametrize(
    ("argv", "agents", "message"),
    [
        (_FRESH, [GpuAgent("gfx1100")], "a bare gfx1100 is not product evidence"),
        (_FRESH, [RX_7900_XTX], "a bare gfx1100 is not product evidence"),
        ([*_FRESH, "--no-kernel"], [W7800], "a bare gfx1100 is not product evidence"),
        (_W7900_VLLM, [RX_7900_XTX], "but the visible GPU is 'AMD Radeon RX 7900 XTX'"),
        (_W7900_VLLM, [R9700], "disagrees with the probed r9700"),
        (_W7900_VLLM, [W7900D], "disagrees with the probed w7900d"),
        ([*_FRESH, "--gpu-type", "r9700"], [W7900_DS], "disagrees with the probed w7900"),
        ([*_FRESH, "--gpu-type", "w7900d"], [W7900_DS], "disagrees with the probed w7900"),
        (_W7900_VLLM, [GpuAgent("gfx1201")], "but the visible GPU reports gfx1201"),
        (
            ["--model", "/m", "--framework", "sglang", "--gpu-type", "w7900"],
            [W7900_DS],
            "w7900 is currently validated only for vllm",
        ),
        (_W7900_VLLM, None, "but rocminfo could not run"),
        ([*_W7900_VLLM, "--no-kernel"], [], "but rocminfo lists no GPU"),
        ([*_W7900_VLLM, "--no-kernel"], [RX_7900_XTX_CHIP_ONLY], "but the visible GPU has PCI ID 1002:744c"),
        ([*_W7900_VLLM, "--no-kernel"], [W7900_DS, RX_7900_XTX], f"'AMD Radeon RX 7900 XTX'. {_MASK_HINT}"),
        (_FRESH, [W7900D, W7900], "which a launch confirms on the card, but the probed board w7900d is another board"),
        (_FRESH, [W7900, W7900D], "but the visible GPU is 'AMD Radeon PRO W7900D'"),
    ],
    ids=[
        "bare-isa",
        "rx-7900-xtx",
        "w7800",
        "w7900-on-rx-7900-xtx",
        "w7900-on-r9700",
        "w7900-on-w7900d",
        "r9700-on-w7900",
        "w7900d-on-w7900",
        "w7900-on-unnamed-gfx1201",
        "sglang",
        "no-rocminfo",
        "no-gpu",
        "no-nameplate-other-chip",
        "mixed-host",
        "w7900d-then-w7900",
        "w7900-then-w7900d",
    ],
)
def test_a_refused_fresh_launch_exits_2_before_model_resolution_or_a_session_dir(launch, capsys, argv, agents, message):
    with pytest.raises(SystemExit) as exc:
        launch.run(argv, agents)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and message in err
    launch.models.assert_not_called()
    preflight._validate_credentials.assert_not_called()
    assert launch.left_no_session()
    step = launch.gpu_step()
    assert (step["category"], step["status"], step["detail"]) == ("check", "failed", {"exit_code": 2})
    assert pending_install_event(launch.args)["ext"]["hard_fail_step_id"] == "check_gpu_product"


@pytest.mark.parametrize(
    ("agents", "seen"), [(None, "could not run"), ([], "lists no GPU")], ids=["no-rocminfo", "no-gpu"]
)
def test_a_w7900_only_rocm_smi_names_must_still_be_confirmed(launch, monkeypatch, capsys, agents, seen):
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda agents=None: "w7900")
    with pytest.raises(SystemExit) as exc:
        launch.run(_FRESH, agents)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith(
        f"ERROR: the probed board w7900 is identified by its nameplate or PCI ID, but rocminfo {seen}, "
    )
    launch.models.assert_not_called()
    assert launch.left_no_session()


def test_runtime_paths_are_derived_before_the_launch_runs_rocminfo(launch, monkeypatch):
    """``$ROCM_PATH/bin`` reaches PATH first, so a rocminfo installed only there is found."""
    order: list[str] = []
    monkeypatch.setattr(preflight, "_derive_runtime_paths", lambda: order.append("derive_runtime_paths"))

    def probe(env=None):
        order.append("rocminfo")
        return provenance.GpuProbe("", [W7900_DS])

    monkeypatch.setattr(provenance, "probe_gpus", probe)
    args = cli._build_parser().parse_args(["optimize", *_W7900_VLLM, "--no-kernel"])
    with patch.dict(os.environ), pytest.raises(PreflightReachedCredentials):
        preflight._preflight(args)
    assert order.index("derive_runtime_paths") < order.index("rocminfo")


@pytest.mark.parametrize(
    ("argv", "agents", "gpu_type"),
    [(["--gpu-type", "r9700"], [MI300X], "mi300x"), (["--gpu-type", "w7900d"], [MI355X], "mi355x")],
    ids=["r9700-flag-on-mi300x", "w7900d-flag-on-mi355x"],
)
def test_a_flag_the_probe_overrides_is_checked_as_the_probed_board(monkeypatch, argv, agents, gpu_type):
    """With the default framework, which the flag's board has not validated and the probed one has."""
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: provenance.GpuProbe("", list(agents)))
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda agents=None: None)
    monkeypatch.delenv("FRAMEWORK", raising=False)
    args = cli._build_parser().parse_args(["optimize", "--model", "/m", *argv])
    assert launch_gpu._check_gpu_product(args)["detail"]["gpu_type"] == gpu_type


def test_the_launch_board_is_what_the_masked_rocminfo_run_names_before_rocm_smi(monkeypatch):
    """rocm-smi lists every card on the host whatever ROCR_VISIBLE_DEVICES says; rocminfo lists the session's."""
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: provenance.GpuProbe("", [W7900D]))
    rocm_smi = Mock(return_value="w7900")
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", rocm_smi)
    args = argparse.Namespace(nodes=1)
    assert launch_gpu.launch_board(args) == "w7900d"
    rocm_smi.assert_not_called()
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: None)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", f"{os.environ.get('ROCR_VISIBLE_DEVICES', '')}7")
    assert launch_gpu.launch_board(args) == launch_gpu.launch_board(args) == "w7900"
    rocm_smi.assert_called_once_with(agents=None)


_FIRST_STEPS = ["load_dotenv", "load_kernel_agent_env", "validate_server_args", "check_gpu_product"]


def test_the_gate_runs_before_any_install_step(launch):
    with pytest.raises(SystemExit):
        launch.run(_W7900_VLLM, [RX_7900_XTX])
    assert launch.step_ids() == _FIRST_STEPS
    launch.preflight(_W7900_VLLM, [W7900_DS])
    assert launch.step_ids()[:6] == [*_FIRST_STEPS, "check_geak_target", "normalize_legacy_deepseek_env"]


# --- the kernel phase: GEAK must name the board it is handed --------------------------------------------------------


def _refused_by_geak(launch, capsys, *, argv=_W7900_VLLM, agents=(W7900_DS,), rocminfo="") -> str:
    with pytest.raises(SystemExit) as exc:
        launch.run(argv, list(agents), rocminfo)
    assert exc.value.code == 2
    launch.models.assert_not_called()
    preflight._validate_credentials.assert_not_called()
    assert launch.left_no_session()
    assert pending_install_event(launch.args)["ext"]["hard_fail_step_id"] == "check_geak_target"
    err = capsys.readouterr().err
    assert err.startswith("ERROR: the kernel phase hands w7900 to GEAK, which checks the card at every handoff, but ")
    assert "Re-run install.sh so it fetches a GEAK whose scripts/gpu_identity.py names the card w7900" in err
    assert "set GEAK_E2E_RUNNER to such a checkout's interface/run_e2e.py (it takes precedence over GEAK_ROOT)" in err
    assert err.rstrip().endswith("or pass --no-kernel. Refusing to start.")
    return err


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        (_prints_identity("unknown"), "identifies this GPU as unknown on gfx1100"),
        (
            "import sys\nprint('E_GPU_IDENTITY: rocminfo reported no non-gfx000 GPU agent', file=sys.stderr)\n"
            "raise SystemExit(2)\n",
            "gpu_identity.py exited 2: E_GPU_IDENTITY: rocminfo reported no non-gfx000 GPU agent",
        ),
        ("print('not json')\n", "gpu_identity.py printed no identity"),
        (None, "has no scripts/gpu_identity.py"),
    ],
    ids=["geak-without-w7900-unknown", "helper-exits-2", "helper-prints-no-json", "geak-without-the-helper"],
)
def test_a_kernel_phase_whose_geak_does_not_identify_the_w7900_exits_2(
    launch, tmp_path, monkeypatch, capsys, helper, message
):
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-under-test", helper))
    assert message in _refused_by_geak(launch, capsys)


def test_a_kernel_phase_with_no_geak_checkout_exits_2(launch, tmp_path, monkeypatch, capsys):
    _use_geak(monkeypatch, tmp_path / "absent")
    assert "no GEAK checkout was found (GEAK_E2E_RUNNER, GEAK_ROOT, HYPERLOOM_CACHE_DIR)" in _refused_by_geak(
        launch, capsys
    )


def test_a_geak_identity_helper_that_hangs_exits_2(launch, tmp_path, monkeypatch, capsys):
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-hangs", "import time\ntime.sleep(30)\n"))
    monkeypatch.setattr(launch_gpu, "_GEAK_IDENTITY_TIMEOUT_SEC", 0.5)
    assert "gpu_identity.py did not finish" in _refused_by_geak(launch, capsys)


def test_an_installs_geak_e2e_runner_wins_over_a_newer_geak_root(launch, tmp_path, monkeypatch, capsys):
    """Exporting GEAK_ROOT alone, as the old refusal said, leaves the install's runner in charge."""
    stale = _use_geak(monkeypatch, _geak_root(tmp_path / "geak-stale", _prints_identity("unknown")))
    monkeypatch.setenv("GEAK_ROOT", str(_geak_root(tmp_path / "geak-new", _prints_identity("w7900"))))
    err = _refused_by_geak(launch, capsys)
    assert f"GEAK at {stale.resolve()} identifies this GPU as unknown on gfx1100" in err


@pytest.mark.parametrize("rocm_smi", [None, "w7900"], ids=["rocminfo-names-it", "rocm-smi-names-it-too"])
def test_a_w7900_found_without_a_flag_gets_the_geak_check(launch, tmp_path, monkeypatch, capsys, rocm_smi):
    if rocm_smi:
        monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda agents=None: rocm_smi)
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-unknown", _prints_identity("unknown")))
    assert "identifies this GPU as unknown on gfx1100" in _refused_by_geak(launch, capsys, argv=_FRESH)


def test_geak_reads_the_launchs_own_rocminfo_output(launch, monkeypatch):
    root = Path(os.environ["GEAK_ROOT"])
    launch.preflight(_W7900_VLLM, [W7900_DS], rocminfo="Agent 2\n  Name: gfx1100\n")
    step = launch.step("check_geak_target")
    assert step is not None
    assert (step["status"], step["target"], step["detail"]) == (
        "applied",
        str(root.resolve()),
        {"geak_target": "w7900", "geak_gfx": "gfx1100"},
    )
    assert (root / "scripts" / "stdin.txt").read_text() == "Agent 2\n  Name: gfx1100\n"


@pytest.mark.parametrize(
    ("argv", "env", "remote", "skip_reason"),
    [
        ([*_W7900_VLLM, "--no-kernel"], {}, False, "no_kernel"),
        (_W7900_VLLM, {"KERNEL_OPT_BACKEND_ORDER": "forge"}, False, "forge_enabled"),
        (_W7900_VLLM, {}, True, "remote_gpu_probe"),
    ],
    ids=["no-kernel", "forge", "remote"],
)
def test_geak_is_not_asked_when_it_will_not_be_handed_the_card(
    launch, tmp_path, monkeypatch, argv, env, remote, skip_reason
):
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-unknown", _prints_identity("unknown")))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(launch_gpu, "gpus_are_remote", lambda args: remote)
    launch.preflight(argv, [W7900_DS])
    step = launch.step("check_geak_target")
    assert step is not None
    assert (step["status"], step["skip_reason"]) == ("skipped", skip_reason)


@pytest.mark.parametrize(("gpu_type", "agents"), [("r9700", [R9700]), ("mi300x", [MI300X]), ("w7900d", [W7900D])])
def test_a_board_geak_does_not_probe_records_no_geak_check(launch, tmp_path, monkeypatch, gpu_type, agents):
    _use_geak(monkeypatch, _geak_root(tmp_path / "geak-unknown", _prints_identity("unknown")))
    launch.preflight([*_FRESH, "--gpu-type", gpu_type], agents)
    assert launch.step("check_geak_target") is None


@pytest.mark.parametrize(
    ("argv", "agents", "detail"),
    [
        (
            _W7900_VLLM,
            [W7900_DS],
            {
                "gpu_type": "w7900",
                "marketing_name": "AMD Radeon Pro W7900 Dual Slot",
                "pci_device_id": "1002:744a",
                "observed_gfx_arch": "gfx1100",
                "kernel_enabled": True,
            },
        ),
        (
            [*_FRESH, "--no-kernel"],
            [W7900, W7900_DS],
            {
                "gpu_type": "w7900",
                "marketing_name": "AMD Radeon Pro W7900",
                "pci_device_id": "1002:7448",
                "observed_gfx_arch": "gfx1100",
                "kernel_enabled": False,
            },
        ),
        (
            [*_FRESH, "--no-kernel"],
            [W7900_DS_CHIP_ONLY],
            {
                "gpu_type": "w7900",
                "marketing_name": None,
                "pci_device_id": "1002:744a",
                "observed_gfx_arch": "gfx1100",
                "kernel_enabled": False,
            },
        ),
        (
            [*_FRESH, "--gpu-type", "w7900d"],
            [W7900D],
            {
                "gpu_type": "w7900d",
                "marketing_name": "AMD Radeon PRO W7900D",
                "pci_device_id": "1002:744b",
                "observed_gfx_arch": "gfx1100",
                "kernel_enabled": True,
            },
        ),
        (
            [*_FRESH, "--gpu-type", "r9700"],
            [R9700],
            {
                "gpu_type": "r9700",
                "marketing_name": "AMD Radeon AI PRO R9700",
                "pci_device_id": None,
                "observed_gfx_arch": "gfx1201",
                "kernel_enabled": True,
            },
        ),
    ],
    ids=["w7900-dual-slot", "w7900-probed", "w7900-chip-only", "w7900d", "r9700"],
)
def test_an_accepted_launch_records_what_rocminfo_showed(launch, argv, agents, detail):
    step = launch.preflight(argv, agents)
    assert (step["status"], step["detail"]) == ("applied", detail)
    launch.models.assert_not_called()


@pytest.mark.parametrize("gpu_type", ["mi300x", "r9700", "w7900d"])
def test_a_host_without_rocminfo_checks_only_the_flags_on_other_boards(launch, gpu_type):
    step = launch.preflight([*_FRESH, "--gpu-type", gpu_type, "--no-kernel"], None)
    assert (step["status"], step["skip_reason"]) == ("skipped", "rocminfo_unavailable")
    assert step["detail"]["gpu_type"] == gpu_type and step["detail"]["observed_gfx_arch"] is None


def test_a_remote_cluster_is_not_judged_by_the_head_nodes_gpu(launch, monkeypatch):
    monkeypatch.setattr(launch_gpu, "gpus_are_remote", lambda args: True)
    step = launch.preflight(_W7900_VLLM, [RX_7900_XTX])
    assert (step["status"], step["skip_reason"]) == ("skipped", "remote_gpu_probe")
    assert step["detail"]["marketing_name"] is None


def _session(tmp_path, **state) -> Path:
    session = tmp_path / "w7900-session"
    session.mkdir()
    (session / "state.json").write_text(json.dumps({"framework": "vllm", "gpu_type": "w7900", **state}))
    return session


@pytest.mark.parametrize(
    ("argv", "nodes", "remote"),
    [(["--nodes", "1"], None, False), (["--nodes", "2"], None, True), ([], 2, True), ([], 1, False)],
    ids=["one-node", "two-nodes", "resumed-two-node-session", "resumed-one-node-session"],
)
def test_the_gpus_are_remote_on_a_multi_node_launch_or_a_resume_of_one(tmp_path, argv, nodes, remote):
    resume = ["--resume-from", str(_session(tmp_path, nodes=nodes))] if nodes else []
    args = cli._build_parser().parse_args(["optimize", "--model", "/m", *argv, *resume])
    assert launch_gpu.gpus_are_remote(args) is remote


@pytest.mark.parametrize(
    ("state", "agents", "message"),
    [
        ({"kernel_enabled": False}, [RX_7900_XTX], "but the visible GPU is 'AMD Radeon RX 7900 XTX'"),
        ({"kernel_enabled": True}, [R9700], "created for w7900, but the probed board is r9700"),
        ({"kernel_enabled": False}, [W7900D], "created for w7900, but the probed board is w7900d"),
        ({"kernel_enabled": False, "framework": "sglang"}, [W7900_DS], "w7900 is currently validated only for vllm"),
    ],
    ids=["moved-to-rx-7900-xtx", "moved-to-r9700", "moved-to-w7900d", "sglang"],
)
def test_a_resume_checks_the_sessions_own_board(launch, tmp_path, capsys, state, agents, message):
    session = _session(tmp_path, **state)
    with pytest.raises(SystemExit) as exc:
        launch.run(["--resume-from", str(session)], agents)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "--gpu-type" not in err
    launch.models.assert_not_called()
    assert sorted(path.name for path in session.iterdir()) == ["state.json"]
    assert launch.left_no_session()


@pytest.mark.parametrize("kernel_enabled", [True, False])
def test_a_resume_of_a_w7900_session_on_its_board_passes(launch, tmp_path, kernel_enabled):
    session = _session(tmp_path, kernel_enabled=kernel_enabled)
    step = launch.preflight(["--resume-from", str(session), "--gpu-type", "r9700"], [W7900_DS])
    assert step["status"] == "applied"
    assert step["detail"]["gpu_type"] == "w7900" and step["detail"]["kernel_enabled"] is kernel_enabled


# --- the identity a session records ---------------------------------------------------------------------------------


def test_a_launch_publishes_the_identity_rocminfo_read(monkeypatch):
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: provenance.GpuProbe("", [W7900_DS]))
    launch_gpu.publish_gpu_identity(launch_gpu.launch_gpu_identity(argparse.Namespace(gpu_type="w7900", nodes=1)))
    assert json.loads(os.environ[provenance.GPU_IDENTITY_ENV]) == {
        "product": "w7900",
        "marketing_name": "AMD Radeon Pro W7900 Dual Slot",
        "pci_device_id": "1002:744a",
        "gfx_arch": "gfx1100",
        "visible_gpus": 1,
    }


def test_a_remote_launch_records_only_the_product(monkeypatch):
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: pytest.fail("the head node's GPU"))
    identity = launch_gpu.launch_gpu_identity(argparse.Namespace(gpu_type="mi355x", nodes=2))
    assert identity == {
        "product": "mi355x",
        "marketing_name": None,
        "pci_device_id": None,
        "gfx_arch": None,
        "visible_gpus": None,
    }


@pytest.mark.parametrize("record", [None, {}, {"product": None, "marketing_name": None}, "w7900"])
def test_nothing_known_publishes_nothing(monkeypatch, record):
    monkeypatch.setenv(provenance.GPU_IDENTITY_ENV, '{"product": "stale"}')
    launch_gpu.publish_gpu_identity(record)
    assert provenance.GPU_IDENTITY_ENV not in os.environ


def test_both_session_paths_publish_the_identity_before_anything_reads_it():
    fresh = cli_call_sequence("_create_fresh_session")
    assert fresh.index("publish_gpu_identity") < fresh.index("write_manifest")
    resumed = cli_call_sequence("_resume_session")
    assert resumed.index("_restore_platform_state") < resumed.index("publish_gpu_identity")
    assert "launch_gpu_identity" in resumed[resumed.index("_restore_platform_state") :]


def test_a_resume_records_the_card_it_now_runs_on_as_the_sessions_gpu_type(monkeypatch):
    """Not the manifest's record from the first leg, and not the flag a resume ignores."""
    monkeypatch.setattr(provenance, "probe_gpus", lambda env=None: provenance.GpuProbe("", [W7900]))
    identity = launch_gpu.launch_gpu_identity(argparse.Namespace(gpu_type="r9700", nodes=1), gpu_type="w7900")
    assert (identity["product"], identity["marketing_name"], identity["pci_device_id"]) == (
        "w7900",
        "AMD Radeon Pro W7900",
        "1002:7448",
    )


class ManifestReached(Exception):
    """The fresh session published its GPU identity; stop before the manifest."""


@pytest.mark.parametrize("flags", [(), ("--no-kernel",)], ids=["kernel-phase", "no-kernel"])
def test_a_launch_runs_rocminfo_once(launch, tmp_path, monkeypatch, flags):
    """The product and GEAK gates, the GPU-type resolution, the ISA check and the recorded identity share one run."""
    listing = _ROCMINFO_4X_W7900.read_text(encoding="utf-8")
    real_run = subprocess.run
    rocminfo_runs: list[list[str]] = []

    def run(argv, *args, **kwargs):
        tool = Path(str(argv[0])).name
        if tool == "rocminfo":
            rocminfo_runs.append(list(argv))
            return subprocess.CompletedProcess(argv, 0, listing, "")
        if tool == "rocm-smi" and "--showproductname" in argv:
            # rocm-smi lists a card the session does not see; the launch's own rocminfo run names the board.
            return subprocess.CompletedProcess(argv, 0, "GPU[0]\t\t: Card Series: \t\tAMD Radeon PRO W7900D\n", "")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", _REAL_AUTODETECT)
    monkeypatch.setattr(provenance, "_rocminfo_binary", lambda: "/opt/rocm/bin/rocminfo")
    for name in ("_enforce_expected_framework", "_apply_operator_supplied_paths"):
        monkeypatch.setattr(cli, name, lambda *a, **kw: None)
    monkeypatch.setattr(cli, "resolve_framework_version", lambda args: "test-version")
    monkeypatch.setattr(cli, "make_session_dir", lambda **kw: tmp_path / "session")
    monkeypatch.setattr(cli, "_acquire_session_lock_or_exit", lambda path: None)
    monkeypatch.setattr(cli, "write_manifest", Mock(side_effect=ManifestReached))
    args = cli._build_parser().parse_args(["optimize", *_W7900_VLLM, *flags])
    monkeypatch.setattr(provenance, "probe_gpus", _REAL_PROBE_GPUS)
    with patch.dict(os.environ):
        launch_shape._resolve_run_topology(args)
        configure_benchmark(args)
        with pytest.raises(PreflightReachedCredentials):
            preflight._preflight(args)
        with pytest.raises(ManifestReached):
            asyncio.run(cli._seed_fresh_session(args, nodes_resolved=1))
        identity = json.loads(os.environ[provenance.GPU_IDENTITY_ENV])
    assert rocminfo_runs == [["/opt/rocm/bin/rocminfo"]]
    assert (args.gpu_type, identity["pci_device_id"], identity["visible_gpus"]) == ("w7900", "1002:744a", 4)
    if not flags:
        assert (Path(os.environ["GEAK_ROOT"]) / "scripts" / "stdin.txt").read_text() == listing


# --- the session's GPU-type resolution ------------------------------------------------------------------------------


def _resolve_target_gpu(monkeypatch, argv, probed):
    """``_resolve_target_gpu`` on a host whose probes name ``probed``; they must be asked once."""
    for name in ("GPU_TYPE", "TARGET_GPU_TYPE"):
        monkeypatch.delenv(name, raising=False)
    args = cli._build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm", *argv])
    configure_benchmark(args)
    autodetect = Mock(return_value=probed)
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", autodetect)
    monkeypatch.setattr(cli, "_check_gfx_arch_resolvable", lambda gpu_type, **_kw: None)
    with patch.dict(os.environ):
        cli._resolve_target_gpu(args, "vllm")
    autodetect.assert_called_once_with(agents=None)
    return args


@pytest.mark.parametrize(
    ("argv", "probed"),
    [
        (["--gpu-type", "r9700"], "w7900"),
        (["--gpu-type", "mi300x", "--no-kernel"], "w7900"),
        (["--gpu-type", "w7900", "--no-kernel"], "r9700"),
        (["--gpu-type", "w7900"], "w7900d"),
        (["--gpu-type", "w7900d"], "w7900"),
    ],
)
def test_a_gpu_type_the_probe_contradicts_exits_2_when_either_is_w7900(monkeypatch, capsys, argv, probed):
    with pytest.raises(SystemExit) as exc:
        _resolve_target_gpu(monkeypatch, argv, probed)
    assert exc.value.code == 2
    assert f"disagrees with the probed {probed}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "probed"),
    [
        ([], "w7900"),
        (["--no-kernel"], "w7900"),
        (["--gpu-type", "w7900"], ""),
        (["--gpu-type", "w7900", "--no-kernel"], ""),
    ],
    ids=["probed", "probed-no-kernel", "flag", "flag-no-kernel"],
)
def test_a_w7900_session_resolves_to_the_w7900(monkeypatch, argv, probed):
    assert _resolve_target_gpu(monkeypatch, argv, probed).gpu_type == "w7900"


def test_the_session_settles_on_the_board_the_launch_gate_checked(monkeypatch):
    """One answer per launch: a second probe could name another board than the one preflight confirmed."""
    for name in ("GPU_TYPE", "TARGET_GPU_TYPE"):
        monkeypatch.delenv(name, raising=False)
    args = cli._build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])
    configure_benchmark(args)
    autodetect = Mock(side_effect=["w7900", "mi355x"])
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", autodetect)
    monkeypatch.setattr(cli, "_check_gfx_arch_resolvable", lambda gpu_type, **_kw: None)
    assert launch_gpu.launch_board(args) == "w7900"
    with patch.dict(os.environ):
        cli._resolve_target_gpu(args, "vllm")
    assert args.gpu_type == "w7900"
    autodetect.assert_called_once_with(agents=None)


@pytest.mark.parametrize(("flag", "gpu_type", "asked"), [(None, "mi355x", True), ("mi300x", "mi300x", False)])
def test_a_multi_node_launch_falls_back_to_the_head_node_only_with_nothing_else(
    monkeypatch, capsys, flag, gpu_type, asked
):
    """With no flag and a remote probe that names no board, the head node's GPU names the type, as on ``main``."""
    from hyperloom.inference_optimizer.multi_node._internal import gpu_probe

    for name in ("GPU_TYPE", "TARGET_GPU_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gpu_probe, "remote_autodetect_gpu_type", lambda **_kw: None)
    head = Mock(return_value="mi355x")
    monkeypatch.setattr(cli, "_autodetect_gpu_type", head)
    monkeypatch.setattr(cli, "_check_gfx_arch_resolvable", lambda gpu_type, **_kw: None)
    args = argparse.Namespace(gpu_type=flag, nodes=2)
    with patch.dict(os.environ):
        cli._resolve_target_gpu(args, "vllm")
    assert args.gpu_type == gpu_type
    assert head.called is asked
    assert ("(head node; the remote probe named no board)" in capsys.readouterr().out) is asked


@pytest.mark.parametrize(("user", "probed"), [("mi300x", "mi355x"), ("w7900d", "r9700"), ("r9700", "w7900d")])
def test_other_probe_disagreements_keep_probe_wins(monkeypatch, capsys, user, probed):
    assert _resolve_target_gpu(monkeypatch, ["--gpu-type", user], probed).gpu_type == probed
    assert f"using probed '{probed}'" in capsys.readouterr().err
