# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CLI surface for offline diffusion (xDiT): parser + descriptor-driven gating."""

from __future__ import annotations

import pytest

from inference_optimizer import framework_registry as fr
from inference_optimizer.cli import _build_parser


def _parse(*extra: str):
    parser = _build_parser()
    return parser.parse_args(["optimize", *extra])


class TestParserAcceptsXdit:
    def test_framework_xdit_is_valid_choice(self):
        ns = _parse("--framework", "xdit", "--run-cmd", "xdit --model flux")
        assert ns.framework == "xdit"
        assert ns.run_cmd == "xdit --model flux"

    def test_run_cmd_defaults_none_without_env(self, monkeypatch):
        monkeypatch.delenv("RUN_CMD", raising=False)
        ns = _parse("--framework", "sglang")
        assert ns.run_cmd is None

    def test_run_cmd_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("RUN_CMD", "xdit --model flux")
        ns = _parse("--framework", "xdit")
        assert ns.run_cmd == "xdit --model flux"

    def test_unknown_framework_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            _parse("--framework", "tensorrt")


class TestConcSweepGatingExpression:
    """The cli computes ``conc_sweep_enabled = flag AND
    get_capabilities(framework).supports_conc_sweep``.

    Pin that composition: frameworks without a conc knob (xdit) force it off
    regardless of the --enable-conc-sweep flag, while LLM frameworks honor it.
    """

    @pytest.mark.parametrize("flag", [True, False])
    def test_no_conc_knob_forces_sweep_off(self, flag):
        enabled = bool(flag) and fr.get_capabilities("xdit").supports_conc_sweep
        assert enabled is False

    def test_online_honors_flag_true(self):
        enabled = True and fr.get_capabilities("sglang").supports_conc_sweep
        assert enabled is True

    def test_online_honors_flag_false(self):
        enabled = False and fr.get_capabilities("sglang").supports_conc_sweep
        assert enabled is False
