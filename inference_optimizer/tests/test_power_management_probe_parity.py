"""Lock the rung-2 capability probe in lock-step with the live executor.

If someone updates the executor's set / reset command strings without
also updating ``scripts/probe_power_management_capability.py``, this
suite fails loudly at unit-test time — far cheaper than discovering
the drift on a live host where the probe says "PASS" but the actual
action then fails on a different sudoers binding.

Plus a smoke-level driver test that runs ``run_probe`` end-to-end with
every subprocess monkeypatched, so we know the orchestration logic
itself (phase gating, fail-fast, finalisation) is sound.

All driver tests force ``os.geteuid → 0`` (root) by default to match
the typical Hyperloom Docker deployment; the explicit non-root case
lives in :class:`TestNonRootElevation`.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import (
    power_management as exec_pm,
)
from inference_optimizer.scripts import (
    probe_power_management_capability as probe,
)


# ---------------------------------------------------------------------------
# autouse: default every test to root unless it explicitly opts out so
# the sudo branches are covered deterministically across runners.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _default_root(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(exec_pm.os, "geteuid", lambda: 0)
    monkeypatch.setattr(probe.os, "geteuid", lambda: 0)


# ---------------------------------------------------------------------------
# Parity: the probe MUST exercise the same command strings the executor
# uses. Drift here = silent runtime breakage.
# ---------------------------------------------------------------------------
class TestExecutorParity:
    def test_rocm_smi_binary_matches(self):
        assert probe.ROCM_SMI_BIN == exec_pm.ROCM_SMI_BIN

    def test_reset_flags_match(self):
        """The executor and probe must reset the same set of knobs."""
        assert probe._RESET_FLAGS == exec_pm._RESET_FLAGS

    @pytest.mark.parametrize("uid,expected_prefix", [(0, ""), (1000, "sudo ")])
    def test_sudo_prefix_matches_executor(
        self, uid, expected_prefix, monkeypatch,
    ):
        monkeypatch.setattr(exec_pm.os, "geteuid", lambda: uid)
        monkeypatch.setattr(probe.os, "geteuid", lambda: uid)
        assert exec_pm._sudo_prefix() == expected_prefix
        assert probe._sudo_prefix() == expected_prefix
        assert exec_pm._sudo_prefix() == probe._sudo_prefix()

    @pytest.mark.parametrize("uid", [0, 1000])
    def test_reset_commands_render_identically(self, uid, monkeypatch):
        """Each rendered reset command must match byte-for-byte across
        the executor and the probe, under both root and non-root."""
        monkeypatch.setattr(exec_pm.os, "geteuid", lambda: uid)
        monkeypatch.setattr(probe.os, "geteuid", lambda: uid)
        assert exec_pm._reset_cmds() == probe._reset_commands()

    def test_apply_variant_cmd_shape_as_root(self, monkeypatch):
        """Under root, --setpoweroverdrive commands should NOT carry sudo."""
        monkeypatch.setattr(exec_pm.os, "geteuid", lambda: 0)
        v = exec_pm.PowerVariant(name="probe", power_cap_w=250, devices=(0, 1))
        cmds = exec_pm._apply_variant_cmds(v)
        assert any("--setpoweroverdrive 250" in c for c in cmds)
        for c in cmds:
            assert not c.startswith("sudo "), c
            assert c.endswith("--autorespond yes"), c

    def test_apply_variant_cmd_shape_as_non_root(self, monkeypatch):
        """Non-root callers should get sudo-prefixed commands."""
        monkeypatch.setattr(exec_pm.os, "geteuid", lambda: 1000)
        v = exec_pm.PowerVariant(name="probe", power_cap_w=250, devices=(0, 1))
        cmds = exec_pm._apply_variant_cmds(v)
        assert any("--setpoweroverdrive 250" in c for c in cmds)
        for c in cmds:
            assert c.startswith("sudo "), c
            assert c.endswith("--autorespond yes"), c

    def test_probe_timeouts_match_executor(self):
        assert probe.PROBE_TIMEOUT_SEC == exec_pm.ROCM_SMI_PROBE_TIMEOUT_SEC
        assert probe.SET_TIMEOUT_SEC == exec_pm.ROCM_SMI_SET_TIMEOUT_SEC


# ---------------------------------------------------------------------------
# Parity: the clock-table parsers are duplicated (the probe imports zero
# Hyperloom modules) — they MUST produce identical output on shared
# fixtures, or the probe's "ladder it would build" report lies about
# what the executor will actually do.
# ---------------------------------------------------------------------------
class TestClockTableParserParity:
    # Representative rocm-smi shapes covering the trailing ``*`` marker,
    # GHz normalisation, multi-card max-per-index reduction, and the
    # single-frequency (no-range) edge.
    _SCLKRANGE_FIXTURES = (
        "GPU[0]: Valid sclk range: 500Mhz - 2400Mhz",
        "Valid sclk range: 500Mhz - 2400Mhz *",
        "range: 0.5Ghz - 2.4Ghz",
        "no frequencies here",
        "",
    )
    _MCLKRANGE_FIXTURES = (
        "Valid mclk range: 900Mhz - 1300Mhz",
        "only one: 2000Mhz",
        "",
    )
    _CLKFRQ_FIXTURES = (
        {"card0": {"sclk[0]": "500Mhz", "sclk[1]": "2400Mhz",
                   "mclk[0]": "900Mhz", "mclk[1]": "1300Mhz"}},
        {"card0": {"mclk[0]": "900Mhz", "mclk[1]": "1200Mhz"},
         "card1": {"mclk[0]": "950Mhz", "mclk[1]": "1300Mhz"}},
        {"card0": {"mclk[0]": "2000Mhz"}},
        {"card0": {"sclk[0]": "1Mhz"}},
        {},
    )

    @pytest.mark.parametrize("text", _SCLKRANGE_FIXTURES)
    def test_sclkrange_top_parity(self, text):
        assert (
            probe._parse_sclkrange_top_mhz(text)
            == exec_pm._parse_sclkrange_top_mhz(text)
        )

    @pytest.mark.parametrize("text", _MCLKRANGE_FIXTURES)
    def test_mclkrange_parity(self, text):
        assert probe._parse_mclkrange(text) == exec_pm._parse_mclkrange(text)

    @pytest.mark.parametrize("data", _CLKFRQ_FIXTURES)
    def test_mclk_levels_parity(self, data):
        assert (
            probe._parse_mclk_levels_from_clkfrq(data)
            == exec_pm._parse_mclk_levels_from_clkfrq(data)
        )

    def test_determinism_pct_ladder_matches(self):
        # The probe's reported ladder fractions must match the executor's
        # so the "ladder it would build" report is truthful.
        assert (
            probe._SETTLE_DETERMINISM_PCTS
            == exec_pm._SETTLE_DETERMINISM_PCTS
        )

    def test_freq_token_parser_parity(self):
        for text in ("2400Mhz", "2.4Ghz", "500 MHz *", "garbage", ""):
            assert (
                probe._freq_tokens_to_mhz(text)
                == exec_pm._freq_tokens_to_mhz(text)
            )


# ---------------------------------------------------------------------------
# Driver smoke — fully monkeypatched subprocess.run, no real shell-out
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Returns a mutable registry of (predicate, response) pairs.

    A ``response`` is ``(returncode, stdout, stderr)``. The probe's
    ``_run`` helper goes through ``subprocess.run`` (via ``shlex.split``)
    so we intercept at that layer for maximal coverage.
    """
    registry: list[tuple[Any, tuple[int, str, str]]] = []

    def fake_run(argv, *, capture_output, text, timeout):
        cmdline = " ".join(argv)
        for predicate, response in registry:
            if predicate(cmdline):
                rc, out, err = response
                return subprocess.CompletedProcess(
                    argv, rc, out, err,
                )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    return registry


def _register(reg: list, substr: str, rc: int, stdout: str = "", stderr: str = ""):
    """Convenience: match by substring."""
    reg.append((lambda cl, s=substr: s in cl, (rc, stdout, stderr)))


_HEALTHY_SHOWMAXPOWER_JSON = (
    '{"card0": {"Max Graphics Package Power (W)": "400.0"}, '
    '"card1": {"Max Graphics Package Power (W)": "380.0"}}'
)
# Most-restrictive ceiling across cards = min(400, 380) = 380.
_HEALTHY_CEILING_W = 380


class TestRunProbeHappyPathRoot:
    """Root-uid Docker case (the typical Hyperloom deployment)."""

    def test_all_pass_without_setter_exercise(self, fake_subprocess, monkeypatch):
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="rocm-smi 6.4.0\n")
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--showmaxpower show ceiling\n--setpoweroverdrive WATTS set cap\n",
        )
        _register(
            fake_subprocess, "--showmaxpower --json", 0,
            stdout=_HEALTHY_SHOWMAXPOWER_JSON,
        )
        # No sudo expected — root path; default fake_run returns rc=0 so
        # the bare "rocm-smi --version" and reset calls all succeed.
        for flag in ("--resetperfdeterminism", "--resetclocks",
                     "--resetpoweroverdrive", "--resetfans"):
            _register(fake_subprocess, flag, 0)

        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "pass", rep.summary
        # No sudo-prefixed command should have run on the root path.
        assert not any(s.cmd.startswith("sudo ") for s in rep.steps)
        # The elevation step should announce root.
        elevation = next(
            s for s in rep.steps if s.name.startswith("running as root")
        )
        assert elevation.status == "pass"
        # And no setter step (opt-in only).
        assert not any(
            "--setpoweroverdrive" in s.cmd and "(no-op)" in s.name
            for s in rep.steps
        )

    def test_all_pass_with_setter_exercise(self, fake_subprocess, monkeypatch):
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="rocm-smi 6.4.0\n")
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--showmaxpower\n--setpoweroverdrive WATTS\n",
        )
        _register(
            fake_subprocess, "--showmaxpower --json", 0,
            stdout=_HEALTHY_SHOWMAXPOWER_JSON,
        )
        for flag in ("--resetperfdeterminism", "--resetclocks",
                     "--resetpoweroverdrive", "--resetfans"):
            _register(fake_subprocess, flag, 0)
        # Setter exercise — note: no sudo prefix because we're root.
        _register(
            fake_subprocess, f"--setpoweroverdrive {_HEALTHY_CEILING_W}", 0,
        )

        rep = probe.run_probe(devices=(), exercise_setter=True)
        assert rep.overall == "pass", rep.summary
        setter_steps = [s for s in rep.steps if "--setpoweroverdrive" in s.cmd]
        # Only the no-op setter exercise step references --setpoweroverdrive
        # (the reset variants use --resetpoweroverdrive).
        assert len(setter_steps) == 1
        assert setter_steps[0].status == "pass"
        assert not setter_steps[0].cmd.startswith("sudo "), setter_steps[0].cmd


class TestRunProbeFailFast:
    def test_missing_rocm_smi_fails_immediately(self, fake_subprocess, monkeypatch):
        monkeypatch.setattr(probe.shutil, "which", lambda _b: None)
        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "fail"
        assert "PATH" in rep.summary
        # No probe of --version etc. should have happened.
        assert all(s.name in {"rocm-smi on PATH"} or s.status == "pass"
                   for s in rep.steps)

    def test_missing_showmaxpower_flag_fails_before_elevation(
        self, fake_subprocess, monkeypatch,
    ):
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="rocm-smi 4.5.0\n")
        # --help missing --showmaxpower (the ceiling probe).
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--setpoweroverdrive WATTS\n",
        )
        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "fail"
        assert "--showmaxpower" in rep.summary
        # Elevation / reset steps must NOT have run.
        assert not any("--reset" in s.cmd for s in rep.steps)
        assert not any(
            "sudo" in s.name or "as root" in s.name for s in rep.steps
        )


class TestNonRootElevation:
    """Non-root case (bare-metal ops, sudo path exercised)."""

    def test_sudo_password_required_fails_before_resets(
        self, fake_subprocess, monkeypatch,
    ):
        monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="rocm-smi 6.4.0\n")
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--showmaxpower\n--setpoweroverdrive WATTS\n",
        )
        _register(
            fake_subprocess, "--showmaxpower --json", 0,
            stdout=_HEALTHY_SHOWMAXPOWER_JSON,
        )
        _register(
            fake_subprocess, "sudo -n true", 1,
            stderr="sudo: a password is required\n",
        )
        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "fail"
        assert "NOPASSWD" in rep.summary or "password" in rep.summary
        assert not any("--reset" in s.cmd for s in rep.steps)

    def test_non_root_happy_path_uses_sudo_prefix(
        self, fake_subprocess, monkeypatch,
    ):
        monkeypatch.setattr(probe.os, "geteuid", lambda: 1000)
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="rocm-smi 6.4.0\n")
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--showmaxpower\n--setpoweroverdrive WATTS\n",
        )
        _register(
            fake_subprocess, "--showmaxpower --json", 0,
            stdout=_HEALTHY_SHOWMAXPOWER_JSON,
        )
        _register(fake_subprocess, "sudo -n true", 0)
        _register(fake_subprocess, "sudo -n rocm-smi --version", 0, stdout="6.4.0\n")
        for flag in ("--resetperfdeterminism", "--resetclocks",
                     "--resetpoweroverdrive", "--resetfans"):
            _register(fake_subprocess, flag, 0)
        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "pass", rep.summary
        # Every reset and the elevation probe must carry sudo -n.
        for s in rep.steps:
            if s.name.startswith(f"{probe.ROCM_SMI_BIN} --reset"):
                assert s.cmd.startswith("sudo -n "), s.cmd


class TestClockTablesStep:
    """`step_clock_tables_parse` reports the GFX/memory ladder the
    settle sweep would build, WARNing (not failing) when it would
    degenerate to no determinism ladder."""

    def test_reports_full_ladder_and_skipped_memory_axis(
        self, fake_subprocess,
    ):
        # MI355X-like: clean 500–2400 sclk range, single 2000 MHz mclk
        # level → 4-row GFX ladder (top 2400), memory axis skipped.
        _register(
            fake_subprocess, "--showsclkrange", 0,
            stdout="GPU[0]: Valid sclk range: 500Mhz - 2400Mhz",
        )
        _register(
            fake_subprocess, "--showclkfrq --json", 0,
            stdout='{"card0": {"sclk[0]": "500Mhz", "mclk[0]": "2000Mhz"}}',
        )
        _register(fake_subprocess, "--showmclkrange", 0, stdout="")
        step = probe.step_clock_tables_parse(())
        assert step.status == "pass"
        assert "top sclk=2400MHz" in step.detail
        assert "det_100=2400MHz" in step.detail
        assert "det_95=2280MHz" in step.detail
        assert "det_90=2160MHz" in step.detail
        assert "det_85=2040MHz" in step.detail
        assert "memory axis: skipped" in step.detail

    def test_reports_capable_memory_axis(self, fake_subprocess):
        _register(
            fake_subprocess, "--showsclkrange", 0,
            stdout="Valid sclk range: 500Mhz - 2400Mhz",
        )
        _register(
            fake_subprocess, "--showclkfrq --json", 0,
            stdout=('{"card0": {"sclk[0]": "2400Mhz", '
                    '"mclk[0]": "1000Mhz", "mclk[1]": "2000Mhz"}}'),
        )
        _register(
            fake_subprocess, "--showmclkrange", 0,
            stdout="Valid mclk range: 1000Mhz - 2000Mhz",
        )
        step = probe.step_clock_tables_parse(())
        assert step.status == "pass"
        assert "memory axis: capable" in step.detail
        assert "2 levels" in step.detail

    def test_warns_when_no_top_sclk(self, fake_subprocess):
        # No parseable sclk range AND no sclk in the DPM table → the
        # determinism ladder would be empty → WARN (not fail), caught
        # before a multi-hour run.
        _register(fake_subprocess, "--showsclkrange", 0, stdout="")
        _register(fake_subprocess, "--showclkfrq --json", 0, stdout="{}")
        _register(fake_subprocess, "--showmclkrange", 0, stdout="")
        step = probe.step_clock_tables_parse(())
        assert step.status == "warn"
        assert "0 rows" in step.detail or "auto/high-only" in step.detail

    def test_warn_does_not_fail_overall_probe(self, fake_subprocess, monkeypatch):
        # A degraded clock ladder is a WARN: the action can still run
        # auto/high-only, so the overall probe stays PASS (with the
        # warning surfaced in the summary).
        monkeypatch.setattr(
            probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi",
        )
        _register(fake_subprocess, "rocm-smi --version", 0, stdout="6.4.0\n")
        _register(
            fake_subprocess, "rocm-smi --help", 0,
            stdout="--showmaxpower\n--setpoweroverdrive WATTS\n",
        )
        _register(
            fake_subprocess, "--showmaxpower --json", 0,
            stdout=_HEALTHY_SHOWMAXPOWER_JSON,
        )
        for flag in ("--resetperfdeterminism", "--resetclocks",
                     "--resetpoweroverdrive", "--resetfans"):
            _register(fake_subprocess, flag, 0)
        # No --showsclkrange / --showclkfrq registered → empty → warn.
        rep = probe.run_probe(devices=(), exercise_setter=False)
        assert rep.overall == "pass"
        assert "warning" in rep.summary.lower()
        ladder = next(
            s for s in rep.steps if s.name == "clock tables → settle ladder"
        )
        assert ladder.status == "warn"


class TestParsing:
    def test_parse_devices_empty(self):
        assert probe._parse_devices("") == ()
        assert probe._parse_devices("   ") == ()

    def test_parse_devices_single(self):
        assert probe._parse_devices("3") == (3,)

    def test_parse_devices_multi(self):
        assert probe._parse_devices("0,1,3,7") == (0, 1, 3, 7)

    def test_parse_devices_bad_token(self):
        with pytest.raises(ValueError):
            probe._parse_devices("0,foo")

    def test_parse_devices_handles_whitespace(self):
        assert probe._parse_devices(" 0 , 1 ,2 ") == (0, 1, 2)


class TestHumanFormat:
    def test_human_format_includes_pass_marker(self, fake_subprocess, monkeypatch):
        monkeypatch.setattr(probe.shutil, "which", lambda _b: "/opt/rocm/bin/rocm-smi")
        rep = probe.ProbeReport()
        rep.add(probe.StepResult(name="step1", status="pass", detail="ok"))
        rep.add(probe.StepResult(name="step2", status="skip", detail="opt-out"))
        rep.overall = "pass"
        rep.summary = "All checks passed"
        out = probe._format_human(rep)
        assert "PASS" in out
        assert "SKIP" in out
        assert "All checks passed" in out

    def test_human_format_surfaces_stderr_on_fail(self):
        rep = probe.ProbeReport()
        rep.add(probe.StepResult(
            name="step", status="fail", detail="rc=1",
            stderr_tail="sudo: a password is required",
        ))
        rep.overall = "fail"
        rep.summary = "FAIL: step — rc=1"
        out = probe._format_human(rep)
        assert "STDERR" in out
        assert "password" in out
