"""The screen may shorten EXPLORE's grid; it may never lose a winner.

Pruning on a cheap probe is the one change here that can make Hyperloom worse
without making it look worse: a dropped variant produces no measurement, so a
discarded winner leaves no trace in the ledger. So most of these tests are about
what the screen refuses to drop -- anything when it is off, when the probe cannot
run, when the grid is too small to be worth pruning, and any individual variant
whose own probe came back unreadable.
"""
from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.actions.executors import _explore_screen
from hyperloom.orchestrator.actions.executors._explore_screen import (
    BASELINE,
    ENV_BACKEND,
    ENV_ENABLED,
    ENV_GPUS,
    ENV_INFERA_ROOT,
    ENV_LAYERS,
    ENV_MARGIN_PCT,
    kernels_from_log,
    screen_enabled,
    screen_variants,
)
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant

CONFIG = """
benchmark:
  framework: vllm
  model: /models/gpt-oss-120b
  envs:
    TP: 8
    CONC: 32
    ISL: 1024
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(CONFIG)
    return path


KERNEL_LOG = ("INFO [rocm.py:556] Using ROCM_AITER_UNIFIED_ATTN backend "
              "(selected via --attention-backend).\n"
              "INFO [mxfp4.py:514] Using 'TRITON' Mxfp4 MoE backend.\n")


@pytest.fixture
def session(tmp_path):
    """A session that has already benchmarked the stack, so its kernels are known."""
    slot = tmp_path / "session" / "explore-001" / "base"
    slot.mkdir(parents=True)
    (slot / "server.log").write_text(KERNEL_LOG)
    return tmp_path / "session"


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_INFERA_ROOT, str(tmp_path))
    return monkeypatch


def variants(n=4):
    return [GridVariant(name=f"v{i}", extra_server_args=f"--max-num-seqs {i}")
            for i in range(n)]


MATCHING_KERNELS = {"attention": "ROCM_AITER_UNIFIED_ATTN", "moe": "TRITON"}


def fake_probe(readings, baseline=10.0, kernels=None):
    """Stand in for the GPU probe: a canned reading and kernel set per variant."""
    resolved = MATCHING_KERNELS if kernels is None else kernels

    def probe(variant, bench, timeout_sec, backend):
        if variant.name == BASELINE:
            return baseline, resolved
        return readings.get(variant.name), resolved
    return probe


# --- the screen stays out of the way ---------------------------------------

def test_disabled_by_default(config, session):
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert [v.name for v in kept] == ["v0", "v1", "v2", "v3"]
    assert dropped == []


def test_enabled_flag_is_explicit(monkeypatch):
    assert screen_enabled() is False
    monkeypatch.setenv(ENV_ENABLED, "1")
    assert screen_enabled() is True
    monkeypatch.setenv(ENV_ENABLED, "0")
    assert screen_enabled() is False


def test_small_grid_is_not_worth_pruning(config, enabled, session):
    kept, dropped = screen_variants(variants(2), config, session_dir=session)
    assert len(kept) == 2 and dropped == []


def test_no_infera_checkout_means_no_screen(config, session, monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.delenv(ENV_INFERA_ROOT, raising=False)
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert len(kept) == 4 and dropped == []


def test_unreadable_config_means_no_screen(tmp_path, enabled, session):
    missing = tmp_path / "absent.yaml"
    kept, dropped = screen_variants(variants(), missing, session_dir=session)
    assert len(kept) == 4 and dropped == []


def test_failed_baseline_probe_benchmarks_the_whole_grid(config, enabled, session,
                                                         monkeypatch):
    """Without a reference reading there is nothing to measure a margin against."""
    monkeypatch.setattr(_explore_screen, "_probe", lambda v, b, t, backend: (None, {}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert len(kept) == 4 and dropped == []


# --- the probe has to be running the deployment's kernels -------------------

def test_a_probe_on_different_kernels_prunes_nothing(config, enabled, session,
                                                     monkeypatch):
    """The failure this exists for, and the one that is expensive to miss.

    Measured on gpt-oss-120b/MI355X at the same TP=8, model and flags: the server
    resolves ROCM_AITER_UNIFIED_ATTN with a Triton MXFP4 MoE, the offline probe
    ROCM_AITER_FA with an AITER one. Read across that gap the screen is not
    noisy but confidently wrong: it ranks a stack the deployment never runs.
    """
    monkeypatch.setattr(_explore_screen, "_probe", fake_probe(
        {"v0": 99.0, "v1": 99.0, "v2": 99.0, "v3": 99.0},
        kernels={"attention": "ROCM_AITER_FA", "moe": "AITER_MXFP4_BF16"}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert len(kept) == 4 and dropped == []


def test_one_matching_kernel_is_not_enough(config, enabled, session, monkeypatch):
    """Pinning attention alone left the same lever reading 65% worse."""
    monkeypatch.setattr(_explore_screen, "_probe", fake_probe(
        {"v0": 99.0, "v1": 9.0, "v2": 9.0, "v3": 9.0},
        kernels={"attention": "ROCM_AITER_UNIFIED_ATTN", "moe": "AITER_MXFP4_BF16"}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert len(kept) == 4 and dropped == []


def test_no_pruning_when_nothing_says_what_the_deployment_runs(config, enabled,
                                                               tmp_path, monkeypatch):
    """An unverifiable regime is not a matching one."""
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 99.0, "v1": 9.0, "v2": 9.0, "v3": 9.0}))
    kept, dropped = screen_variants(variants(), config, session_dir=tmp_path / "empty")
    assert len(kept) == 4 and dropped == []


def test_target_kernels_come_from_a_benchmark_already_paid_for(session):
    """EXPLORE boots the stack before it proposes anything; the answer is on disk."""
    assert _explore_screen._target_kernels(session) == MATCHING_KERNELS


def test_kernels_are_read_from_either_thing_vllm_logs():
    assert kernels_from_log(
        "Overriding with ROCM_AITER_FA out of potential backends") == {
            "attention": "ROCM_AITER_FA"}
    assert kernels_from_log(
        "Using TRITON_ATTN backend (selected via --attention-backend).\n"
        "Using 'TRITON' Mxfp4 MoE backend.") == {
            "attention": "TRITON_ATTN", "moe": "TRITON"}
    assert kernels_from_log("nothing about kernels here") == {}


def test_target_backend_is_read_from_the_deployments_own_flags(tmp_path, enabled,
                                                               monkeypatch):
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    path = tmp_path / "pinned.yaml"
    path.write_text(CONFIG + "    EXTRA_VLLM_ARGS: --attention-backend TRITON_ATTN\n")
    with open(path) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    assert _explore_screen._target_backend(bench, None) == "TRITON_ATTN"


def test_the_pin_precedes_the_variants_own_flags(config, enabled):
    """A variant testing a backend must override the pin, not be overridden by it."""
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    variant = GridVariant(name="v", extra_server_args="--attention-backend TRITON_ATTN")
    cmd = _explore_screen._probe_command(variant, bench, "/tmp/o.json",
                                         "ROCM_AITER_UNIFIED_ATTN")
    args = next(c for c in cmd if c.startswith("--server-args="))
    assert args.endswith("--attention-backend TRITON_ATTN")


# --- the screen cuts only the decisive losers -------------------------------

def test_cuts_only_what_is_beyond_the_margin(config, enabled, session, monkeypatch):
    # Baseline 10ms, margin 10%: 11.0ms survives, 11.1ms does not.
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 14.0, "v1": 9.0, "v2": 11.0, "v3": 11.1}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert [v.name for v in kept] == ["v1", "v2"]
    assert {d["name"] for d in dropped} == {"v0", "v3"}
    assert all(d["reason"] == "screen_decisively_slower" for d in dropped)


def test_a_grid_of_near_ties_is_forwarded_whole(config, enabled, session, monkeypatch):
    """Differences the screen cannot resolve are left to the real benchmark."""
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 10.4, "v1": 9.7, "v2": 10.9, "v3": 10.1}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert len(kept) == 4 and dropped == []


def test_survivors_keep_the_grid_order(config, enabled, session, monkeypatch):
    """The screen prunes; it does not get to decide what EXPLORE tries first."""
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 10.5, "v1": 9.0, "v2": 20.0, "v3": 9.5}))
    kept, _ = screen_variants(variants(), config, session_dir=session)
    assert [v.name for v in kept] == ["v0", "v1", "v3"]


def test_margin_is_configurable(config, enabled, session, monkeypatch):
    """A wider margin trusts the screen less: v3, cut at the default, survives."""
    monkeypatch.setenv(ENV_MARGIN_PCT, "30")
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 14.0, "v1": 9.0, "v2": 11.0, "v3": 11.1}))
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert [v.name for v in kept] == ["v1", "v2", "v3"]
    assert [d["name"] for d in dropped] == ["v0"]


def test_nonsense_margin_falls_back_to_the_measured_one(config, enabled, session,
                                                        monkeypatch):
    monkeypatch.setenv(ENV_MARGIN_PCT, "not-a-number")
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 14.0, "v1": 9.0, "v2": 11.0, "v3": 10.5}))
    kept, _ = screen_variants(variants(), config, session_dir=session)
    assert [v.name for v in kept] == ["v1", "v2", "v3"]


def test_unreadable_variant_is_kept_not_dropped(config, enabled, session, monkeypatch):
    """A probe that fails for one variant must not decide against it."""
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 90.0, "v1": 3.0, "v2": 7.0}))  # v3 unreadable
    kept, dropped = screen_variants(variants(), config, session_dir=session)
    assert "v3" in {v.name for v in kept}
    assert "v3" not in {d["name"] for d in dropped}


def test_dropped_records_carry_the_measurement(config, enabled, session, monkeypatch):
    monkeypatch.setattr(_explore_screen, "_probe",
                        fake_probe({"v0": 14.0, "v1": 9.0, "v2": 7.0, "v3": 1.0}))
    _, dropped = screen_variants(variants(), config, session_dir=session)
    assert dropped[0]["detail"] == "probe decode 14.000 ms vs baseline 10.000 ms (+40%)"


# --- the probe carries the variant's own levers -----------------------------

def test_probe_command_applies_variant_flags_and_env(config, enabled, tmp_path):
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    variant = GridVariant(
        name="aiter_off",
        extra_server_args="--max-num-seqs 512 --enable-chunked-prefill",
        extra_envs={"VLLM_ROCM_USE_AITER": "0"},
    )
    cmd = _explore_screen._probe_command(variant, bench, "/tmp/out.json", None)
    assert "--server-args=--max-num-seqs 512 --enable-chunked-prefill" in cmd
    assert "VLLM_ROCM_USE_AITER=0" in cmd
    # The probe runs at the deployment's target TP but on fewer GPUs, which is
    # where the saving comes from.
    assert cmd[cmd.index("--tp") + 1] == "8"
    assert cmd[cmd.index("--benchmark-gpus") + 1] == "1"
    assert cmd[cmd.index("--batches") + 1] == "32"


def test_probe_gpu_count_and_depth_are_configurable(config, enabled, monkeypatch):
    monkeypatch.setenv(ENV_GPUS, "2")
    monkeypatch.setenv(ENV_LAYERS, "8")
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    cmd = _explore_screen._probe_command(GridVariant(name="v"), bench, "/tmp/o.json", None)
    assert cmd[cmd.index("--benchmark-gpus") + 1] == "2"
    assert cmd[cmd.index("--num-hidden-layers") + 1] == "8"


def test_probe_never_asks_for_more_gpus_than_the_target_has(config, enabled,
                                                            monkeypatch):
    monkeypatch.setenv(ENV_GPUS, "16")
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    cmd = _explore_screen._probe_command(GridVariant(name="v"), bench, "/tmp/o.json", None)
    assert cmd[cmd.index("--benchmark-gpus") + 1] == "8"


def test_probe_reads_decode_and_kernels_from_its_own_run(config, enabled, tmp_path,
                                                         monkeypatch):
    """The real _probe, with the subprocess replaced by a canned run."""
    artifact = {"sweep": [{"batch": 32, "decode_ms": 4.25}]}

    class Done:
        returncode = 0
        stdout = KERNEL_LOG
        stderr = ""

    def fake_run(cmd, **kwargs):
        json.dump(artifact, open(cmd[cmd.index("--save") + 1], "w"))
        return Done()

    monkeypatch.setattr(_explore_screen.subprocess, "run", fake_run)
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    reading, kernels = _explore_screen._probe(GridVariant(name="v"), bench, 60, None)
    assert reading == pytest.approx(4.25)
    assert kernels == MATCHING_KERNELS


def test_probe_failure_returns_no_reading(config, enabled, monkeypatch):
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(_explore_screen.subprocess, "run", lambda cmd, **kw: Failed())
    with open(config) as fh:
        import yaml
        bench = yaml.safe_load(fh)["benchmark"]
    reading, _ = _explore_screen._probe(GridVariant(name="v"), bench, 60, None)
    assert reading is None
