# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the shared provenance builder (hyperloom.common.provenance).

All hermetic: env is injected as a dict and ``probe=False`` disables every
subprocess/package/marker probe, so no ROCm/git/pkg presence is assumed.
"""

from __future__ import annotations

import argparse
import json
import sys

from hyperloom.common.provenance import (
    PROVENANCE_SOURCE,
    build_provenance,
    detect_gfx_arch,
    server_args_hash,
)


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_env_only_mapping():
    env = {
        "FRAMEWORK": "sglang",
        "GPU_TYPE": "MI355X",
        "HYPERLOOM_GFX_ARCH": "gfx950",
        "HYPERLOOM_GRAPH_MODE": "graph_capture",
        "TP": "1",
        "EP": "8",
        "PRECISION": "bf16",
        "CONC": "64",
        "ISL": "1024",
        "OSL": "1024",
        "MAX_MODEL_LEN": "2048",
    }
    p = build_provenance(args=None, env=env, probe=False)
    assert p["framework"] == "sglang"
    assert p["gpu_type"] == "MI355X"
    assert p["gfx_arch"] == "gfx950"
    assert p["graph_mode"] == "graph_capture"
    assert p["tp"] == 1 and p["ep"] == 8
    assert p["dtype"] == "bf16"
    assert p["concurrency"] == 64
    assert p["isl"] == 1024 and p["osl"] == 1024 and p["max_model_len"] == 2048


def test_args_override_env():
    env = {"FRAMEWORK": "sglang", "GPU_TYPE": "MI300X", "TP": "8", "ISL": "512"}
    args = _ns(
        framework="vllm",
        gpu_type="MI355X",
        tp=1,
        ep=4,
        isl=1024,
        osl=2048,
        precision="fp8",
        graph_mode="eager",
        model="/models/DeepSeek-V4-Flash",
        model_display_name="DeepSeek-V4-Flash",
    )
    p = build_provenance(args=args, env=env, probe=False)
    assert p["framework"] == "vllm"
    assert p["gpu_type"] == "MI355X"
    assert p["tp"] == 1 and p["ep"] == 4
    assert p["isl"] == 1024 and p["osl"] == 2048
    assert p["dtype"] == "fp8"
    assert p["graph_mode"] == "eager"
    assert p["model_path"] == "/models/DeepSeek-V4-Flash"
    assert p["model_name"] == "DeepSeek-V4-Flash"


def test_model_name_falls_back_to_basename():
    args = _ns(model="/shared_nfs/models/Qwen3-14B")
    p = build_provenance(args=args, env={}, probe=False)
    assert p["model_path"] == "/shared_nfs/models/Qwen3-14B"
    assert p["model_name"] == "Qwen3-14B"


def test_missing_degrades_to_none_never_raises():
    p = build_provenance(args=None, env={}, probe=False)
    for key in (
        "model_name",
        "model_path",
        "framework",
        "gpu_type",
        "gfx_arch",
        "graph_mode",
        "tp",
        "ep",
        "dtype",
        "concurrency",
        "isl",
        "osl",
        "max_model_len",
    ):
        assert p[key] is None
    assert p["stack_fingerprint"] == {"rocm": "unknown", "aiter": "unknown", "sglang": "unknown", "vllm": "unknown"}
    assert p["server_args"] == []
    assert p["server_args_hash"] == ""
    assert p["code_revision"] == ""
    assert p["image"] is None


def test_gfx_env_normalized_and_fallback():
    # A decorated arch string is normalized to the bare gfx token.
    assert detect_gfx_arch({"HYPERLOOM_GFX_ARCH": "gfx950:sramecc+:xnack-"}, probe=False) == "gfx950"
    # Priority falls back to GFX_ARCH.
    assert detect_gfx_arch({"GFX_ARCH": "gfx942"}, probe=False) == "gfx942"
    # No env + no probe -> None (never shells out to rocminfo).
    assert detect_gfx_arch({}, probe=False) is None


def test_gfx_ignores_build_target_env():
    """PYTORCH_ROCM_ARCH names compile targets, not the installed device.

    Both shapes were wrong on an MI355X: the multi-arch list resolved to its
    first entry (gfx90a, MI200), and a single-valued vendor-image setting was
    wrong while looking plausible. Either must fall through to the probe.
    """
    build_list = {"PYTORCH_ROCM_ARCH": "gfx90a;gfx942;gfx950;gfx1100"}
    assert detect_gfx_arch(build_list, probe=False) is None
    assert detect_gfx_arch({"PYTORCH_ROCM_ARCH": "gfx942"}, probe=False) is None
    # A real override still wins over the build target.
    assert detect_gfx_arch({**build_list, "HYPERLOOM_GFX_ARCH": "gfx950"}, probe=False) == "gfx950"


def test_gfx_resolves_from_gpu_type_without_probing():
    """--gpu-type answers the question rocminfo would, and is always present.

    Excluding PYTORCH_ROCM_ARCH left detection resting entirely on rocminfo,
    which is not on PATH after either install script -- so bare-metal nodes went
    from a wrong arch to no arch. gpu_type is fixed for the session and already
    the KB's hardware dimension, so it resolves without shelling out at all.
    """
    assert detect_gfx_arch({}, gpu_type="mi355x", probe=False) == "gfx950"
    assert detect_gfx_arch({}, gpu_type="MI300X", probe=False) == "gfx942"
    # The session env carries it too, when args are not threaded through.
    assert detect_gfx_arch({"GPU_TYPE": "mi325x"}, probe=False) == "gfx942"
    # An unrecognised board must not invent an answer.
    assert detect_gfx_arch({}, gpu_type="h100", probe=False) is None


def test_gfx_override_beats_gpu_type():
    """An explicit operator override outranks the board table."""
    assert detect_gfx_arch({"HYPERLOOM_GFX_ARCH": "gfx950"}, gpu_type="mi300x", probe=False) == "gfx950"


def test_gfx_gpu_type_short_circuits_the_probe(monkeypatch):
    """A resolvable gpu_type must not spawn rocminfo."""

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("rocminfo probed despite a known gpu_type")

    monkeypatch.setattr("hyperloom.common.provenance.subprocess.run", _boom)
    assert detect_gfx_arch({}, gpu_type="mi355x", probe=True) == "gfx950"


def test_server_args_hash_stable_and_sensitive():
    a = ["--tp", "1", "--quant", "fp8"]
    b = ["--tp", "1", "--quant", "fp8"]
    c = ["--tp", "8", "--quant", "fp8"]
    assert server_args_hash(a) == server_args_hash(b)
    assert server_args_hash(a) != server_args_hash(c)
    assert server_args_hash([]) == ""


def test_server_args_from_env_string_tokenized():
    p = build_provenance(args=None, env={"SERVER_ARGS": "--tp 1 --mem-fraction 0.9"}, probe=False)
    assert p["server_args"] == ["--tp", "1", "--mem-fraction", "0.9"]
    assert p["server_args_hash"]  # non-empty


def test_server_args_from_args_list_preferred_over_env():
    args = _ns(server_args=["--x", "1"])
    p = build_provenance(args=args, env={"SERVER_ARGS": "--y 2"}, probe=False)
    assert p["server_args"] == ["--x", "1"]


def test_source_tag_default_and_custom():
    assert build_provenance(env={}, probe=False)["_provenance_source"] == PROVENANCE_SOURCE
    assert build_provenance(env={}, probe=False, source="wp1_stub")["_provenance_source"] == "wp1_stub"


def test_int_coercion_rejects_garbage():
    p = build_provenance(env={"ISL": "abc", "TP": " 4 ", "OSL": ""}, probe=False)
    assert p["isl"] is None
    assert p["tp"] == 4
    assert p["osl"] is None


def test_stack_fingerprint_from_env():
    env = {"ROCM_VERSION": "6.5.0", "AITER_COMMIT": "abc123", "SGLANG_VERSION": "0.5.12"}
    p = build_provenance(env=env, probe=False)
    fp = p["stack_fingerprint"]
    assert fp["rocm"] == "6.5.0"
    assert fp["aiter"] == "abc123"
    assert fp["sglang"] == "0.5.12"
    assert fp["vllm"] == "unknown"


def test_json_serializable():
    p = build_provenance(args=_ns(model="/m/x", framework="sglang"), env={"TP": "1"}, probe=False)
    json.dumps(p)


# --- probe/marker branch coverage (WP-0) -----------------------------------
from types import SimpleNamespace  # noqa: E402

import hyperloom.common.provenance as _prov  # noqa: E402


def test_gfx_arch_probe_via_rocminfo(monkeypatch):
    # empty env -> falls through to the rocminfo subprocess probe.
    monkeypatch.setattr(
        _prov.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="  Name:  gfx950  \n"),
    )
    assert detect_gfx_arch({}, probe=True) == "gfx950"


def test_gfx_arch_probe_subprocess_absent(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("rocminfo not installed")

    monkeypatch.setattr(_prov.subprocess, "run", _boom)
    assert detect_gfx_arch({}, probe=True) is None


def test_gfx_arch_probe_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(_prov.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert detect_gfx_arch({}, probe=True) is None


def test_read_first_line(tmp_path):
    f = tmp_path / "v.txt"
    f.write_text("\n\n  first real line \nsecond\n", encoding="utf-8")
    assert _prov._read_first_line(f) == "first real line"
    # non-existent path degrades to "" (never raises)
    assert _prov._read_first_line(tmp_path / "nope.txt") == ""


def test_detect_image_none_when_absent(monkeypatch):
    # no image env vars and no marker files -> None (markers absent on runner).
    monkeypatch.setattr(_prov, "_read_first_line", lambda p: "")
    assert _prov.detect_image({}) is None


def test_detect_image_from_marker(monkeypatch):
    monkeypatch.setattr(_prov, "_read_first_line", lambda p: "myrepo/img:tag")
    assert _prov.detect_image({}) == "myrepo/img:tag"


def test_detect_image_probe_false_skips_markers(monkeypatch):
    # probe=False must be hermetic: no marker reads even if a marker would match
    # (else provenance becomes host-dependent and non-reproducible for hashing).
    def _boom(p):
        raise AssertionError("marker file must not be read when probe=False")

    monkeypatch.setattr(_prov, "_read_first_line", _boom)
    assert _prov.detect_image({}, probe=False) is None


def test_build_provenance_probe_false_image_hermetic(monkeypatch):
    # build_provenance(probe=False) must not read image markers.
    def _boom(p):
        raise AssertionError("marker read under probe=False")

    monkeypatch.setattr(_prov, "_read_first_line", _boom)
    prov = _prov.build_provenance(None, env={}, probe=False)
    assert prov["image"] is None


def test_stack_fingerprint_reads_rocm_marker(monkeypatch):
    # empty env + probe -> rocm resolves from the /opt/rocm marker file.
    monkeypatch.setattr(_prov, "_read_first_line", lambda p: "6.2.0")
    fp = _prov.detect_stack_fingerprint({}, probe=True)
    assert fp["rocm"] == "6.2.0"


def test_code_revision_falls_back_when_git_absent(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(_prov.subprocess, "run", _boom)
    assert _prov.detect_code_revision({"HYPERLOOM_CODE_REVISION": "envrev"}, probe=True) == "envrev"


# --- isolated framework venv ------------------------------------------------


def _installed(venv_root, name: str, version: str) -> str:
    """A distribution installed under ``venv_root`` and nowhere this process looks.

    Builds the ``bin/python -> python3.12 -> <base>`` symlink chain a real venv
    has, so a lookup that resolves the interpreter path lands on the base
    prefix (``/usr``) instead of the venv and finds no ``site-packages``. A
    fixture of plain non-existent paths cannot catch that: ``Path.resolve()``
    does not follow symlinks that are not there.

    Returns the interpreter path preflight would publish for that venv.
    """
    site = venv_root / "lib" / "python3.12" / "site-packages"
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n")
    bin_dir = venv_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python3.12").symlink_to(sys.executable)
    (bin_dir / "python").symlink_to("python3.12")
    return str(bin_dir / "python")


def test_a_framework_in_its_own_venv_is_still_versioned(tmp_path):
    """``--framework-env isolated`` is the default for vLLM, whose ROCm wheel
    pins its own torch. The orchestrator's interpreter cannot see that venv, so
    without following the interpreter preflight resolved, every bare-metal vLLM
    report recorded the framework it actually served with as "unknown"."""
    python_exe = _installed(tmp_path / "vllm-venv", "vllm", "0.27.1+rocm723")
    fp = _prov.detect_stack_fingerprint(
        {"HYPERLOOM_RESOLVED_FRAMEWORK": "vllm", "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON": python_exe}, probe=True
    )
    assert fp["vllm"] == "0.27.1+rocm723"


def test_an_operator_pin_still_wins_over_the_resolved_interpreter(tmp_path):
    python_exe = _installed(tmp_path / "vllm-venv", "vllm", "0.27.1+rocm723")
    fp = _prov.detect_stack_fingerprint(
        {
            "HYPERLOOM_RESOLVED_FRAMEWORK": "vllm",
            "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON": python_exe,
            "VLLM_VERSION": "0.28.0-rc1",
        },
        probe=True,
    )
    assert fp["vllm"] == "0.28.0-rc1"


def test_an_interpreter_whose_prefix_yields_nothing_falls_back(tmp_path):
    """No ``site-packages`` under the prefix means the derivation failed.

    A system prefix keeps packages in ``dist-packages`` and a vanished venv has
    no tree at all; treating either as an authoritative empty answer would
    report "unknown" for a framework this process can see. Only a prefix that
    really yields ``site-packages`` speaks for the run.
    """
    fp = _prov.detect_stack_fingerprint(
        {
            "HYPERLOOM_RESOLVED_FRAMEWORK": "vllm",
            "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON": str(tmp_path / "gone" / "bin" / "python"),
        },
        probe=True,
    )
    assert fp["vllm"] == _prov.detect_stack_fingerprint({}, probe=True)["vllm"]


def test_a_real_venv_without_the_distribution_is_authoritative(tmp_path):
    """A resolved venv that genuinely lacks the package must not be papered over.

    This process may well have its own copy, but the run is not served by it.
    """
    venv_root = tmp_path / "vllm-venv"
    (venv_root / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    bin_dir = venv_root / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)

    fp = _prov.detect_stack_fingerprint(
        {
            "HYPERLOOM_RESOLVED_FRAMEWORK": "vllm",
            "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON": str(bin_dir / "python"),
        },
        probe=True,
    )
    assert fp["vllm"] == "unknown"


def test_an_installer_venv_root_is_not_consulted(tmp_path):
    """Provenance records a resolution, it does not perform one.

    ``$VLLM_VENV_ROOT`` is host state the installer only ever writes, so on its
    own it cannot say whether that tree still holds vLLM. Preflight leads its
    candidate list with it and probes it; only that outcome is published here.
    Reading the raw variable would be a second, weaker discovery path.
    """
    root = tmp_path / "installer-venv"
    _installed(root, "vllm", "0.27.1+rocm723")
    fp = _prov.detect_stack_fingerprint({"VLLM_VENV_ROOT": str(root)}, probe=True)
    assert fp["vllm"] != "0.27.1+rocm723"


def test_the_venv_is_not_scanned_under_probe_false(tmp_path):
    """probe=False is the hermetic contract: env only, no filesystem."""
    python_exe = _installed(tmp_path / "vllm-venv", "vllm", "0.27.1+rocm723")
    fp = _prov.detect_stack_fingerprint(
        {"HYPERLOOM_RESOLVED_FRAMEWORK": "vllm", "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON": python_exe}, probe=False
    )
    assert fp["vllm"] == "unknown"


def test_the_lever_is_what_came_back_not_what_was_asked_for():
    """A config deliverable is a config lever, whoever dispatched it.

    Observed on a real MI355X session: the local-exploration arm asked for a
    source patch and the specialist returned ``--max-num-batched-tokens``. It
    was recorded as a framework source lesson with an empty ``changed_files``,
    and had it been kept, the source arm would have been credited for a
    configuration win.
    """
    from hyperloom.inference_optimizer.breakdown.agent_ownership import (
        LEVER_CONFIG,
        LEVER_SOURCE_PATCH,
        patch_lever_kind,
    )

    config_deliverable = {
        "framework_agent_candidate_id": "local_explore:0",
        "specialist_task_id": "t1",
        "extra_server_args": "--max-num-batched-tokens 16384",
        "extra_envs": {},
        "patch_name": "",
    }
    assert patch_lever_kind(config_deliverable) == LEVER_CONFIG

    # The same arm, when it really does write a diff.
    source_deliverable = {
        "framework_agent_candidate_id": "local_explore:0",
        "specialist_task_id": "t1",
        "patches_applied": ["/w/a.patch"],
        "extra_server_args": "",
    }
    assert patch_lever_kind(source_deliverable) == LEVER_SOURCE_PATCH


def test_an_explicit_stamp_still_outranks_the_derivation():
    """Callers that do know the lever keep saying so.

    An upstream PR carries server args in its integrate params on the way to
    the bench; the fetched diff is still what it moves.
    """
    from hyperloom.inference_optimizer.breakdown.agent_ownership import (
        LEVER_UPSTREAM_PR,
        patch_lever_kind,
    )

    assert patch_lever_kind({"lever_kind": "upstream_pr", "extra_server_args": "--x"}) == LEVER_UPSTREAM_PR
