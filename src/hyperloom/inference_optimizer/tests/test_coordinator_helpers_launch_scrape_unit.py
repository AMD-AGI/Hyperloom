# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for launch-flag and config-blob helper primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common.launch_log_evidence import launch_argv_from_log, split_launch_flags
from hyperloom.orchestrator.loop import coordinator_helpers as ch


# ── _split_env_and_flags ──────────────────────────────────────────────────


def test_split_env_and_flags_mixed_tokens() -> None:
    envs, flags = ch._split_env_and_flags("FOO=1 BAR=baz --chunked-prefill-size=2048 --disable-radix-cache")
    assert envs == {"FOO": "1", "BAR": "baz"}
    assert flags == "--chunked-prefill-size=2048 --disable-radix-cache"


def test_split_env_and_flags_empty_input() -> None:
    assert ch._split_env_and_flags("") == ({}, "")
    assert ch._split_env_and_flags(None) == ({}, "")


def test_split_env_and_flags_only_env() -> None:
    envs, flags = ch._split_env_and_flags("A=1 B=2")
    assert envs == {"A": "1", "B": "2"}
    assert flags == ""


def test_split_env_and_flags_only_flags() -> None:
    envs, flags = ch._split_env_and_flags("--flag-a --flag-b=1")
    assert envs == {}
    assert flags == "--flag-a --flag-b=1"


def test_split_env_and_flags_falls_back_on_shlex_error() -> None:
    # An unbalanced quote makes shlex.split raise, so the ``.split()`` fallback runs; the unterminated token starts
    # with "-" and lands in ``flags``.
    envs, flags = ch._split_env_and_flags('FOO=1 --flag="unterminated')
    assert envs["FOO"] == "1"
    assert flags == '--flag="unterminated'


def test_accepted_config_uses_published_env_map_for_revalidation() -> None:
    from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    config = {
        "flags": "--mem-fraction-static 0.95",
        "env": 'SGLANG_USE_AITER=1 RUN_EVAL=true; BACKEND=sglang; EXTRA_ENV=")." --discarded-prose',
        "env_map": {"SGLANG_USE_AITER": "1", "RUN_EVAL": "true", "BACKEND": "sglang"},
        "env_unparsed": ["EXTRA_ENV=).", "--discarded-prose"],
    }
    flags, envs = KernelPhase._parse_geak_accepted_config({"accepted_config": config})
    variant = GridVariant(name="geak_revalidate", extra_server_args=flags, extra_envs=envs)
    assert variant.extra_server_args == config["flags"]
    assert variant.extra_envs == config["env_map"]
    assert variant.extra_envs["RUN_EVAL"] == "true"
    assert not ch._geak_result_has_material({"accepted_config": config}, prev_best_flags=flags, prev_best_envs=envs)


def test_empty_accepted_env_map_does_not_restore_raw_assignments() -> None:
    config = {"env": "RUN_EVAL=false SGLANG_USE_AITER=1 --disable-cuda-graph", "env_map": {}}
    assert ch._accepted_config_as_variant(config) == ("--disable-cuda-graph", {})
    assert ch._geak_result_has_material({"accepted_config": config}, prev_best_envs={"RUN_EVAL": "true"})


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("unparsed", [False, True])
def test_legacy_flags_and_discarded_text_are_reported(caplog, mapped, unparsed):
    config = {"env": "SGLANG_USE_AITER=1 --disable-cuda-graph"}
    if mapped:
        config["env_map"] = {}
    if unparsed:
        config["env_unparsed"] = ["discarded prose"]
    assert ch._accepted_config_as_variant(config) == (
        "--disable-cuda-graph",
        {} if mapped else {"SGLANG_USE_AITER": "1"},
    )
    assert "retaining them alongside accepted_config.flags" in caplog.text
    assert ("reports discarded source text" in caplog.text) == unparsed
    assert "using only validated env_map" not in caplog.text


@pytest.mark.parametrize("artifact", ["accepted_kernels", "accepted_heads", "final_overlay", "final_patch"])
def test_invalid_config_does_not_hide_artifact_materiality(artifact):
    assert ch._geak_result_has_material({artifact: ["product"], "accepted_config": {"env_map": None}})


def test_invalid_config_is_not_reported_as_no_material():
    with pytest.raises(ValueError, match="env_map"):
        ch._geak_result_has_material({"accepted_config": {"env_map": None}})


@pytest.mark.parametrize("removals", [None, [], ["--disable-radix-cache"]])
def test_complete_config_materiality_inherits_omitted_removals(removals):
    config = {"flags": "--mem-fraction-static 0.95", "env_map": {}, "args_mode": "replace"}
    prior = {"args_mode": "replace", "remove_args": ["--disable-radix-cache"]}
    if removals is not None:
        config["remove_args"] = removals
    assert ch._geak_result_has_material(
        {"accepted_config": config},
        prev_best_flags=config["flags"],
        prev_best_controls=prior,
    ) == (removals == [])


def test_accepted_env_map_preserves_values_and_filters_loader_keys() -> None:
    value = '{"path": "a b", "pattern": "x=y;z"}'
    config = {"env_map": {"SGLANG_TEST_CONFIG": value, "PYTHONPATH": "/untrusted"}}
    assert ch._accepted_config_as_variant(config) == ("", {"SGLANG_TEST_CONFIG": value})


@pytest.mark.parametrize("env_map", [None, "RUN_EVAL=true", [], {"RUN_EVAL": True}, {1: "x"}, {"bad-key": "1"}])
def test_invalid_accepted_env_map_does_not_fall_back_to_raw_string(env_map) -> None:
    with pytest.raises(ValueError, match="env_map must map strings to strings"):
        ch._accepted_config_as_variant({"env": "RUN_EVAL=true;", "env_map": env_map})


def test_legacy_accepted_config_still_parses_env_and_flags() -> None:
    assert ch._accepted_config_as_variant(
        {"flags": "--mem-fraction-static 0.95", "env": "SGLANG_USE_AITER=1 --disable-radix-cache"}
    ) == ("--mem-fraction-static 0.95 --disable-radix-cache", {"SGLANG_USE_AITER": "1"})


@pytest.mark.parametrize("mapped", [False, True])
def test_legacy_flags_keep_values_and_literal_json(mapped):
    import shlex

    value = '{"path": "a b", "pattern": "x=y"}'
    config = {"env": shlex.join(["SGLANG_USE_AITER=1", "--limit", "64", "--config", value])}
    if mapped:
        config["env_map"] = {}
    flags, envs = ch._accepted_config_as_variant(config)
    assert shlex.split(flags) == ["--limit", "64", "--config", value]
    assert envs == ({} if mapped else {"SGLANG_USE_AITER": "1"})


# ── _geak_sweep_measured_tput ─────────────────────────────────────────────


def test_geak_sweep_measured_tput_prefers_the_promotion_measurement() -> None:
    res = {
        "promotion_measurement": {"output_throughput": 150.0},
        "points": [{"status": "succeeded", "output_throughput": 999.0}],
    }
    assert ch._geak_sweep_measured_tput(res) == 150.0


def test_geak_sweep_measured_tput_none_when_not_dict() -> None:
    assert ch._geak_sweep_measured_tput(None) is None
    assert ch._geak_sweep_measured_tput([]) is None  # type: ignore[arg-type]


def test_geak_sweep_measured_tput_none_when_no_positive_throughput() -> None:
    res = {
        "promotion_measurement": {"output_throughput": 0},
        "points": [{"status": "succeeded", "output_throughput": -1}],
    }
    assert ch._geak_sweep_measured_tput(res) is None


# ── split_launch_flags ────────────────────────────────────────────────────


def test_split_launch_flags_strips_run_specific_space_form() -> None:
    argv = "--model-path /models/x --tensor-parallel-size 8 --mem-fraction-static 0.9"
    assert split_launch_flags(argv) == "--mem-fraction-static 0.9"


def test_split_launch_flags_strips_equals_form() -> None:
    argv = "--host=0.0.0.0 --port=30000 --disable-radix-cache"
    assert split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_strips_profiling_flags() -> None:
    argv = "--enable-profile --chunked-prefill-size 2048"
    assert split_launch_flags(argv) == "--chunked-prefill-size 2048"


def test_split_launch_flags_handles_valueless_run_specific_flag() -> None:
    # ``--pid`` followed by another flag: the run-specific flag is dropped without eating the next flag.
    argv = "--pid --disable-radix-cache"
    assert split_launch_flags(argv) == "--disable-radix-cache"


def test_split_launch_flags_falls_back_on_shlex_error() -> None:
    out = split_launch_flags('--mem-fraction-static 0.9 "unterminated')
    assert "--mem-fraction-static" in out


# ── launch_argv_from_log ──────────────────────────────────────────────────


def test_launch_argv_from_log_extracts_and_strips(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "some preamble\n"
        "+ python3 -m sglang.launch_server --model-path /models/x "
        "--tensor-parallel-size 8 --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "sglang")
    assert flags == "--mem-fraction-static 0.9"


def test_launch_argv_from_log_returns_empty_when_marker_absent(
    tmp_path: Path,
) -> None:
    log = tmp_path / "server.log"
    log.write_text("no engine launch here\n", encoding="utf-8")
    assert launch_argv_from_log(str(log), "sglang") == ""


def test_launch_argv_from_log_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert launch_argv_from_log(str(tmp_path / "nope.log"), "sglang") == ""


def test_launch_argv_from_log_returns_empty_for_unmarked_framework(
    tmp_path: Path,
) -> None:
    # A framework with no registered argv marker never reads the log.
    log = tmp_path / "server.log"
    log.write_text(
        "+ python3 -m sglang.launch_server --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    assert launch_argv_from_log(str(log), "xdit") == ""
    assert launch_argv_from_log(str(log), "") == ""


def test_launch_argv_from_log_falls_back_to_double_dash_scan(
    tmp_path: Path,
) -> None:
    # No regex match, but the line has a "--" run after the marker → the ``line.find("--")`` fallback path is
    # exercised.
    log = tmp_path / "server.log"
    log.write_text(
        "vllm serve --model-path /models/x --mem-fraction-static 0.9\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "vllm")
    assert "--mem-fraction-static 0.9" in flags


# ── vLLM: the launch line this reader has to be able to pass ──────────────


def test_vllm_module_form_yields_observed_flags(tmp_path: Path) -> None:
    """Keyed on ``--model-path``, this line could never pass the gate, so every
    vLLM session produced empty observed flags and a verdict that was
    insufficient by construction rather than by evidence."""
    log = tmp_path / "server.log"
    log.write_text(
        "INFO 09-13 10:00:00 [api_server.py:1] python3 -m vllm.entrypoints.openai.api_server "
        "--model /models/glm5 --max-num-seqs 256 --enable-chunked-prefill\n",
        encoding="utf-8",
    )
    flags = launch_argv_from_log(str(log), "vllm")
    assert flags == "--max-num-seqs 256 --enable-chunked-prefill"
    assert "/models/glm5" not in flags


def test_vllm_serve_form_yields_observed_flags_without_the_model(tmp_path: Path) -> None:
    """``vllm serve <model>`` carries the model as a positional, which the
    run-specific FLAG list cannot reach; left in it would put a host model path
    in the durable record."""
    log = tmp_path / "server.log"
    log.write_text("INFO: vllm serve /models/glm5 --max-num-seqs 256 --tensor-parallel-size 8\n", encoding="utf-8")
    flags = launch_argv_from_log(str(log), "vllm")
    assert flags == "--max-num-seqs 256"
    assert "/models/glm5" not in flags and "serve" not in flags


def test_a_line_that_names_no_model_is_not_a_launch_line(tmp_path: Path) -> None:
    """The gate still has to reject a passing mention of the marker; it was
    widened to every model spelling, not removed."""
    log = tmp_path / "server.log"
    log.write_text("INFO: vllm is starting up --max-num-seqs 256\n", encoding="utf-8")
    assert launch_argv_from_log(str(log), "vllm") == ""


def test_a_model_prefixed_flag_does_not_pass_the_gate(tmp_path: Path) -> None:
    """``--model`` is a prefix of ``--model-loader-extra-config``; a substring
    test would read that as the model operand."""
    log = tmp_path / "server.log"
    log.write_text("INFO: vllm --model-loader-extra-config {} --max-num-seqs 256\n", encoding="utf-8")
    assert launch_argv_from_log(str(log), "vllm") == ""


def test_sglang_model_path_still_passes_the_gate(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "INFO: python3 -m sglang.launch_server --model-path /models/glm5 --chunked-prefill-size 2048\n",
        encoding="utf-8",
    )
    assert launch_argv_from_log(str(log), "sglang") == "--chunked-prefill-size 2048"


# ── vLLM: the record it actually writes ───────────────────────────────────


_VLLM_NON_DEFAULT = (
    "(APIServer pid=1) INFO 09-14 07:03:07 [utils.py:233] non-default args: "
    "{'model_tag': '/models/m', 'port': 38035, 'model': '/models/m', "
    "'trust_remote_code': True, 'max_model_len': 4096, 'tensor_parallel_size': 4, "
    "'kv_cache_dtype': 'fp8', 'gpu_memory_utilization': 0.95, "
    "'compilation_config': CompilationConfig(level=3, backend='inductor')}\n"
)


def test_vllm_identity_is_read_from_the_record_vllm_actually_writes(tmp_path: Path) -> None:
    """vLLM prints no argv line in any log, successful or failed. It prints the
    RESOLVED argument dict, which is a better observed record than a command
    line -- it is what the parser produced rather than what was typed.

    Without this reader ``observed_server_launch_flags`` and
    ``observed_model_binding`` are empty for every vLLM session, every requested
    setting is judged unconfirmed, and the replay verdict is insufficient by
    construction rather than by evidence.
    """
    from hyperloom.common.launch_log_evidence import observed_vllm_server_identity_from_log

    log = tmp_path / "server.log"
    log.write_text(_VLLM_NON_DEFAULT, encoding="utf-8")
    identity = observed_vllm_server_identity_from_log(str(log))

    assert identity["model"] == "/models/m"
    assert identity["tensor_parallel_size"] == 4
    assert identity["max_model_len"] == 4096
    assert identity["kv_cache_dtype"] == "fp8"
    # There is no argv line to find, and the argv reader must not invent one.
    assert launch_argv_from_log(str(log), "vllm") == ""


def test_a_non_literal_value_skips_its_key_rather_than_the_whole_record(tmp_path: Path) -> None:
    """vLLM prints object reprs inside that dict -- ``CompilationConfig(...)``.
    A single ``literal_eval`` of the dict raises on the first one and yields
    nothing, which is what makes the whole record look unreadable."""
    from hyperloom.common.launch_log_evidence import observed_vllm_server_identity_from_log

    log = tmp_path / "server.log"
    log.write_text(_VLLM_NON_DEFAULT, encoding="utf-8")
    identity = observed_vllm_server_identity_from_log(str(log))

    assert "compilation_config" not in identity
    assert identity["model"] == "/models/m", "the literal keys must survive the non-literal one"


def test_the_vllm_binding_carries_the_width_and_digests_the_model(tmp_path: Path) -> None:
    import hashlib

    from hyperloom.common.launch_log_evidence import observed_vllm_server_identity_from_log
    from hyperloom.orchestrator.actions.executors._launch_evidence import _binding_from_vllm_identity

    log = tmp_path / "server.log"
    log.write_text(_VLLM_NON_DEFAULT, encoding="utf-8")
    binding = _binding_from_vllm_identity(observed_vllm_server_identity_from_log(str(log)))

    assert binding["tp"] == "4"
    assert binding["model_digest"] == "sha256:" + hashlib.sha256(b"/models/m").hexdigest()
    assert "/models/m" not in str(binding), "a host model path must not travel in the binding"


def test_a_log_with_no_such_record_yields_no_vllm_identity(tmp_path: Path) -> None:
    from hyperloom.common.launch_log_evidence import observed_vllm_server_identity_from_log

    log = tmp_path / "server.log"
    log.write_text("(APIServer pid=1) INFO starting up\n", encoding="utf-8")
    assert observed_vllm_server_identity_from_log(str(log)) == {}


def _vllm_slot(tmp_path, log_text: str):
    """A minimal measured-launch slot with a vLLM server log."""
    slot = tmp_path / "slot"
    slot.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    log = slot / "server.log"
    log.write_text(log_text, encoding="utf-8")
    return config, slot, str(log)


def test_a_vllm_launch_reaches_the_evidence_through_the_production_entry_point(tmp_path):
    """Pins the WIRING, not just the reader.

    The reader and the binding helper were each covered directly, so deleting
    the whole vLLM branch out of ``build_launch_evidence`` left every one of
    those tests passing while no vLLM launch was bound to anything. This goes
    through the production entry point, which is the only thing that fails when
    the branch is removed.
    """
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, slot, log = _vllm_slot(
        tmp_path,
        "INFO 09-14 07:00:00 [config.py:1] non-default args: "
        "{'model': '/models/m', 'tensor_parallel_size': 4, 'quantization': 'fp8'}\n",
    )
    evidence = build_launch_evidence(
        config_path=config,
        actual_server_log=log,
        framework="vllm",
        slot=slot,
        model_path="/models/m",
    )
    identity = evidence["observed_server_identity"]
    assert identity["tensor_parallel_size"] == 4
    assert identity["quantization"] == "fp8"
    assert evidence["observed_model_binding"]


def test_a_quoted_marker_is_not_accepted_as_a_vllm_launch_through_the_entry_point(tmp_path):
    """User- or attacker-supplied text echoed into the log quotes the marker
    but carries no launch record; binding to it reports settings the server
    never ran with."""
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, slot, log = _vllm_slot(
        tmp_path,
        "WARNING 09-14 07:00:00 [x.py:1] ignored user text: "
        "\"non-default args: {'model': '/wanted', 'tensor_parallel_size': 4}\"\n",
    )
    evidence = build_launch_evidence(
        config_path=config,
        actual_server_log=log,
        framework="vllm",
        slot=slot,
        model_path="/wanted",
    )
    assert not evidence["observed_server_identity"]


def test_a_preceding_dict_does_not_displace_the_real_vllm_record(tmp_path):
    """Extraction anchored at the line's first brace reads an unrelated dict
    and ignores the actual record that follows it on the same line."""
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, slot, log = _vllm_slot(
        tmp_path,
        "INFO 09-14 07:00:00 [config.py:1] context={'model': '/wanted', 'tensor_parallel_size': 8} "
        "non-default args: {'model': '/actual', 'tensor_parallel_size': 2}\n",
    )
    evidence = build_launch_evidence(
        config_path=config,
        actual_server_log=log,
        framework="vllm",
        slot=slot,
        model_path="/actual",
    )
    assert evidence["observed_server_identity"]["tensor_parallel_size"] == 2


def test_a_brace_inside_a_vllm_model_path_does_not_truncate_the_record(tmp_path):
    """A ``}`` inside a string is not structure; ending the payload there drops
    every field after it, including the parallelism the decision compares."""
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config, slot, log = _vllm_slot(
        tmp_path,
        "INFO 09-14 07:00:00 [config.py:1] non-default args: {'model': '/models/m}', 'tensor_parallel_size': 4}\n",
    )
    evidence = build_launch_evidence(
        config_path=config,
        actual_server_log=log,
        framework="vllm",
        slot=slot,
        model_path="/models/m}",
    )
    assert evidence["observed_server_identity"]["tensor_parallel_size"] == 4
