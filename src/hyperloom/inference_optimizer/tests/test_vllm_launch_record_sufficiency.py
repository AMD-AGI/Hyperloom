# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A vLLM launch with ``--compilation-config`` is judged on what it ran.

The record below is a real vLLM 0.30 server log line, with the model path
shortened. Its ``compilation_config`` carries enum reprs Python cannot parse,
which used to make the whole record unreadable: every requested setting was
then unconfirmed and no such recipe could be judged sufficient.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.launch_log_evidence import observed_server_identity_from_log
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.enablement.recipe.steps import build_recipe_steps
from hyperloom.orchestrator.enablement.recipe.sufficiency import evaluate_replay_sufficiency
from hyperloom.orchestrator.enablement.recipe.projections import project_launch_evidence

MODEL = "/models/DeepSeek-V4-Pro"

VLLM_030_LOG = (
    "WARNING 10-11 08:31:35 [argparse_utils.py:441] Found duplicate keys --no-disable-hybrid-kv-cache-manager, "
    "--tokenizer-mode, --async-scheduling, --compilation-config, --distributed-executor-backend, "
    "--reasoning-parser, --enable-prefix-caching, --moe-backend, --data-parallel-size, --tool-call-parser, "
    "--enable-auto-tool-choice, --speculative-config\n"
    "(APIServer pid=87233) INFO 10-11 08:31:35 [api_utils.py:286] non-default args: "
    "{'model_tag': '/models/DeepSeek-V4-Pro', 'enable_auto_tool_choice': True, 'tool_call_parser': 'deepseek_v4', "
    "'model': '/models/DeepSeek-V4-Pro', 'tokenizer_mode': 'deepseek_v4', 'trust_remote_code': True, "
    "'max_model_len': 1048576, 'reasoning_parser': 'deepseek_v4', 'distributed_executor_backend': 'mp', "
    "'tensor_parallel_size': 8, 'gpu_memory_utilization': 0.86, 'kv_cache_dtype': 'fp8', "
    "'enable_prefix_caching': True, 'max_num_batched_tokens': 8192, 'max_num_seqs': 64, "
    "'disable_hybrid_kv_cache_manager': False, 'async_scheduling': True, 'moe_backend': 'aiter', "
    "'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2}, "
    "'compilation_config': {'mode': <CompilationMode.VLLM_COMPILE: 3>, 'debug_dump_path': None, 'cache_dir': '', "
    "'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'ir_enable_torch_wrap': None, "
    "'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, "
    "'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, "
    "'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': None, 'compile_ranges_endpoints': None, "
    "'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, "
    "'benchmark_combo_kernel': True}, 'inductor_passes': {}, "
    "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 0, "
    "'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, "
    "'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, "
    "'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, "
    "'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, "
    "'static_all_moe_layers': []}}\n"
    "(APIServer pid=87233) INFO 10-11 08:31:40 [model.py:1] Resolved architecture: DeepseekV4ForCausalLM\n"
)

#: The accepted launch the same session requested.
REQUESTED = (
    "--async-scheduling --distributed-executor-backend mp --kv-cache-dtype fp8 --max-num-batched-tokens 8192 "
    '--data-parallel-size 1 --moe-backend aiter --compilation-config {"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"} '
    '--speculative-config {"method":"mtp","num_speculative_tokens":2} --tokenizer-mode deepseek_v4 '
    "--tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice --enable-prefix-caching "
    "--no-disable-hybrid-kv-cache-manager --max-num-seqs 64"
)


def _log(tmp_path: Path, text: str = VLLM_030_LOG) -> str:
    path = tmp_path / "server.log"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _launch_codes(tmp_path: Path, requested: str = REQUESTED, log_text: str = VLLM_030_LOG) -> list[tuple[str, str]]:
    """Judge one launch through the production entry points, KEEP to verdict."""
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    evidence = build_launch_evidence(
        RequestedLaunch(config, "vllm", server_args=requested, model_path=MODEL),
        actual_server_log=_log(tmp_path, log_text),
        slot=tmp_path / "slot",
    )
    projected, refused = project_launch_evidence(evidence)
    assert refused is False
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    decision = evaluate_replay_sufficiency({}, steps=build_recipe_steps({}), section=section)
    return [
        (r["code"], r["scope"])
        for r in decision["reasons"]
        if r["code"] in ("activation_incomplete", "launch_evidence_mismatch")
    ]


def test_enum_reprs_do_not_void_the_vllm_record(tmp_path: Path) -> None:
    identity = observed_server_identity_from_log(_log(tmp_path), "vllm")

    assert identity["model"] == MODEL
    assert identity["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 2}
    # Kept verbatim: the value still says which member it was.
    assert identity["compilation_config"]["mode"] == "<CompilationMode.VLLM_COMPILE: 3>"
    assert identity["compilation_config"]["dynamic_shapes_config"]["type"] == "<DynamicShapesType.BACKED: 'backed'>"
    # Every non-default setting, not an allowlisted few.
    for name in ("async_scheduling", "moe_backend", "tool_call_parser", "disable_hybrid_kv_cache_manager"):
        assert name in identity, name


def test_angle_brackets_inside_a_string_are_left_alone(tmp_path: Path) -> None:
    log = _log(tmp_path, "INFO non-default args: {'model': '/m', 'chat_template': 'a <b> c', 'x': <E.Y: 1>}\n")
    identity = observed_server_identity_from_log(log, "vllm")
    assert identity == {"model": "/m", "chat_template": "a <b> c", "x": "<E.Y: 1>"}


def test_the_real_vllm_launch_confirms_every_requested_setting(tmp_path: Path) -> None:
    """Before: 16 ``activation_incomplete`` reasons, one per requested setting."""
    assert _launch_codes(tmp_path) == []


def test_a_width_absent_from_the_record_is_vllms_default(tmp_path: Path) -> None:
    """``--data-parallel-size 1`` is a default, so vLLM never records it."""
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    evidence = build_launch_evidence(
        RequestedLaunch(config, "vllm"), actual_server_log=_log(tmp_path), slot=tmp_path / "slot"
    )
    assert evidence["observed_model_binding"]["tp"] == "8"
    assert evidence["observed_model_binding"]["dp"] == "1"


def test_a_json_setting_is_compared_as_structure_and_can_still_contradict(tmp_path: Path) -> None:
    wrong_mode = REQUESTED.replace('{"mode":3,', '{"mode":2,')
    assert _launch_codes(tmp_path, wrong_mode) == [("launch_evidence_mismatch", "requested:compilation_config")]

    wrong_graph = REQUESTED.replace('"FULL_AND_PIECEWISE"', '"PIECEWISE"')
    assert _launch_codes(tmp_path, wrong_graph) == [("launch_evidence_mismatch", "requested:compilation_config")]

    wrong_spec = REQUESTED.replace('"num_speculative_tokens":2', '"num_speculative_tokens":3')
    assert _launch_codes(tmp_path, wrong_spec) == [("launch_evidence_mismatch", "requested:speculative_config")]


def test_strings_inside_a_json_setting_compare_exactly(tmp_path: Path) -> None:
    """An architecture name is case-sensitive; folding case passed a launch that ran another one."""
    record = VLLM_030_LOG.replace(
        "'max_num_seqs': 64,", "'max_num_seqs': 64, 'hf_overrides': {'architectures': ['WrongCase']},"
    )
    asked = REQUESTED + ' --hf-overrides {"architectures":["wrongcase"]}'
    assert _launch_codes(tmp_path, asked, record) == [("launch_evidence_mismatch", "requested:hf_overrides")]
    exact = REQUESTED + ' --hf-overrides {"architectures":["WrongCase"]}'
    assert _launch_codes(tmp_path, exact, record) == []


def test_a_boolean_flag_must_be_recorded_as_set(tmp_path: Path) -> None:
    off = VLLM_030_LOG.replace("'enable_prefix_caching': True", "'enable_prefix_caching': False")
    assert _launch_codes(tmp_path, log_text=off) == [("launch_evidence_mismatch", "requested:enable_prefix_caching")]

    on = VLLM_030_LOG.replace("'disable_hybrid_kv_cache_manager': False", "'disable_hybrid_kv_cache_manager': True")
    assert _launch_codes(tmp_path, log_text=on) == [
        ("launch_evidence_mismatch", "requested:no_disable_hybrid_kv_cache_manager")
    ]


def test_a_requested_setting_the_record_lacks_is_still_unconfirmed(tmp_path: Path) -> None:
    missing = VLLM_030_LOG.replace("'moe_backend': 'aiter', ", "")
    assert _launch_codes(tmp_path, log_text=missing) == [("activation_incomplete", "requested:moe_backend")]


def test_the_published_identity_holds_only_what_the_launch_asked_for(tmp_path: Path) -> None:
    """A setting nobody requested is not published, whatever the record holds."""
    record = VLLM_030_LOG.replace(
        "'max_num_seqs': 64,",
        "'max_num_seqs': 64, 'kv_transfer_config': {'kv_connector_extra_config': "
        "{'socket_path': 'unix:///run/private.sock'}},",
    )
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    evidence = build_launch_evidence(
        RequestedLaunch(config, "vllm", server_args=REQUESTED),
        actual_server_log=_log(tmp_path, record),
        slot=tmp_path / "slot",
    )
    assert "kv_transfer_config" in evidence["observed_server_identity"], "observed in full"
    identity = project_launch_evidence(evidence)[0]["observed_server_identity"]
    assert "kv_transfer_config" not in identity and "private.sock" not in str(identity)
    assert identity["disable_hybrid_kv_cache_manager"] is False, "--no-X publishes X"
    assert identity["compilation_config"]["mode"] == "<CompilationMode.VLLM_COMPILE: 3>"
    # The record's other settings, requested by nobody, are not published either.
    assert "gpu_memory_utilization" not in identity and "trust_remote_code" not in identity


def test_the_published_identity_carries_no_host_path_or_secret(tmp_path: Path) -> None:
    log_text = VLLM_030_LOG.replace(
        "'max_num_seqs': 64,",
        "'max_num_seqs': 64, 'api_key': 'shh', 'chat_template': '/srv/t.jinja', "
        "'hf_overrides': {'host_origin': {'trace': '/srv/private.log'}},",
    )
    config = tmp_path / "bench.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    evidence = build_launch_evidence(
        # Requested, so it is the sanitizer and not the requested-only filter that removes them.
        RequestedLaunch(
            config, "vllm", server_args="--max-num-seqs 64 --api-key x --chat-template t.jinja --hf-overrides {}"
        ),
        actual_server_log=_log(tmp_path, log_text),
        slot=tmp_path / "slot",
    )
    projected, _ = project_launch_evidence(evidence)
    identity = projected["observed_server_identity"]
    assert identity["max_num_seqs"] == 64
    assert not {"model", "model_tag", "api_key", "chat_template"}.intersection(identity)
    assert MODEL not in str(projected) and "shh" not in str(projected) and "/srv/" not in str(projected)
