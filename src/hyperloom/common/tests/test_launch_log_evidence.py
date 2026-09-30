# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regressions for bounded SGLang configuration evidence from launch logs."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import launch_log_evidence as evidence


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_server_args_records_preserve_typed_identity(tmp_path: Path, form: str) -> None:
    fields = {
        "model_path": "/models/qwen",
        "tokenizer_path": "/models/tokenizer",
        "served_model_name": "qwen",
        "tp_size": 1,
        "dp_size": 1,
        "mem_fraction_static": 0.8,
        "context_length": 6144,
        "chunked_prefill_size": 16384,
        "quantization": None,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "attention_backend": "aiter",
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        "disable_radix_cache": True,
        "trust_remote_code": False,
    }
    record = (
        repr(fields) if form == "dictionary" else f"ServerArgs({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"[2026-09-17 00:00:00] server_args={record}\n", encoding="utf-8")

    identity = evidence.observed_sglang_server_identity_from_log(str(log))

    assert identity == fields
    assert list(identity) == sorted(fields)
    assert type(identity["tp_size"]) is int
    assert type(identity["mem_fraction_static"]) is float
    assert identity["disable_radix_cache"] is True
    assert identity["trust_remote_code"] is False
    assert identity["quantization"] is None
    assert evidence.launch_argv_from_log(str(log), "sglang") == ""


@pytest.mark.parametrize(
    "record",
    [
        """{
    'model_path': '/models/{qwen}(v1)[test]',
    'served_model_name': "qwen's } ) ] \\\"alias\\\"",
    'tp_size': 8,
    'unknown_config': {'nested': [(1, 2), {'delimiters': '}])'}]},
}""",
        """ServerArgs(
    model_path='/models/{qwen}(v1)[test]',
    served_model_name="qwen's } ) ] \\\"alias\\\"",
    tp_size=8,
    unknown_config={'nested': [(1, 2), {'delimiters': '}])'}]},
)""",
    ],
    ids=["dictionary", "server_args"],
)
def test_multiline_records_ignore_quoted_delimiters(tmp_path: Path, record: str) -> None:
    log = tmp_path / "server.log"
    log.write_text(f"startup\nINFO server_args = {record} trailing log text\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {
        "model_path": "/models/{qwen}(v1)[test]",
        "served_model_name": 'qwen\'s } ) ] "alias"',
        "tp_size": 8,
    }


def test_dictionary_records_exclude_sensitive_and_unknown_fields(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "server_args={'tp_size': 1, 'api_key': 'secret', 'host': 'private-host', "
        "'port': 30000, 'disable_cuda_graph': False, 'max_running_requests': None, "
        "'max_total_tokens': None, 'env': {'ACCESS_TOKEN': 'secret'}, "
        "'future_option': future_runtime_object()}\n",
        encoding="utf-8",
    )

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {"tp_size": 1}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_executable_values_are_never_evaluated(tmp_path: Path, form: str) -> None:
    sentinel = tmp_path / "executed"
    executable = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    record = f"{{'model_path': {executable}}}" if form == "dictionary" else f"ServerArgs(model_path={executable})"
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "record",
    [
        "{'tp_size': 1",
        "{'tp_size': 1]}",
        "{'model_path': 'unterminated}",
        "{'tp_size' 1}",
        "{'tp_size': 1, **other}",
        "{'tp_size': 1, key: 'value'}",
        "{'tp_size': 1, 2: 'value'}",
        "{key: 1 for key in ['tp_size']}",
        "ServerArgs(tp_size=1, **other)",
        "ServerArgs(tp_size=1]",
    ],
    ids=[
        "unclosed",
        "mismatched",
        "unterminated-string",
        "syntax",
        "unpacking",
        "nonliteral-key",
        "nonstring-key",
        "comprehension",
        "legacy-unpacking",
        "legacy-mismatched",
    ],
)
def test_malformed_or_nonliteral_records_are_unavailable(tmp_path: Path, record: str) -> None:
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
@pytest.mark.parametrize(
    "value",
    [
        "runtime_name",
        "object.attribute",
        "1 + 1",
        "b'bytes'",
        "{1, 2}",
        "[[[[[1]]]]]",
        repr("x" * 4097),
        repr([1] * 65),
        repr({str(index): index for index in range(65)}),
        "{1: 'nonstring nested key'}",
        "[" * 250 + "1" + "]" * 250,
        "+".join(["1"] * 10000),
    ],
    ids=[
        "name",
        "attribute",
        "expression",
        "bytes",
        "set",
        "too-deep",
        "long-string",
        "long-list",
        "large-mapping",
        "nested-nonstring-key",
        "parser-nesting-limit",
        "parser-recursion-limit",
    ],
)
def test_disallowed_or_oversized_values_reject_the_record(tmp_path: Path, form: str, value: str) -> None:
    record = (
        f"{{'tp_size': 1, 'served_model_name': {value}}}"
        if form == "dictionary"
        else f"ServerArgs(tp_size=1, served_model_name={value})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_supported_nested_values_remain_bounded_and_json_safe(tmp_path: Path, form: str) -> None:
    value = "{'z': ('name', None), 'a': [True, {'size': -1}]}"
    record = f"{{'served_model_name': {value}}}" if form == "dictionary" else f"ServerArgs(served_model_name={value})"
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {
        "served_model_name": {"a": [True, {"size": -1}], "z": ["name", None]}
    }


@pytest.mark.parametrize("form", ["dictionary", "server_args"])
def test_top_level_field_count_is_capped(tmp_path: Path, form: str) -> None:
    fields = {"tp_size": 1, **{f"unknown_{index}": None for index in range(2048)}}
    record = (
        repr(fields) if form == "dictionary" else f"ServerArgs({', '.join(f'{k}={v!r}' for k, v in fields.items())})"
    )
    log = tmp_path / "server.log"
    log.write_text(f"server_args={record}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == {}


@pytest.mark.parametrize("extra", [0, 1], ids=["at-cap", "past-cap"])
def test_log_character_scan_is_capped(tmp_path: Path, extra: int) -> None:
    record = "server_args={'tp_size': 1}"
    log = tmp_path / "server.log"
    log.write_text(" " * (512 * 1024 - len(record) + extra) + record, encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == ({"tp_size": 1} if extra == 0 else {})


@pytest.mark.parametrize("lines", [2047, 2048], ids=["at-cap", "past-cap"])
def test_log_line_scan_is_capped(tmp_path: Path, lines: int) -> None:
    log = tmp_path / "server.log"
    log.write_text("startup noise\n" * lines + "server_args={'tp_size': 1}\n", encoding="utf-8")

    assert evidence.observed_sglang_server_identity_from_log(str(log)) == ({"tp_size": 1} if lines == 2047 else {})


def test_missing_log_is_unavailable(tmp_path: Path) -> None:
    assert evidence.observed_sglang_server_identity_from_log(str(tmp_path / "missing.log")) == {}


@pytest.mark.parametrize(
    ("framework", "line", "record", "identity"),
    [
        (
            "sglang",
            "[2026-09-17 00:00:00] server_args={record}",
            {"tp_size": 8, "enable_hierarchical_cache": True, "hicache_ratio": 1.5, "cuda_graph_bs": [1, 2, 4]},
            {"tp_size": 8},
        ),
        (
            "vllm",
            "INFO 09-17 00:00:00 non-default args: {record}",
            {"tensor_parallel_size": 8, "enable_prefix_caching": True, "moe_backend": "aiter"},
            {"tensor_parallel_size": 8},
        ),
    ],
    ids=["sglang", "vllm"],
)
def test_the_config_read_keeps_the_settings_identity_discards(
    tmp_path: Path, framework: str, line: str, record: dict[str, object], identity: dict[str, object]
) -> None:
    """The filter needs the tuned knobs, which the identity allowlist drops."""
    log = tmp_path / "server.log"
    log.write_text(line.format(record=repr(record)) + "\n", encoding="utf-8")

    assert evidence.observed_server_config_from_log(str(log), framework) == record
    # The same record through the identity read keeps only its allowlisted key.
    assert evidence.observed_server_identity_from_log(str(log), framework) == identity


def test_a_non_literal_value_does_not_cost_the_rest_of_the_config(tmp_path: Path) -> None:
    """An object repr among 500 settings skips its key, not the whole record."""
    log = tmp_path / "server.log"
    log.write_text(
        "[2026-09-17 00:00:00] server_args={'tp_size': 8, 'sampling': Config(x=1), 'dp_size': 4}\n",
        encoding="utf-8",
    )

    assert evidence.observed_server_config_from_log(str(log), "sglang") == {"dp_size": 4, "tp_size": 8}


@pytest.mark.parametrize("flag", ["--dp", "--dp-size", "--data-parallel-size"])
def test_every_spelling_of_one_knob_names_one_setting(flag: str) -> None:
    """SGLang launches with ``--dp`` where a variant proposes ``--dp-size``."""
    assert evidence.launch_flag_setting_name(flag, "sglang") == "dp_size"


def test_each_engine_names_parallelism_the_way_it_reports_it() -> None:
    """One flag, two keys: folding it to SGLang's name hides it from a vLLM record."""
    assert evidence.launch_flag_setting_name("--tensor-parallel-size", "sglang") == "tp_size"
    assert evidence.launch_flag_setting_name("--tensor-parallel-size", "vllm") == "tensor_parallel_size"


@pytest.mark.parametrize(
    ("framework", "marker"),
    [("sglang", "server_args={'tp_size': 99}"), ("vllm", "non-default args: {'tensor_parallel_size': 99}")],
)
def test_a_marker_echoed_inside_a_quoted_string_is_not_a_launch(tmp_path: Path, framework: str, marker: str) -> None:
    """A log may quote the marker while recording no launch, and that is not evidence.

    User- or attacker-supplied text echoed into a log would otherwise hand the
    decision an identity the server never ran with.
    """
    log = tmp_path / "server.log"
    log.write_text(f'WARNING ignored user text: "{marker}"\n', encoding="utf-8")

    assert evidence.observed_server_identity_from_log(str(log), framework) == {}
    assert evidence.observed_server_config_from_log(str(log), framework) == {}


@pytest.mark.parametrize(
    ("framework", "line", "expected"),
    [
        ("sglang", "context={'a': 1} server_args={'tp_size': 8}", {"tp_size": 8}),
        ("vllm", "context={'a': 1} non-default args: {'tensor_parallel_size': 8}", {"tensor_parallel_size": 8}),
    ],
)
def test_an_unrelated_dict_earlier_on_the_line_is_not_the_record(
    tmp_path: Path, framework: str, line: str, expected: dict[str, object]
) -> None:
    """The payload is anchored at the marker, not at the first delimiter on the line."""
    log = tmp_path / "server.log"
    log.write_text(line + "\n", encoding="utf-8")

    assert evidence.observed_server_identity_from_log(str(log), framework) == expected


def test_an_engine_that_records_nothing_leaves_the_launch_unobserved(tmp_path: Path) -> None:
    """A workload with no row reads as unknown rather than as a misparsed record."""
    log = tmp_path / "server.log"
    log.write_text("server_args={'tp_size': 8}\n", encoding="utf-8")

    assert evidence.observed_server_identity_from_log(str(log), "atom") == {}
    assert evidence.observed_server_config_from_log(str(log), "custom") == {}


def test_an_sglang_log_yields_no_argv(tmp_path: Path) -> None:
    """The only ``launch_server`` token an SGLang log carries names no model, so it is rejected."""
    log = tmp_path / "server.log"
    log.write_text(
        "'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.\n",
        encoding="utf-8",
    )

    assert evidence.launch_argv_from_log(str(log), "sglang") == ""
