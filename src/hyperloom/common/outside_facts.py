# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Facts about projects outside this repository that more than one of our readers depends on, each stated once.

aiter, SGLang, vLLM, PyTorch, the HIP runtime, argparse, Transformers and InferenceX decide these; we only repeat them.
Their values are not written here: they are read from the rule pack (``hyperloom.common.rule_pack``), where each fact
carries the project that owns it, the reason it exists and its examples. This module binds each fact to the name the
readers import, so a reader in ``hyperloom.measure``, ``hyperloom.common``, ``hyperloom.sandbox`` or
``hyperloom_multibranch.toolbox`` never holds a copy. The pack is loaded when this module is first imported, because the
readers build module-level patterns from these values; a pack that cannot be used stops that import, naming the entry.
The module lives in ``hyperloom.common`` because that is the one layer all of those readers may import.

Two kinds of copy cannot import it and are pinned to it by a test instead: a YAML data file (``profile_*.yaml``), and a
file that is copied into a sandbox and runs there with the standard library only (``aiter_merge_hook``).
"""

from __future__ import annotations

from hyperloom.common.rule_pack import default_pack

__all__ = [
    "AIPERF_ERROR_CLASS_KEY",
    "AIPERF_ERROR_KEY",
    "AIPERF_EXPORT_DIR",
    "AIPERF_EXPORT_FILE",
    "AIPERF_INVALID_REPLY_ERROR",
    "AIPERF_METADATA_KEY",
    "AIPERF_METRICS_KEY",
    "AIPERF_OUTPUT_LENGTH_METRIC",
    "AIPERF_PHASE_KEY",
    "AIPERF_PROFILING_PHASE",
    "AITER_LOG_TUNED_CONFIG_ENV",
    "AITER_MERGED_VIEW_DIR",
    "AITER_TUNED_HIT",
    "AITER_TUNED_HIT_MARKER",
    "ARGPARSE_INVALID_CHOICE",
    "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE",
    "CONTEXT_LENGTH_FLAGS",
    "ENV_CONC",
    "ENV_EP",
    "ENV_ISL",
    "ENV_MAGPIE_EVAL_LIMIT",
    "ENV_MAGPIE_EVAL_TASKS",
    "ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS",
    "ENV_MAX_MODEL_LEN",
    "ENV_NUM_PROMPTS",
    "ENV_NUM_WARMUPS",
    "ENV_OSL",
    "ENV_RANDOM_RANGE_RATIO",
    "ENV_RUN_EVAL",
    "ENV_TP",
    "HIP_ERROR_PREFIX",
    "HIP_GRAPH_PACKET_CAPTURE_ENV",
    "HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE",
    "HIP_NO_BINARY_FOR_GPU",
    "HIP_NO_KERNEL_IMAGE",
    "HIP_UNDEFINED_SYMBOL",
    "HSA_STATUS_ERROR",
    "PY_IMPORT_ERROR",
    "PY_MODULE_NOT_FOUND_ERROR",
    "ROCR_VISIBLE_DEVICES_ENV",
    "SGLANG_IMAGE_AITER_CONFIGS",
    "SGLANG_IMAGE_WORKSPACE",
    "TRANSFORMERS_DOES_NOT_RECOGNIZE",
    "TRANSFORMERS_UNKNOWN_ARCHITECTURE",
    "TUNABLEOP_ENABLED_ENV",
    "TUNABLEOP_TABLE_ENV",
    "TUNABLEOP_TUNING_ENV",
    "spaced",
]

_PACK = default_pack()

_aiter = _PACK.names("aiter_merged_view", "dir")
AITER_MERGED_VIEW_DIR = _aiter["dir"]

AITER_LOG_TUNED_CONFIG_ENV = _PACK.names("aiter_log_tuned_config", "env")["env"]
AITER_TUNED_HIT = _PACK.pattern("aiter_tuned_hit")
AITER_TUNED_HIT_MARKER = _PACK.pattern("aiter_tuned_hit_marker")

_image = _PACK.names("sglang_image_layout", "workspace", "aiter_configs")
SGLANG_IMAGE_WORKSPACE = _image["workspace"]
SGLANG_IMAGE_AITER_CONFIGS = f"{SGLANG_IMAGE_WORKSPACE}/{_image['aiter_configs']}"

_context = _PACK.names("context_length_flags", "sglang", "vllm")
CONTEXT_LENGTH_FLAGS = (_context["sglang"], _context["vllm"])

HIP_UNDEFINED_SYMBOL = _PACK.pattern("hip_undefined_symbol")
HIP_NO_KERNEL_IMAGE = _PACK.pattern("hip_no_kernel_image")
HIP_NO_BINARY_FOR_GPU = _PACK.pattern("hip_no_binary_for_gpu")
HSA_STATUS_ERROR = _PACK.pattern("hsa_status_error")
HIP_ERROR_PREFIX = _PACK.pattern("hip_error_prefix")
ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE = _PACK.pattern("argparse_unrecognized_arguments")
ARGPARSE_INVALID_CHOICE = _PACK.pattern("argparse_invalid_choice")
TRANSFORMERS_UNKNOWN_ARCHITECTURE = _PACK.pattern("transformers_unknown_architecture")
TRANSFORMERS_DOES_NOT_RECOGNIZE = _PACK.pattern("transformers_does_not_recognize")

_import_errors = _PACK.names("python_import_errors", "import_error", "module_not_found_error")
PY_IMPORT_ERROR = _import_errors["import_error"]
PY_MODULE_NOT_FOUND_ERROR = _import_errors["module_not_found_error"]

_graph_replay = _PACK.env_default("hip_graph_packet_capture")
HIP_GRAPH_PACKET_CAPTURE_ENV = _graph_replay.name
HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE = _graph_replay.value

ROCR_VISIBLE_DEVICES_ENV = _PACK.names("rocr_visible_devices", "env")["env"]

_tunableop = _PACK.names("pytorch_tunableop_env", "enabled", "tuning", "table")
TUNABLEOP_ENABLED_ENV = _tunableop["enabled"]
TUNABLEOP_TUNING_ENV = _tunableop["tuning"]
TUNABLEOP_TABLE_ENV = _tunableop["table"]

_inferencex = _PACK.names(
    "inferencex_benchmark_envs",
    "isl",
    "osl",
    "random_range_ratio",
    "conc",
    "run_eval",
    "num_prompts",
    "num_warmups",
    "max_model_len",
    "tp",
    "ep",
    "magpie_eval_tasks",
    "magpie_eval_limit",
    "magpie_eval_tokenized_requests",
)
ENV_ISL = _inferencex["isl"]
ENV_OSL = _inferencex["osl"]
ENV_RANDOM_RANGE_RATIO = _inferencex["random_range_ratio"]
ENV_CONC = _inferencex["conc"]
ENV_RUN_EVAL = _inferencex["run_eval"]
ENV_NUM_PROMPTS = _inferencex["num_prompts"]
ENV_NUM_WARMUPS = _inferencex["num_warmups"]
ENV_MAX_MODEL_LEN = _inferencex["max_model_len"]
ENV_TP = _inferencex["tp"]
ENV_EP = _inferencex["ep"]
ENV_MAGPIE_EVAL_TASKS = _inferencex["magpie_eval_tasks"]
ENV_MAGPIE_EVAL_LIMIT = _inferencex["magpie_eval_limit"]
ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS = _inferencex["magpie_eval_tokenized_requests"]

_aiperf_export = _PACK.names(
    "aiperf_request_export",
    "dir",
    "file",
    "metadata",
    "metrics",
    "error",
    "error_class",
    "phase",
    "profiling_phase",
    "output_length",
)
AIPERF_EXPORT_DIR = _aiperf_export["dir"]
AIPERF_EXPORT_FILE = _aiperf_export["file"]
AIPERF_METADATA_KEY = _aiperf_export["metadata"]
AIPERF_METRICS_KEY = _aiperf_export["metrics"]
AIPERF_ERROR_KEY = _aiperf_export["error"]
AIPERF_ERROR_CLASS_KEY = _aiperf_export["error_class"]
AIPERF_PHASE_KEY = _aiperf_export["phase"]
AIPERF_PROFILING_PHASE = _aiperf_export["profiling_phase"]
AIPERF_OUTPUT_LENGTH_METRIC = _aiperf_export["output_length"]

AIPERF_INVALID_REPLY_ERROR = _PACK.names("aiperf_invalid_reply", "error_class")["error_class"]


def spaced(phrase: str) -> str:
    """Regular-expression source for ``phrase`` in which each run of whitespace matches any run of whitespace."""
    return r"\s+".join(phrase.split())
