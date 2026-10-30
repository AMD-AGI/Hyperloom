# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Facts about projects outside this repository that more than one of our readers depends on, each stated once.

aiter, SGLang, vLLM, PyTorch, the HIP runtime, argparse, Transformers and InferenceX decide these; we only repeat them.
Their values are not written here: they are read from the rule pack (``hyperloom.common.outside_facts.rule_pack``),
where each fact carries the project that owns it, the reason it exists and its examples. This package binds each fact
to the name the readers import, so a reader in ``hyperloom.measure``, ``hyperloom.common``, ``hyperloom.sandbox`` or
``hyperloom_multibranch.toolbox`` never holds a copy. The pack is loaded when this package is first imported, because
the readers build module-level patterns from these values; a pack that cannot be used stops that import, naming the
entry. The package lives in ``hyperloom.common`` because that is the one layer all of those readers may import.

Two kinds of copy cannot import it and are pinned to it by a test instead: a YAML data file (``profile_*.yaml``), and a
file that is copied into a sandbox and runs there with the standard library only (``aiter_merge_hook``).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from hyperloom.common.outside_facts.component_layout import ComponentLayoutSpec
from hyperloom.common.outside_facts.rule_pack import default_pack

__all__ = [
    "AIPERF_ERROR_CLASS_KEY",
    "AIPERF_ERROR_KEY",
    "AIPERF_ERROR_OUTPUT_LENGTH_METRIC",
    "AIPERF_EXPORT_DIR",
    "AIPERF_EXPORT_FILE",
    "AIPERF_INVALID_REPLY_ERROR",
    "AIPERF_METADATA_KEY",
    "AIPERF_METRICS_KEY",
    "AIPERF_OUTPUT_LENGTH_FILE",
    "AIPERF_PARSER_FILE",
    "AIPERF_PHASE_KEY",
    "AIPERF_PROFILING_PHASE",
    "AIPERF_SUBMODULE",
    "AITER_DEVELOP_MODE",
    "AITER_INSTALL_MODE_FILE",
    "AITER_LOG_TUNED_CONFIG_ENV",
    "AITER_MERGED_VIEW_DIR",
    "AITER_META_DIR_ENV",
    "AITER_META_MARKER_DIR",
    "AITER_META_WHEEL_DIR",
    "AITER_TUNED_HIT",
    "AITER_TUNED_HIT_MARKER",
    "ARGPARSE_INVALID_CHOICE",
    "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE",
    "ASM_LIBTYPE",
    "CONTEXT_LENGTH_FLAGS",
    "DRM_CLIENT_ID_KEY",
    "DRM_FDINFO_DIR",
    "DRM_FD_DIR",
    "DRM_RENDER_NODE_PREFIX",
    "DRM_VRAM_KEY",
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
    "FORGE_REL_NORM_ERR_COLUMN",
    "HIP_ERROR_PREFIX",
    "HIP_GRAPH_PACKET_CAPTURE_ENV",
    "HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE",
    "HIP_NO_BINARY_FOR_GPU",
    "HIP_NO_KERNEL_IMAGE",
    "HIP_UNDEFINED_SYMBOL",
    "HSA_STATUS_ERROR",
    "INITIAL_PID_NAMESPACE",
    "KFD_NODES",
    "KFD_NODE_GPU_ID",
    "KFD_NODE_PROPERTIES",
    "KFD_PROCESSES",
    "KFD_QUEUES",
    "KFD_QUEUE_GPU_ID",
    "KFD_RENDER_MINOR_KEY",
    "KFD_ROOT",
    "KFD_VRAM_PREFIX",
    "PID_NAMESPACE_LINK",
    "PY_IMPORT_ERROR",
    "PY_MODULE_NOT_FOUND_ERROR",
    "ROCR_VISIBLE_DEVICES_ENV",
    "SGLANG_CUDA_GRAPH_CAPTURE_END",
    "SGLANG_CUDA_GRAPH_DECODE_BS",
    "SGLANG_IMAGE_AITER_CONFIGS",
    "SGLANG_IMAGE_WORKSPACE",
    "SHORT_PROMPT_CHECK",
    "SHORT_PROMPT_TOUCH",
    "TRANSFORMERS_DOES_NOT_RECOGNIZE",
    "TRANSFORMERS_UNKNOWN_ARCHITECTURE",
    "TUNABLEOP_ENABLED_ENV",
    "TUNABLEOP_TABLE_ENV",
    "TUNABLEOP_TUNING_ENV",
    "TUNED_LIBTYPE_COLUMN",
    "TUNED_SPLITK_COLUMN",
    "UPSTREAM_PR_REPOS",
    "component_layouts",
    "spaced",
]

_PACK = default_pack()


def component_layouts() -> Mapping[str, ComponentLayoutSpec]:
    """The layout of every component the pack describes, by the name a recipe calls the component."""
    return MappingProxyType(
        {
            "sglang": _PACK.component_layout("component_layout_sglang"),
            "vllm": _PACK.component_layout("component_layout_vllm"),
            "aiter": _PACK.component_layout("component_layout_aiter"),
            "atom": _PACK.component_layout("component_layout_atom"),
            "xdit": _PACK.component_layout("component_layout_xdit"),
            "triton": _PACK.component_layout("component_layout_triton"),
            "torch": _PACK.component_layout("component_layout_torch"),
            "sgl_kernel": _PACK.component_layout("component_layout_sgl_kernel"),
            "flydsl": _PACK.component_layout("component_layout_flydsl"),
        }
    )


_aiter = _PACK.names("aiter_merged_view", "dir")
AITER_MERGED_VIEW_DIR = _aiter["dir"]

_meta = _PACK.names("aiter_meta_dir", "variable", "wheel_dir", "marker_dir", "install_mode_file", "develop_mode")
AITER_META_DIR_ENV = _meta["variable"]
AITER_META_WHEEL_DIR = _meta["wheel_dir"]
AITER_META_MARKER_DIR = _meta["marker_dir"]
AITER_INSTALL_MODE_FILE = _meta["install_mode_file"]
AITER_DEVELOP_MODE = _meta["develop_mode"]

_tuned = _PACK.names("aiter_tuned_row_columns", "libtype_column", "splitk_column", "asm_libtype")
TUNED_LIBTYPE_COLUMN = _tuned["libtype_column"]
TUNED_SPLITK_COLUMN = _tuned["splitk_column"]
ASM_LIBTYPE = _tuned["asm_libtype"]

UPSTREAM_PR_REPOS = tuple(_PACK.names("upstream_pr_repos", "sglang", "vllm", "aiter", "pytorch").values())

FORGE_REL_NORM_ERR_COLUMN = _PACK.names("forge_rel_norm_err", "column")["column"]

AITER_LOG_TUNED_CONFIG_ENV = _PACK.names("aiter_log_tuned_config", "env")["env"]
AITER_TUNED_HIT = _PACK.pattern("aiter_tuned_hit")
AITER_TUNED_HIT_MARKER = _PACK.pattern("aiter_tuned_hit_marker")

SGLANG_CUDA_GRAPH_CAPTURE_END = _PACK.pattern("sglang_cuda_graph_capture_end")
SGLANG_CUDA_GRAPH_DECODE_BS = _PACK.pattern("sglang_cuda_graph_decode_bs")

SHORT_PROMPT_CHECK = _PACK.names("short_prompt_check", "max_new_tokens", "phase_cap_sec", "always_purposes")
SHORT_PROMPT_TOUCH = _PACK.pattern("short_prompt_touch")

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

_kfd = _PACK.names(
    "kfd_process_listing",
    "root",
    "processes",
    "nodes",
    "node_gpu_id",
    "queues",
    "queue_gpu_id",
    "vram_prefix",
    "node_properties",
    "render_minor_key",
)
KFD_ROOT = _kfd["root"]
KFD_PROCESSES = _kfd["processes"]
KFD_NODES = _kfd["nodes"]
KFD_NODE_GPU_ID = _kfd["node_gpu_id"]
KFD_QUEUES = _kfd["queues"]
KFD_QUEUE_GPU_ID = _kfd["queue_gpu_id"]
KFD_VRAM_PREFIX = _kfd["vram_prefix"]
KFD_NODE_PROPERTIES = _kfd["node_properties"]
KFD_RENDER_MINOR_KEY = _kfd["render_minor_key"]

_drm = _PACK.names("drm_fdinfo", "fd_dir", "fdinfo_dir", "render_node_prefix", "client_id_key", "vram_key")
DRM_FD_DIR = _drm["fd_dir"]
DRM_FDINFO_DIR = _drm["fdinfo_dir"]
DRM_RENDER_NODE_PREFIX = _drm["render_node_prefix"]
DRM_CLIENT_ID_KEY = _drm["client_id_key"]
DRM_VRAM_KEY = _drm["vram_key"]

_pid_namespace = _PACK.names("linux_pid_namespace", "link", "initial")
PID_NAMESPACE_LINK = _pid_namespace["link"]
INITIAL_PID_NAMESPACE = _pid_namespace["initial"]

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
    "error_output_length",
)
AIPERF_EXPORT_DIR = _aiperf_export["dir"]
AIPERF_EXPORT_FILE = _aiperf_export["file"]
AIPERF_METADATA_KEY = _aiperf_export["metadata"]
AIPERF_METRICS_KEY = _aiperf_export["metrics"]
AIPERF_ERROR_KEY = _aiperf_export["error"]
AIPERF_ERROR_CLASS_KEY = _aiperf_export["error_class"]
AIPERF_PHASE_KEY = _aiperf_export["phase"]
AIPERF_PROFILING_PHASE = _aiperf_export["profiling_phase"]
AIPERF_ERROR_OUTPUT_LENGTH_METRIC = _aiperf_export["error_output_length"]

AIPERF_INVALID_REPLY_ERROR = _PACK.names("aiperf_invalid_reply", "error_class")["error_class"]

_aiperf_source = _PACK.names("aiperf_source", "submodule", "parser_file", "output_length_file")
AIPERF_SUBMODULE = _aiperf_source["submodule"]
AIPERF_PARSER_FILE = _aiperf_source["parser_file"]
AIPERF_OUTPUT_LENGTH_FILE = _aiperf_source["output_length_file"]

MAGPIE_AGENTX_CLIENT_LOG = _PACK.names("magpie_agentx_client_log", "file")["file"]
HUB_RATE_LIMITED = _PACK.pattern("hub_rate_limited")


def spaced(phrase: str) -> str:
    """Regular-expression source for ``phrase`` in which each run of whitespace matches any run of whitespace."""
    return r"\s+".join(phrase.split())
