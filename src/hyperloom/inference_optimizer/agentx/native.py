# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validation helpers for Magpie's native InferenceX AgentX path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, MutableMapping

from hyperloom.common.agentx_mode import native_agentx_enabled
from .magpie_source_identity import MAGPIE_SOURCE_IDENTITY_CODE

_ENCODING = "utf-8"
_SGLANG_AGENTIC_SCRIPT = "srt_agentic.sh"
_GIT = "git"
_GIT_TIMEOUT_SEC = 30
_GIT_HEAD = "HEAD"
_COMMIT_SHA_PATTERN = "[0-9a-f]{40}"


_AGENTIC_PREFIX = ("single_node", "agentic")
_RESOLVER_SENTINEL = "HYPERLOOM_AGENTX_RECIPE="
_NATIVE_CHECKOUT_PATH_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./-]+$")


# Ambient values consumed by the launchers in the exact-ref manifest below,
# plus loader/search and ROCm/framework controls inherited by their children.
# Keep this allowlist explicit: hashing every login-shell variable would make a
# resume depend on unrelated values such as PWD/SHLVL, while omitting a real
# launcher knob would let the workload change without changing its identity.
_NATIVE_AMBIENT_CONTROL_NAMES = frozenset(
    {
        "AGENTIC_VENV",
        "BASH_ENV",
        "CHUNKED_PREFILL_SIZE",
        "CONC",
        "CPU_BYTES_PER_RANK",
        "CPU_OFFLOAD_BYTES",
        "CUDAGRAPH_CAPTURE_SIZES",
        "CUDA_GRAPH_CAPTURE_SIZES",
        "CUDA_GRAPH_MAX_BS",
        "DCP_SIZE",
        "DP_ATTENTION",
        "DRAFT_MODEL",
        "DRAFT_MODEL_PATH",
        "DURATION",
        "EP_SIZE",
        "ENV",
        "EVAL_ONLY",
        "FRAMEWORK",
        "GPU_MEMORY_UTILIZATION",
        "GPU_MEM_UTIL",
        "HF_MODEL_ID",
        "IMAGE",
        "IS_AGENTIC",
        "IS_MULTINODE",
        "ISL",
        "GCONV_PATH",
        "KV_OFFLOADING",
        "KV_OFFLOAD_BACKEND",
        "L3_PER_RANK_GB",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_PRELOAD",
        "LIBRARY_PATH",
        "MAX_CUDAGRAPH_CAPTURE_SIZE",
        "MAX_NUM_SEQS",
        "MAX_RUNNING_REQUESTS",
        "MAX_MODEL_LEN",
        "MEM_FRACTION_STATIC",
        "MODEL_DOWNLOAD_LOCK_TIMEOUT",
        "MODEL",
        "MODEL_NAME",
        "MODEL_PATH",
        "MODEL_PREFIX",
        "NUM_SPEC_TOKENS",
        "PATH",
        "PCP_SIZE",
        "PORT",
        "PP_SIZE",
        "PRECISION",
        "PROFILE",
        "PYTHONHASHSEED",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "SCHEDULER_RECV_INTERVAL",
        "SCENARIO_TYPE",
        "SERVED_MODEL_NAME",
        "SHELLOPTS",
        "SHM_CAP_GB",
        "SIMPLE_LAZY_OFFLOAD",
        "SPEC_DECODING",
        "SPEC_NUM_TOKENS",
        "SYNTHETIC_ACCEPT_LEN",
        "TOTAL_CPU_DRAM_GB",
        "TOTAL_CPU_DRAM_PARTITION_GB",
        "TP",
        "RECIPE_FINGERPRINT",
        "RUNNER_TYPE",
        "VLLM_ENGINE_READY_TIMEOUT_S",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
        "VIRTUAL_ENV",
        "WITH_LM_EVAL",
        "WITH_PR4521",
        "WRITABLE_MODELS_DIR",
        "WS",
        "OSL",
    }
)
_NATIVE_AMBIENT_CONTROL_PREFIXES = (
    "AITER_",
    "CUDA_",
    "DECODE_",
    "EVAL_",
    "HICACHE_",
    "HIP_",
    "HSA_",
    "INFERENCEX_",
    "KV_",
    "LD_",
    "LMCACHE_",
    "MC_",
    "MIOPEN_",
    "MOONCAKE_",
    "NCCL_",
    "OMP_",
    "PREFILL_",
    "PYTORCH_",
    "RCCL_",
    "ROCM_",
    "ROCR_",
    "SGLANG_",
    "SWEBENCH_",
    "TILELANG_",
    "TORCH_",
    "TRITON_",
    "UCX_",
    "VLLM_",
)

# Values that define the exact native AgentX arm selected for a session.  The
# CLI snapshots every non-empty value at seed time and restores it before a
# fresh-shell resume can materialize another benchmark.  Keep the resolved
# fingerprint and physical GPU count here too: the former detects recipe drift
# and the latter deliberately differs from recipe ``TP`` when PP/PCP > 1.
AGENTX_RUNTIME_PIN_NAMES = (
    "AGENTX_MODEL_ID",
    "AGENTX_SERVER_SCRIPT",
    "INFERENCEX_PATH",
    "HYPERLOOM_IMAGE",
    "AGENTX_MODE",
    "AGENTX_RECIPE",
    "AGENTX_CONFIG_FILE",
    "AGENTX_SELECTOR",
    "AGENTX_FAILED_REQUEST_THRESHOLD",
    "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT",
    "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT",
    "HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT",
    "HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT",
    "HYPERLOOM_AGENTX_GPU_COUNT",
    "MAGPIE_REF",
    "INFERENCEX_REF",
)

# These four values existed before the native Magpie bridge and remain the
# minimum needed by the legacy/pre-baseline path.  An accepted native baseline
# is held to the stronger fingerprint set in ``_restore_agentx_runtime_pins``.
AGENTX_REQUIRED_RUNTIME_PIN_NAMES = (
    "AGENTX_MODEL_ID",
    "AGENTX_SERVER_SCRIPT",
    "INFERENCEX_PATH",
    "HYPERLOOM_IMAGE",
)

# Persisted epoch-2 sessions use these exact shell launcher bytes. New sessions
# resolve Magpie's server-launch-spec and never consult this historical mapping.
_NATIVE_LAUNCHER_MANIFEST: dict[tuple[str, str], tuple[str, str]] = {
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "qwen3.5-fp8-mi325x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/qwen3.5_fp8_mi325x_mtp.sh",
        "227452668acec427535a80aea00736271742f506a9f739ac3f8621e516443e8e",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "qwen3.5-fp4-mi355x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh",
        "889f82eab6e05bc81b77e374eb71853d8c9701fd17246aa9e9be38437cb8f161",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "qwen3.5-fp8-mi300x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/qwen3.5_fp8_mi300x_mtp.sh",
        "a56d14c8056cdebfc9d5f1f53b60c0279f12b85f062d987b6ac593bd1d3e8694",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "kimik3-fp4-mi355x-vllm-agentic-mtp",
    ): (
        "single_node/agentic/kimik3_fp4_mi355x_mtp.sh",
        "4de24764a0bdf0ea9a21d5f5c5c304d2754097fb8398da10d60793ff85427125",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "dsv4-fp4-mi355x-vllm-agentic-mtp",
    ): (
        "single_node/agentic/dsv4_fp4_mi355x_vllm_mtp.sh",
        "80f26dfce576b0e1375510303f96089f92df416f52adb26a019449ebb2e85d3f",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "minimaxm3-fp8-mi300x-vllm-agentic-mtp",
    ): (
        "single_node/agentic/minimaxm3_fp8_mi300x_mtp.sh",
        "8fc7a35d28ea2f6ccf353c176e2d0c813d1a989f64ebf8c9cda26f83e05bb278",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "glm5.2-fp8-mi325x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/glm5.2_fp8_mi325x_mtp.sh",
        "e48b43d03c71c439833d7369002ef92bf50fabc3b0a20c738dc751b2dbbe8ef7",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "minimaxm3-fp8-mi325x-vllm-agentic-mtp",
    ): (
        "single_node/agentic/minimaxm3_fp8_mi325x_mtp.sh",
        "99f47d000f94445bcf66882438388d9a14ed52b7cdaef7b8c4bdcd3d5cfe67f2",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "minimaxm3-fp4-mi355x-vllm-agentic-mtp",
    ): (
        "single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh",
        "630cf96f51d2a288212c389a05b7df312745f4bac074e34306bf154d8cb40b47",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "glm5.2-fp4-mi355x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh",
        "5083b395d4af15cd6b668c16f744954ee0d90c2dc607e585b8610072ffc5e099",
    ),
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "dsv4-fp4-mi355x-sglang-agentic-mtp",
    ): (
        "single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh",
        "da70afbb14b48766a0fe1e1d590ec123eda6654a4c1ca419961354545f629ddc",
    ),
}

_NATIVE_LAUNCHER_TRANSITIVE_INPUTS: dict[tuple[str, str], tuple[str, ...]] = {
    (
        "3d5581562f643f9bdeb8410cd924e2c70906c966",
        "single_node/agentic/kimik3_fp4_mi355x_mtp.sh",
    ): ("benchmarks/single_node/agentic/apply_k3_container_patches.sh",),
}

# Persisted epoch-3 sessions retain the launch contract of their accepted pin.
_NATIVE_LAUNCH_CONTRACT_REFS: frozenset[str] = frozenset({"421312f8984c2152f4b8eafefc93ea2fa598e80f"})
_NATIVE_LAUNCH_CONTRACT_INPUTS = (
    "configs/agentx-launchers.json",
    "utils/agentic/server_launch.py",
    "utils/agentic/custom_model.py",
)

_RESOLVER_CODE = (
    MAGPIE_SOURCE_IDENTITY_CODE
    + r"""
import hashlib
import json
import re
from pathlib import Path
import sys

import Magpie
from Magpie.modes.benchmark import BenchmarkConfig
from Magpie.modes.benchmark.agentx import resolve_agentx_recipe


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


package_root = Path(Magpie.__file__).resolve().parent
source_commit, source_url = _resolve_magpie_source_identity(package_root)
magpie_tree = _validate_magpie_execution_tree(package_root, source_commit)
magpie_identity = {
    "source_commit": source_commit,
    "source_url": source_url,
    "published_execution_tree": magpie_tree,
}
magpie_execution = {"package_root": str(package_root), **magpie_identity}
magpie_execution["fingerprint"] = hashlib.sha256(
    json.dumps(magpie_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

payload = json.load(sys.stdin)
config = BenchmarkConfig.from_dict(payload["benchmark"])
spec = resolve_agentx_recipe(
    config,
    payload["inferencex_path"],
    runner_type=str(config.runner_type or ""),
)
result = {
    "benchmark": config.to_dict(),
    "recipe": spec.recipe,
    "config_file": spec.config_file,
    "entry": spec.entry,
    "magpie_execution": magpie_execution,
}
print("HYPERLOOM_AGENTX_RECIPE=" + json.dumps(result, separators=(",", ":")))
"""
)


def _run_magpie_recipe_resolver(
    benchmark: MutableMapping[str, Any],
    *,
    inferencex_path: Path,
) -> dict[str, Any]:
    """Resolve through the same interpreter that will execute Magpie."""
    from hyperloom.common.env_safety import scrub_benchmark_process_env
    from hyperloom.inference_optimizer.benchmark_interpreter import (
        resolve_magpie_python,
    )

    inferencex_path = validate_native_checkout_path(inferencex_path)
    interpreter = resolve_magpie_python()
    payload = json.dumps(
        {
            "benchmark": {**benchmark, "inferencex_path": str(inferencex_path)},
            "inferencex_path": str(inferencex_path),
        }
    )
    try:
        completed = subprocess.run(
            [interpreter, "-c", _RESOLVER_CODE],
            input=payload,
            text=True,
            capture_output=True,
            timeout=120,
            env=scrub_benchmark_process_env(dict(os.environ)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Unable to run Magpie AgentX recipe resolver with {interpreter!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Magpie AgentX recipe resolution failed in the benchmark interpreter {interpreter!r}: {detail[-2000:]}"
        )
    encoded = next(
        (
            line[len(_RESOLVER_SENTINEL) :]
            for line in reversed((completed.stdout or "").splitlines())
            if line.startswith(_RESOLVER_SENTINEL)
        ),
        "",
    )
    if not encoded:
        raise RuntimeError("Magpie AgentX recipe resolver returned no structured result")
    try:
        result = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Magpie AgentX recipe resolver returned invalid JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("benchmark"), dict):
        raise RuntimeError("Magpie AgentX recipe resolver returned an invalid result")
    from .identity import has_launch_contract

    if has_launch_contract(benchmark) and not has_launch_contract(result["benchmark"]):
        raise RuntimeError("Pinned Magpie does not support the native AgentX optimization launch contract")
    return result


def validate_native_checkout_path(inferencex_path: str | Path) -> Path:
    """Normalize managed projects; keep historical shell launchers path-safe."""
    raw = str(inferencex_path or "")
    if not raw:
        raise ValueError("Native AgentX requires a non-empty InferenceX checkout path")
    if raw != raw.strip():
        raise ValueError("Native AgentX InferenceX checkout path is shell-unsafe for the pinned Magpie launcher")
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Native AgentX InferenceX checkout path is not resolvable") from exc
    from .managed import project_root

    resolved = project_root(resolved)
    managed = (resolved / "benchmarks/srt_agentic.sh").is_file()
    if not managed and not _NATIVE_CHECKOUT_PATH_RE.fullmatch(str(resolved)):
        raise ValueError(
            f"Native AgentX InferenceX checkout path is shell-unsafe for the pinned Magpie launcher: {resolved}"
        )
    return resolved


def native_runtime_search_paths() -> dict[str, str]:
    """Pin the same runtime imports during recipe selection and materialization."""
    paths = {
        name: value
        for name in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LIBRARY_PATH")
        if (value := os.environ.get(name, "").strip())
    }
    if paths.get("PATH") and paths["PATH"].split(":", 1)[0] != "/opt/venv/bin":
        paths["PATH"] = f"/opt/venv/bin:{paths['PATH']}"
    return paths


def validate_native_replay_env(env: Mapping[str, Any], *, source: str) -> None:
    """Reject legacy replay settings that native AgentX cannot honour."""
    overrides = sorted(
        str(name)
        for name, value in env.items()
        if value is not None
        and str(value).strip()
        and (
            str(name).upper().startswith(("AIPERF_", "AGENTIC_"))
            or str(name).upper() in {"AGENTX_DATASET", "AGENTX_WARMUP_REQUESTS_PER_LANE", "WEKA_LOADER_OVERRIDE"}
        )
    )
    if overrides:
        raise ValueError(
            f"Native AgentX does not support legacy replay controls in {source}: "
            + ", ".join(overrides)
            + ". Remove them; use benchmark.agentx and the resolved InferenceX recipe for replay settings."
        )


def scrub_native_agentx_ambient_env(env: MutableMapping[str, str]) -> None:
    """Remove generic replay controls and inherited Bash functions."""
    for name in tuple(env):
        upper = str(name).upper()
        if upper.startswith(("AIPERF_", "AGENTIC_", "BASH_FUNC_")) or upper == "WEKA_LOADER_OVERRIDE":
            env.pop(name, None)


def native_launch_environment_identity(
    resolved_benchmark: Mapping[str, Any],
    *,
    launch_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash the effective child environment that can alter native execution.

    This mirrors the real process boundary: Hyperloom first removes child
    startup hooks and control-plane credentials, then removes generic AgentX
    replay overrides, and finally Magpie overlays ``benchmark.envs``.  Only
    audited launcher/framework controls are retained in the identity; cache
    routing and authentication values such as HF_HOME/HF_TOKEN remain usable
    without being persisted or making a fresh-shell resume incomparable.
    """
    from hyperloom.common.env_safety import scrub_benchmark_process_env

    source = os.environ if launch_env is None else launch_env
    effective = scrub_benchmark_process_env({str(key): str(value) for key, value in source.items()})
    scrub_native_agentx_ambient_env(effective)
    # Hyperloom deliberately drops its remote model identifier here: native
    # launchers interpret any non-empty MODEL_PATH as a local directory.
    effective.pop("MODEL_PATH", None)
    raw_envs = resolved_benchmark.get("envs")
    benchmark_envs = raw_envs if isinstance(raw_envs, Mapping) else {}
    effective.update({str(key): str(value) for key, value in benchmark_envs.items()})
    controls = {
        name: value
        for name, value in effective.items()
        if name.upper() in _NATIVE_AMBIENT_CONTROL_NAMES or name.upper().startswith(_NATIVE_AMBIENT_CONTROL_PREFIXES)
    }
    encoded = json.dumps(controls, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(_ENCODING)
    return {
        "names": sorted(controls),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_native_launcher_name(script_name: str) -> PurePosixPath:
    """Validate an explicit launcher relative to ``InferenceX/benchmarks``."""
    raw = str(script_name or "").strip()
    path = PurePosixPath(raw)
    if raw == _SGLANG_AGENTIC_SCRIPT:
        return path
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != _AGENTIC_PREFIX
        or path.suffix != ".sh"
        or len(path.parts) != 3
    ):
        raise ValueError(
            "AGENTX_SERVER_SCRIPT must name one explicit InferenceX launcher "
            "under single_node/agentic/ (for example "
            "single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh)"
        )
    return path


def resolve_native_launcher(
    *,
    inferencex_path: str | Path,
    benchmark_script: str,
) -> Path:
    """Resolve an explicit native launcher without modifying the checkout."""
    rel = validate_native_launcher_name(benchmark_script)
    root = validate_native_checkout_path(inferencex_path)
    benchmarks_root = (root / "benchmarks").resolve()
    lexical_launcher = benchmarks_root / Path(*rel.parts)
    if _contains_symlink(benchmarks_root, lexical_launcher):
        raise ValueError(f"AgentX launcher path must not contain symlinks: {lexical_launcher}")
    launcher = lexical_launcher.resolve()
    agentic_root = (
        benchmarks_root
        if rel.as_posix() == _SGLANG_AGENTIC_SCRIPT
        else (benchmarks_root / "single_node" / "agentic").resolve()
    )
    try:
        launcher.relative_to(agentic_root)
    except ValueError as exc:
        raise ValueError(f"AgentX launcher escapes {agentic_root}: {launcher}") from exc
    if not launcher.is_file():
        raise FileNotFoundError(
            f"Configured native AgentX launcher does not exist: {launcher}. "
            "AGENTX_SERVER_SCRIPT must match the pinned InferenceX checkout."
        )
    return launcher


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_benchmark_sha256(benchmark: Mapping[str, Any]) -> str:
    """Hash the resolved Magpie launch config, including forwarded envs."""
    try:
        payload = json.dumps(
            dict(benchmark),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode(_ENCODING)
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved native AgentX benchmark is not canonically serializable") from exc
    return hashlib.sha256(payload).hexdigest()


def _contains_symlink(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _head_blob(root: Path, relative_path: str) -> bytes:
    completed = subprocess.run(
        [_GIT, "-C", str(root), "show", f"HEAD:{relative_path}"],
        capture_output=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"native AgentX execution input is not tracked at HEAD: {relative_path}")
    return completed.stdout


def _checkout_head(root: Path, *, label: str) -> str:
    """Return one exact git commit, or reject an unverifiable checkout."""
    try:
        completed = subprocess.run(
            [_GIT, "-C", str(root), "rev-parse", _GIT_HEAD],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot read {label} git identity at {root}: {exc}") from exc
    head = (completed.stdout or "").strip().lower()
    if completed.returncode != 0 or not re.fullmatch(_COMMIT_SHA_PATTERN, head):
        raise ValueError(f"native AgentX requires a readable {label} git checkout at {root}")
    return head


def _launcher_manifest(root: Path, head: str) -> dict[str, tuple[str, str]]:
    if head not in _NATIVE_LAUNCH_CONTRACT_REFS:
        return {recipe: item for (commit, recipe), item in _NATIVE_LAUNCHER_MANIFEST.items() if commit == head}
    manifest_path = root / "configs" / "agentx-launchers.json"
    if _contains_symlink(root, manifest_path) or manifest_path.read_bytes() != _head_blob(
        root, "configs/agentx-launchers.json"
    ):
        raise ValueError("Native AgentX launcher manifest changed from its pinned source")
    manifest = json.loads(manifest_path.read_text(encoding=_ENCODING))
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(manifest.get("recipes"), dict):
        raise ValueError("Native AgentX launcher manifest has an unsupported schema")
    generic = manifest.get("generic", {})
    if not isinstance(generic, dict) or manifest["recipes"].keys() & generic.keys():
        raise ValueError("Native AgentX launcher manifest has invalid generic capabilities")
    entries: dict[str, tuple[str, str]] = {}
    for recipe, entry in {**manifest["recipes"], **generic}.items():
        if not isinstance(entry, dict) or entry.get("launch_overrides_version") != 1:
            raise ValueError(f"Native AgentX recipe {recipe!r} has no supported launch contract")
        script = validate_native_launcher_name(str(entry.get("benchmark_script") or "")).as_posix()
        entries[recipe] = (script, hashlib.sha256(_head_blob(root, f"benchmarks/{script}")).hexdigest())
    return entries


def validate_native_recipe_launcher(
    *,
    inferencex_path: str | Path,
    recipe: str,
    benchmark_script: str,
) -> dict[str, str]:
    """Verify the managed client or an existing session's pinned shell launcher."""
    root = validate_native_checkout_path(inferencex_path)
    head = _checkout_head(root, label="InferenceX")
    recipe_name = str(recipe or "").strip()
    if benchmark_script == _SGLANG_AGENTIC_SCRIPT:
        launcher = resolve_native_launcher(inferencex_path=root, benchmark_script=benchmark_script)
        from .managed import head_blob

        if launcher.read_bytes() != head_blob(root, "benchmarks/srt_agentic.sh"):
            raise ValueError("Managed AgentX client differs from pinned HEAD")
        return {
            "inferencex_commit": head,
            "recipe": recipe_name,
            "launcher": benchmark_script,
            "launcher_sha256": _sha256_file(launcher),
        }
    manifest_entry = _launcher_manifest(root, head).get(recipe_name)
    if manifest_entry is None:
        raise ValueError(
            "Native AgentX has no audited launcher mapping for "
            f"InferenceX {head} recipe {recipe_name!r}. Update the exact-ref "
            "compatibility manifest instead of guessing a launcher."
        )
    expected_script, expected_sha256 = manifest_entry
    actual_script = validate_native_launcher_name(benchmark_script).as_posix()
    if actual_script != expected_script:
        raise ValueError(
            f"AgentX recipe {recipe_name!r} requires audited launcher "
            f"{expected_script!r} at InferenceX {head}; got {actual_script!r}"
        )
    launcher = resolve_native_launcher(
        inferencex_path=root,
        benchmark_script=actual_script,
    )
    actual_sha256 = _sha256_file(launcher)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Native AgentX launcher bytes do not match the audited manifest: "
            f"{actual_script} expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "inferencex_commit": head,
        "recipe": recipe_name,
        "launcher": actual_script,
        "launcher_sha256": actual_sha256,
    }


def native_execution_identity(
    *,
    inferencex_path: str | Path,
    benchmark_script: str,
    config_file: str,
    resolved_benchmark: Mapping[str, Any],
    expected_ref: str = "",
    magpie_execution: Mapping[str, Any] | None = None,
    expected_magpie_ref: str = "",
    launch_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind a native run to clean executable inputs from one git checkout."""
    root = validate_native_checkout_path(inferencex_path)
    from .managed import execution_identity, identity_benchmark, server_spec

    if server_spec(resolved_benchmark) is not None:
        identity_config = identity_benchmark(
            resolved_benchmark,
            str(
                (os.environ if launch_env is None else launch_env).get("HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT")
                or ""
            ),
        )
        return execution_identity(
            root=root,
            benchmark=identity_config,
            config_file=config_file,
            expected_ref=expected_ref,
            magpie=magpie_execution or {},
            expected_magpie_ref=expected_magpie_ref,
            launch_environment=native_launch_environment_identity(resolved_benchmark, launch_env=launch_env),
        )
    launcher = resolve_native_launcher(
        inferencex_path=root,
        benchmark_script=benchmark_script,
    )
    benchmark_lib = root / "benchmarks" / "benchmark_lib.sh"
    runners_config = root / "configs" / "runners.yaml"
    raw_config = PurePosixPath(str(config_file or "").strip())
    if not str(raw_config) or raw_config.is_absolute() or ".." in raw_config.parts:
        raise ValueError("resolved AgentX config_file must be relative to the InferenceX checkout")
    lexical_config = root / Path(*raw_config.parts)
    if _contains_symlink(root, lexical_config):
        raise ValueError(f"AgentX config_file path must not contain symlinks: {lexical_config}")
    config_path = lexical_config.resolve()
    try:
        config_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"AgentX config_file escapes {root}: {config_path}") from exc
    missing = [str(path) for path in (benchmark_lib, runners_config, config_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Native AgentX execution input is missing: " + ", ".join(missing))
    for execution_path in (launcher, benchmark_lib, runners_config, config_path):
        if _contains_symlink(root, execution_path):
            raise ValueError(f"native AgentX execution inputs must not contain symlinks: {execution_path}")

    head = _checkout_head(root, label="InferenceX")
    wanted = str(expected_ref or "").strip().lower()
    if wanted and not re.fullmatch(_COMMIT_SHA_PATTERN, wanted):
        raise ValueError(
            f"Native AgentX requires INFERENCEX_REF to be an immutable 40-character commit SHA; got {expected_ref!r}"
        )
    if wanted and head != wanted:
        raise ValueError(f"InferenceX HEAD {head} does not match the pinned ref {wanted}")
    from .identity import has_launch_contract

    if has_launch_contract(resolved_benchmark) and head not in _NATIVE_LAUNCH_CONTRACT_REFS:
        raise ValueError("Pinned InferenceX does not support the native AgentX optimization launch contract")

    launcher_manifest_entries = [
        (recipe, expected_sha256)
        for recipe, (script, expected_sha256) in _launcher_manifest(root, head).items()
        if script == benchmark_script
    ]
    if not launcher_manifest_entries:
        raise ValueError(
            f"native AgentX launcher is not audited for this exact InferenceX commit: {head} {benchmark_script!r}"
        )
    launcher_sha256 = _sha256_file(launcher)
    if all(launcher_sha256 != expected_sha256 for _recipe, expected_sha256 in launcher_manifest_entries):
        raise ValueError(f"native AgentX launcher bytes do not match the exact-ref manifest: {benchmark_script}")
    transitive_relative = sorted(set(_NATIVE_LAUNCHER_TRANSITIVE_INPUTS.get((head, benchmark_script), ())))
    if head in _NATIVE_LAUNCH_CONTRACT_REFS:
        transitive_relative = sorted(
            set(transitive_relative)
            | set(_NATIVE_LAUNCH_CONTRACT_INPUTS)
            | {
                path
                for (_commit, script), paths in _NATIVE_LAUNCHER_TRANSITIVE_INPUTS.items()
                if script == benchmark_script
                for path in paths
            }
        )
    transitive_paths = [root / Path(*PurePosixPath(item).parts) for item in transitive_relative]
    missing_transitive = [str(path) for path in transitive_paths if not path.is_file()]
    if missing_transitive:
        raise FileNotFoundError("Native AgentX transitive launcher input is missing: " + ", ".join(missing_transitive))

    relative_inputs = (
        launcher.relative_to(root).as_posix(),
        benchmark_lib.relative_to(root).as_posix(),
        runners_config.relative_to(root).as_posix(),
        config_path.relative_to(root).as_posix(),
        "utils",
        *transitive_relative,
    )
    try:
        status_proc = subprocess.run(
            [
                _GIT,
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
                "--",
                *relative_inputs,
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot verify native AgentX execution inputs at {root}: {exc}") from exc
    dirty = (status_proc.stdout or "").strip()
    if status_proc.returncode != 0:
        detail = (status_proc.stderr or dirty).strip()
        raise ValueError(f"cannot verify native AgentX execution inputs at {root}: {detail}")
    if dirty:
        changed = ", ".join(line[3:].strip() for line in dirty.splitlines() if line.strip())
        raise ValueError(
            f"native AgentX requires clean pinned launcher/replay inputs; modified paths in {root}: {changed}"
        )

    for execution_path in (
        launcher,
        benchmark_lib,
        runners_config,
        config_path,
        *transitive_paths,
    ):
        if _contains_symlink(root, execution_path):
            raise ValueError(f"native AgentX execution inputs must not contain symlinks: {execution_path}")
        relative = execution_path.relative_to(root).as_posix()
        if execution_path.read_bytes() != _head_blob(root, relative):
            raise ValueError(f"native AgentX execution input differs from the pinned HEAD blob: {relative}")

    gitlink_proc = subprocess.run(
        [_GIT, "-C", str(root), "ls-tree", _GIT_HEAD, "utils/aiperf"],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    gitlink_fields = (gitlink_proc.stdout or "").strip().split()
    if (
        gitlink_proc.returncode != 0
        or len(gitlink_fields) < 3
        or gitlink_fields[0] != "160000"
        or not re.fullmatch(_COMMIT_SHA_PATTERN, gitlink_fields[2])
    ):
        raise ValueError("InferenceX utils/aiperf is not a pinned git submodule")
    aiperf_commit = gitlink_fields[2]
    aiperf_root = root / "utils" / "aiperf"
    if not (aiperf_root / "pyproject.toml").is_file():
        raise ValueError(
            "InferenceX utils/aiperf submodule is not initialized; run git submodule update --init -- utils/aiperf"
        )
    submodule_head = subprocess.run(
        [_GIT, "-C", str(aiperf_root), "rev-parse", _GIT_HEAD],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    actual_aiperf_commit = (submodule_head.stdout or "").strip().lower()
    if submodule_head.returncode != 0 or actual_aiperf_commit != aiperf_commit:
        raise ValueError(f"InferenceX utils/aiperf does not match its pinned gitlink {aiperf_commit}")
    submodule_status = subprocess.run(
        [
            _GIT,
            "-C",
            str(aiperf_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    if submodule_status.returncode != 0 or (submodule_status.stdout or "").strip():
        raise ValueError("InferenceX utils/aiperf submodule is not clean")

    magpie = dict(magpie_execution or {})
    magpie_fingerprint = str(magpie.get("fingerprint") or "").strip().lower()
    magpie_commit = str(magpie.get("source_commit") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", magpie_fingerprint):
        raise ValueError("Magpie resolver did not provide a valid execution fingerprint")
    wanted_magpie = str(expected_magpie_ref or "").strip().lower()
    if wanted_magpie and not re.fullmatch(_COMMIT_SHA_PATTERN, wanted_magpie):
        raise ValueError(
            f"Native AgentX requires MAGPIE_REF to be an immutable 40-character commit SHA; got {expected_magpie_ref!r}"
        )
    if wanted_magpie and magpie_commit != wanted_magpie:
        raise ValueError(
            f"Magpie source commit {magpie_commit or '<unknown>'} does not match the pinned ref {wanted_magpie}"
        )

    identity = {
        "inferencex_commit": head,
        "launcher": launcher.relative_to(root).as_posix(),
        "launcher_sha256": launcher_sha256,
        "benchmark_lib_sha256": _sha256_file(benchmark_lib),
        "runners_config_sha256": _sha256_file(runners_config),
        "config_file": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "aiperf_commit": aiperf_commit,
        "magpie_fingerprint": magpie_fingerprint,
        "magpie_commit": magpie_commit,
        "transitive_inputs": {path.relative_to(root).as_posix(): _sha256_file(path) for path in transitive_paths},
    }
    # This stable layer is established by the CLI before workload
    # materialization adds generated fields (for example ROCR_VISIBLE_DEVICES
    # and warm-up controls).  It pins the executable sources without falsely
    # treating those deterministic additions as checkout drift.
    identity["static_execution_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(_ENCODING)
    ).hexdigest()
    from .identity import has_launch_contract, native_workload_fingerprint

    if has_launch_contract(resolved_benchmark):
        identity["workload_fingerprint"] = native_workload_fingerprint(
            resolved_benchmark, identity["static_execution_fingerprint"]
        )
    # The complete resolved BenchmarkConfig is accepted only after
    # materialization and persisted beside the recipe.  Magpie overlays its
    # envs onto the InferenceX launcher environment, so this second layer also
    # binds PATH/PYTHONPATH and framework-specific env changes which do not
    # affect Magpie's recipe fingerprint.
    identity["launch_config_sha256"] = _canonical_benchmark_sha256(resolved_benchmark)
    identity["launch_environment"] = native_launch_environment_identity(
        resolved_benchmark,
        launch_env=launch_env,
    )
    identity["execution_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(_ENCODING)
    ).hexdigest()
    return identity


def preview_native_recipe(
    benchmark: MutableMapping[str, Any],
    *,
    inferencex_path: str | Path,
) -> dict[str, Any]:
    """Resolve and validate the recipe identity before GPU scheduling.

    This is intentionally read-only with respect to ``benchmark``.  The CLI
    uses it to learn the physical GPU count (TP x PP x PCP) before it exports
    workload defaults or asks the scheduler for resources.  Full launch-time
    validation, including the visible-device mask and outer-image attestation,
    remains in :func:`resolve_native_recipe`.
    """
    root = validate_native_checkout_path(inferencex_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Native AgentX recipe resolution requires an existing InferenceX checkout; got {root}")
    resolved_payload = _run_magpie_recipe_resolver(
        benchmark,
        inferencex_path=root,
    )
    resolved_benchmark = resolved_payload["benchmark"]
    if isinstance(benchmark.get("workload_spec"), dict):
        resolved_benchmark["workload_spec"] = dict(benchmark["workload_spec"])
    run_mode = str(resolved_benchmark.get("run_mode") or "local").lower()
    if run_mode != "local":
        raise ValueError(
            "Hyperloom's native AgentX bridge requires benchmark.run_mode=local; "
            "the optimizer already runs inside the recipe image"
        )
    entry_raw = resolved_payload.get("entry")
    if not isinstance(entry_raw, dict):
        raise RuntimeError("Magpie AgentX recipe resolver omitted the resolved entry")
    entry = dict(entry_raw)
    validate_native_recipe_launcher(
        inferencex_path=root,
        recipe=str(resolved_payload.get("recipe") or ""),
        benchmark_script=str(resolved_benchmark.get("benchmark_script") or ""),
    )
    try:
        tp = int(entry["tp"])
        pp = int(entry.get("pp", 1))
        pcp_size = int(entry.get("pcp-size", 1))
        ep = int(entry.get("ep", 1))
        resolved_envs = resolved_benchmark.get("envs")
        env_conc = resolved_envs.get("CONC") if isinstance(resolved_envs, dict) else None
        raw_conc = entry.get("conc", env_conc)
        if raw_conc is None:
            raise ValueError("Resolved AgentX recipe is missing concurrency")
        conc = int(raw_conc)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resolved AgentX recipe has invalid topology metadata") from exc
    if min(tp, pp, pcp_size, ep, conc) <= 0:
        raise ValueError("Resolved AgentX recipe topology and concurrency values must be positive")
    topology = {
        "tp": tp,
        "pp": pp,
        "pcp_size": pcp_size,
        "ep": ep,
        "gpu_count": tp * pp * pcp_size,
        "conc": conc,
        "duration_seconds": int(entry.get("duration", 0) or 0),
        "recipe_fingerprint": str(entry.get("recipe-fingerprint") or ""),
    }
    return {
        **resolved_payload,
        "benchmark": resolved_benchmark,
        "entry": entry,
        "topology": topology,
        "inferencex_path": str(root),
    }


def resolve_native_recipe(
    benchmark: MutableMapping[str, Any],
    *,
    inferencex_path: str | Path,
    expected_gpu_count: int,
    outer_image: str | None = None,
) -> dict[str, Any]:
    """Resolve one native AgentX point through the installed Magpie API.

    Hyperloom must know the concrete recipe topology *before* its scheduler
    accounts for the serving run.  Leaving resolution until Magpie starts is
    unsafe: InferenceX may select TP x PP x PCP greater than Hyperloom's outer
    ``--tp`` and the result would then be graded with aggregate throughput as
    though it came from one GPU.

    The resolved Magpie fields are persisted back into ``benchmark``.  Unknown
    Hyperloom metadata (notably ``workload_spec``) is retained.
    """
    resolved_payload = preview_native_recipe(
        benchmark,
        inferencex_path=inferencex_path,
    )
    resolved_benchmark = resolved_payload["benchmark"]
    entry = resolved_payload["entry"]
    topology = resolved_payload["topology"]
    tp = int(topology["tp"])
    pp = int(topology["pp"])
    pcp_size = int(topology["pcp_size"])
    ep = int(topology["ep"])
    gpu_count = int(topology["gpu_count"])
    recipe_fingerprint = str(topology.get("recipe_fingerprint") or "").strip()
    expected_fingerprint = os.environ.get("HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT", "").strip()
    if expected_fingerprint and recipe_fingerprint != expected_fingerprint:
        raise ValueError(
            "Native AgentX recipe changed after session finalization: "
            f"expected fingerprint {expected_fingerprint!r}, resolved "
            f"{recipe_fingerprint!r}. Refusing to mix recipe arms in one session."
        )
    if int(expected_gpu_count) != gpu_count:
        raise ValueError(
            "Native AgentX requires Hyperloom --tp to equal the recipe's total "
            f"GPU count (TP x PP x PCP): requested {expected_gpu_count}, "
            f"resolved {tp} x {pp} x {pcp_size} = {gpu_count}. This release "
            "does not support topology-changing AgentX rounds."
        )

    resolved_envs = resolved_benchmark.get("envs")
    if not isinstance(resolved_envs, dict):
        raise RuntimeError("Magpie AgentX recipe resolver returned invalid envs")
    raw_mask = str(resolved_envs.get("ROCR_VISIBLE_DEVICES") or "").strip()
    mask = [part.strip() for part in raw_mask.split(",") if part.strip()]
    expected_mask = [str(index) for index in range(gpu_count)]
    if mask != expected_mask:
        raise ValueError(
            "Pinned InferenceX AgentX launchers overwrite logical "
            "HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES. Until that upstream "
            f"bug is fixed, the mask must be exactly {','.join(expected_mask)!r}; "
            f"got {raw_mask!r}."
        )
    # The zero-based boundary is authoritative.  Do not let Magpie's idle-GPU
    # probe replace it with nonzero physical IDs that the pinned launcher would
    # incorrectly reuse as logical HIP indices.
    raw_selection = resolved_benchmark.get("gpu_selection")
    selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
    selection["auto"] = False
    resolved_benchmark["gpu_selection"] = selection

    image = str(resolved_benchmark.get("docker_image") or entry.get("image") or "").strip()
    declared_outer_image = str(
        outer_image if outer_image is not None else os.environ.get("HYPERLOOM_IMAGE", "")
    ).strip()
    if not declared_outer_image:
        raise ValueError(
            "Native AgentX local mode requires HYPERLOOM_IMAGE to name the "
            f"already-running outer image; the resolved recipe expects {image!r}."
        )
    if declared_outer_image != image:
        raise ValueError(
            "Native AgentX outer image does not match the resolved recipe: "
            f"HYPERLOOM_IMAGE={declared_outer_image!r}, recipe image={image!r}"
        )

    for key, value in resolved_benchmark.items():
        benchmark[key] = value

    workload = benchmark.get("workload_spec")
    if not isinstance(workload, dict):
        workload = {}
        benchmark["workload_spec"] = workload
    topology = {
        "tp": tp,
        "pp": pp,
        "pcp_size": pcp_size,
        "ep": ep,
        "gpu_count": gpu_count,
        "conc": int(topology["conc"]),
        "duration_seconds": int(topology["duration_seconds"]),
        "recipe_fingerprint": str(entry.get("recipe-fingerprint") or ""),
    }
    workload["resolved_topology"] = topology
    workload["recipe"] = {
        "name": str(resolved_payload.get("recipe") or ""),
        "config_file": str(resolved_payload.get("config_file") or ""),
        "recipe_fingerprint": str(entry.get("recipe-fingerprint") or ""),
        "image": image,
        "runner": str(entry.get("runner") or ""),
        "model": str(entry.get("model") or ""),
        "model_prefix": str(entry.get("model-prefix") or ""),
        "framework": str(entry.get("framework") or ""),
        "precision": str(entry.get("precision") or ""),
        "concurrency": int(topology["conc"]),
        "duration_seconds": int(topology["duration_seconds"]),
        "launcher": str(resolved_benchmark.get("benchmark_script") or ""),
    }
    workload["outer_image"] = declared_outer_image
    # This proves config equality only.  Hyperloom cannot introspect the digest
    # of the already-running outer container from inside a generic pod.
    workload["outer_image_config_pinned"] = bool(declared_outer_image)
    expected_execution_fingerprint = os.environ.get("HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT", "").strip()
    if expected_execution_fingerprint:
        execution = native_execution_identity(
            inferencex_path=inferencex_path,
            benchmark_script=str(resolved_benchmark.get("benchmark_script") or ""),
            config_file=str(resolved_payload.get("config_file") or ""),
            resolved_benchmark=resolved_benchmark,
            expected_ref=os.environ.get("INFERENCEX_REF", ""),
            magpie_execution=resolved_payload.get("magpie_execution"),
            expected_magpie_ref=os.environ.get("MAGPIE_REF", ""),
        )
        actual_execution_fingerprint = execution["static_execution_fingerprint"]
        if actual_execution_fingerprint != expected_execution_fingerprint:
            raise ValueError(
                "Native AgentX execution inputs changed after session finalization: "
                f"expected {expected_execution_fingerprint!r}, resolved "
                f"{actual_execution_fingerprint!r}."
            )
        from .identity import has_launch_contract

        optimizing = has_launch_contract(resolved_benchmark)
        pinned_field = "workload_fingerprint" if optimizing else "execution_fingerprint"
        pin_name = (
            "HYPERLOOM_AGENTX_EXPECTED_WORKLOAD_FINGERPRINT"
            if optimizing
            else "HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT"
        )
        materialized_fingerprint = execution[pinned_field]
        accepted_materialized_fingerprint = os.environ.get(pin_name, "").strip()
        if accepted_materialized_fingerprint and materialized_fingerprint != accepted_materialized_fingerprint:
            raise ValueError(
                "Native AgentX fixed benchmark workload changed after its first "
                "materialization: expected fingerprint "
                f"{accepted_materialized_fingerprint!r}, resolved "
                f"{materialized_fingerprint!r}."
            )
        os.environ[pin_name] = materialized_fingerprint
        workload["execution"] = execution
    return topology


__all__ = [
    "AGENTX_REQUIRED_RUNTIME_PIN_NAMES",
    "AGENTX_RUNTIME_PIN_NAMES",
    "native_agentx_enabled",
    "native_execution_identity",
    "native_launch_environment_identity",
    "native_runtime_search_paths",
    "preview_native_recipe",
    "resolve_native_launcher",
    "resolve_native_recipe",
    "scrub_native_agentx_ambient_env",
    "validate_native_recipe_launcher",
    "validate_native_checkout_path",
    "validate_native_launcher_name",
]
