"""Capture native campaign allocation evidence inside an assigned Ray actor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

VISIBLE = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
           "GPU_DEVICE_ORDINAL", "HSA_VISIBLE_DEVICES")
RECEIPT_ENV = "HYPERLOOM_RAY_ALLOCATION_RECEIPT"
SHA_ENV = "HYPERLOOM_RAY_ALLOCATION_SHA256"


def capture_launch_environment(caller_env):
    """Add a fresh allocation receipt without changing any GPU mask."""
    if not caller_env or "HYPERLOOM_PAIR_GPU_BINDING_JSON" not in caller_env:
        return caller_env
    import ray

    context = ray.get_runtime_context()
    resources = context.get_assigned_resources()
    actor_id = context.get_actor_id()
    if not actor_id or resources.get("GPU") != 4 or resources.get("serving_slot") != 1:
        raise ValueError("native benchmark requires an assigned four-GPU serving actor")
    job = os.environ.get("SPUR_JOB_ID") or os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal():
        raise ValueError("native Ray actor has no scheduler identity")
    root = Path("/output") / ("native-" + job)
    admission_path = root / "admitted.json"
    admission_raw = admission_path.read_bytes()
    admission = json.loads(admission_raw)
    if admission["job_id"] != job or admission["gpu_mapping"] != json.loads(caller_env["HYPERLOOM_PAIR_GPU_BINDING_JSON"]):
        raise ValueError("native admission and actor binding disagree")
    policy_path = Path(caller_env["COMPARISON_SERVING_POLICY"])
    arm = policy_path.parent.name
    expected_policy = root / "comparison/native-serving" / arm / "policy.json"
    if arm not in ("control", "treatment") or policy_path != expected_policy or policy_path.resolve() != policy_path:
        raise ValueError("native serving policy is outside this job/arm")
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw)
    if policy["native_rocr"] != ",".join(map(str, admission["gpu_mapping"]["serving_physical_ids"])):
        raise ValueError("native policy and admission GPU order disagree")
    launch_id = uuid.uuid4().hex
    record = {"schema": "native-ray-allocation-v1", "launch_id": launch_id,
              "native_job_id": job, "native_config_sha256": admission["config_sha256"],
              "arm": arm, "admission_path": str(admission_path),
              "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
              "policy_path": str(policy_path), "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
              "Ray_job_id": context.get_job_id(), "Ray_node_id": context.get_node_id(),
              "Ray_actor_id": actor_id, "assigned_resources": resources,
              "accelerator_ids": context.get_accelerator_ids(), "actor_pid": os.getpid(),
              "actor_start_ticks": Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "worker_visibility": {name: os.environ.get(name) for name in VISIBLE},
              "capture_point": "ServingActor.run_blocking before _run_subprocess_worker"}
    directory = policy_path.parent / "ray-allocations"
    directory.mkdir(mode=0o700, exist_ok=True)
    target = directory / (launch_id + ".json")
    raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    result = dict(caller_env)
    result[RECEIPT_ENV] = str(target)
    result[SHA_ENV] = hashlib.sha256(raw).hexdigest()
    return result
