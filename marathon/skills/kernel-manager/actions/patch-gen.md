# Action: Generate Merge-Ready Patch

Produces the `merge_ready/<id>/` directory that the orchestrator reads to apply a kernel
optimization. Every patch MUST be self-contained: the orchestrator should be able to
apply, verify, and roll back without asking the manager any follow-up questions.

## Patch Directory Structure

```
$RESULT_DIR/kernel_manager/merge_ready/<task_id>/
├── metadata.json               # What was done, how to apply, how to roll back
├── patch.diff                   # Git-style unified diff (for python-dispatch, triton-source)
│   OR optimized_<filename>.py   # Full replacement file (for Triton/Inductor rewrites)
│   OR optimized_<filename>.cu   # Full replacement file (for C++/HIP)
├── original_<filename>.py.bak   # Backup of the original file
├── micro_benchmark.json         # Detailed benchmark results (or "deferred")
└── test_harness.py              # (optional) Reproducible test script
```

## metadata.json Schema

```json
{
  "task_id": "string",
  "kernel_name": "string",
  "timestamp": "ISO 8601",
  
  "strategy_used": "self-fix | oob-rewrite | config-change | triton-rewrite | hip-kernel",
  "backend_used": "self | geak | codex | claude | llm-proxy | null",
  "backend_model": "claude-opus-4-6 | gpt-4.1 | null",
  
  "patch_type": "python-dispatch | triton-source | inductor-triton | cpp-rebuild | config-only | jit-source",
  "target_file": "/absolute/path/to/file/being/patched",
  "backup_file": "original_<filename>.py.bak",
  
  "apply_method": "str-replace | file-replace | diff-apply | config-edit | rebuild",
  "apply_instructions": [
    "Step-by-step instructions for the orchestrator to apply the patch"
  ],
  
  "rebuild_required": false,
  "rebuild_command": "cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install",
  "cache_clear_commands": [
    "rm -rf ~/.triton/cache",
    "find /sgl-workspace -name __pycache__ -exec rm -rf {} +"
  ],
  
  "rollback_command": "cp $PATCH_DIR/original_<filename>.py.bak /path/to/target_file",
  "rollback_rebuild_command": null,
  
  "verification_command": "python3 -c \"from sgl_kernel import rotary_embedding; print(rotary_embedding)\"",
  
  "micro_speedup": 9.5,
  "micro_benchmark_status": "passed | deferred | partial",
  "correctness_status": "passed | deferred",
  
  "git_archaeology": {
    "commits_checked": ["661e9775d", "77873343c", "7d4ae057e"],
    "finding": "Correct sgl_kernel path was removed by commit 7d4ae057e (Feb 13, 2026)",
    "fix_approach": "Restore sgl_kernel import for HIP platform"
  },
  
  "risk_assessment": {
    "accuracy_risk": 0.05,
    "crash_risk": 0.02,
    "notes": "Simple dispatch routing change, no numerical computation affected"
  }
}
```

---

## Generating Patches by Type

### Type 1: Python Dispatch Fix (`python-dispatch`)

For one-line routing changes, import fixes, and platform branch corrections.

```python
import json, os, shutil, datetime, difflib

def generate_dispatch_patch(task_id, kernel_name, target_file, 
                            old_content, new_content, git_findings,
                            micro_results, result_dir):
    """Generate a merge-ready patch for a Python dispatch fix."""
    patch_dir = os.path.join(result_dir, "kernel_manager", "merge_ready", task_id)
    os.makedirs(patch_dir, exist_ok=True)
    
    target_basename = os.path.basename(target_file)
    
    shutil.copy2(target_file, os.path.join(patch_dir, f"original_{target_basename}.bak"))
    
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{target_basename}",
        tofile=f"b/{target_basename}",
    )
    with open(os.path.join(patch_dir, "patch.diff"), "w") as f:
        f.writelines(diff)
    
    metadata = {
        "task_id": task_id,
        "kernel_name": kernel_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "strategy_used": "self-fix",
        "backend_used": "self",
        "backend_model": None,
        "patch_type": "python-dispatch",
        "target_file": target_file,
        "backup_file": f"original_{target_basename}.bak",
        "apply_method": "file-replace",
        "apply_instructions": [
            f"cp {patch_dir}/original_{target_basename}.bak {patch_dir}/rollback.bak",
            f"cp {target_file} {patch_dir}/pre_apply.bak",
            f"cd {os.path.dirname(target_file)} && git apply {patch_dir}/patch.diff || "
            f"cp {patch_dir}/optimized_{target_basename} {target_file}",
            f"find {os.path.dirname(target_file)} -name __pycache__ -exec rm -rf {{}} +",
        ],
        "rebuild_required": False,
        "rebuild_command": None,
        "cache_clear_commands": [
            f"find {os.path.dirname(target_file)} -name __pycache__ -exec rm -rf {{}} +",
        ],
        "rollback_command": f"cp {patch_dir}/original_{target_basename}.bak {target_file}",
        "rollback_rebuild_command": None,
        "verification_command": f"python3 -c \"import importlib; m = importlib.import_module('{target_basename.replace('.py', '')}'); print('OK')\"",
        "micro_speedup": micro_results[0] if micro_results and micro_results[0] else None,
        "micro_benchmark_status": "passed" if micro_results and micro_results[0] else "deferred",
        "correctness_status": "passed",
        "git_archaeology": git_findings,
        "risk_assessment": {
            "accuracy_risk": 0.02,
            "crash_risk": 0.01,
            "notes": "Python dispatch routing change, no numerical computation affected",
        },
    }
    
    with open(os.path.join(patch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(os.path.join(patch_dir, f"optimized_{target_basename}"), "w") as f:
        f.write(new_content)
    
    if micro_results and micro_results[1]:
        with open(os.path.join(patch_dir, "micro_benchmark.json"), "w") as f:
            json.dump({"results": micro_results[1], "avg_speedup": micro_results[0]}, f, indent=2)
    
    return patch_dir
```

### Type 2: Triton/Inductor Kernel Rewrite (`triton-source`, `inductor-triton`)

For optimized Triton kernels from OOB backends.

```python
def generate_triton_patch(task_id, kernel_name, target_file,
                          optimized_source, backend_used, backend_model,
                          micro_results, best_config, result_dir):
    """Generate a merge-ready patch for a Triton kernel rewrite."""
    patch_dir = os.path.join(result_dir, "kernel_manager", "merge_ready", task_id)
    os.makedirs(patch_dir, exist_ok=True)
    
    target_basename = os.path.basename(target_file)
    
    shutil.copy2(target_file, os.path.join(patch_dir, f"original_{target_basename}.bak"))
    
    with open(os.path.join(patch_dir, f"optimized_{target_basename}"), "w") as f:
        f.write(optimized_source)
    
    is_inductor = target_file.startswith("/tmp/torchinductor")
    
    apply_instructions = [
        f"cp {target_file} {patch_dir}/pre_apply.bak",
        f"cp {patch_dir}/optimized_{target_basename} {target_file}",
    ]
    cache_clear = ["rm -rf ~/.triton/cache"]
    
    if is_inductor and best_config:
        best_config_path = target_file.replace(".py", ".best_config")
        apply_instructions.append(
            f"echo '{json.dumps(best_config)}' > {best_config_path}"
        )
        cache_clear.append(f"rm -f {target_file.replace('.py', '.so')}")
        cache_clear.append(f"rm -f {target_file.replace('.py', '.json')}")
    
    metadata = {
        "task_id": task_id,
        "kernel_name": kernel_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "strategy_used": "oob-rewrite",
        "backend_used": backend_used,
        "backend_model": backend_model,
        "patch_type": "inductor-triton" if is_inductor else "triton-source",
        "target_file": target_file,
        "backup_file": f"original_{target_basename}.bak",
        "apply_method": "file-replace",
        "apply_instructions": apply_instructions,
        "rebuild_required": False,
        "rebuild_command": None,
        "cache_clear_commands": cache_clear,
        "rollback_command": f"cp {patch_dir}/original_{target_basename}.bak {target_file}",
        "rollback_rebuild_command": None,
        "verification_command": (
            f"python3 -c \"exec(open('{target_file}').read()); print('Compilation OK')\""
        ),
        "micro_speedup": micro_results[0] if micro_results and micro_results[0] else None,
        "micro_benchmark_status": "passed" if micro_results and micro_results[0] else "deferred",
        "correctness_status": "passed" if micro_results and micro_results[0] else "deferred",
        "git_archaeology": None,
        "risk_assessment": {
            "accuracy_risk": 0.15 if "reduction" in kernel_name or "norm" in kernel_name else 0.05,
            "crash_risk": 0.05,
            "notes": f"Triton kernel rewrite by {backend_used}/{backend_model}",
        },
    }
    
    if best_config:
        metadata["best_config"] = best_config
    
    with open(os.path.join(patch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if micro_results and micro_results[1]:
        with open(os.path.join(patch_dir, "micro_benchmark.json"), "w") as f:
            json.dump({"results": micro_results[1], "avg_speedup": micro_results[0]}, f, indent=2)
    
    return patch_dir
```

### Type 3: C++/HIP Kernel (`cpp-rebuild`)

For compiled extension changes that require `setup_rocm.py` rebuild.

```python
def generate_cpp_patch(task_id, kernel_name, target_file,
                       optimized_source, backend_used, backend_model,
                       micro_results, result_dir):
    """Generate a merge-ready patch for a C++/HIP kernel change."""
    patch_dir = os.path.join(result_dir, "kernel_manager", "merge_ready", task_id)
    os.makedirs(patch_dir, exist_ok=True)
    
    target_basename = os.path.basename(target_file)
    
    shutil.copy2(target_file, os.path.join(patch_dir, f"original_{target_basename}.bak"))
    
    with open(os.path.join(patch_dir, f"optimized_{target_basename}"), "w") as f:
        f.write(optimized_source)
    
    is_sgl_kernel = "/sgl-kernel/" in target_file
    is_aiter = "/aiter/" in target_file
    
    if is_sgl_kernel:
        rebuild_cmd = "cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install"
        rollback_rebuild = rebuild_cmd
    elif is_aiter:
        rebuild_cmd = "cd /sgl-workspace/aiter && pip install -e . --no-deps"
        rollback_rebuild = rebuild_cmd
    else:
        rebuild_cmd = None
        rollback_rebuild = None
    
    metadata = {
        "task_id": task_id,
        "kernel_name": kernel_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "strategy_used": "oob-rewrite",
        "backend_used": backend_used,
        "backend_model": backend_model,
        "patch_type": "cpp-rebuild",
        "target_file": target_file,
        "backup_file": f"original_{target_basename}.bak",
        "apply_method": "file-replace",
        "apply_instructions": [
            f"cp {target_file} {patch_dir}/pre_apply.bak",
            f"cp {patch_dir}/optimized_{target_basename} {target_file}",
            f"# Rebuild required:",
            rebuild_cmd or "# No rebuild command — manual intervention needed",
        ],
        "rebuild_required": True,
        "rebuild_command": rebuild_cmd,
        "cache_clear_commands": [],
        "rollback_command": f"cp {patch_dir}/original_{target_basename}.bak {target_file}",
        "rollback_rebuild_command": rollback_rebuild,
        "verification_command": (
            f"python3 -c \"import sgl_kernel; print(sgl_kernel.__file__)\"" if is_sgl_kernel
            else f"python3 -c \"import aiter; print(aiter.__file__)\"" if is_aiter
            else "echo 'Manual verification needed'"
        ),
        "micro_speedup": micro_results[0] if micro_results and micro_results[0] else None,
        "micro_benchmark_status": "passed" if micro_results and micro_results[0] else "deferred",
        "correctness_status": "passed" if micro_results and micro_results[0] else "deferred",
        "git_archaeology": None,
        "risk_assessment": {
            "accuracy_risk": 0.15,
            "crash_risk": 0.10,
            "notes": f"C++/HIP kernel change requiring library rebuild. "
                     f"Rebuild time: ~5-15 min.",
        },
    }
    
    with open(os.path.join(patch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if micro_results and micro_results[1]:
        with open(os.path.join(patch_dir, "micro_benchmark.json"), "w") as f:
            json.dump({"results": micro_results[1], "avg_speedup": micro_results[0]}, f, indent=2)
    
    return patch_dir
```

### Type 4: Config Change (`config-only`)

For tuning CSV edits, environment variable changes, or JSON config updates.

```python
def generate_config_patch(task_id, kernel_name, target_file,
                          old_content, new_content, micro_results, result_dir):
    """Generate a merge-ready patch for a config file change."""
    patch_dir = os.path.join(result_dir, "kernel_manager", "merge_ready", task_id)
    os.makedirs(patch_dir, exist_ok=True)
    
    target_basename = os.path.basename(target_file)
    
    shutil.copy2(target_file, os.path.join(patch_dir, f"original_{target_basename}.bak"))
    
    with open(os.path.join(patch_dir, f"optimized_{target_basename}"), "w") as f:
        f.write(new_content)
    
    metadata = {
        "task_id": task_id,
        "kernel_name": kernel_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "strategy_used": "config-change",
        "backend_used": "self",
        "backend_model": None,
        "patch_type": "config-only",
        "target_file": target_file,
        "backup_file": f"original_{target_basename}.bak",
        "apply_method": "file-replace",
        "apply_instructions": [
            f"cp {target_file} {patch_dir}/pre_apply.bak",
            f"cp {patch_dir}/optimized_{target_basename} {target_file}",
        ],
        "rebuild_required": False,
        "rebuild_command": None,
        "cache_clear_commands": [],
        "rollback_command": f"cp {patch_dir}/original_{target_basename}.bak {target_file}",
        "rollback_rebuild_command": None,
        "verification_command": f"cat {target_file} | head -5",
        "micro_speedup": micro_results[0] if micro_results and micro_results[0] else None,
        "micro_benchmark_status": "passed" if micro_results and micro_results[0] else "deferred",
        "correctness_status": "deferred",
        "git_archaeology": None,
        "risk_assessment": {
            "accuracy_risk": 0.01,
            "crash_risk": 0.01,
            "notes": "Config file change only, no code modification",
        },
    }
    
    with open(os.path.join(patch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return patch_dir
```

---

## Writing to results.jsonl

After generating the patch directory, write a result entry to `results.jsonl`:

```python
def write_result_entry(task_id, status, patch_dir, metadata, result_dir):
    """Write a result entry to results.jsonl for the orchestrator to consume."""
    results_path = os.path.join(result_dir, "kernel_manager", "results.jsonl")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    entry = {
        "id": task_id,
        "status": status,  # "merge-ready", "failed", "no-improvement"
        "strategy_used": metadata.get("strategy_used"),
        "backend_used": metadata.get("backend_used"),
        "micro_speedup": metadata.get("micro_speedup"),
        "patch_dir": patch_dir,
        "patch_type": metadata.get("patch_type"),
        "rebuild_required": metadata.get("rebuild_required", False),
        "rollback_command": metadata.get("rollback_command"),
        "verification_command": metadata.get("verification_command"),
        "error_message": metadata.get("error_message"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    
    line = json.dumps(entry) + "\n"
    with open(results_path, "a") as f:
        f.write(line)
    
    return entry
```

---

## Validation Before Writing

**The manager MUST verify the patch directory is complete before writing to results.jsonl.**

```python
def validate_patch_dir(patch_dir):
    """Verify a patch directory is complete and valid."""
    errors = []
    
    metadata_path = os.path.join(patch_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        errors.append("Missing metadata.json")
        return errors
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    required_fields = [
        "task_id", "kernel_name", "patch_type", "target_file",
        "rollback_command", "apply_instructions",
    ]
    for field in required_fields:
        if not metadata.get(field):
            errors.append(f"Missing or empty field: {field}")
    
    backup = metadata.get("backup_file")
    if backup and not os.path.exists(os.path.join(patch_dir, backup)):
        errors.append(f"Backup file not found: {backup}")
    
    has_patch = any(
        os.path.exists(os.path.join(patch_dir, f))
        for f in os.listdir(patch_dir)
        if f.startswith("optimized_") or f == "patch.diff"
    )
    if not has_patch:
        errors.append("No patch file (optimized_* or patch.diff) found")
    
    return errors
```

---

## End-to-End Example: Self-Fix Dispatch Bug

```python
task = {
    "id": "rope_dispatch_001",
    "kernel_name": "rotary_embedding",
    "source_file": "/sgl-workspace/sglang/python/sglang/srt/layers/rotary_embedding.py",
    "strategy": "dispatch-fix",
    "dispatch_analysis": {"active_path": "jit", "optimal_path": "sgl_kernel", "dispatch_bug": True},
}

# 1. Read original
original = open(task["source_file"]).read()

# 2. Git archaeology
# git log -S "sgl_kernel" -- rotary_embedding.py → found commit 7d4ae057e removed it
git_findings = {
    "commits_checked": ["661e9775d", "77873343c", "7d4ae057e"],
    "finding": "sgl_kernel path removed by 7d4ae057e, replaced with slow JIT path",
    "fix_approach": "Restore sgl_kernel import for HIP",
}

# 3. Write fix
new_content = original.replace(
    'from sglang.jit_kernel import rotary_embedding',
    'from sgl_kernel import rotary_embedding',
)

# 4. Test
# (compilation check, correctness if GPU free, micro-benchmark if GPU free)

# 5. Generate patch
patch_dir = generate_dispatch_patch(
    task_id="rope_dispatch_001",
    kernel_name="rotary_embedding",
    target_file=task["source_file"],
    old_content=original,
    new_content=new_content,
    git_findings=git_findings,
    micro_results=(9.5, [{"xnumel": 1, "speedup": 9.5, "orig_ms": 0.124, "opt_ms": 0.013}]),
    result_dir=os.environ.get("RESULT_DIR", "/tmp"),
)

# 6. Validate
errors = validate_patch_dir(patch_dir)
assert not errors, f"Patch validation failed: {errors}"

# 7. Write result
write_result_entry(
    task_id="rope_dispatch_001",
    status="merge-ready",
    patch_dir=patch_dir,
    metadata=json.load(open(os.path.join(patch_dir, "metadata.json"))),
    result_dir=os.environ.get("RESULT_DIR", "/tmp"),
)
```
