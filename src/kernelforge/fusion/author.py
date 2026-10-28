# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3: author an env-gated fused kernel from a self-discovered recipe."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.session_resume import is_api_failure

from .author_hooks import target_file_hooks
from .emit import _FUSED_MODULE_MARKERS, _FUSED_MODULE_PREFIXES, _is_fused_module_name
from .llm_failure import (
    DEFAULT_BASE_DELAY_SEC,
    DEFAULT_MAX_DELAY_SEC,
    RETRYABLE_KINDS,
    classify_llm_error,
    env_setting,
    is_agent_safety_error,
    is_agent_timeout_error,
    retry_delay,
)
from .harness_contract import harness_contract, trace_kernels_block
from .validate import DEFAULT_TARGET_SPEEDUP
from kernelforge.tools.git import git_bytes
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS

log = logging.getLogger("forge_fusion")

# Process-style author return codes.
AUTHOR_RC_OK = 0
AUTHOR_RC_FAILED = 1
AUTHOR_RC_SAFETY = 3
AUTHOR_RC_TIMEOUT = 124

# An author session costs minutes and gates a lane that costs hours, so a transport that
# never delivered an answer is worth one more call. Two attempts by default: past that the
# upstream is down rather than flaky, and the lane is better off reporting it.
DEFAULT_AUTHOR_ATTEMPTS = 2


def proven_fusion_fewshot() -> str:
    """Few-shot block of serving-validated decode fusions (worked examples)."""
    return """## Proven fusion examples (few-shot — these ALL passed real sglang serving e2e)
- ZAYA CCA QK post-processing (`ZAYA_FUSED_QK`): fold `_add_grouped_qk_means` +
  `_normalize_qk` (~15-20 tiny fp32 view/mean/add/mul/pow/sum/rsqrt ops) into ONE
  Triton kernel, one program per (token, k-head). +14.7% e2e alone.
- ZAYA ResidualScaling (`ZAYA_FUSED_RESIDUAL`): dual affine `(x+bias)*scale` on the
  hidden AND residual streams in ONE launch, bf16->fp32 in-kernel. QK+Residual
  together = +34.5% e2e.
- LFM2 (`LFM2_FUSED_RESIDUAL` / `LFM2_FUSED_SILU`): thread the per-layer residual
  adds into the next RMSNorm; merge w1|w3 SwiGLU into one GEMM + fused SiluAndMul.
  ~+16% e2e.
- Granite (`GRANITE_FUSED_RESIDUAL`): `scaled_add_rmsnorm` = `rmsnorm(x*scale + r)`,
  folding scalar-mul + residual-add + RMSNorm into ONE kernel; ~5e-9 vs eager.

## MANDATORY patterns from these serving-validated kernels
## (violating them passes microbench but CRASHES real serving — do NOT):
1. env-gated: with the flag UNSET the path is bit-for-bit the original eager code.
2. fp32 accumulation INSIDE the Triton kernel (cast bf16->fp32 in-kernel, not outside).
3. ONE Triton launch replacing the whole tiny-op chain (fewer launches = the win).
4. CUDA-GRAPH SAFE (CRITICAL): the kernel runs INSIDE the captured decode CUDA graph.
   Preallocate ALL outputs; use tl.constexpr for shapes; NO python-side allocation,
   NO `.item()`/`.cpu()`/torch host sync, NO data-dependent shapes in the decode hot
   path. A kernel that allocates or host-syncs per call passes a standalone microbench
   but SIGQUIT-crashes the sglang scheduler decode loop.
5. import the REAL eager op as the parity oracle; keep public signatures/imports intact.
6. ROCm-native Triton only; never reuse a CUDA-only framework fused op.

"""


def _arch_phrase(gpu_arch: str) -> str:
    """How to name the target GPU in a prompt."""
    arch = (gpu_arch or "").strip().lower()
    marketing = {"gfx950": "MI355X", "gfx942": "MI300X/MI325X"}.get(arch, "")
    if not arch:
        return "an AMD ROCm GPU"
    return f"AMD {marketing} ({arch})" if marketing else f"an AMD GPU ({arch})"


def _model_dir_block(model_path: str) -> str:
    """Name the model directory, because the alternative is that it gets searched for."""
    if not model_path:
        return ""
    return (
        f"- Model directory (config.json, tokenizer, weights): {model_path}\n"
        "  Read the model's own files from there rather than searching for them: "
        "`/` on this host includes multi-terabyte network mounts, and a single "
        "`find /` can outlast the whole authoring budget.\n"
    )


@dataclass(frozen=True)
class AuthorBrief:
    """What the author is told besides the recipe: framework, how it is validated, the bar, harness, GPU and model."""

    framework: str
    ab_hint: str
    target_speedup: float = DEFAULT_TARGET_SPEEDUP
    harness_path: str = ""
    gpu_arch: str = ""
    model_path: str = ""


def build_author_prompt(recipe: dict, brief: AuthorBrief) -> str:
    """Build the authoring prompt from a recipe dict (``Recipe.to_dict()``)."""
    framework, ab_hint, target_speedup = brief.framework, brief.ab_hint, brief.target_speedup
    harness_path, gpu_arch, model_path = brief.harness_path, brief.gpu_arch, brief.model_path
    shapes = recipe.get("shapes", {})
    hints = recipe.get("source_hints", [])
    env_flag = recipe.get("env_flag", "FUSED")
    harness_block = harness_contract(harness_path, env_flag) if harness_path else ""
    candidate_kind = str(recipe.get("candidate_kind") or "new_fusion")
    existing_operator = str(recipe.get("existing_operator") or "").strip()
    integration_block = ""
    if candidate_kind == "integration" and existing_operator:
        integration_block = f"""## Existing operator integration (MANDATORY first path)
- Candidate kind: integration
- Existing ROCm operator: `{existing_operator}`
- Reproduce the eager boundary, then benchmark and wire the existing operator first.
- Do not author a replacement kernel unless the existing operator is incompatible or
  loses the exact-shape microbenchmark; record that evidence before falling back.
- The operator's numerics were never verified against THIS model's eager path. Before
  keeping it, record parity against the eager reference at the framework's own
  tolerance (reuse the rtol/atol the framework's tests use for this dtype; do not
  invent a looser one). A microbenchmark win alone is NOT sufficient to keep it.
- Report the measured max relative error alongside the speedup, so a fast but
  numerically worse operator is visible as such.

"""
    rocm_line = (
        "- TARGET IS ROCm (AMD GPU): author a ROCm-native Triton (or aiter) kernel. "
        "Do NOT reuse a framework CUDA-only fused op; verify it BUILDS and RUNS.\n"
        if recipe.get("rocm_native", True)
        else ""
    )
    return f"""You are optimizing the {framework} model file for a decode-path kernel fusion on
{_arch_phrase(gpu_arch)}, bf16 serving. Work autonomously; do not ask questions.

## Target
- Framework source file to edit: {recipe.get("source_file") or "(resolve it under the framework model dir)"}
{_model_dir_block(model_path)}- Fusion pattern: {recipe.get("pattern")}
- {recipe.get("description")}
{trace_kernels_block(recipe.get("trace_kernels"))}
## What to fuse (the recipe)
{recipe.get("fusion_math")}

## Representative decode shapes (from the model config + trace)
{shapes}

## How to localize it in the source
Grep the model file for these anchors and fuse the chain they mark:
{chr(10).join(f"  - {h}" for h in hints)}

{integration_block}{proven_fusion_fewshot()}## Engineering discipline (MUST follow)
- The kernel MUST be REACHED by the decode path, not merely defined and exposed.
  Assigning it onto another module (`other_mod.my_fused_op = ...`) does nothing
  unless something already reads that name -- check that a reader exists before
  you rely on it. When the chain you are fusing lives in a file you may not edit,
  take over the call site from the file you may: rebind the method or attribute
  the chain already goes through, so the existing callers reach your kernel
  without changing any signature. Publishing a new name that nothing calls scores
  as a failed attempt, and it is the most common way a correct kernel is wasted.
- If you cannot reach the chain from the files you are allowed to touch, say so
  explicitly in your final message instead of leaving an unreachable kernel
  behind. That answer is useful; an inert one is not.
- The fusion MUST be env-gated by `{env_flag}`. With the flag UNSET the code path
  stays bit-for-bit the original eager path.
{rocm_line}- Cast to fp32 inside the fused kernel; one launch instead of the multi-op chain.
- NET launch count must DROP. Count the kernels the eager path launches per decode
  step and the kernels the fused path launches; the fused number must be strictly
  smaller. A scratch-fill, a separate cast or a contiguous copy added to feed your
  kernel can cancel the launches it saved while the chain alone still benchmarks
  faster. Report both counts alongside the speedup.
- CUDA-graph safe: no Python-side dynamic allocation or host sync in the decode
  hot path (preallocate outputs; use tl.constexpr for shapes).
- Keep all public function/class signatures and imports intact.
- Add a pure-torch reference and assert parity BEFORE trusting the kernel.
  {recipe.get("eager_reference_hint")}
- If Triton is unavailable, fall back to eager (never crash).
{harness_block}
## How to validate (the ONLY success signal)
Run this A/B (boots the model twice; eager vs `{env_flag}=1`), decode-step median:
    {ab_hint}
- SUCCESS = fused clearly faster than eager (target speedup >= {target_speedup:.2f}x).
- Also confirm correctness: greedy output with the flag on stays coherent vs off.
- Iterate: edit -> run A/B -> read speedup -> fix -> repeat until the target is met.

When done, print: `AUTHORING_RESULT: env_flag={env_flag} speedup=<X>x files=<edited files>` and stop.
Do not edit the A/B harness or hard-code any numbers.
"""


def build_multi_author_prompt(recipes: list[dict], brief: AuthorBrief) -> str:
    """Prompt to author SEVERAL confirmed fusions in one pass (each env-gated)."""
    if len(recipes) == 1:
        return build_author_prompt(recipes[0], brief)
    framework, ab_hint, target_speedup = brief.framework, brief.ab_hint, brief.target_speedup
    harness_path, gpu_arch, model_path = brief.harness_path, brief.gpu_arch, brief.model_path
    blocks = []
    all_flags = []
    for i, r in enumerate(recipes, 1):
        all_flags.append(r.get("env_flag", "FUSED"))
        blocks.append(
            f"### Fusion {i}: {r.get('pattern')} (env gate `{r.get('env_flag')}`)\n"
            f"{r.get('description')}\n"
            f"Math: {r.get('fusion_math')}\n"
            f"Source anchors: {', '.join(r.get('source_hints', []))}\n"
            f"Eager reference: {r.get('eager_reference_hint')}\n"
            f"Candidate kind: {r.get('candidate_kind', 'new_fusion')}\n"
            f"Existing operator: {r.get('existing_operator', '')}\n"
        )
    src = recipes[0].get("source_file") or "(resolve under the framework model dir)"
    shapes = recipes[0].get("shapes", {})
    harness_block = harness_contract(harness_path, " ".join(all_flags)) if harness_path else ""
    existing_operators = [
        str(recipe.get("existing_operator"))
        for recipe in recipes
        if recipe.get("candidate_kind") == "integration" and recipe.get("existing_operator")
    ]
    integration_block = ""
    if existing_operators:
        integration_block = f"""## Existing operator integrations (MANDATORY first paths)
Benchmark and wire these existing ROCm operators before authoring replacements:
{chr(10).join(f"- `{operator}`" for operator in existing_operators)}
Do not author replacement kernels unless an operator is incompatible or loses its
exact-shape microbenchmark; record that evidence before falling back.
For each integrated operator, record parity against the eager reference at the
framework's own rtol/atol for this dtype and report the measured max relative error
next to the speedup. A microbenchmark win alone is NOT sufficient to keep it.

"""
    rocm_line = (
        "- TARGET IS ROCm (AMD GPU): author ROCm-native Triton/aiter kernels; do NOT "
        "reuse a framework CUDA-only fused op; verify each BUILDS and RUNS.\n"
        if any(r.get("rocm_native", True) for r in recipes)
        else ""
    )
    return f"""You are optimizing the {framework} model file `{src}` with SEVERAL decode-path
kernel fusions on {_arch_phrase(gpu_arch)}, bf16 serving. Work autonomously; no questions.

{_model_dir_block(model_path)}{trace_kernels_block(recipes[0].get("trace_kernels"))}
## Representative decode shapes (model config + trace)
{shapes}

## Fusions to author (each INDEPENDENTLY env-gated, default OFF = original eager path)
{chr(10).join(blocks)}

{integration_block}{proven_fusion_fewshot()}## Engineering discipline (MUST follow for every fusion)
- Each fusion is env-gated; with its flag UNSET the path is bit-for-bit eager.
{rocm_line}- Cast to fp32 inside the fused kernel; one launch instead of the op chain.
- CUDA-graph safe (preallocate outputs; tl.constexpr shapes; no host sync in decode).
- Keep public signatures/imports intact; import the REAL eager op for the parity ref.
- Add a parity self-check before trusting each kernel; fall back to eager if Triton is missing.
{harness_block}
## Validate (the ONLY success signal) — all flags ON together:
    {ab_hint}
- SUCCESS = combined fused clearly faster than eager (target speedup >= {target_speedup:.2f}x).
- Confirm greedy output stays coherent with all flags on. Iterate until the target is met.

When done print: `AUTHORING_RESULT: env_flags={" ".join(all_flags)} speedup=<X>x files=<edited>` and stop.
Do not edit the A/B harness or hard-code numbers.
"""


class AuthorSafetyError(RuntimeError):
    """Report an author workspace state that cannot be safely accepted."""

    def __init__(
        self,
        message: str,
        *,
        paths: Optional[list[str]] = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.paths = list(paths or [])
        self.transient = bool(transient)


@dataclass(frozen=True)
class _WorkspacePathState:
    """Exact restorable state for one Git-visible worktree path."""

    kind: str
    data: bytes = b""
    mode: int = 0


@dataclass(frozen=True)
class _AuthorWorkspaceOutcome:
    """What one guarded author run left behind once restoration finished."""

    violations: tuple[str, ...] = ()
    created: tuple[str, ...] = ()


def _git_bytes(
    root: Path,
    *args: str,
    input_data: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run one bounded Git plumbing command without decoding path bytes."""
    try:
        result = git_bytes("-C", str(root), *args, input=input_data, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        # Pure I/O, and ``SubprocessError`` covers ``TimeoutExpired``: a 60s ``git ls-files`` timeout against an NFS
        # worktree under a concurrent serving campaign is weather, not a verdict on what the author did.
        raise AuthorSafetyError(
            f"author workspace Git command failed: git {' '.join(args[:3])}: {type(exc).__name__}",
            transient=True,
        ) from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()[-400:]
        raise AuthorSafetyError(
            f"author workspace Git command failed: git {' '.join(args[:3])}: {detail or f'exit {result.returncode}'}"
        )
    return result


def _decode_git_path(value: bytes) -> str:
    """Decode a NUL-delimited Git path without losing arbitrary bytes."""
    return value.decode(errors="surrogateescape")


def _nul_git_paths(output: bytes) -> set[str]:
    return {_decode_git_path(value) for value in output.split(b"\0") if value}


def _capture_path_state(path: Path) -> _WorkspacePathState:
    """Capture a regular file, symlink, or absence without following symlinks."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _WorkspacePathState("absent")
    except OSError as exc:
        # Reading the worktree is the guard's own bookkeeping, not a verdict on what the author did, and it recovers
        # on its own -- same reason the Git command and index-lock paths are marked retryable.
        raise AuthorSafetyError(
            f"cannot inspect author workspace path: {path}",
            transient=True,
        ) from exc
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        try:
            return _WorkspacePathState(
                "symlink",
                os.fsencode(os.readlink(path)),
                mode,
            )
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot read author workspace symlink: {path}",
                transient=True,
            ) from exc
    if stat.S_ISREG(metadata.st_mode):
        try:
            return _WorkspacePathState("file", path.read_bytes(), mode)
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot snapshot author workspace file: {path}",
                transient=True,
            ) from exc
    if stat.S_ISDIR(metadata.st_mode):
        return _WorkspacePathState("directory", mode=mode)
    return _WorkspacePathState("unsupported", mode=mode)


FUSION_SCRATCH_DIRNAME = ".forge_fusion"


def _is_fusion_scratch_relpath(rel: str) -> bool:
    """Whether a repo-relative path is forge-fusion's own staging directory."""
    parts = Path(rel).parts
    return len(parts) > 1 and parts[0] == FUSION_SCRATCH_DIRNAME


class _AuthorWorkspaceGuard:
    """Restore every Git-visible mutation outside the author's writable scope."""

    def __init__(self, workdir: str, target_files: list[str], new_module_dirs: list[str]) -> None:
        cwd = Path(workdir).expanduser().resolve()
        root_result = _git_bytes(cwd, "rev-parse", "--show-toplevel")
        root_text = root_result.stdout.decode(errors="surrogateescape").strip()
        if not root_text:
            raise AuthorSafetyError("registered authoring requires a Git worktree")
        self.root = Path(root_text).resolve()
        self.cwd = cwd
        self.target_files = [str(self._contained_path(value, "author target")) for value in target_files]
        self.target_relpaths = {Path(path).relative_to(self.root).as_posix() for path in self.target_files}

        # Name inventory per nominated directory.
        self.new_module_baselines: dict[str, frozenset[str]] = {}
        for value in new_module_dirs:
            directory = self._contained_path(value, "author module directory")
            try:
                entries = frozenset(os.listdir(directory))
            except FileNotFoundError as exc:
                # An empty inventory is the most permissive scope there is -- every name in it counts as absent -- and
                # the prompt would then advertise a directory the author cannot write into anyway.
                raise AuthorSafetyError(
                    f"author module directory does not exist: {directory}",
                    paths=[str(directory)],
                ) from exc
            except OSError as exc:
                # A missing directory above is a verdict: it is absent on every attempt.
                raise AuthorSafetyError(
                    f"cannot inventory author module directory: {directory}",
                    paths=[str(directory)],
                    transient=True,
                ) from exc
            self.new_module_baselines[directory.relative_to(self.root).as_posix()] = entries
        self.new_module_dirs = [str(self.root / rel) for rel in self.new_module_baselines]

        self.baseline_head = self._head()
        self.baseline_branch = self._branch()
        self.baseline_index_entries = self._index_entries()
        self.baseline_index_flags = self._index_flags()
        self.index_path = self._index_path()
        self.index_lock_path = Path(f"{self.index_path}.lock")
        if self.index_lock_path.exists():
            # ``index.lock`` exists for milliseconds whenever anything else runs a Git command in this worktree, so
            # the next attempt very likely finds it gone.
            raise AuthorSafetyError(
                "Git index is locked before authoring; workspace snapshot is unsafe",
                paths=[str(self.index_lock_path)],
                transient=True,
            )
        try:
            self.baseline_index_bytes = self.index_path.read_bytes()
            self.baseline_index_mode = stat.S_IMODE(self.index_path.stat().st_mode)
        except OSError as exc:
            raise AuthorSafetyError(f"cannot snapshot Git index: {self.index_path}") from exc
        hidden_index_paths = {rel for rel, flag in self.baseline_index_flags.items() if flag != "H"}
        self.baseline_states: dict[str, _WorkspacePathState] = {}
        for rel in self._status_paths() | hidden_index_paths:
            state = _capture_path_state(self._path(rel))
            if state.kind in {"directory", "unsupported"}:
                raise AuthorSafetyError(
                    f"cannot safely snapshot Git-visible path type: {rel}",
                    paths=[rel],
                )
            self.baseline_states[rel] = state

    def _contained_path(self, value: str, what: str) -> Path:
        """Resolve one caller-named path, refusing one outside the worktree or reached through a symlink."""
        raw = Path(value).expanduser()
        lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else self.cwd / raw)))
        resolved = lexical.resolve(strict=False) if lexical.is_relative_to(self.root) else lexical
        if not resolved.is_relative_to(self.root):
            raise AuthorSafetyError(f"{what} is outside the Git worktree: {lexical}", paths=[str(lexical)])
        if lexical.is_symlink() or resolved != lexical:
            raise AuthorSafetyError(f"{what} symlink/path escape is not allowed: {lexical}", paths=[str(lexical)])
        return resolved

    def _head(self) -> bytes:
        return _git_bytes(
            self.root,
            "rev-parse",
            "--verify",
            "HEAD",
        ).stdout.strip()

    def _branch(self) -> bytes:
        result = _git_bytes(
            self.root,
            "symbolic-ref",
            "-q",
            "HEAD",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else b""

    def _index_path(self) -> Path:
        raw = _git_bytes(self.root, "rev-parse", "--git-path", "index").stdout.decode(errors="surrogateescape").strip()
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def _index_entries(self) -> dict[str, tuple[tuple[str, str, int], ...]]:
        entries: dict[str, list[tuple[str, str, int]]] = {}
        output = _git_bytes(
            self.root,
            "ls-files",
            "--stage",
            "-z",
        ).stdout
        for record in output.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                raw_mode, raw_oid, raw_stage = metadata.split()
                entry = (
                    raw_mode.decode("ascii"),
                    raw_oid.decode("ascii"),
                    int(raw_stage),
                )
            except (ValueError, UnicodeError) as exc:
                raise AuthorSafetyError("cannot parse Git index metadata for author workspace") from exc
            entries.setdefault(_decode_git_path(raw_path), []).append(entry)
        return {path: tuple(sorted(values, key=lambda item: item[2])) for path, values in entries.items()}

    def _index_flags(self) -> dict[str, str]:
        flags: dict[str, str] = {}
        output = _git_bytes(self.root, "ls-files", "-v", "-z").stdout
        for record in output.split(b"\0"):
            if not record:
                continue
            if len(record) < 3 or record[1:2] != b" ":
                raise AuthorSafetyError("cannot parse Git index flags for author workspace")
            flags[_decode_git_path(record[2:])] = record[:1].decode(
                "ascii",
                errors="replace",
            )
        return flags

    def _status_paths(self) -> set[str]:
        paths: set[str] = set()
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "--",
                    ".",
                ).stdout
            )
        )
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "diff",
                    "--cached",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "--",
                    ".",
                ).stdout
            )
        )
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ).stdout
            )
        )
        return paths

    def _path(self, rel: str) -> Path:
        raw = Path(rel)
        if raw.is_absolute() or ".." in raw.parts:
            raise AuthorSafetyError(
                f"unsafe Git path returned by workspace query: {rel}",
                paths=[rel],
            )
        return self.root / raw

    def _remove_path(self, rel: str) -> None:
        path = self._path(rel)
        self._ensure_safe_parent(path)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot inspect path during author rollback: {rel}",
                paths=[rel],
            ) from exc
        try:
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot remove path during author rollback: {rel}",
                paths=[rel],
            ) from exc

    def _ensure_safe_parent(self, path: Path) -> None:
        relative = path.parent.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                current.mkdir()
                continue
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                continue
            rel = current.relative_to(self.root).as_posix()
            if rel in self.preserved_relpaths:
                raise AuthorSafetyError(
                    f"cannot restore non-target below a preserved path: {rel}",
                    paths=[rel],
                )
            try:
                current.unlink()
                current.mkdir()
            except OSError as exc:
                raise AuthorSafetyError(
                    f"cannot repair unsafe parent during author rollback: {rel}",
                    paths=[rel],
                ) from exc

    def _restore_state(self, rel: str, state: _WorkspacePathState) -> None:
        path = self._path(rel)
        self._remove_path(rel)
        if state.kind == "absent":
            return
        self._ensure_safe_parent(path)
        try:
            if state.kind == "file":
                path.write_bytes(state.data)
                path.chmod(state.mode)
            elif state.kind == "symlink":
                os.symlink(os.fsdecode(state.data), path)
            else:
                raise AuthorSafetyError(
                    f"unsupported baseline path type during rollback: {rel}",
                    paths=[rel],
                )
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot restore path after rejected author run: {rel}",
                paths=[rel],
            ) from exc

    def _restore_index(
        self,
        post_entries: dict[str, tuple[tuple[str, str, int], ...]],
        preserved_targets: set[str],
    ) -> None:
        if self.index_lock_path.exists():
            # Held by another Git command, not by the author: retryable for the same reason as the pre-run check
            # above.
            raise AuthorSafetyError(
                "Git index became locked during authoring; restoration is unsafe",
                paths=[str(self.index_lock_path)],
                transient=True,
            )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".forge-index-",
            dir=str(self.index_path.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self.baseline_index_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, self.baseline_index_mode)
            os.replace(temporary, self.index_path)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise AuthorSafetyError("cannot restore the Git index exactly") from exc

        for rel in sorted(preserved_targets):
            desired = post_entries.get(rel, ())
            baseline = self.baseline_index_entries.get(rel, ())
            if desired == baseline:
                continue
            if any(stage != 0 for _mode, _oid, stage in desired):
                raise AuthorSafetyError(
                    f"cannot preserve conflicted target index state: {rel}",
                    paths=[rel],
                )
            _git_bytes(
                self.root,
                "update-index",
                "--force-remove",
                "--",
                rel,
            )
            if not desired:
                continue
            if len(desired) != 1:
                raise AuthorSafetyError(
                    f"cannot preserve target index state: {rel}",
                    paths=[rel],
                )
            mode, oid, _stage = desired[0]
            _git_bytes(
                self.root,
                "update-index",
                "--add",
                "--cacheinfo",
                mode,
                oid,
                rel,
            )

    def _restore_clean_tracked(self, rel: str) -> None:
        entries = self.baseline_index_entries.get(rel, ())
        if len(entries) != 1 or entries[0][2] != 0:
            raise AuthorSafetyError(
                f"cannot reconstruct tracked baseline path: {rel}",
                paths=[rel],
            )
        mode = entries[0][0]
        if mode == "160000":
            raise AuthorSafetyError(
                f"cannot safely restore changed submodule path: {rel}",
                paths=[rel],
            )
        path = self._path(rel)
        self._remove_path(rel)
        self._ensure_safe_parent(path)
        _git_bytes(
            self.root,
            "checkout-index",
            "--force",
            "--",
            rel,
        )

    def _allowed_path_is_unsafe(self, rel: str) -> bool:
        path = self._path(rel)
        # Replacing an allowlisted path or one of its parent directories with a symlink changes what that path
        # resolves to.
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return True
        state = _capture_path_state(path)
        return path.is_symlink() or resolved != path or state.kind not in {"absent", "file"}

    def _permits_new_relpath(self, rel: str) -> bool:
        """Whether one repo-relative path is a fused module the author may add."""
        path = Path(rel)
        entries = self.new_module_baselines.get(path.parent.as_posix())
        if entries is None or path.name in entries:
            return False
        return path.suffix == ".py" and _is_fused_module_name(path.name)

    def permits_new_path(self, value: str) -> bool:
        """Whether one filesystem path lies in the permitted new-module scope."""
        raw = Path(str(value)).expanduser()
        lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else self.cwd / raw)))
        try:
            relative = lexical.relative_to(self.root)
        except ValueError:
            return False
        rel = relative.as_posix()
        # Checked before the nomination gate: staging is the pipeline's own and exists whether or not this run
        # nominates a module directory.
        if _is_fusion_scratch_relpath(rel):
            return True
        if not self.new_module_baselines:
            return False
        return self._permits_new_relpath(rel)

    def _baseline_has_path_object(self, rel: str) -> bool:
        state = self.baseline_states.get(rel)
        if state is not None:
            return state.kind != "absent"
        return rel in self.baseline_index_entries

    def _minimal_restore_paths(self, violations: set[str]) -> set[str]:
        """Remove redundant parent/child paths according to baseline structure."""
        selected = set(violations)
        ordered = sorted(violations, key=lambda value: len(Path(value).parts))
        for index, parent in enumerate(ordered):
            parent_parts = Path(parent).parts
            for child in ordered[index + 1 :]:
                child_parts = Path(child).parts
                if child_parts[: len(parent_parts)] != parent_parts:
                    continue
                if self._baseline_has_path_object(child) and not (self._baseline_has_path_object(parent)):
                    selected.discard(parent)
                else:
                    selected.discard(child)
        return selected

    def _changed_paths(
        self,
        entries: dict[str, tuple[tuple[str, str, int], ...]],
        flags: dict[str, str],
        status: set[str],
    ) -> set[str]:
        """Git-visible paths whose index entry, index flag or worktree state differs from the baseline snapshot."""
        baseline_entries, baseline_flags = self.baseline_index_entries, self.baseline_index_flags
        changed = {
            rel
            for rel in baseline_entries.keys() | entries.keys()
            if baseline_entries.get(rel, ()) != entries.get(rel, ())
        }
        changed.update(rel for rel in baseline_flags.keys() | flags.keys() if baseline_flags.get(rel) != flags.get(rel))
        changed.update(status - self.baseline_states.keys())
        changed.update(
            rel for rel, state in self.baseline_states.items() if _capture_path_state(self._path(rel)) != state
        )
        return changed

    def enforce(self) -> _AuthorWorkspaceOutcome:
        """Restore out-of-scope deltas and report rejections plus new modules."""
        if self._head() != self.baseline_head or self._branch() != self.baseline_branch:
            raise AuthorSafetyError(
                "author changed Git HEAD or branch; automatic restoration is unsafe",
                paths=["<git-head>"],
            )

        post_entries = self._index_entries()
        post_flags = self._index_flags()
        changed = self._changed_paths(post_entries, post_flags, self._status_paths())
        # A new fused helper module inside the nominated scope is part of the authored fusion, so it is allowed to
        # survive.
        scratch = {rel for rel in changed if _is_fusion_scratch_relpath(rel)}
        created = {
            rel
            for rel in changed - self.target_relpaths - scratch
            if rel not in post_entries and self._permits_new_relpath(rel)
        }
        allowed = self.target_relpaths | created | scratch
        unsafe_allowed = {rel for rel in changed & allowed if self._allowed_path_is_unsafe(rel)}
        violations = (changed - allowed) | unsafe_allowed
        if not violations:
            return _AuthorWorkspaceOutcome(created=tuple(sorted(created)))

        preserved = allowed - unsafe_allowed
        unsafe_flags = {rel for rel in preserved if self.baseline_index_flags.get(rel) != post_flags.get(rel)}
        violations |= unsafe_flags
        preserved -= unsafe_flags
        # Paths the current transaction must not clobber while restoring others.
        self.preserved_relpaths = preserved

        self._restore_index(post_entries, preserved)
        restore_paths = self._minimal_restore_paths(violations)
        for rel in sorted(restore_paths, key=lambda value: len(Path(value).parts)):
            baseline = self.baseline_states.get(rel)
            if baseline is not None:
                self._restore_state(rel, baseline)
            elif rel in self.baseline_index_entries:
                self._restore_clean_tracked(rel)
            else:
                self._remove_path(rel)

        failed = self._changed_paths(self._index_entries(), self._index_flags(), self._status_paths()) - preserved
        if failed:
            raise AuthorSafetyError(
                "could not prove exact restoration of non-target paths",
                paths=sorted(failed),
            )
        return _AuthorWorkspaceOutcome(
            violations=tuple(sorted(violations)),
            created=tuple(sorted(created - violations)),
        )


_AUTHOR_SYSTEM_PROMPT = """\
You are the authoring stage of KernelForge forge-fuse. Implement and validate
the requested source-level fusion autonomously. Keep system instructions separate
from the user task. Modify only the exact target files supplied by the caller;
never modify tests, benchmark oracles, git state, or unrelated source. Create the
target validation harness only when the user prompt requests it. Use the requested
working directory and finish with the result contract specified in the user prompt.
"""

_AUTHOR_NEW_MODULE_CLAUSE = """\
You may additionally create NEW fused-kernel helper modules that a target file
imports, but only directly inside: {directories}. Each one must be a .py file whose
name marks it as fused — it must contain {markers} or start with {prefixes} (e.g.
qwen3_fused_ops.py) — must not reuse the name of a file that already exists in that
directory, and must not be added to the git index. Anything else you create is
reverted and the whole attempt is rejected with it.
"""


def _quoted_name_list(values: tuple[str, ...]) -> str:
    """Render name fragments as an ``a, b or c`` list for the prompt."""
    quoted = [f"`{value}`" for value in values]
    if len(quoted) < 2:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def _author_system_prompt(new_module_dirs: list[str]) -> str:
    """State exactly the write scope the workspace guard is going to accept."""
    if not new_module_dirs:
        return _AUTHOR_SYSTEM_PROMPT
    clause = _AUTHOR_NEW_MODULE_CLAUSE.format(
        directories=", ".join(new_module_dirs),
        markers=_quoted_name_list(_FUSED_MODULE_MARKERS),
        prefixes=_quoted_name_list(_FUSED_MODULE_PREFIXES),
    )
    return f"{_AUTHOR_SYSTEM_PROMPT}{clause}"


def _write_registered_author_log(
    log_path: str,
    progress: list[str],
    *,
    text: str = "",
    error: str = "",
    created: tuple[str, ...] = (),
) -> None:
    """Persist streamed progress plus the final Agent result."""
    lines = list(progress)
    if created:
        lines.append(f"created new fusion module(s): {', '.join(created)}")
    if text.strip():
        lines.append(text.strip())
    if error.strip():
        lines.append(f"error: {error.strip()}")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(
        ("\n".join(lines).strip() + "\n") if lines else "",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class AuthorAgent:
    """The agent that authors: its backend (or a factory for one), model, device, turn cap and per-attempt timeout."""

    backend: Any
    gpu: str = "0"
    model: Optional[str] = None
    max_turns: int = 100
    timeout_s: int = 7200


@dataclass(frozen=True)
class AuthorWorkspace:
    """Where an authoring session runs and logs, the files it may edit, and where it may add new modules."""

    workdir: str
    log_path: str
    target_files: Sequence[str] = ()
    new_module_dirs: Sequence[str] = ()


def _run_registered_author(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> int:
    """Run authoring, retrying only a transport that never delivered an answer."""
    timeout_s = agent.timeout_s
    attempts = max(1, int(env_setting("FORGE_FUSION_AUTHOR_ATTEMPTS", DEFAULT_AUTHOR_ATTEMPTS, cast=int)))
    base_delay = float(env_setting("FORGE_FUSION_LLM_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, cast=float))
    max_delay = float(env_setting("FORGE_FUSION_LLM_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, cast=float))
    # llm_failure's 1800s default is sized for the short calls discover.py makes; an author attempt
    # is allowed timeout_s, which defaults to 7200. Defaulting to it here would make the gate below
    # read `elapsed + delay + 7200 >= 1800` on the first failure, so the retry could never start.
    # The default budget is what the configured attempts can legitimately cost; an operator who
    # sets the deadline still gets it enforced.
    default_deadline = attempts * (float(timeout_s) + max_delay)
    deadline = float(env_setting("FORGE_LLM_RETRY_DEADLINE_SEC", default_deadline, cast=float))
    started_at = time.monotonic()
    for attempt in range(1, attempts + 1):
        rc, retryable = _run_registered_author_once(prompt, workspace, agent)
        if not retryable or attempt >= attempts:
            return rc
        delay = retry_delay(attempt, base_sec=base_delay, max_sec=max_delay)
        elapsed = time.monotonic() - started_at
        # A retry that cannot plausibly finish must not be started: each attempt spends up to the
        # per-attempt timeout, so the clock is checked against what the next one would cost.
        if deadline > 0 and elapsed + delay + timeout_s >= deadline:
            log.warning(
                "author transport failed on attempt %d/%d with %.0fs of the %.0fs retry budget spent; not retrying",
                attempt,
                attempts,
                elapsed,
                deadline,
            )
            return rc
        log.warning(
            "author attempt %d/%d failed on the transport; retrying in %.0fs",
            attempt,
            attempts,
            delay,
        )
        time.sleep(delay)
    return rc


def _record_author_log(log_path: str, progress: list[str], **fields: Any) -> None:
    """Write the author log, best-effort.

    A log this run could not write is not a reason to lose the verdict it was
    recording, so the failure is reported and the caller carries on.
    """
    try:
        _write_registered_author_log(log_path, progress, **fields)
    except OSError:
        log.warning("could not write registered author log %s", log_path)


def _safety_detail(exc: AuthorSafetyError) -> str:
    """The guard's complaint, with the paths it named."""
    detail = str(exc)
    return f"{detail}; paths={', '.join(exc.paths[:20])}" if exc.paths else detail


def _author_run_error_verdict(run_error: Exception, *, backend: Any, timeout_s: int) -> tuple[int, bool]:
    """The exit code for an agent run that raised, and whether another attempt is worth spending."""
    # Checked ahead of the timeout markers: a provider safety stop is final even if its message happens to mention
    # a clock, and retrying one is exactly the anti-pattern the session-resume allowlist already refuses.
    if is_agent_safety_error(run_error):
        log.error(
            "%s author was stopped by a provider safety guard: %s: %s",
            backend.name,
            type(run_error).__name__,
            run_error,
        )
        return AUTHOR_RC_SAFETY, False
    if is_agent_timeout_error(run_error):
        log.warning(
            "%s author timed out after %ss: %s: %s",
            backend.name,
            timeout_s,
            type(run_error).__name__,
            run_error,
        )
        # The attempt just spent the whole per-attempt budget; another would spend it again.
        return AUTHOR_RC_TIMEOUT, False
    log.error("%s author failed: %s: %s", backend.name, type(run_error).__name__, run_error)
    return AUTHOR_RC_FAILED, classify_llm_error(run_error) in RETRYABLE_KINDS


def _author_result_verdict(result: Any, *, backend: Any) -> tuple[int, bool]:
    """The exit code for an agent run that returned, and whether another attempt is worth spending."""
    if result.end_reason == "agent_stopped":
        return AUTHOR_RC_OK, False
    log.warning(
        "%s author ended without success (end_reason=%s subtype=%s)",
        backend.name,
        result.end_reason,
        result.subtype or "none",
    )
    # The backends flatten a transport failure the SDK swallowed into this end_reason rather than an exception, so
    # session_resume owns which of them is worth another attempt; a turn cap is the task's own answer.
    return AUTHOR_RC_FAILED, is_api_failure(result)


def _run_author_session(
    prompt: str, workdir: str, agent: AuthorAgent, guard: _AuthorWorkspaceGuard, progress: list[str]
) -> Any:
    """Run one agent session on the author's GPU, allowed to write exactly what the guard will accept."""
    session_timeout = max(1, int(agent.timeout_s))
    spec = AgentRunSpec(
        system_prompt=_author_system_prompt(guard.new_module_dirs),
        user_prompt=prompt,
        role="fusion author",
        cwd=workdir,
        model=agent.model,
        writable=True,
        timeout_sec=session_timeout,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=max(1, int(agent.max_turns)),
        ),
        target_files=guard.target_files,
        allow_dirty_targets=True,
        allow_untracked=True,
        # The author phase runs in a worktree a long campaign has already left dirty in ways it never touches, so the
        # backend has to judge this turn against the pre-run snapshot rather than against a clean HEAD.
        allow_dirty_baseline=True,
        # Forge's measurement protections, and nothing beyond them; the outer provider-neutral transaction enforces
        # the exact target allowlist.
        protected_path_rules=FORGE_PROTECTED_PATHS,
        protected_globs=[],
        hooks=target_file_hooks(guard.target_files, workdir, allows_new_path=guard.permits_new_path),
        progress_log=progress,
    )
    previous_gpu = os.environ.get("HIP_VISIBLE_DEVICES")
    os.environ["HIP_VISIBLE_DEVICES"] = agent.gpu
    try:
        return asyncio.run(asyncio.wait_for(agent.backend.run(spec), timeout=watchdog_timeout_sec(session_timeout)))
    finally:
        if previous_gpu is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = previous_gpu


def _run_registered_author_once(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> tuple[int, bool]:
    """One authoring run, with whether its failure was the transport's rather than the task's."""
    progress: list[str] = []
    requested_targets = list(dict.fromkeys(str(path) for path in workspace.target_files if str(path)))
    requested_module_dirs = list(dict.fromkeys(str(path) for path in workspace.new_module_dirs if str(path)))
    try:
        guard = _AuthorWorkspaceGuard(workspace.workdir, requested_targets, requested_module_dirs)
    except AuthorSafetyError as exc:
        detail = _safety_detail(exc)
        heading = (
            "author workspace could not be inspected before run"
            if exc.transient
            else "author workspace safety rejected before run"
        )
        _record_author_log(workspace.log_path, progress, error=f"{heading}: {detail}")
        log.error("%s: %s", heading, detail)
        # A workspace the guard could not read is the box's problem, not the transport's.
        return (AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY), False
    result = None
    run_error: BaseException | None = None
    try:
        result = _run_author_session(prompt, workspace.workdir, agent, guard, progress)
    except BaseException as exc:  # isolation boundary: one author attempt; restoration must always run
        log.warning("author agent run failed for %s", workspace.workdir, exc_info=True)
        run_error = exc
    text = str(getattr(result, "text", "") or "")

    def _with_run_error(reason: str) -> str:
        """Keep the session's own failure beside a verdict about the workspace."""
        if run_error is None:
            return reason
        return f"{reason}; the agent run also failed: {type(run_error).__name__}: {run_error}"

    def _rejected(reason: str, rc: int) -> tuple[int, bool]:
        """Record a workspace verdict and end the attempt; the retry loop never spends another one on it."""
        reason = _with_run_error(reason)
        _record_author_log(workspace.log_path, progress, text=text, error=reason)
        log.error("%s", reason)
        if run_error is not None and not isinstance(run_error, Exception):
            raise run_error
        return rc, False

    try:
        enforcement = guard.enforce()
    except AuthorSafetyError as exc:
        heading = (
            "author workspace restoration could not complete"
            if exc.transient
            else "author workspace safety restoration failed"
        )
        return _rejected(f"{heading}: {_safety_detail(exc)}", AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY)
    except OSError as exc:
        detail = f"{type(exc).__name__}: internal workspace guard failure"
        # The agent-facing log stays content-free (a guard defect is not something the author can act on), but the
        # operator needs the traceback to fix it.
        log.exception("author workspace safety restoration failed: %s", _with_run_error(detail))
        return _rejected(f"author workspace safety rejection: {detail}", AUTHOR_RC_SAFETY)

    if enforcement.violations:
        violations = enforcement.violations
        detail = ", ".join(violations[:20])
        if len(violations) > 20:
            detail += f", ... ({len(violations)} paths)"
        # Deterministic: the same guard, worktree and prompt reject the next attempt the same way, and the loop is
        # told so rather than spending one on it.
        return _rejected(f"author workspace rejected and restored non-target paths: {detail}", AUTHOR_RC_SAFETY)

    if enforcement.created:
        log.info(
            "author created new fusion module(s): %s",
            ", ".join(enforcement.created),
        )

    if run_error is not None:
        if not isinstance(run_error, Exception):
            raise run_error
        _record_author_log(
            workspace.log_path,
            progress,
            error=f"{type(run_error).__name__}: {run_error}",
            created=enforcement.created,
        )
        return _author_run_error_verdict(run_error, backend=agent.backend, timeout_s=agent.timeout_s)

    assert result is not None
    _record_author_log(workspace.log_path, progress, text=text, created=enforcement.created)
    return _author_result_verdict(result, backend=agent.backend)


def run_author(prompt: str, workspace: AuthorWorkspace, agent: AuthorAgent) -> int:
    """Drive the selected Agent backend and return the legacy process-style code."""
    backend = agent.backend
    if callable(backend) and not hasattr(backend, "run"):
        backend = backend()
    selected_model = (
        str(agent.model or "").strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    )
    log.info(
        "running %s Agent author (model=%s max_turns=%d workdir=%s)",
        backend.name,
        selected_model,
        agent.max_turns,
        workspace.workdir,
    )
    return _run_registered_author(prompt, workspace, dataclasses.replace(agent, backend=backend, model=selected_model))
