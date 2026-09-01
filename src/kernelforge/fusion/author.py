# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 3: author an env-gated fused kernel from a self-discovered recipe.

Turns a :class:`~kernelforge.fusion.models.Recipe` into an authoring prompt and drives a
registered Agent backend to write integration wiring or a fused kernel into the
framework source. Discovery may attach a semantically retrieved existing ROCm
operator; integration recipes must benchmark and wire that operator before
authoring a replacement. The historical bare ``claude`` helper remains only for
direct-call compatibility; the forge-fuse CLI always injects a registered
backend shared with discovery.

A provider-neutral transaction (:class:`_AuthorWorkspaceGuard`) wraps every run and
restores whatever the session changed outside its writable scope: the caller's exact
target files, plus new fused-kernel helper modules inside the directories the caller
nominates. The SDK edit hook calls the guard's own predicate, and the system prompt
is built from the guard's directory list and the same :mod:`emit` naming constants
the predicate matches on, so the agent is never rejected for obeying its
instructions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from kernelforge.agent_backends.base import (
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)

from .emit import _FUSED_MODULE_MARKERS, _FUSED_MODULE_PREFIXES, _is_fused_module_name
from .llm_failure import is_agent_safety_error, is_agent_timeout_error
from .harness_contract import harness_contract
from .validate import DEFAULT_TARGET_SPEEDUP
from kernelforge.llm.git import git

log = logging.getLogger("forge_fusion")

# Process-style author return codes. ``run_author`` has always answered with a
# single integer, and the fusion loop has to tell a deterministic workspace-safety
# rejection -- identical on every retry -- from a transient failure, so the class
# travels as a dedicated code rather than as a second return value.
#
# ``AUTHOR_RC_SAFETY`` is reserved for verdicts about the worktree's CONTENT: an
# ``enforce()`` violation, a target or module path that cannot be validated, a
# moved HEAD or branch, a provider safety stop. The guard failing at its own
# bookkeeping -- a Git query that timed out, an index lock another process held --
# reports ``AUTHOR_RC_FAILED``, because abandoning a recipe over that costs every
# remaining attempt for a condition the next attempt very likely will not see.
AUTHOR_RC_OK = 0
AUTHOR_RC_FAILED = 1
AUTHOR_RC_SAFETY = 3
AUTHOR_RC_TIMEOUT = 124


def proven_fusion_fewshot() -> str:
    """Few-shot block of serving-validated decode fusions (worked examples).

    Every fusion below was authored AND validated on the REAL sglang serving path
    (CUDA graph ON, MI325X/ROCm), so the author mimics patterns that survive
    production serving. A from-scratch kernel that passes a standalone microbench
    but ignores these (especially CUDA-graph safety) SIGQUIT-crashes the sglang
    decode loop — this has happened, so the rules below are mandatory.
    """
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
    """How to name the target GPU in a prompt.

    Hardcoding one chip here would tell the author to tune for hardware the run
    is not on: tile shapes, warp counts and intrinsics are all chosen per ISA,
    which is the same reason the knowledge base treats arch as a hard filter.
    An unknown arch says nothing rather than guessing.
    """
    arch = (gpu_arch or "").strip().lower()
    marketing = {"gfx950": "MI355X", "gfx942": "MI300X/MI325X"}.get(arch, "")
    if not arch:
        return "an AMD ROCm GPU"
    return f"AMD {marketing} ({arch})" if marketing else f"an AMD GPU ({arch})"


def _model_dir_block(model_path: str) -> str:
    """Name the model directory, because the alternative is that it gets searched for.

    An author that needs `config.json` and has not been told where the model lives
    reaches for `find / -name config.json`, and on a serving host `/` includes
    multi-terabyte network mounts: one such search ran 43 minutes and consumed the
    authoring attempt it was issued from.
    """
    if not model_path:
        return ""
    return (
        f"- Model directory (config.json, tokenizer, weights): {model_path}\n"
        "  Read the model's own files from there rather than searching for them: "
        "`/` on this host includes multi-terabyte network mounts, and a single "
        "`find /` can outlast the whole authoring budget.\n"
    )


def build_author_prompt(
    recipe: dict,
    *,
    framework: str,
    ab_hint: str,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    harness_path: str = "",
    gpu_arch: str = "",
    model_path: str = "",
) -> str:
    """Build the authoring prompt from a recipe dict (``Recipe.to_dict()``).

    Everything model-specific comes from the recipe fields, so this carries no
    per-model literals.
    """
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


def build_multi_author_prompt(
    recipes: list[dict],
    *,
    framework: str,
    ab_hint: str,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    harness_path: str = "",
    gpu_arch: str = "",
    model_path: str = "",
) -> str:
    """Prompt to author SEVERAL confirmed fusions in one pass (each env-gated).

    A model often has more than one launch-bound chain worth fusing (e.g. LFM2's
    residual+rmsnorm AND swiglu). Authoring them together lets the A/B measure the
    combined gain, matching how the fusions were originally validated.
    """
    if len(recipes) == 1:
        return build_author_prompt(
            recipes[0],
            framework=framework,
            ab_hint=ab_hint,
            target_speedup=target_speedup,
            harness_path=harness_path,
            gpu_arch=gpu_arch,
            model_path=model_path,
        )
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

{_model_dir_block(model_path)}
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
    """Report an author workspace state that cannot be safely accepted.

    ``transient`` separates the two things this class carries. Almost every
    instance is a verdict about the worktree's CONTENT -- a path outside the
    writable scope, a moved HEAD, a restoration that could not be proved -- which
    the same guard reaches identically on the next attempt, so the loop is right
    to abandon the recipe. A few are the guard failing at its own bookkeeping: a
    Git query that timed out, an index lock another process held for a
    millisecond. Those say nothing about the author and recover on their own, so
    they must reach the loop as retryable.
    """

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
        result = git(
            "-C",
            str(root),
            *args,
            input=input_data,
            check=False,
            text=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Pure I/O, and ``SubprocessError`` covers ``TimeoutExpired``: a 60s
        # ``git ls-files`` timeout against an NFS worktree under a concurrent
        # serving campaign is weather, not a verdict on what the author did.
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
        # Reading the worktree is the guard's own bookkeeping, not a verdict on
        # what the author did, and it recovers on its own -- same reason the Git
        # command and index-lock paths are marked retryable.
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
    """Whether a repo-relative path is forge-fusion's own staging directory.

    The validation harness is staged inside the worktree because the author
    sandbox is workspace-write and cannot reach outside it, so the author
    writing there is what the directory exists for rather than an out-of-scope
    edit. Nothing under it reaches the exported patch, and ``cli`` removes it
    once the turn ends.
    """
    parts = Path(rel).parts
    return len(parts) > 1 and parts[0] == FUSION_SCRATCH_DIRNAME


class _AuthorWorkspaceGuard:
    """Restore every Git-visible mutation outside the author's writable scope.

    The writable scope is the caller's exact target files plus, when the caller
    nominates directories in ``new_module_dirs``, new fused-kernel helper modules
    created directly inside them. That second part is not a convenience: a
    source-level fusion routinely lives in its own module that the target file
    imports, and export/teardown already expect it, so a guard that forbids it
    rejects the very output the pipeline asked for.
    """

    def __init__(
        self,
        workdir: str,
        target_files: list[str],
        new_module_dirs: Optional[list[str]] = None,
    ) -> None:
        cwd = Path(workdir).expanduser().resolve()
        root_result = _git_bytes(cwd, "rev-parse", "--show-toplevel")
        root_text = root_result.stdout.decode(errors="surrogateescape").strip()
        if not root_text:
            raise AuthorSafetyError("registered authoring requires a Git worktree")
        self.root = Path(root_text).resolve()
        self.cwd = cwd
        self.target_relpaths: set[str] = set()
        self.target_files: list[str] = []
        for value in target_files:
            if not value:
                continue
            raw = Path(value).expanduser()
            lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else cwd / raw)))
            try:
                lexical.relative_to(self.root)
            except ValueError as exc:
                raise AuthorSafetyError(
                    f"author target is outside the Git worktree: {lexical}",
                    paths=[str(lexical)],
                ) from exc
            resolved = lexical.resolve(strict=False)
            try:
                relative = resolved.relative_to(self.root)
            except ValueError as exc:
                raise AuthorSafetyError(
                    f"author target is outside the Git worktree: {lexical}",
                    paths=[str(lexical)],
                ) from exc
            if lexical.is_symlink() or resolved != lexical:
                raise AuthorSafetyError(
                    f"author target symlink/path escape is not allowed: {lexical}",
                    paths=[str(lexical)],
                )
            rel = relative.as_posix()
            self.target_relpaths.add(rel)
            self.target_files.append(str(resolved))

        self.new_module_dirs: list[str] = []
        # Name inventory per nominated directory. Membership is what makes an
        # existing framework module unwritable through the creation door: a file
        # such as ``fused_moe.py`` matches the fused-module marker but shipped with
        # the framework, and overwriting it is not creating a helper.
        self.new_module_baselines: dict[str, frozenset[str]] = {}
        for value in new_module_dirs or []:
            if not value:
                continue
            raw = Path(value).expanduser()
            lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else cwd / raw)))
            resolved = lexical.resolve(strict=False)
            try:
                relative = resolved.relative_to(self.root)
            except ValueError as exc:
                raise AuthorSafetyError(
                    f"author module directory is outside the Git worktree: {lexical}",
                    paths=[str(lexical)],
                ) from exc
            if lexical.is_symlink() or resolved != lexical:
                raise AuthorSafetyError(
                    f"author module directory symlink/path escape is not allowed: {lexical}",
                    paths=[str(lexical)],
                )
            try:
                entries = frozenset(os.listdir(resolved))
            except FileNotFoundError as exc:
                # An empty inventory is the most permissive scope there is -- every
                # name in it counts as absent -- and the prompt would then advertise
                # a directory the author cannot write into anyway.
                raise AuthorSafetyError(
                    f"author module directory does not exist: {lexical}",
                    paths=[str(lexical)],
                ) from exc
            except OSError as exc:
                # A missing directory above is a verdict: it is absent on every
                # attempt. Any other listdir failure is the guard failing to read,
                # which the next attempt very likely does not hit.
                raise AuthorSafetyError(
                    f"cannot inventory author module directory: {lexical}",
                    paths=[str(lexical)],
                    transient=True,
                ) from exc
            self.new_module_baselines[relative.as_posix()] = entries
            if str(resolved) not in self.new_module_dirs:
                self.new_module_dirs.append(str(resolved))

        # Paths the current transaction must not clobber while restoring others.
        # Starts as the targets and grows with the creations enforce() accepts.
        self.preserved_relpaths: set[str] = set(self.target_relpaths)

        self.baseline_head = self._head()
        self.baseline_branch = self._branch()
        self.baseline_index_entries = self._index_entries()
        self.baseline_index_flags = self._index_flags()
        self.index_path = self._index_path()
        self.index_lock_path = Path(f"{self.index_path}.lock")
        if self.index_lock_path.exists():
            # ``index.lock`` exists for milliseconds whenever anything else runs a
            # Git command in this worktree, so the next attempt very likely finds
            # it gone.
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
        self.baseline_status_paths = self._status_paths() | hidden_index_paths
        self.baseline_states: dict[str, _WorkspacePathState] = {}
        for rel in self.baseline_status_paths:
            state = _capture_path_state(self._path(rel))
            if state.kind in {"directory", "unsupported"}:
                raise AuthorSafetyError(
                    f"cannot safely snapshot Git-visible path type: {rel}",
                    paths=[rel],
                )
            self.baseline_states[rel] = state

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
            # Held by another Git command, not by the author: retryable for the
            # same reason as the pre-run check above.
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
        # Replacing an allowlisted path or one of its parent directories with a
        # symlink changes what that path resolves to.
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return True
        state = _capture_path_state(path)
        return path.is_symlink() or resolved != path or state.kind not in {"absent", "file"}

    def _permits_new_relpath(self, rel: str) -> bool:
        """Whether one repo-relative path is a fused module the author may add.

        Deliberately narrow on four axes: the ``*_fused*``/``*_fusion*`` naming
        convention :func:`emit._is_fused_module_name` already defines (no second
        rule to drift from), a ``.py`` module because that is the only shape the
        export path emits, a directory the caller nominated, and a name that was
        absent when the run started.
        """
        path = Path(rel)
        entries = self.new_module_baselines.get(path.parent.as_posix())
        if entries is None or path.name in entries:
            return False
        return path.suffix == ".py" and _is_fused_module_name(path.name)

    def permits_new_path(self, value: str) -> bool:
        """Whether one filesystem path lies in the permitted new-module scope.

        Lexical on purpose: this answers a tool argument before the file exists, so
        there is nothing to resolve, and resolving would follow a symlink the agent
        just created. ``enforce`` re-checks the materialized path.
        """
        raw = Path(str(value)).expanduser()
        lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else self.cwd / raw)))
        try:
            relative = lexical.relative_to(self.root)
        except ValueError:
            return False
        rel = relative.as_posix()
        # Checked before the nomination gate: staging is the pipeline's own and
        # exists whether or not this run nominates a module directory.
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

    def enforce(self) -> _AuthorWorkspaceOutcome:
        """Restore out-of-scope deltas and report rejections plus new modules."""
        if self._head() != self.baseline_head or self._branch() != self.baseline_branch:
            raise AuthorSafetyError(
                "author changed Git HEAD or branch; automatic restoration is unsafe",
                paths=["<git-head>"],
            )

        post_entries = self._index_entries()
        post_flags = self._index_flags()
        post_status = self._status_paths()
        index_paths = set(self.baseline_index_entries) | set(post_entries)
        flag_paths = set(self.baseline_index_flags) | set(post_flags)
        index_changed = {
            rel for rel in index_paths if self.baseline_index_entries.get(rel, ()) != post_entries.get(rel, ())
        }
        flag_changed = {rel for rel in flag_paths if self.baseline_index_flags.get(rel) != post_flags.get(rel)}
        candidates = self.baseline_status_paths | post_status | index_changed | flag_changed
        worktree_changed: set[str] = set()
        for rel in candidates:
            baseline = self.baseline_states.get(rel)
            if baseline is not None:
                if _capture_path_state(self._path(rel)) != baseline:
                    worktree_changed.add(rel)
            elif rel in post_status:
                worktree_changed.add(rel)

        changed = worktree_changed | index_changed | flag_changed
        # A new fused helper module inside the nominated scope is part of the
        # authored fusion, so it is allowed to survive. Staging it is not: the
        # exported patch reaches an untracked new module through
        # ``git diff --no-index``, and an indexed one would silently drop out of the
        # handoff, so an index entry keeps the creation a rejection.
        # Staging is the pipeline's own scratch: it must survive the transaction
        # without being restored as a foreign write, and it must stay out of
        # ``created`` so it is never reported or handed off as an authored module.
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
        if flag_changed & preserved:
            unsafe_flags = sorted(flag_changed & preserved)
            violations.update(unsafe_flags)
            preserved.difference_update(unsafe_flags)
        self.preserved_relpaths = set(preserved)

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

        final_entries = self._index_entries()
        final_flags = self._index_flags()
        final_status = self._status_paths()
        verify_paths = (
            self.baseline_status_paths | final_status | set(self.baseline_index_entries) | set(final_entries)
        ) - preserved
        failed: set[str] = set()
        for rel in verify_paths:
            if self.baseline_index_entries.get(rel, ()) != final_entries.get(
                rel,
                (),
            ):
                failed.add(rel)
                continue
            if self.baseline_index_flags.get(rel) != final_flags.get(rel):
                failed.add(rel)
                continue
            baseline = self.baseline_states.get(rel)
            if baseline is not None:
                if _capture_path_state(self._path(rel)) != baseline:
                    failed.add(rel)
            elif (rel in self.baseline_status_paths) != (rel in final_status):
                failed.add(rel)
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
    """State exactly the write scope the workspace guard is going to accept.

    Generated from the guard's own scope rather than written alongside it: an
    author told to touch nothing but the target files, and then rejected for the
    helper module its target imports, has burned a whole attempt learning a rule
    the prompt could have stated. Both halves of the scope come from the guard --
    the directories from the caller's nomination, the naming rule from the same
    :mod:`emit` constants :func:`emit._is_fused_module_name` matches on -- so
    adding a marker cannot leave the prompt describing the previous rule.
    """
    if not new_module_dirs:
        return _AUTHOR_SYSTEM_PROMPT
    clause = _AUTHOR_NEW_MODULE_CLAUSE.format(
        directories=", ".join(new_module_dirs),
        markers=_quoted_name_list(_FUSED_MODULE_MARKERS),
        prefixes=_quoted_name_list(_FUSED_MODULE_PREFIXES),
    )
    return f"{_AUTHOR_SYSTEM_PROMPT}{clause}"


def _target_file_hooks(
    target_files: list[str],
    workdir: str,
    allows_new_path: Optional[Callable[[str], bool]] = None,
) -> AgentHooks | None:
    """Deny direct SDK edit tools outside the caller's writable path set.

    ``allows_new_path`` is the workspace guard's own predicate for a permitted new
    fused module, so this hook never blocks a path the transaction would keep --
    a hook stricter than the guard makes the guard's allowance unreachable.
    """
    if not target_files:
        return None
    root = Path(workdir).resolve()
    targets = {(Path(value) if Path(value).is_absolute() else root / value).resolve() for value in target_files}

    async def _deny_non_target(input_data: dict, _tool_use_id: Any, _context: Any) -> dict:
        tool_input = input_data.get("tool_input") or {}
        raw_path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path") or ""
        if raw_path:
            candidate = Path(str(raw_path)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.resolve() in targets:
                return {}
            if allows_new_path is not None and allows_new_path(str(candidate)):
                return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Authoring may edit only the exact target files supplied by "
                    "forge-fuse, plus any new fused-kernel module the run "
                    "explicitly permits; tests, harness oracles, and unrelated "
                    "source are protected."
                ),
            }
        }

    return AgentHooks(
        pre_tool_use=[
            AgentHook(
                matcher="Edit|Write|MultiEdit|NotebookEdit",
                callback=_deny_non_target,
            )
        ]
    )


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


def _run_registered_author(
    backend: Any,
    prompt: str,
    *,
    workdir: str,
    log_path: str,
    gpu: str,
    model: str,
    max_turns: int,
    timeout_s: int,
    target_files: list[str],
    new_module_dirs: list[str],
) -> int:
    """Run authoring through one already-created registered Agent backend."""
    progress: list[str] = []
    requested_targets = list(dict.fromkeys(str(path) for path in target_files if str(path)))
    requested_module_dirs = list(dict.fromkeys(str(path) for path in new_module_dirs if str(path)))
    try:
        guard = _AuthorWorkspaceGuard(workdir, requested_targets, requested_module_dirs)
    except AuthorSafetyError as exc:
        detail = str(exc)
        if exc.paths:
            detail = f"{detail}; paths={', '.join(exc.paths[:20])}"
        heading = (
            "author workspace could not be inspected before run"
            if exc.transient
            else "author workspace safety rejected before run"
        )
        try:
            _write_registered_author_log(log_path, progress, error=f"{heading}: {detail}")
        except OSError:
            log.warning("could not write registered author log %s", log_path)
        log.error("%s: %s", heading, detail)
        return AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY
    targets = guard.target_files
    spec = AgentRunSpec(
        system_prompt=_author_system_prompt(guard.new_module_dirs),
        user_prompt=prompt,
        cwd=workdir,
        model=model,
        writable=True,
        timeout_sec=max(1, int(timeout_s)),
        reasoning_effort="max",
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=max(1, int(max_turns)),
        ),
        target_files=targets,
        allow_dirty_targets=True,
        allow_untracked=True,
        # The author phase runs in a worktree a long campaign has already left
        # dirty in ways it never touches, so the backend has to judge this turn
        # against the pre-run snapshot rather than against a clean HEAD.
        allow_dirty_baseline=True,
        # Keep the backend's built-in measurement protections; the outer
        # provider-neutral transaction enforces the exact target allowlist.
        protected_globs=[],
        hooks=_target_file_hooks(targets, workdir, allows_new_path=guard.permits_new_path),
        progress_log=progress,
    )

    async def _run() -> Any:
        return await asyncio.wait_for(
            backend.run(spec),
            timeout=watchdog_timeout_sec(max(1, int(timeout_s))),
        )

    previous_gpu = os.environ.get("HIP_VISIBLE_DEVICES")
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    result = None
    run_error: BaseException | None = None
    try:
        result = asyncio.run(_run())
    except BaseException as exc:  # noqa: BLE001 - safety restoration must always run
        run_error = exc
    finally:
        if previous_gpu is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = previous_gpu

    def _with_run_error(reason: str) -> str:
        """Keep the session's own failure beside a verdict about the workspace.

        ``enforce()`` is judged before ``run_error`` is examined, and a rejected
        turn that also ran out of clock returns from one of the branches below --
        so without this the operator sees the violation and no sign the session
        never finished.
        """
        if run_error is None:
            return reason
        return f"{reason}; the agent run also failed: {type(run_error).__name__}: {run_error}"

    try:
        enforcement = guard.enforce()
    except AuthorSafetyError as exc:
        detail = str(exc)
        if exc.paths:
            detail = f"{detail}; paths={', '.join(exc.paths[:20])}"
        heading = (
            "author workspace restoration could not complete"
            if exc.transient
            else "author workspace safety restoration failed"
        )
        reason = _with_run_error(f"{heading}: {detail}")
        try:
            _write_registered_author_log(
                log_path,
                progress,
                text=str(getattr(result, "text", "") or ""),
                error=reason,
            )
        except OSError:
            log.warning("could not write registered author log %s", log_path)
        log.error("%s", reason)
        return AUTHOR_RC_FAILED if exc.transient else AUTHOR_RC_SAFETY
    except Exception as exc:  # noqa: BLE001 - fail closed on guard defects
        detail = f"{type(exc).__name__}: internal workspace guard failure"
        try:
            _write_registered_author_log(
                log_path,
                progress,
                text=str(getattr(result, "text", "") or ""),
                error=_with_run_error(f"author workspace safety rejection: {detail}"),
            )
        except OSError:
            log.warning("could not write registered author log %s", log_path)
        # The agent-facing log stays content-free (a guard defect is not something
        # the author can act on), but the operator needs the traceback to fix it.
        log.exception("author workspace safety restoration failed: %s", _with_run_error(detail))
        return AUTHOR_RC_SAFETY

    if enforcement.violations:
        violations = enforcement.violations
        detail = ", ".join(violations[:20])
        if len(violations) > 20:
            detail += f", ... ({len(violations)} paths)"
        reason = _with_run_error(f"author workspace rejected and restored non-target paths: {detail}")
        try:
            _write_registered_author_log(
                log_path,
                progress,
                text=str(getattr(result, "text", "") or ""),
                error=reason,
            )
        except OSError:
            log.warning("could not write registered author log %s", log_path)
        log.error("%s", reason)
        # Deterministic: the same guard, worktree and prompt reject the next attempt
        # the same way, and the loop is told so rather than spending one on it.
        return AUTHOR_RC_SAFETY

    if enforcement.created:
        log.info(
            "author created new fusion module(s): %s",
            ", ".join(enforcement.created),
        )

    if run_error is not None:
        if not isinstance(run_error, Exception):
            raise run_error
        try:
            _write_registered_author_log(
                log_path,
                progress,
                error=f"{type(run_error).__name__}: {run_error}",
                created=enforcement.created,
            )
        except OSError:
            log.warning("could not write registered author log %s", log_path)
        # Checked ahead of the timeout markers: a provider safety stop is final even
        # if its message happens to mention a clock, and retrying one is exactly the
        # anti-pattern the session-resume allowlist already refuses. Safe to keep
        # first because the classifier now requires the provider's explicit
        # rejection marker, so a rollback that merely failed on the way out of a
        # timeout no longer reaches this branch at all.
        if is_agent_safety_error(run_error):
            log.error(
                "%s author was stopped by a provider safety guard: %s: %s",
                backend.name,
                type(run_error).__name__,
                run_error,
            )
            return AUTHOR_RC_SAFETY
        if is_agent_timeout_error(run_error):
            log.warning(
                "%s author timed out after %ss: %s: %s",
                backend.name,
                timeout_s,
                type(run_error).__name__,
                run_error,
            )
            return AUTHOR_RC_TIMEOUT
        log.error(
            "%s author failed: %s: %s",
            backend.name,
            type(run_error).__name__,
            run_error,
        )
        return AUTHOR_RC_FAILED

    assert result is not None
    final_text = str(getattr(result, "text", "") or "")
    try:
        _write_registered_author_log(
            log_path,
            progress,
            text=final_text,
            created=enforcement.created,
        )
    except OSError:
        log.warning("could not write registered author log %s", log_path)
    end_reason = str(getattr(result, "end_reason", "agent_stopped") or "agent_stopped")
    subtype = str(getattr(result, "subtype", "") or "")
    ok = end_reason == "agent_stopped" and subtype in {"", "success"}
    if not ok:
        log.warning(
            "%s author ended without success (end_reason=%s subtype=%s)",
            backend.name,
            end_reason,
            subtype or "none",
        )
    return AUTHOR_RC_OK if ok else AUTHOR_RC_FAILED


def run_author(
    prompt: str,
    *,
    workdir: str,
    log_path: str,
    gpu: str = "0",
    model: Optional[str] = None,
    max_turns: int = 100,
    timeout_s: int = 7200,
    backend: Any,
    target_files: Optional[list[str]] = None,
    new_module_dirs: Optional[list[str]] = None,
) -> int:
    """Drive the selected Agent backend and return the legacy process-style code.

    Args:
        prompt: The authoring prompt (from :func:`build_author_prompt`).
        workdir: Working dir (the framework repo root, e.g. the sglang checkout).
        log_path: File to capture the agent's stdout/stderr.
        backend: Registered backend reused from discovery, or a zero-argument
            factory returning one.
        target_files: Exact editable source/harness files for the workspace guard.
        new_module_dirs: Directories in which the author may create NEW fused
            helper modules. Pass the directories the export path scans, so a module
            the guard keeps is a module the emitted patch carries; omitting them
            forbids creation entirely. Each one must exist -- a nominated directory
            that does not is rejected rather than treated as an empty (and
            therefore maximally permissive) name inventory.

    Returns:
        One of ``AUTHOR_RC_OK``, ``AUTHOR_RC_FAILED`` (transient, including a
        workspace guard that could not complete its own bookkeeping),
        ``AUTHOR_RC_SAFETY`` (a verdict about the worktree's content or a provider
        safety stop, identical on every retry), or ``AUTHOR_RC_TIMEOUT``. Callers
        driving a retry loop must distinguish ``AUTHOR_RC_SAFETY``; anything else
        may be attempted again.
    """
    if callable(backend) and not hasattr(backend, "run"):
        backend = backend()
    selected_model = str(model or "").strip() or str(getattr(getattr(backend, "runtime", None), "model", "")).strip()
    log.info(
        "running %s Agent author (model=%s max_turns=%d workdir=%s)",
        backend.name,
        selected_model,
        max_turns,
        workdir,
    )
    return _run_registered_author(
        backend,
        prompt,
        workdir=workdir,
        log_path=log_path,
        gpu=gpu,
        model=selected_model,
        max_turns=max_turns,
        timeout_s=timeout_s,
        target_files=list(target_files or []),
        new_module_dirs=list(new_module_dirs or []),
    )
