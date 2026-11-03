"""EvolveOrchestrator — drives parallel MAP-Elites island agents for evolve-asm.

Each kernel task gets its own orchestrator instance.  Islands run as async
Claude agent SDK calls, one per GPU.  Results are parsed, gated, committed to
git, and optionally deployed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pack-relative import setup — must happen before intra-pack imports
# ---------------------------------------------------------------------------
_PACK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_PACK_DIR.parent.parent))

from packs.evolve_asm.policy.island_db import IslandDB, Candidate          # noqa: E402
from packs.evolve_asm.validator.gate import (                               # noqa: E402
    ROOFLINE_TARGET,
    GateDecision,
    measure_noise_floor,
    run_gate,
)
from packs.evolve_asm.validator.roofline_tracker import (                   # noqa: E402
    load_trajectory,
    update_tracker,
)
from packs.evolve_asm.policy.task_loader import load_task                   # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = ["native", "peer", "asm", "crosskernel", "algorithmic"]

ESCALATE_AT: Dict[str, int] = {
    "native":      3,
    "peer":        5,
    "asm":         5,
    "crosskernel": 5,
    "algorithmic": 5,
}

_EVOLVE_REPO = Path.home()
_STOP_FILE   = Path("/tmp/evolve_stop")

log = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Knowledge-base helper
# ---------------------------------------------------------------------------

def _kb_search(kb_path: str, query: str, top: int = 5) -> str:
    """Run kb_search.py and return the top-*top* results as a formatted string."""
    kb_script = Path(kb_path).parent / "kb_search.py"
    if not kb_script.exists():
        # Fallback: try a sibling scripts directory
        kb_script = Path(kb_path) / "kb_search.py"
    if not kb_script.exists():
        return "(kb_search not found — skipping knowledge-base lookup)"

    try:
        result = subprocess.run(
            [sys.executable, str(kb_script), "--kb", kb_path,
             "--query", query, "--top", str(top)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return f"(kb_search returned no results for: {query!r})"
    except subprocess.TimeoutExpired:
        return "(kb_search timed out)"
    except Exception as exc:  # noqa: BLE001
        return f"(kb_search error: {exc})"


# ---------------------------------------------------------------------------
# Island prompt builder
# ---------------------------------------------------------------------------

def _island_prompt(
    config: Dict[str, Any],
    island: int,
    tier: str,
    generation: int,
    parent: Optional[Candidate],
    kb_snippet: str,
    noise_floor_us: float,
    run_dir: Path,
) -> str:
    task_id   = config["task_id"]
    op_class  = config["op_class"]
    ref_co    = config["ref_co"]
    bench_cmd = config["bench_cmd"]
    snr_cmd   = config["snr_cmd"]
    snr_thresh = config["snr_threshold"]
    workspace = config["workspace"]
    kernel_lang = config["kernel_lang"]

    parent_info = ""
    if parent is not None:
        parent_info = f"""
## Parent candidate
- ID:           {parent.id}
- Latency (µs): {parent.weakest_us:.3f}
- Roofline:     {parent.roofline_frac:.3%}
- Tier:         {parent.tier}
- Label:        {parent.label}
- sig:          vgpr={parent.sig.get('vgpr', '?')}  barriers={parent.sig.get('barriers', '?')}  spills={parent.sig.get('spills', '?')}  mfma={parent.sig.get('mfma', '?')}
- Source:       {parent.s_path}
"""

    tier_instructions: Dict[str, str] = {
        "native": (
            "Sweep ALL compile parameters: tile_m/n/k, waves_per_eu, k_wave, b_nt, "
            "xcd_swizzle, async_copy, gate_mode, persist_m, cu_num_mul. Try every "
            "combination that changes the structural signature (barriers/VGPR). "
            "Do not stop at the first win — exhaust the space."
        ),
        "peer": (
            "Cross-compile to the best language for this op_class. Extract the .amdgcn "
            "binary and set it as the new ASM baseline. Compare structural signatures "
            "(barriers, VGPR, MFMA count) to understand what changed."
        ),
        "asm": """HAND-TUNED AMDGCN ASM — be AGGRESSIVE. Do NOT just remove s_nop 0.
Attack every axis:

1. VGPR REDUCTION (biggest lever — each wave needs fewer regs → more waves/SIMD)
   - Profile with rocprofv3 SQ_INSTS_VALU, count live VGPRs in disassembly
   - Reorder instructions to shorten live ranges (use a reg sooner, free it sooner)
   - Replace v_mov with alias where possible
   - Use AGPR as scratch if AGPR count < VGPR count (CDNA4: 512 AGPR available)
   - Target: if VGPR=512 → try to get to <384 (3 waves) or <256 (4 waves/SIMD)

2. s_waitcnt TIGHTENING (reduce serialization, expose ILP)
   - Map every s_waitcnt to what it's waiting for (vmcnt vs lgkmcnt vs expcnt)
   - If vmcnt(N) waits for N loads but only M<N are actually needed before next use,
     change to vmcnt(N-M) — this is often safe and hides latency
   - Split long waitcnt chains: use partial waits to allow earlier instructions to proceed
   - Check if any vmcnt(0) can become vmcnt(1) or vmcnt(2)

3. MFMA SCHEDULING (hide 64-cycle MFMA latency)
   - MFMA on gfx950 has 64-cycle latency before accumulator is ready
   - Issue buffer_load / ds_read in the MFMA shadow: move loads BEFORE the MFMA
     that produces data they don't need yet
   - If there are consecutive MFMAs, interleave with address computation
   - Check s_nop count between MFMA and next use of accumulator: should be
     exactly what ISA requires (check METHODOLOGY.md), no more

4. LDS OPTIMIZATION
   - Find bank conflicts: if two threads in the same wave read same LDS bank → stall
   - Add padding (e.g., LDS stride +1 element) to eliminate bank conflicts
   - Use ds_read_b128 instead of 4x ds_read_b32 where alignment allows
   - Use ds_read2_b32/b64 for strided access patterns

5. SOFTWARE PREFETCHING (hide HBM latency, critical for BW-bound kernels)
   - Issue buffer_load N iterations ahead of where the data is consumed
   - On gfx950, HBM latency is ~300-500 cycles — prefetch enough iterations ahead
   - Use s_waitcnt to enforce ordering but allow OOO load execution
   - Double-buffer: load tile A[k+1] while computing with tile A[k]

6. PACKED / WIDER INSTRUCTIONS
   - Replace scalar address math with packed ops where possible
   - Use v_pk_fma_f16/v_pk_mul_f16 for pairs of FP16 values
   - Use buffer_load_b128 (4 dwords) instead of buffer_load_b32 x4
   - Check if v_dot2c_f32_f16 can replace v_fma sequences

7. s_setprio / WAVE SCHEDULING
   - s_setprio 3 at start of hot loop: gives this wave higher CU priority
   - s_setprio 0 in epilog: release priority for other waves

8. BARRIER ELIMINATION
   - Every s_barrier serializes all waves in a workgroup — expensive
   - Check if producer-consumer pairs can use a lighter fence instead
   - If some barriers guard data that's always ready, they may be removable
   - Use lgkmcnt instead of s_barrier when only LDS sync is needed

9. OPCODES — use the fastest instruction for each operation
   - v_fma_f32 is one instruction (fused), not v_mul + v_add
   - v_mad_f32 for multiply-add without fusing (when FMA precision not needed)
   - v_rcp_f32 costs 4 cycles — see if reciprocal can be precomputed or approximated
   - v_sqrt_f32: only use if needed, not a cheap instruction
   - Use v_cndmask_b32 instead of branch+mov sequences

THINK BIG — NOP REMOVAL IS NOT ENOUGH:
   Removing s_nop 0 is 1-2% gain at best. The real gains are:

   A. VGPR REDUCTION: if VGPR=512, reducing to <256 unlocks 4 waves/SIMD (2x occupancy).
      This alone can be 30-50% speedup. Reorder ALL instruction sequences to shorten live
      ranges. Rename registers to free them earlier. This takes real work.

   B. SOFTWARE PIPELINING: Issue buffer_load for tile K+1 BEFORE computing tile K.
      On gfx950 HBM latency is 300-500 cycles. A properly pipelined kernel hides
      this entirely. This is a structural rewrite, not a tweak.

   C. MFMA PIPELINE RESTRUCTURING: Pack MFMAs tighter, eliminate dead cycles in the
      accumulator pipeline. The 64-cycle MFMA latency window is where loads should go.

   D. LDS BANK CONFLICT ELIMINATION: Add stride padding, switch to ds_read_b128.
      Bank conflicts serialize waves — eliminating them is 10-30% gain.

   E. LOOP UNROLLING + PREFETCH DEPTH: Manually unroll the K-loop 2x or 4x to expose
      more ILP. Double-buffer: keep two tiles in flight at once.

   NOP REMOVAL, s_setprio, and waitcnt loosening are SMALL wins (<3% each).
   Do those AFTER you attempt A-E. Focus on structural changes first.
   A 30-min session that reduces VGPR from 512 to 200 is worth 10 sessions of NOP removal.

BENCH PATH — USE KERNEL-ONLY TIMING (not full pipeline):
   The default bench measures _flydsl_stage2_wrapper which on some nodes uses
   persist=True kernel that runs 170µs instead of 8-9µs. Always pass the known
   fast baseline explicitly when benching moe2:

   docker exec CONTAINER HIP_VISIBLE_DEVICES=GPU \\
     python3 /arbor/packs/evolve-asm/validator/bench_dsv41flash_moe.py \\
     --kernel moe2 \\
     --kernel-name flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic \\
     --M 32 --N 5120 --K 2304 --n-experts 384 --top-k 6 --n-bench 500

   The 8.445µs baseline uses this kernel. Your ASM edits should start from this .co.
   If you see 170µs, you are benchmarking the wrong kernel — use --kernel-name above.

CRITICAL KNOWN BLOCKER — READ BEFORE STARTING:
   The `llvm-objdump -d` disassembly uses raw branch offsets (e.g. s_cbranch_vccnz 10790)
   NOT symbolic labels. If you remove/add instructions, branch offsets break → illegal instruction.

   SOLUTION — get a labeled .s via one of these methods:
   A) FLYDSL_DUMP_IR: set FLYDSL_DUMP_IR=1 before compile, then find the .mlir pipeline
      and run the final lowering step manually with llc --emit-labels
   B) Use llvm-mc with --position-independent to emit relocatable code, then link
   C) Work ONLY in the instruction body between two adjacent branch targets (safe zone)
   D) Replace NOPs with same-size instructions (1:1 substitution) — no offsets change

   DO NOT attempt NOP deletion using raw objdump output — it will fail on branches.
   DO use method D (1:1 substitution) as the safest first approach.

MANDATORY FIRST STEP — PROFILE BEFORE TOUCHING ANYTHING:
   Run rocprofv3 with the full PMC counter set and read EVERY number:

   docker exec CONTAINER rocprofv3 \\
     --pmc SQ_INSTS_VMEM SQ_WAIT_INST_ANY SQ_INSTS_LDS SQ_ACTIVE_INST_ANY \\
           TCP_GATE_EN1 SQ_INSTS_VALU SQ_INSTS_SALU SQ_WAVES \\
           TCP_TOTAL_CACHE_ATCS TCP_TCP_LATENCY SQ_INSTS_EXP \\
     -- python3 BENCH_SCRIPT --n-bench 50   (use the bench_cmd above, reduce n-bench)

   Then compute and interpret:
     WAIT/VMEM ratio  = SQ_WAIT_INST_ANY / SQ_INSTS_VMEM
       > 10×  → memory latency stall — need prefetching or waitcnt loosening
     LDS/VMEM ratio   = SQ_INSTS_LDS / SQ_INSTS_VMEM
       > 5×   → LDS contention — bank conflicts or too many LDS ops
     VALU/VMEM ratio  = SQ_INSTS_VALU / SQ_INSTS_VMEM
       < 2×   → compute-starved, VMEM dominates — reorder to hide latency
     ACTIVE/WAIT      = SQ_ACTIVE_INST_ANY / SQ_WAIT_INST_ANY
       < 2×   → nearly all time is waiting — issue more loads earlier
     TCP_LATENCY      → L2 cache latency in cycles (high = L2 miss → HBM)
     TCP_GATE_EN1     → L2 cache hit rate proxy

   THEN disassemble and annotate every instruction with its cost:
   docker exec {container} llvm-objdump -d --mcpu=gfx950 {ref_co} > /tmp/kernel.s

   Count and report BEFORE any edit:
   - Total instructions in hot loop
   - s_waitcnt count and vmcnt/lgkmcnt distribution
   - s_barrier count and placement
   - MFMA count and spacing (cycles between consecutive MFMAs)
   - buffer_load count and prefetch depth
   - VGPR register numbering (find highest VGPR used = register pressure)
   - Any spill instructions (scratch_load/store)

   ONLY after understanding the profile, pick the highest-leverage edit.
   Make ONE change per iteration. Bench. Report.

MINIMUM TIME: You must run for AT LEAST 60 MINUTES before stopping. Check the wall clock.
If you have been running less than 60 minutes, you CANNOT write island_stopped.json.
You CANNOT declare "parameter space exhausted" or "at ceiling" before 60 minutes.

MINIMUM ITERATIONS: Run at least 20 distinct ASM edits before stopping. Each edit must:
  - Profile with rocprofv3 PMC first if not already done this session
  - Target a different axis than the previous edit (rotate through the 9 axes above)
  - Bench with n_bench=200 minimum
  - Report sig_vgpr/barriers/spills/mfma

NULL STREAK RULE: You may only write island_stopped.json after BOTH:
  1. null_streak >= 15 (15 consecutive attempts with <0.5% gain)
  2. At least 60 minutes have elapsed since session start

If FlyDSL exhausted → move to ASM. If one ASM axis is stuck → try a different axis.
If blocked on branch offsets → use 1:1 instruction substitution (same size, no offset change).
If GPU throttling → wait 60s, check sclk, retry before declaring failure.

NEVER say "at ceiling". NEVER exit early. The ceiling is 85% MAF roofline, proven by rocprofv3.""",
        "crosskernel": (
            "Read campaign_archive/ for techniques that changed structural signatures "
            "on OTHER kernels. Apply the most promising to THIS kernel's .s. "
            "Focus on: barrier patterns, LDS padding, prefetch depth, VGPR layout."
        ),
        "algorithmic": (
            "Fundamentally different algorithm: persistent kernel, fused stages, "
            "different reduction order, MFMA for outer product, XCD locality "
            "reorder. Implement in peer language → compile → extract .amdgcn → "
            "set as new ASM baseline, then apply ASM tier optimizations."
        ),
    }
    tier_hint = tier_instructions.get(tier, "")

    return f"""# Island {island} — evolve-asm agent
## Task
- task_id:    {task_id}
- op_class:   {op_class}
- tier:       {tier}  (generation {generation})
- GPU:        {config['gpus'][island % len(config['gpus'])]}
- workspace:  {workspace}
- kernel_lang:{kernel_lang}
- run_dir:    {run_dir}

## Objectives
1. Produce a kernel variant that is FASTER than the current best.
2. The variant must pass the SNR gate (threshold {snr_thresh} dB).
3. Target roofline fraction >= {ROOFLINE_TARGET:.0%}.
4. Noise floor measured at {noise_floor_us:.3f} µs — differences smaller than
   this are noise.

## Tier guidance — {tier}
{tier_hint}

{parent_info}

## Reference kernel
{ref_co if ref_co else '(none — start from scratch or KB snippets)'}

## Benchmark / SNR commands
```
{bench_cmd}
{snr_cmd}
```

## Knowledge-base snippets
{kb_snippet}

## Required output format (emit EXACTLY this block, filled in):
```
RESULT
gpu_us=<float>
sig_vgpr=<int>
sig_barriers=<int>
sig_spills=<int>
sig_mfma=<int>
co_path=<absolute path to .co>
s_path=<absolute path to .s>
label=<one-line description>
END_RESULT
```

Write the new kernel, compile it, bench it, emit the RESULT block.
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EvolveOrchestrator:
    """Drive MAP-Elites island evolution for a single kernel task."""

    def __init__(self, config: Dict[str, Any], run_dir: Path) -> None:
        self.config  = config
        self.run_dir = run_dir
        self.task_id = config["task_id"]

        self.db: IslandDB
        self.noise_floor_us: float = 0.0
        self._tier_keeps: Dict[int, int] = {i: 0 for i in range(config["n_islands"])}
        self._null_counts: Dict[int, int] = {i: 0 for i in range(config["n_islands"])}
        self._stop: bool = False

        # Load tier + generation from orch_state.json if it exists (survives restarts)
        orch_state_path = self.run_dir / "orch_state.json"
        saved_tier = "native"
        saved_gen  = 0
        if orch_state_path.exists():
            try:
                _s = json.loads(orch_state_path.read_text())
                saved_tier = _s.get("tier", "native")
                saved_gen  = _s.get("generation", 0)
                if saved_tier not in TIERS:
                    saved_tier = "native"
                log.info("[%s] Resuming from orch_state: tier=%s gen=%d",
                         self.task_id, saved_tier, saved_gen)
            except Exception:
                pass
        self._current_tier: List[str] = [saved_tier] * config["n_islands"]
        self._generation: List[int]   = [saved_gen]  * config["n_islands"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Initialise DB, measure noise floor, set up logging."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.run_dir / f"{self.task_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        log.setLevel(logging.DEBUG)

        db_dir = self.run_dir / "db"
        self.db = IslandDB(
            num_islands=self.config["n_islands"],
            db_path=db_dir,
        )

        log.info("[%s] Measuring noise floor …", self.task_id)
        try:
            self.noise_floor_us = await asyncio.to_thread(
                measure_noise_floor,
                self.config["container"],
                self.config["bench_cmd"],
                n_runs=self.config.get("n_bench_runs", 3),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("noise_floor measurement failed (%s) — defaulting to 0", exc)
            self.noise_floor_us = 0.0
        log.info("[%s] Noise floor: %.3f µs", self.task_id, self.noise_floor_us)

    async def run_forever(self) -> None:
        """Main loop — runs until STOP file, roofline target, or max_runtime."""
        await self.init()
        t_start = time.time()
        max_s   = self.config["max_runtime_h"] * 3600

        while not self._stop:
            # Honour stop file
            if _STOP_FILE.exists():
                log.info("[%s] STOP file detected — exiting.", self.task_id)
                break

            # Timeout guard
            if time.time() - t_start > max_s:
                log.info("[%s] Max runtime reached.", self.task_id)
                break

            # Maybe migrate
            if self.db.should_migrate():
                n = self.db.migrate()
                log.info("[%s] Migration: %d candidates moved.", self.task_id, n)

            # Maybe escalate each island's tier
            for isl in range(self.config["n_islands"]):
                self._maybe_escalate(isl)

            # Run one round across all islands
            await self._run_island_round()

    async def _run_island_round(self) -> None:
        """Spawn one island agent per GPU and process results."""
        tasks = [
            self._run_island_agent(island=i)
            for i in range(self.config["n_islands"])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error("[%s] Island %d error: %s", self.task_id, i, result)
            else:
                await self._process_island_result(island=i, output=result)

    async def _run_island_agent(self, island: int) -> str:
        """Invoke a Claude agent SDK instance for one island and return its output."""
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore

        kb_query   = f"{self.config['op_class']} {self.config['kernel_lang']} optimisation"
        kb_snippet = await asyncio.to_thread(
            _kb_search, self.config["kb_path"], kb_query
        )

        parent = self.db.sample_parent(island)
        tier   = self._current_tier[island]
        gen    = self._generation[island]

        # Track consecutive null rounds per island — escalate pressure each time
        null_count = self._null_counts.get(island, 0)
        orch_pressure = ""
        if tier == "asm" and null_count > 0:
            orch_pressure = f"""
## ORCHESTRATOR OVERRIDE — Generation {gen} (null streak: {null_count})
The previous island STOPPED TOO EARLY. Minimum is 60 minutes and 20 iterations.
Check elapsed time — if under 60 minutes you MUST keep working.
There is ALWAYS more to optimize in ASM. You have {null_count} failed attempts.

MANDATORY THIS RUN — you must attempt ALL of these before reporting:
1. Run rocprofv3 PMC and compute WAIT/VMEM, LDS/VMEM, VALU/VMEM ratios
2. Attempt VGPR reduction: rearrange at least 3 instruction sequences to shorten live ranges
3. Attempt waitcnt loosening: find at least 2 vmcnt(0) and try vmcnt(1) or vmcnt(2)
4. Attempt software prefetch: move at least 1 buffer_load 4+ instructions earlier
5. Try packed instructions: find at least 1 case for buffer_load_b128 or v_pk_fma

If you cannot improve, you must explain WHY each of these 5 failed with the
specific instruction addresses and dependency chains that block the optimization.
Vague "kernel is HBM-bound" answers are not acceptable. Show the work.
"""
        prompt = _island_prompt(
            config=self.config,
            island=island,
            tier=tier,
            generation=gen,
            parent=parent,
            kb_snippet=kb_snippet,
            noise_floor_us=self.noise_floor_us,
            run_dir=self.run_dir,
        ) + orch_pressure

        log_path = self.run_dir / f"island_{island}_{tier}.log"
        # cwd must be a real host path (not a container path like /workspace)
        host_cwd = str(Path.home())
        opts = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            cwd=host_cwd,
            model=os.environ.get("EVOLVE_MODEL", "Claude-Opus-5"),
            env={k: v for k, v in {
                "ANTHROPIC_BASE_URL":        os.environ.get("ANTHROPIC_BASE_URL", ""),
                "ANTHROPIC_API_KEY":         os.environ.get("ANTHROPIC_API_KEY", ""),
                "ANTHROPIC_CUSTOM_HEADERS":  os.environ.get("ANTHROPIC_CUSTOM_HEADERS", ""),
            }.items() if v},
        )

        collected: list[str] = []
        async for event in query(prompt=prompt, options=opts):
            # Collect text output from the agent stream
            text = getattr(event, "text", None) or getattr(event, "content", "")
            if text:
                collected.append(str(text))

        full_output = "\n".join(collected)

        # Persist log
        try:
            log_path.write_text(full_output)
        except OSError:
            pass

        log.debug("[%s] Island %d agent completed (%d chars).",
                  self.task_id, island, len(full_output))
        return full_output

    async def _process_island_result(self, island: int, output: str) -> None:
        """Parse agent output, run gate, record KEEP/REJECT."""
        parsed = _parse_result_block(output)
        if parsed is None:
            log.warning("[%s] Island %d: no RESULT block found.", self.task_id, island)
            # Count nulls — orchestrator increases pressure each time
            self._null_counts[island] = self._null_counts.get(island, 0) + 1
            return

        gpu_us       = parsed["gpu_us"]
        co_path      = parsed["co_path"]
        s_path       = parsed["s_path"]
        label        = parsed["label"]
        sig = {
            "vgpr":     parsed["sig_vgpr"],
            "barriers": parsed["sig_barriers"],
            "spills":   parsed["sig_spills"],
            "mfma":     parsed["sig_mfma"],
        }

        tier = self._current_tier[island]
        gen  = self._generation[island]
        cid  = f"{self.task_id}_isl{island}_g{gen}_{uuid.uuid4().hex[:6]}"

        parent = self.db.sample_parent(island)

        candidate = Candidate(
            id            = cid,
            weakest_us    = gpu_us,
            roofline_frac = 0.0,       # filled in by gate
            parent_id     = parent.id if parent else None,
            generation    = gen,
            island        = island,
            tier          = tier,
            label         = label,
            sig           = sig,
            co_path       = co_path,
            s_path        = s_path,
        )

        # Run gate — returns a GateDecision dataclass, not a tuple.
        # Pass current campaign best so gate can compare candidate vs best,
        # not candidate vs itself.
        current_best = self.db.best()
        gate_config = {**self.config,
                       "best_us": current_best.weakest_us if current_best else None}
        try:
            gate: GateDecision = await asyncio.to_thread(
                run_gate,
                candidate,
                gate_config,
                self.noise_floor_us,
            )
            decision     = gate.decision
            roofline_frac = gate.roofline_frac or 0.0
            # Surface gate reason in log for every non-KEEP result
            if decision != "KEEP":
                log.info("[%s] Island %d gate reason: %s", self.task_id, island, gate.reason)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] Island %d gate error: %s", self.task_id, island, exc)
            # On exception, fall back to island's own measurement as baseline if no DB best
            if self.db.best() is None and gpu_us > 5.0:
                log.warning("[%s] Island %d gate exception fallback — using island "
                            "measurement %.3fµs as baseline", self.task_id, island, gpu_us)
                candidate.roofline_frac = 0.0
                self.db.record_keep(candidate)
                self._generation[island] = gen + 1
            return

        candidate.roofline_frac = roofline_frac

        if decision == "KEEP":
            log.info(
                "[%s] Island %d KEEP  gpu_us=%.3f  roofline=%.2f%%  tier=%s  gen=%d",
                self.task_id, island, gpu_us, roofline_frac * 100, tier, gen,
            )
            self.db.record_keep(candidate)
            # Persist tier so restarts resume in the right tier
            try:
                _sp = self.run_dir / "orch_state.json"
                _state = json.loads(_sp.read_text()) if _sp.exists() else {}
                _state.update({"tier": tier, "generation": gen, "best_us": candidate.weakest_us})
                _sp.write_text(json.dumps(_state, indent=2))
            except Exception:
                pass
            self._tier_keeps[island] = self._tier_keeps.get(island, 0) + 1
            self._generation[island] = gen + 1

            # Update roofline tracker
            try:
                update_tracker(str(self.run_dir), candidate.to_dict())
            except Exception:  # noqa: BLE001
                pass

            await asyncio.to_thread(self._git_commit_keep, candidate)
            await asyncio.to_thread(self._deploy_keep, candidate)

            # Check stop condition — only on trustworthy roofline measurements.
            # l2_resident=True means analytic hbm_bytes >> real HBM traffic (small-M
            # MoE, few active experts). In that case roofline_frac is capped at 1.0
            # but does NOT reflect actual hardware efficiency. Per KernelForge: use
            # rocprofv3 FETCH_SIZE+WRITE_SIZE for accurate fracs before declaring done.
            l2_resident_flag = getattr(gate, "l2_resident", False)
            if roofline_frac >= ROOFLINE_TARGET and not l2_resident_flag:
                log.info(
                    "[%s] ROOFLINE TARGET %.0f%% reached (verified, not L2-inflated) — stop.",
                    self.task_id, ROOFLINE_TARGET * 100,
                )
                self._stop = True
            elif roofline_frac >= ROOFLINE_TARGET and l2_resident_flag:
                log.warning(
                    "[%s] roofline_frac=%.2f%% but l2_resident=True — analytic model "
                    "invalid for small-M MoE (few active experts << analytic estimate). "
                    "Continuing campaign. Use rocprofv3 FETCH_SIZE+WRITE_SIZE for real frac.",
                    self.task_id, roofline_frac * 100,
                )

        else:
            log.info(
                "[%s] Island %d REJECT gpu_us=%.3f  roofline=%.2f%%  tier=%s  gen=%d",
                self.task_id, island, gpu_us, roofline_frac * 100, tier, gen,
            )
            self._generation[island] = gen + 1
            self._null_counts[island] = self._null_counts.get(island, 0) + 1

        # Dream (meta-reflection) every 20 generations
        if self._generation[island] % 20 == 0:
            await asyncio.to_thread(self._dream, island)

    # ------------------------------------------------------------------
    # Git commit
    # ------------------------------------------------------------------

    def _git_commit_keep(self, candidate: Candidate) -> None:
        """Stage all changes in $HOME/Evolve and commit."""
        repo = _EVOLVE_REPO
        if not repo.exists():
            log.warning("Evolve repo not found at %s — skipping commit.", repo)
            return
        try:
            # Skip files written by container as root (permission denied on git add)
            subprocess.run(
                ["git", "add", "--ignore-errors", "-A"], cwd=repo,
                capture_output=True,  # don't check=True — permission errors are non-fatal
            )
            msg = (
                f"KEEP {candidate.id}: {candidate.label}\n\n"
                f"task_id={self.task_id}  island={candidate.island}  "
                f"tier={candidate.tier}  gen={candidate.generation}\n"
                f"gpu_us={candidate.weakest_us:.3f}  "
                f"roofline={candidate.roofline_frac:.4f}\n"
                f"vgpr={candidate.sig.get('vgpr',0)}  "
                f"barriers={candidate.sig.get('barriers',0)}  "
                f"spills={candidate.sig.get('spills',0)}  "
                f"mfma={candidate.sig.get('mfma',0)}\n\n"
                f"Co-Authored-By: Claude <noreply@anthropic.com>"
            )
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=repo, check=True, capture_output=True,
            )
            log.info("[%s] Git commit for %s succeeded.", self.task_id, candidate.id)
        except subprocess.CalledProcessError as exc:
            log.error(
                "[%s] Git commit failed: %s\nstdout=%s\nstderr=%s",
                self.task_id, exc,
                exc.stdout.decode(errors="replace") if exc.stdout else "",
                exc.stderr.decode(errors="replace") if exc.stderr else "",
            )

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def _deploy_keep(self, candidate: Candidate) -> None:
        """Copy .co to deployment_path and optionally update deployment_csv."""
        deploy_path = self.config.get("deployment_path", "")
        deploy_csv  = self.config.get("deployment_csv", "")
        if not deploy_path or not candidate.co_path:
            return

        import shutil
        try:
            dst = Path(deploy_path)
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.co_path, dst / Path(candidate.co_path).name)
            log.info("[%s] Deployed %s → %s", self.task_id, candidate.co_path, dst)
        except OSError as exc:
            log.error("[%s] Deploy failed: %s", self.task_id, exc)

        if deploy_csv:
            _append_csv(deploy_csv, candidate)

    # ------------------------------------------------------------------
    # Tier escalation
    # ------------------------------------------------------------------

    def _maybe_escalate(self, island: int) -> None:
        """Advance tier if quota met. ASM tier never escalates — stays there forever."""
        tier  = self._current_tier[island]

        # ASM is the terminal tier — never escalate away from it.
        # Even if the island declares it's stuck, the orchestrator disagrees
        # and re-launches it with a stronger prompt (see _build_island_prompt).
        if tier == "asm":
            return

        quota = ESCALATE_AT.get(tier, 999)
        if self._tier_keeps.get(island, 0) >= quota:
            idx = TIERS.index(tier)
            if idx + 1 < len(TIERS):
                new_tier = TIERS[idx + 1]
                log.info(
                    "[%s] Island %d escalating %s → %s after %d KEEPs.",
                    self.task_id, island, tier, new_tier, quota,
                )
                self._current_tier[island] = new_tier
                self._tier_keeps[island]   = 0

    # ------------------------------------------------------------------
    # Dream (meta-reflection)
    # ------------------------------------------------------------------

    def _dream(self, island: int) -> None:
        """Write a brief meta-reflection to encourage diversity."""
        lb = self.db.leaderboard(n=5)
        dream_path = self.run_dir / f"dream_island{island}.json"
        try:
            dream_path.write_text(json.dumps(lb, indent=2))
            log.debug("[%s] Island %d dream written to %s.", self.task_id, island, dream_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_result_block(output: str) -> Optional[Dict[str, Any]]:
    """Extract timing from agent output — tries structured RESULT block first,
    then falls back to scanning for gpu_us: lines (which bench scripts emit)."""

    # --- Primary: structured RESULT...END_RESULT block ---
    m = re.search(r"RESULT\s*\n(.*?)\nEND_RESULT", output, re.DOTALL)
    if m:
        body = m.group(1)
        def _get(key: str, cast=str, default=None):
            km = re.search(rf"^{key}=(.+)$", body, re.MULTILINE)
            if km:
                try: return cast(km.group(1).strip())
                except (ValueError, TypeError): return default
            return default
        gpu_us = _get("gpu_us", float)
        if gpu_us is not None:
            return {"gpu_us": gpu_us,
                    "sig_vgpr":     _get("sig_vgpr",    int, 0),
                    "sig_barriers": _get("sig_barriers", int, 0),
                    "sig_spills":   _get("sig_spills",   int, 0),
                    "sig_mfma":     _get("sig_mfma",     int, 0),
                    # Validate paths exist — truncated paths fail the rebuild guard
                    "co_path": (_get("co_path", str, "") or "")
                               if _get("co_path", str, "") and
                               __import__('os').path.exists(_get("co_path", str, ""))
                               else "",
                    "s_path":  (_get("s_path",  str, "") or "")
                               if _get("s_path",  str, "") and
                               __import__('os').path.exists(_get("s_path",  str, ""))
                               else "",
                    "label":        _get("label",        str, "")}

    # --- Fallback: scan for any gpu_us: XX.XX line the bench scripts emit ---
    # Take the MINIMUM (best candidate) the island found this iteration.
    gpu_us_matches = re.findall(r"gpu_us[=:\s]+([\d.]+)", output)
    if not gpu_us_matches:
        # Also accept µs pattern: "7.33 µs" or "22.37µs"
        gpu_us_matches = re.findall(r"([\d.]+)\s*µs", output)
    if gpu_us_matches:
        us_values = [float(x) for x in gpu_us_matches if 5.0 < float(x) < 100_000]
        if us_values:
            best_us = min(us_values)
            label_m = re.search(r"(flydsl_\w+|ck_\w+|triton_\w+|t\d+x\d+\w+)", output)
            label = label_m.group(1) if label_m else f"best={best_us:.3f}µs"
            # Extract .co/.s paths — use longest match to avoid truncation
            co_matches = re.findall(r"(/[\w/.-]+\.co)\b", output)
            s_matches  = re.findall(r"(/[\w/.-]+\.s)\b", output)
            # Prefer the longest path (most likely to be complete, not truncated)
            co_m_val = max(co_matches, key=len) if co_matches else ""
            s_m_val  = max(s_matches,  key=len) if s_matches  else ""
            # Also validate the path exists — if truncated it won't exist
            import os as _os
            if co_m_val and not _os.path.exists(co_m_val):
                co_m_val = ""  # don't pass a broken path to rebuild guard
            if s_m_val and not _os.path.exists(s_m_val):
                s_m_val = ""
            co_m = type('M', (), {'group': lambda self, n: co_m_val})() if co_m_val else None
            s_m  = type('M', (), {'group': lambda self, n: s_m_val})()  if s_m_val  else None
            return {"gpu_us":      best_us,
                    "sig_vgpr":    0, "sig_barriers": 0,
                    "sig_spills":  0, "sig_mfma":     0,
                    "co_path":     co_m.group(1) if co_m else "",
                    "s_path":      s_m.group(1)  if s_m  else "",
                    "label":       label}

    return None


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _append_csv(csv_path: str, candidate: Candidate) -> None:
    p = Path(csv_path)
    header = not p.exists()
    try:
        with open(p, "a") as f:
            if header:
                f.write("ts,id,island,tier,generation,weakest_us,roofline_frac,"
                        "vgpr,barriers,spills,mfma,label\n")
            f.write(
                f"{candidate.ts},{candidate.id},{candidate.island},{candidate.tier},"
                f"{candidate.generation},{candidate.weakest_us:.3f},"
                f"{candidate.roofline_frac:.4f},"
                f"{candidate.sig.get('vgpr',0)},{candidate.sig.get('barriers',0)},"
                f"{candidate.sig.get('spills',0)},{candidate.sig.get('mfma',0)},"
                f'"{candidate.label}"\n'
            )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Async multi-kernel runner
# ---------------------------------------------------------------------------

async def _run_all_kernels(configs: List[Dict[str, Any]], run_dir: Path) -> None:
    """Run all kernel orchestrators in parallel."""
    orchestrators = [
        EvolveOrchestrator(cfg, run_dir / cfg["task_id"])
        for cfg in configs
    ]
    await asyncio.gather(*[o.run_forever() for o in orchestrators])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="evolve-asm orchestrator — drive MAP-Elites island evolution"
    )
    parser.add_argument(
        "task_yaml",
        help="Path to task YAML (or JSON) file describing the kernels to evolve.",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        metavar="TASK_ID",
        help="Restrict to specific task_ids (default: all kernels in the YAML).",
    )
    parser.add_argument(
        "--run-dir",
        default=str(Path.home() / "evolve_runs"),
        help="Directory for logs, DB, and artefacts (default: ~/evolve_runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run_dir (DB is loaded automatically).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    configs = load_task(args.task_yaml)

    if args.kernels:
        configs = [c for c in configs if c["task_id"] in args.kernels]
        if not configs:
            parser.error(
                f"None of the requested kernels {args.kernels} found in {args.task_yaml}"
            )

    run_dir = Path(args.run_dir)
    if not args.resume and run_dir.exists():
        log.info("run_dir %s already exists; use --resume to continue.", run_dir)

    log.info("Starting evolve-asm for %d kernel(s) in %s", len(configs), run_dir)
    asyncio.run(_run_all_kernels(configs, run_dir))


if __name__ == "__main__":
    main()
