"""Multi-island ASM kernel evolution orchestrator.

Integrates with the Arbor harness: uses CampaignState for persistence,
arbor.execute.workspace for git commits on KEEP, and the pack's validator
shell scripts for bench and correctness.

Tier ladder (code-enforced):
  native (3 nulls) → peer (5) → asm (5) → crosskernel (5) → algorithmic (5) → native

Stop when roofline_frac >= 0.85 on any weakest-of-3 KEEP, or STOP file exists.

Usage:
    python3 -m arbor.policy.evolve_asm.orchestrator \
        --task /path/to/evolve_task.yaml [--run-dir experiments/...] [--resume]
"""
from __future__ import annotations

import asyncio, json, re, subprocess, sys, time, uuid
from pathlib import Path
from typing import Optional

from arbor.policy.evolve_asm.island_db import Candidate, IslandDB
from arbor.policy.evolve_asm.roofline_tracker import (
    load_trajectory, render_summary, update_tracker,
)

ROOFLINE_TARGET = 0.85
TIERS           = ['native', 'peer', 'asm', 'crosskernel', 'algorithmic']
ESCALATE_AT     = {'native': 3, 'peer': 5, 'asm': 5, 'crosskernel': 5, 'algorithmic': 5}
PEER_LANG       = {'gemm': 'flydsl', 'norm': 'ck', 'attention': 'triton',
                   'recurrent': 'triton', 'moe': 'flydsl', 'elementwise': 'flydsl'}

_ARBOR_ROOT = Path(__file__).parent.parent.parent.parent   # Arbor/


def _kb_search(kb_path: str, query: str, top: int = 5) -> str:
    """Query Evolve KB index. Returns formatted passages or empty string."""
    for candidate in [
        Path(kb_path).parent.parent / 'kb_search.py',
        _ARBOR_ROOT.parent / 'Evolve/knowledge_base/kb_search.py',
    ]:
        if candidate.exists():
            try:
                r = subprocess.run(
                    [sys.executable, str(candidate), 'query', query, '--top', str(top)],
                    capture_output=True, text=True, timeout=30, cwd=str(candidate.parent),
                )
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
            except Exception:
                pass
    return ''


def _git_commit(workspace_host: str, message: str) -> Optional[str]:
    """Commit all changes in the host workspace git repo. Returns sha or None."""
    try:
        subprocess.run(['git', '-C', workspace_host, 'add', '-A'],
                       check=True, capture_output=True)
        r = subprocess.run(['git', '-C', workspace_host, 'commit', '-m', message],
                           capture_output=True, text=True)
        if r.returncode == 0:
            sha = r.stdout.strip().split()[1] if r.stdout.strip() else 'unknown'
            return sha
    except Exception as e:
        print(f'[git] commit failed: {e}', file=sys.stderr)
    return None


def _build_island_prompt(*, task_id: str, role: str, tier: str, island: int,
                          gpu: int | str, container: str, workspace: str,
                          kb_path: str, run_dir: str, op_class: str,
                          kernel_lang: str, ref_co: str, bench_cmd: str,
                          snr_cmd: str, snr_threshold: float, best_us: float,
                          roofline_frac: Optional[float], population: list[dict],
                          lineage_tail: list[dict], analysis: dict,
                          null_streak: int, generation: int) -> str:

    pop_lines = '\n'.join(
        f"  {c['weakest_us']:.3f}µs  tier={c['tier']}  {c['label'][:55]}"
        for c in population[:8])
    lineage_lines = '\n'.join(
        f"  gen {e.get('generation','?')}: {e.get('weakest_us','?')}µs "
        f"[{e.get('decision','?')}] {e.get('label','')[:55]}"
        for e in lineage_tail[-6:])
    bottleneck = analysis.get('bottleneck', '')
    tried = json.dumps(analysis.get('tried_approaches', [])[-8:], indent=2)

    kb_query   = (f'{bottleneck} {op_class} {kernel_lang} gfx950' if bottleneck
                  else f'{op_class} {kernel_lang} gfx950 MFMA optimization')
    kb_results = _kb_search(kb_path, kb_query)
    kb_section = f"\n--- KB RESULTS: '{kb_query}' ---\n{kb_results}\n" if kb_results else ''

    tier_guide = {
        'native': (
            f"TIER 1: Exhaust {kernel_lang} compile parameters.\n"
            f"  FlyDSL: k_wave=[1,2,4,7] tile_m=[32,64,128] gui_mode async_copy persist\n"
            f"  Triton: BK BV num_warps num_stages\n"
            f"  CK: block_size vec_width num_waves\n"
            f"Accept if code STRUCTURE changes (barriers±20%, VGPR±10). "
            f"Exit on null_streak={ESCALATE_AT['native']}."
        ),
        'peer': (
            f"TIER 2: Cross-compile to {PEER_LANG.get(op_class,'flydsl')} for better .amdgcn.\n"
            f"Compile → extract .amdgcn → set as new ASM baseline via patch_co.py.\n"
            f"Exit on null_streak={ESCALATE_AT['peer']}."
        ),
        'asm': (
            f"TIER 3: Hand-tuned AMDGCN ASM. This is the real optimization.\n"
            f"Read KB instructions in {kb_path}/instructions/\n"
            f"Run rocprofv3 PMC: SQ_INSTS_VMEM SQ_WAIT_INST_ANY SQ_INSTS_LDS SQ_ACTIVE_INST_ANY\n"
            f"One change per iteration. patch_co.py → SNR check → bench.\n"
            f"Exit on null_streak={ESCALATE_AT['asm']}."
        ),
        'crosskernel': (
            f"TIER 4: Apply techniques from campaign_archive/ that worked on other kernels.\n"
            f"Exit on null_streak={ESCALATE_AT['crosskernel']}."
        ),
        'algorithmic': (
            f"TIER 5: Fundamentally different algorithm → peer lang → new .amdgcn → ASM.\n"
            f"Exit on null_streak={ESCALATE_AT['algorithmic']}."
        ),
    }

    return f"""You are an AMDGCN ASM kernel optimization island agent.
{kb_section}
Task: {task_id}  Role: {role}  Island: {island}  Tier: {tier}
GPU: {gpu}  (HIP_VISIBLE_DEVICES={gpu})

GOAL: Push to {ROOFLINE_TARGET*100:.0f}% of MAF roofline (hand-tuned AMDGCN ASM).
NEVER say "at ceiling". NEVER park idle.

--- STATE ---
Best:         {best_us:.3f} µs
Roofline:     {f'{roofline_frac*100:.1f}%' if roofline_frac else 'not measured'} (target {ROOFLINE_TARGET*100:.0f}%)
null_streak:  {null_streak} / {ESCALATE_AT.get(tier, 5)}
generation:   {generation}
Bottleneck:   {bottleneck or 'unknown'}

--- TOP POPULATION ---
{pop_lines or '  (empty — first iteration)'}

--- RECENT LINEAGE ---
{lineage_lines or '  (empty)'}

--- TRIED ---
{tried}

--- RESOURCES ---
Container:   {container}   (docker exec {container} bash -c '...')
Workspace:   {workspace}
Ref .co:     {ref_co}
KB:          {kb_path}
patch_co.py: /arbor/packs/evolve-asm/validator/patch_co.py
bench:       /arbor/packs/evolve-asm/validator/bench_moe_fp8fp4.py
LLVM:        $(find /opt/rocm* -name llvm-mc 2>/dev/null | head -1 | xargs dirname)
Bench cmd:   {bench_cmd}
SNR cmd:     {snr_cmd}  (threshold: {snr_threshold} dB)

--- TIER INSTRUCTIONS ---
{tier_guide.get(tier, tier_guide['asm'])}

--- LOOP ---
READ → HYPOTHESIZE → IMPLEMENT → bench → SNR → RECORD → repeat

After EVERY bench print:
  gpu_us: <X.X>
  sig_vgpr: <N>   sig_barriers: <N>   sig_spills: <N>   sig_mfma: <N>

Write {run_dir}/island_{island}_{role}_progress.md after every iteration:
  ## Iteration N — [timestamp]
  **Hypothesis:** ...  **KB ref:** ...  **Change:** ...
  **Result:** XX.Xµs or FAILED   **Learned:** one sentence

On null_streak == threshold: write {run_dir}/island_{island}_{role}_stopped.json:
  {{"reason": "...", "bottleneck": "...", "open_directions": [...]}}
Then EXIT.

When done with any change: DONE: <one-line description>
"""


class EvolveOrchestrator:
    def __init__(self, task: dict, run_dir: str):
        self.task         = task
        self.run_dir      = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.task_id      = task['task_id']
        self.container    = task['container']
        self.workspace    = task.get('workspace', '/workspace')
        self.kb_path      = task.get('kb_path', '/workspace/knowledge_base/aiter/intellikit-asm-skills')
        self.gpus         = task.get('gpus', [0])
        self.op_class     = task.get('op_class', 'gemm')
        self.kernel_lang  = task.get('kernel_lang', 'flydsl')
        self.ref_co       = task.get('ref_co', '')
        self.bench_cmd    = task['bench_cmd']
        self.snr_cmd      = task['snr_cmd']
        self.snr_threshold = task.get('snr_threshold', 30.0)
        self.n_islands    = task.get('n_islands', len(self.gpus))
        self.n_bench_runs = task.get('n_bench_runs', 3)
        self.primary_M    = (task.get('shapes', {}).get('primary', {}).get('M') or
                             task.get('M', 32))
        # workspace on host (for git commits)
        self.workspace_host = task.get('workspace_host',
                                        str(_ARBOR_ROOT.parent / 'Evolve'))

        self.db       = IslandDB(num_islands=self.n_islands,
                                  db_path=str(self.run_dir / 'db'))
        self.tier     = 'native'
        self.null_streak = 0
        self.generation  = 0
        self.best_us: Optional[float] = None
        self.noise_floor = 0.005
        self.analysis: dict = {}

        self._state_path = self.run_dir / 'orch_state.json'
        if self._state_path.exists():
            self._load_state()

        self._log_file = open(self.run_dir / 'orch.log', 'a', buffering=1)

    def _resolved_bench_cmd(self, gpu: int | str) -> str:
        return (self.bench_cmd
                .replace('{gpu}', str(gpu))
                .replace('{M}', str(self.primary_M)))

    def _resolved_snr_cmd(self, gpu: int | str) -> str:
        return (self.snr_cmd
                .replace('{gpu}', str(gpu))
                .replace('{M}', str(self.primary_M)))

    def log(self, msg: str) -> None:
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        line = f'[{ts}] {msg}'
        print(line, flush=True)
        self._log_file.write(line + '\n')

    # ── Init ─────────────────────────────────────────────────────────────────

    def _run_gate(self, gpu: int | str) -> dict:
        """Run bench.sh inside container, return parsed gate result."""
        bench_cmd = self._resolved_bench_cmd(gpu)
        env = {
            'BENCH_OUTPUT': '/tmp/evolve_bench_output.json',
            'CONTAINER': self.container,
            'KERNEL_BENCH_CMD': bench_cmd,
            'N_BENCH': str(self.n_bench_runs * 100),   # more for noise floor
        }
        bench_sh = str(_ARBOR_ROOT / 'packs/evolve-asm/validator/bench.sh')
        env_args = [item for k, v in env.items() for item in [f'{k}={v}']]
        r = subprocess.run(['env'] + env_args + ['bash', bench_sh],
                           capture_output=True, text=True, timeout=600)
        out_path = Path('/tmp/evolve_bench_output.json')
        if out_path.exists():
            try:
                return json.loads(out_path.read_text())
            except Exception:
                pass
        return {'status': 'failed', 'error': r.stderr[-300:] or 'bench.sh failed'}

    def init(self) -> None:
        self.log('=== INIT: measuring baseline ===')
        gpu = self.gpus[0]
        result = self._run_gate(gpu)
        if result.get('status') == 'ok':
            self.best_us = result['weakest_us']
            rf = result.get('roofline_frac')
            self.log(f'Baseline: {self.best_us:.3f}µs'
                     + (f' | roofline {rf*100:.1f}%' if rf else ''))
            baseline = Candidate(id=str(uuid.uuid4()), weakest_us=self.best_us,
                                 roofline_frac=rf, parent_id=None, generation=0,
                                 island=0, tier='baseline', label='baseline',
                                 co_path=self.ref_co)
            self.db.add(baseline)
            update_tracker(str(self.run_dir), {
                'decision': 'BASELINE', 'generation': 0, 'tier': 'baseline',
                'weakest_us': self.best_us, 'roofline_frac': rf,
                'label': 'baseline', 'ts': time.time(),
            })
        else:
            self.log(f'WARN: baseline failed ({result.get("error","?")}) — proceeding')
            self.best_us = 999999.0
        self._save_state()

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        self.log('=== CAMPAIGN START ===')
        self.log(f'Tier ladder: {" → ".join(f"{t}({ESCALATE_AT[t]})" for t in TIERS)}')
        self.log(f'Stop: roofline >= {ROOFLINE_TARGET*100:.0f}% or touch {self.run_dir}/STOP')

        if self.best_us is None:
            self.init()
        else:
            self.log(f'\n{load_trajectory(str(self.run_dir))}')

        while True:
            if (self.run_dir / 'STOP').exists():
                self.log('STOP file — exiting.')
                break

            self._maybe_escalate()

            if self.db.should_migrate():
                n = self.db.migrate()
                self.log(f'Migration: {n} candidates moved between islands')

            rf = self._current_roofline_frac()
            self.log(f'--- Gen {self.generation} | tier={self.tier} | '
                     f'null_streak={self.null_streak}/{ESCALATE_AT.get(self.tier,"?")} | '
                     f'best={self.best_us:.3f}µs | rf={f"{rf*100:.1f}%" if rf else "?"} ---')
            for i in range(self.n_islands):
                ib = self.db.island_best(i)
                if ib:
                    self.log(f'  island {i}: {ib.weakest_us:.3f}µs tier={ib.tier} '
                             f'rf={f"{ib.roofline_frac*100:.1f}%" if ib.roofline_frac else "?"}')

            results = await self._run_island_round()
            for r in results:
                self._process_result(r)

            if self._campaign_done():
                self.log(f'\n=== DONE: reached {ROOFLINE_TARGET*100:.0f}% MAF roofline ===')
                self.log(self.db.leaderboard())
                break

            self.generation += 1
            self._save_state()

    async def _run_island_round(self) -> list[dict]:
        tasks = []
        for i in range(self.n_islands):
            gpu = self.gpus[i % len(self.gpus)]
            role = ['algorithmic', 'occupancy', 'peer', 'crosskernel'][i % 4]
            prompt = _build_island_prompt(
                task_id=self.task_id, role=role, tier=self.tier,
                island=i, gpu=gpu, container=self.container,
                workspace=self.workspace, kb_path=self.kb_path,
                run_dir=str(self.run_dir), op_class=self.op_class,
                kernel_lang=self.kernel_lang, ref_co=self.ref_co,
                bench_cmd=self._resolved_bench_cmd(gpu),
                snr_cmd=self._resolved_snr_cmd(gpu),
                snr_threshold=self.snr_threshold,
                best_us=self.best_us or 0.0,
                roofline_frac=self._current_roofline_frac(),
                population=self.db.population_snapshot(),
                lineage_tail=self._lineage_tail(),
                analysis=self.analysis,
                null_streak=self.null_streak,
                generation=self.generation,
            )
            tasks.append(self._run_island(i, role, gpu, prompt))
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _run_island(self, island: int, role: str, gpu: int | str,
                           prompt: str) -> dict:
        log_path     = self.run_dir / f'island_{island}_{role}.log'
        stopped_path = self.run_dir / f'island_{island}_{role}_stopped.json'
        stopped_path.unlink(missing_ok=True)
        self.log(f'  Spawning island {island} ({role}) GPU={gpu}')
        output = ''
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
            opts = ClaudeAgentOptions(
                cwd=str(_ARBOR_ROOT),
                allowed_tools=['Bash', 'Read', 'Edit', 'Write'],
                permission_mode='bypassPermissions',
            )
            parts = []
            with open(log_path, 'a', buffering=1) as lf:
                async for msg in query(prompt=prompt, options=opts):
                    if hasattr(msg, 'content'):
                        for block in msg.content:
                            if hasattr(block, 'text') and block.text:
                                lf.write(block.text); parts.append(block.text)
            output = ''.join(parts)
        except Exception as e:
            self.log(f'  Island {island} ({role}) error: {e}')

        stopped = {}
        if stopped_path.exists():
            try: stopped = json.loads(stopped_path.read_text())
            except Exception: pass

        return {'island': island, 'role': role, 'gpu': gpu,
                'output': output, 'stopped': stopped}

    def _process_result(self, result: dict) -> None:
        island = result['island']
        role   = result['role']
        output = result['output']

        # Parse all gpu_us: lines from island output
        samples = [float(m) for m in re.findall(r'gpu_us[=:\s]+([\d.]+)', output)]
        if not samples:
            self.log(f'  Island {island} ({role}): no timing — null')
            self.null_streak += 1
            return

        weakest_us = max(samples)   # gate on worst run
        median_us  = sorted(samples)[len(samples) // 2]

        # Parse roofline_frac
        rf_m = re.search(r'roofline[_\s]+(\d+\.?\d*)%', output, re.IGNORECASE)
        roofline_frac = float(rf_m.group(1)) / 100.0 if rf_m else None

        # Parse structural signature
        sig = {}
        for key in ('vgpr', 'barriers', 'spills', 'mfma'):
            m = re.search(rf'sig_{key}[=:\s]+(\d+)', output, re.IGNORECASE)
            if m: sig[key] = int(m.group(1))

        # Gate decision
        if self.best_us is None:
            decision = 'KEEP'; delta_pct = 0.0; reason = 'baseline'
        else:
            improvement = (self.best_us - weakest_us) / self.best_us
            min_improvement = self.noise_floor
            if weakest_us < self.best_us and improvement >= min_improvement:
                decision = 'KEEP'
                delta_pct = improvement * 100.0
                reason = f'+{delta_pct:.1f}% vs {self.best_us:.3f}µs'
            else:
                decision = 'REVERT'
                delta_pct = (self.best_us - weakest_us) / self.best_us * 100.0
                reason = f'{delta_pct:+.1f}% — below noise floor or slower'

        if roofline_frac and roofline_frac >= ROOFLINE_TARGET:
            decision = 'STOP'
            reason = f'ROOFLINE TARGET: {roofline_frac*100:.1f}% >= {ROOFLINE_TARGET*100:.0f}%'

        self.log(f'  Island {island} ({role}): {decision} {weakest_us:.3f}µs — {reason}')

        lineage_entry = {
            'ts': time.time(), 'island': island, 'role': role,
            'tier': self.tier, 'generation': self.generation,
            'decision': decision, 'weakest_us': weakest_us,
            'median_us': median_us, 'best_us': self.best_us,
            'delta_pct': delta_pct, 'roofline_frac': roofline_frac,
            'reason': reason, 'label': reason, 'sig': sig,
        }

        if decision in ('KEEP', 'STOP'):
            self.best_us = weakest_us
            self.null_streak = 0
            dashboard = update_tracker(str(self.run_dir), lineage_entry)
            self.log(f'\n{dashboard}')

            # Git commit the KEEP to the host workspace
            label = result['stopped'].get('reason', role)[:60]
            sha = _git_commit(self.workspace_host,
                              f'KEEP gen{self.generation}: {self.task_id} {weakest_us:.3f}µs '
                              f'({delta_pct:+.1f}%) tier={self.tier} rf={f"{roofline_frac*100:.1f}%" if roofline_frac else "?"}\n\n'
                              f'{label}\n\nCo-Authored-By: Claude <noreply@anthropic.com>')
            if sha:
                self.log(f'  Git commit: {sha}')

            candidate = Candidate(
                id=str(uuid.uuid4()), weakest_us=weakest_us,
                roofline_frac=roofline_frac, parent_id=None,
                generation=self.generation, island=island,
                tier=self.tier, label=reason, sig=sig,
            )
            self.db.record_keep(candidate)
            self.log(f'  {self.db.leaderboard()}')
        else:
            self.null_streak += 1
            with open(self.run_dir / 'lineage.jsonl', 'a') as f:
                f.write(json.dumps(lineage_entry) + '\n')

    def _maybe_escalate(self) -> None:
        threshold = ESCALATE_AT.get(self.tier, 5)
        if self.null_streak < threshold:
            return
        idx = TIERS.index(self.tier) if self.tier in TIERS else 0
        new_tier = TIERS[(idx + 1) % len(TIERS)]
        self.log(f'ESCALATE: {self.tier}(null_streak={self.null_streak}) → {new_tier}')
        if new_tier == 'native':   # full cycle — run dream
            self._dream()
        self.tier = new_tier
        self.null_streak = 0

    def _dream(self) -> None:
        """Synthesise learnings from lineage on cycle completion."""
        self.log('DREAM: synthesising campaign learnings...')
        p = self.run_dir / 'lineage.jsonl'
        if not p.exists(): return
        entries = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        self.analysis['tried_approaches'] = [e.get('reason', '') for e in entries[-50:]]
        self.analysis['dream'] = (
            f"{sum(1 for e in entries if e.get('decision')=='KEEP')} KEEPs, "
            f"{sum(1 for e in entries if e.get('decision')=='REVERT')} REVERTs "
            f"over {self.generation} generations. Best: {self.best_us:.3f}µs."
        )
        (self.run_dir / 'analysis.json').write_text(
            json.dumps(self.analysis, indent=2) + '\n')
        self.log(f'DREAM: {self.analysis["dream"]}')

    def _campaign_done(self) -> bool:
        rf = self._current_roofline_frac()
        return rf is not None and rf >= ROOFLINE_TARGET

    def _current_roofline_frac(self) -> Optional[float]:
        b = self.db.best()
        return b.roofline_frac if b else None

    def _lineage_tail(self, n: int = 8) -> list[dict]:
        p = self.run_dir / 'lineage.jsonl'
        if not p.exists(): return []
        lines = p.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps({
            'tier': self.tier, 'null_streak': self.null_streak,
            'generation': self.generation, 'best_us': self.best_us,
            'noise_floor': self.noise_floor, 'analysis': self.analysis,
        }, indent=2) + '\n')

    def _load_state(self) -> None:
        s = json.loads(self._state_path.read_text())
        self.tier        = s.get('tier', 'native')
        self.null_streak = s.get('null_streak', 0)
        self.generation  = s.get('generation', 0)
        self.best_us     = s.get('best_us')
        self.noise_floor = s.get('noise_floor', 0.005)
        self.analysis    = s.get('analysis', {})
        self.log(f'Resumed: gen={self.generation} tier={self.tier} best={self.best_us}µs')


# ── Entry point ───────────────────────────────────────────────────────────────

def load_task(yaml_path: str, kernel_index: int = 0) -> dict:
    """Load a kernel task from a DSv4.1-Flash-style evolve YAML."""
    import yaml
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    kernels = doc.get('kernels', [])
    if not kernels:
        raise ValueError(f"No kernels: in {yaml_path}")
    k = kernels[kernel_index % len(kernels)]
    return {
        'task_id':       k['task_id'],
        'container':     doc['container'],
        'workspace':     doc.get('workspace', '/workspace'),
        'workspace_host': str(_ARBOR_ROOT.parent / 'Evolve'),
        'kb_path':       doc.get('kb_path', '/workspace/knowledge_base/aiter/intellikit-asm-skills'),
        'gpus':          k.get('gpus', [0]),
        'op_class':      k.get('op_class', 'moe'),
        'kernel_lang':   k.get('kernel_lang', 'flydsl'),
        'ref_co':        k.get('ref_co', ''),
        'bench_cmd':     k['bench_cmd'].strip(),
        'snr_cmd':       k['snr_cmd'].strip(),
        'snr_threshold': k.get('snr_threshold', 30.0),
        'n_bench_runs':  doc.get('n_bench_runs', 3),
        'shapes':        k.get('shapes', {}),
        'deployment_path': k.get('deployment_path', ''),
        'deployment_csv':  k.get('deployment_csv', ''),
    }


def main() -> None:
    import argparse, yaml as _yaml
    p = argparse.ArgumentParser(description='evolve-asm orchestrator')
    p.add_argument('task', help='Task YAML path')
    p.add_argument('--kernel', type=int, default=0, help='Kernel index in kernels: list (default 0)')
    p.add_argument('--run-dir', default=None, help='Run directory (default: auto)')
    p.add_argument('--resume', default=None, help='Resume from existing run dir')
    args = p.parse_args()

    task = load_task(args.task, args.kernel)
    if args.resume:
        run_dir = args.resume
    elif args.run_dir:
        run_dir = args.run_dir
    else:
        ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        run_dir = f"experiments/evolve_{task['task_id']}_{ts}"

    print(f"[evolve-asm] task={task['task_id']} run_dir={run_dir}", flush=True)
    orch = EvolveOrchestrator(task, run_dir)
    asyncio.run(orch.run_forever())


if __name__ == '__main__':
    main()
