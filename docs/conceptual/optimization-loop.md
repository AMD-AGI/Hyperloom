---
myst:
    html_meta:
        "description": "Understand the Hyperloom optimization loop: runtime contracts, phase order (PRELUDE through CLOSE), orchestration model, feedback loops, and session artifacts."
        "keywords": "Hyperloom, optimization loop, PRELUDE, ENABLEMENT, FRAMEWORK_AGENT, KERNEL_AGENT, SWEEP, CLOSE, orchestration, session artifacts, AMD GPU, ROCm, LLM inference, PolicyGate, enablement, targeted build, escalation ladder, runnable gate"
---
# Hyperloom optimization loop

This topic describes the current Hyperloom agentic code optimizer loop from the
runtime contracts outward. It intentionally avoids retired action names
and old DFS demo mechanics; the live action catalog, phase allowlist,
PolicyGate, and session artifacts are the source of truth. This optimization
loop runs alongside the agentic kernel optimizer.

```{image} ../images/Hyperloom_optimization_loop.png
:alt: Hyperloom's control plane delegates kernel optimization to GEAK's execution plane through handoff, result, and kernel-journey artifacts. The current phase order and write-path contracts are described below.
:class: hl-lightbox-trigger
```

```{raw} html
<style>
  img.hl-lightbox-trigger { cursor: zoom-in; }
  dialog.hl-lightbox-overlay {
    max-width: 90vw; max-height: 90vh; padding: 0; border: none;
    background: transparent; overflow: visible;
  }
  dialog.hl-lightbox-overlay::backdrop { background: rgba(0,0,0,0.75); }
  dialog.hl-lightbox-overlay img { max-width: 90vw; max-height: 90vh; display: block; cursor: zoom-out; }
</style>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    if (window._hlLightboxInit) return;
    window._hlLightboxInit = true;
    document.querySelectorAll('img.hl-lightbox-trigger').forEach(function(img) {
      img.addEventListener('click', function() {
        var dialog = document.createElement('dialog');
        dialog.className = 'hl-lightbox-overlay';
        var clone = new Image();
        clone.src = img.src;
        clone.alt = img.alt;
        dialog.appendChild(clone);
        dialog.addEventListener('click', function() { dialog.close(); dialog.remove(); });
        document.body.appendChild(dialog);
        dialog.showModal();
      });
    });
  });
</script>
```

## Runtime contract

The optimizer is launched through `python -m hyperloom optimize`. A run
must be able to:

- Create or resume a session directory,
- Write `manifest.json`, `state.json`, `storage/coordinator.db`, action
  run workspaces, reports, and `session_breakdown.json`,
- Route intents through the Orchestration and Critic LLM roles,
  and dispatch kernel work to programmatic Python handlers,
- Produce a final report and a dashboard-consumable breakdown.

Private helper names and internal prompt wording are not contracts. The
observable session artifacts and subprocess JSON bridges are.

## Phase order

The Coordinator advances through the live phase chain:

```text
PRELUDE -> ENABLEMENT -> FRAMEWORK_AGENT -> KERNEL_AGENT -> SWEEP -> CLOSE
```

Cyclic macro-cycling is always enabled. After SWEEP, the Coordinator can
`cycle_reloop` back to `FRAMEWORK_AGENT` for another pass while
budget and leverage remain. The effective minimum remaining budget to justify
opening a new cycle scales with session length (capped at the 3-hour absolute
floor), so shorter sessions can also reloop when they have proportionally
sufficient time left. The 24-hour threshold only selects long-run budget
accounting: short bounded runs keep charge-back phase budgeting, while long /
unbounded runs use the fixed per-cycle budget window.

Whether another cycle is feasible is surfaced as `cycle_reloop_feasible` in
the ``=== Phase ===`` block for the five middle phases.

`machine_state.PHASE_ALLOWED_ACTIONS` is a cross-phase transition survival
filter, not the set the LLM can propose. The LLM-proposable set is
`allowed_actions_for(phase)`. Coordinator-owned actions such as analysis
refreshes and close sequencing may be enqueued internally even when the LLM
is not allowed to propose them.

Every phase transition is a GPU barrier: the Coordinator stops admitting new
work, drops queued work the next phase does not support, and commits the
transition only once the actions already running have finished and been
booked. A phase budget never cancels running work -- the last action in
flight, such as a GEAK rebench, completes and settles its result. Running
work is cancelled only when the run is already over: a terminal transition to
CLOSE (a recorded stop reason such as an unhealthy backend, a failed baseline,
convergence), an exhausted session budget, a shutdown or an
operator interrupt. Each
phase therefore starts on quiet GPUs, and its entry hook runs on the
transition itself.

## PRELUDE

PRELUDE establishes the session baseline:

1. `target_analysis` writes the reference summary and source-backed
   `competitor_target.json`. With no external GPU configured, it writes a
   no-target marker and clears the competitor target. Advisory and final-report
   comparisons read the same target file; missing targets remain unavailable.
   AgentX uses accepted `current_best` metrics, normalized by `state.tp` and
   matched at `state.conc`, without rereading benchmark artifacts.
   `--no-target-advisory` disables prompt hints, not the final comparison.
   External references never change Objective, scoring, or KEEP/REVERT.
2. `baseline` measures the starting throughput and records the benchmark
   invocation needed to reproduce it.
3. `roofline` or `profile` captures the first performance analysis.
   `roofline` is the preferred composite path when enabled; it wraps
   profiling, trace analysis, and `analysis.md` snapshot publication.

`model_class` is supplied by the launcher or derived once from model
metadata at boot. There is no separate live `classify` action.

## ENABLEMENT

ENABLEMENT is entered when a baseline fails and enablement is admitted
(`--enablement` is not `off` and the run is not multi-node). It is skipped
entirely on runs where the baseline boots on the first attempt.

Admission is controlled by `--enablement {off,launch,eval,all}`,
defaulting to `all`. `launch` admits the boot-failure lane, `eval` the
accuracy-failure lane, `all` both. With `off`, a baseline that keeps failing
terminates with `stop_reason='baseline_failed'` rather than opening an
authoring loop.

A KEEP in ENABLEMENT is graded on runnability and the accuracy floor, not
throughput. The phase exits normally when `baseline_tput > 0`, the
revalidation window is closed, and all in-flight enablement work is drained.
It exits terminally on `server_argv_invalid`, `environment_fault`, or
`enablement_attempts_exhausted` (eight consecutive failed rounds).

Only the enablement lane authors rounds here; the LLM proposes `target_analysis`.
A round that reaches its executor after the lane stopped taking rounds is skipped.

A KEEP or an advance ends the held round whoever authored it, since the wall
that round was aimed at is behind the boot. A specialist still queued under that
round is cancelled rather than left to author against the old wall; one already
running reports normally.

### Enablement escalation ladder

Enablement admission is controlled by `--enablement {off,launch,eval,all}`,
defaulting to `all`. `launch` admits the boot-failure lane, `eval` the
accuracy-failure lane, `all` both. With `off`, neither engages and a baseline
that keeps failing terminates the run with `stop_reason='baseline_failed'`
rather than opening an authoring loop.

When the admitted lane fires — a `(model, backend)` combination cannot launch, or
it launches but fails its accuracy eval (`accuracy_below_floor` /
`eval_runtime_failure`) — enablement repairs it along two axes. **Diagnosis (once):** work out which capability layer is
missing — read the failure signature, the model's `config.json` architecture,
the framework's supported-architecture registry and installed version, and
upstream (whether the capability already exists and in which version/PR). That
picks the entry rung. **Climb (as needed):** start at the lowest plausible rung
and go up only when the current rung cannot make it boot; after each cleared
boot failure, re-diagnose the new (deeper) failure and pick a rung again
(serial enablement — each round's fix is cumulative). A model whose architecture is
already supported but merely un-wired needs only the cheap top rungs; a
genuinely-new architecture climbs higher.

The full rendered methodology (the advisory "ladder book") is the canonical
text, built by `build_enablement_ladder_book` in
`hyperloom.orchestrator.enablement.mandate` and injected into the enablement
authoring specialist's prompt. The rungs, in increasing complexity:

0. **Rung 0 — diagnose / capability-gap localization.** Read-only:
   classify the failure, read `config.json`, check the supported-arch registry
   and version, and look up upstream. Output: the missing layer and the entry
   rung.
1. **Rung 1 — serve-flag / config wire-up.** The architecture is supported and
   only a serve flag / env / tokenizer-mode / trivial registration alias is
   missing. No new code or dependencies.
2. **Rung 2 — in-tree source patch.** A unified diff against the source tree
   the server imports: register the arch, a small forward/config/tokenizer
   bridge, or backport a merged PR. Pure Python, no compile.
3. **Rung 3 — attempt-scoped runtime.** The specialist names the version
   that carries the fix as a `needs_runtime` request in its
   `specialist_done`: a framework ref (PR, tag, branch or sha) installed
   editable, pip packages, or both. No operator setting or failure kind
   gates it, and vLLM accepts packages only, since a vLLM checkout compiles
   (Rung 5). `integrate_patch` installs the request into an isolated
   per-attempt venv that sees the installed stack, with pip held to the
   image's ROCm builds — torch, torchvision, torchaudio, triton and every
   distribution whose version carries a ROCm label — so a request that needs
   a different torch fails provision instead of pulling a CUDA build. That
   venv is activated only through the
   per-variant YAML benchmark envs; the Coordinator never mutates its own
   process environment to point at an attempt runtime. While one is active,
   the source tree the server imports is that runtime's own — the editable
   checkout, or the venv's site-packages when the framework itself was
   installed there — so Rung 2 and Rung 4 patches land there. A kept runtime
   is re-provisioned for every later round, whatever that round's failure or
   request.
4. **Rung 4 — source localization.** Localize a merged-PR or vendored
   closure into the source tree the server imports. Only the files under the
   framework's package are localized. A change that touches compiled or
   build-backend files cannot be satisfied by a plain source edit, and one
   that is oversize or does not apply to the tree is not backported either;
   in each case the round runs on its own patches, and a compiled gap is
   left to Rung 5.
5. **Rung 5 — off-loop compiled build.** Perform a compiled-component build
   (AITER kernels, sgl-kernel, or vLLM-from-source). Builds run *off* the
   coordinator tick loop on a dedicated single-slot build lane: each build
   is spawned detached and reaped across later ticks against a wall-clock
   budget, so a long compile never blocks the loop. A build installs under
   the same ROCm pins as Rung 3, minus the component it builds. Auto-escalation into
   Rung 5 can be disabled with the `HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD`
   environment variable (see
   [Targeted builds (Rung 5)](../reference/environment-variables.md#targeted-builds-rung-5)).

### Latency budget (constraint on KEEP)

`--max-latency-ms` sets a ceiling on mean end-to-end latency, for **scriptable
workloads only** (`xdit`, `custom`); the CLI refuses it for a serving framework.
A scriptable workload grades on output throughput alone, and it is the only
kind compute partitioning places work for, so it is where a throughput-only
gate can buy throughput with per-request latency. An AgentX serving session
already refuses that trade on interactivity.

It is a constraint rather than an objective: it does not decide when the run
stops, only which winners are admissible, so it composes with whichever
`--target-*` is in use. It rides the same verdict the gain gates decide — a
candidate that would otherwise KEEP and breaks the ceiling is a REVERT, with a
`veto_reason` (`latency_budget_exceeded` or `latency_unmeasured`). A candidate
that did not gain carries no veto, so the ledger names the gate that refused
it. Every KEEP decision a scriptable session reaches reads that verdict —
explore, a framework source patch, and a kernel integration — so a lane
reverts its own over-budget change rather than leaving it on disk.

The constraint exists because a throughput-only comparison does not merely
tolerate a latency-for-throughput trade, it selects for the worst one on
offer: facing a lever that raises aggregate throughput *by* making each stream
slower, the largest regression is where the most throughput is.

It fails closed. A candidate that reported no end-to-end latency is refused,
since a constraint nobody measured is not one anybody satisfied — which is why
every lane copies `e2el_mean_ms` onto the dict it promotes. It fails closed at
the boundary too: if the baseline itself exceeds the ceiling, or reported no
end-to-end latency, the run stops with `baseline_over_latency_budget` rather
than spending its whole budget refusing every candidate to learn what was
knowable at launch. Off by default, leaving KEEP behaviour unchanged when
unset.

### GPU power: measured and asserted

Each round's GPU power is sampled by Hyperloom itself, with read-only
`amd-smi metric --power --mem-usage --clock` every 2 s, only while the round is in its
measured phase (from server ready, and from AIPerf's measured-phase line under
AgentX, until the client exits or the eval starts). It is averaged over the
serving cards: those in the visible-device mask that held at least 10% of VRAM
during that phase, so a TP4 round on an unpinned eight-card host is averaged
over its four cards. The result is the round's `gpu_power.json`, carried as
`gpu_power_avg_w` on the measurement. A round that was sampled but had no
serving card or no power reading stays unmeasured. The benchmark report's
`gpu_monitor` block (`gpu_metrics.json`) is only the fallback for rounds no
sampler ran on: on one node it reads a single card over the whole process
lifetime, boot and idle tail included.

The sampler also reports how the cards ran under their power cap. Each card in
`gpu_power.json` carries its mean graphics clock over the measured phase while
busy (`sclk_mean_mhz`; an XCD in deep sleep, below its `min_clk`, is idle time
and left out), its clock limit (`sclk_max_mhz`, unread when the limit changed
during the phase), and the share of the measured phase it spent held back by its
power limit (`power_throttle_fraction`, from the `amd-smi metric --violation`
accumulators read as each measured window opens and closes; idle stretches in
the phase count toward that share's whole). A window whose counters could not be
read leaves the share unread rather than counting only the windows that were:
every card's when the read failed (counted in `throttle_query_errors`, apart
from the power reads' `query_errors`), one card's when it was missing from
either end or both, or its counters went back or stopped advancing, and every
card's when the sampler was still blocked in `amd-smi` as the round closed. A
serving card power throttled for at least 10% of the phase is listed in
`power_throttled_gpus`, and the round logs a warning naming each such card's
throttled share and mean clock: its throughput was set by the power cap, not by
its clock limit. This is only reported; nothing changes the cards.

Node Power Management (MI350 series) holds the node to its own limit by
lowering every card's power, and the cards' throttle accumulators do not count
that: on an MI350X node held to 2600 W, every card ran near 430 W against a
1000 W cap with a `power_throttle_fraction` under 1%. So while NPM is enabled,
each sample also reads the node's power and the limit firmware enforces on it
(`board/node_power` and `board/cur_node_power_limit` in sysfs; amdgpu publishes
them on one card of an XGMI hive, so on one card of an MI350 node, and on every
card of a node without a hive), and `gpu_power.json` carries a `node` block: `limit_w`
(unread when the limit changed during the phase), `avg_power_w`, `max_power_w`,
`samples` and `limited_fraction`, the share of samples at 99% of the limit or
above. A held node reads at its limit, or above it once every card is down at
its minimum power. When that share is at least 10%, the round logs a warning
naming the limit and the share. The block is `null` on a node without NPM or
with it disabled. Failed reads, and a reading that reports NPM enabled with no
limit, are counted in `node_query_errors`. Where
the node limit equals the sum of the cards' caps (8000 W against eight 1000 W
caps), both limits are reached together and both are reported.

The optimizer never changes power settings during a session. The power cap, the
DPM performance level and the graphics clock's maximum (sclk) decide how much of
the card's throughput is available at what power, so they are part of the
measurement contract, and setting them is privileged and card-wide. By default,
set them with `amd-smi set --power-cap` / `--perf-level` / `-L sclk max` before
launch; `--gpu-power-cap-w`, `--gpu-perf-level` and `--gpu-sclk-max-mhz` then
assert them, the way `--compute-partition-mode` asserts a partition mode: the
session refuses to start (and to resume) if a card it uses is at a different
value. The observed cap, perf level and sclk limit are recorded in the platform
fingerprint whether or not they are asserted. None can be checked on a
multi-node session, so asserting one there refuses. The sclk limit is read from
`amd-smi metric --clock`, where a limit set with `-L` shows; it is a ceiling,
not a locked clock, so under load a card may run below it. The limit is only
ever asserted: `--apply-gpu-power-settings` sets the cap and perf level, not the
limit.

Node Power Management (NPM, MI350 series) is the node-wide counterpart: when it
is enabled, firmware holds the sum of every card's power to a node limit and
moves power between cards, so a card's throughput also depends on what the
other cards draw. Its state and the limit firmware enforces, as
`amd-smi node -p` reports them, are recorded in the same record under `node`
(left out where `amd-smi` has no `node` command, before ROCm 7.2).
`--node-power-management enabled|disabled` and `--node-power-limit-w` assert
them the same way, refusing a launch or resume whose node differs or cannot be
read. They are only ever asserted: NPM is enabled and its limit set on the
BMC, and `--apply-gpu-power-settings` never changes them. The enforced limit
can sit below the one the BMC was asked for, since firmware clamps it to the
node's maximum, so assert the value `amd-smi` reports.

Because NPM moves power between cards, a session on part of an NPM-enabled node
is measured against whatever runs on the rest of it. So when `amd-smi` reports
NPM enabled, a launch or resume whose visible-device mask leaves any of the
node's GPUs out is refused, naming the cards left out, whether or not they are
busy at that moment: an idle neighbour at launch says nothing about the hours
after it. A mask that is set but whose cards cannot be named is refused the same
way. The check is on the mask: a session with no mask counts as having every
card, as everywhere else, even if its tensor parallelism uses fewer.
`--allow-shared-npm-node` measures anyway and records the cards left out as a
list of amd-smi indices under `shared_npm_node` (`null` when the mask's cards
cannot be named). Like `--apply-gpu-power-settings`, the permission is not kept:
pass it again on resume. A node with NPM disabled, or without it, and a
multi-node session are not checked.

### Applying the power settings (opt-in)

`--apply-gpu-power-settings` permits the session to set the declared
`--gpu-power-cap-w` / `--gpu-perf-level` itself, once at launch, and to keep
them for the whole session. The same read-back check then verifies them, and
the fingerprint records them as applied by Hyperloom along with the originals.
It needs root, as `amd-smi set` does: a launch without root refuses (exit 2)
before it writes any power-settings record, with the cards unchanged. It also
refuses, with nothing left changed:

- without an integer `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` mask. It
  sets exactly the masked cards, never the whole host. A mask number is a ROCm
  number, not an amd-smi index: ROCm numbers cards in KFD node order and amd-smi
  in PCI order, and on MI300X/MI355X boards the two differ, so each masked card
  is matched to amd-smi through the KFD node id `amd-smi list` reports, with
  `HIP_VISIBLE_DEVICES` (or `CUDA_VISIBLE_DEVICES`) indexing into what
  `ROCR_VISIBLE_DEVICES` left; `HSA_VISIBLE_DEVICES` and `GPU_DEVICE_ORDINAL`,
  which ROCm ignores, are not read. The read-only assertion, the per-GPU budget
  and the power sampler use the same translation. A mask that is set but empty,
  is not written as plain comma-separated numbers (GPU UUIDs, spaces, `;`,
  leading zeros, which HIP reads differently), repeats a card or names
  one that does not exist is refused;
- on a card not in SPX compute-partition mode: a cap is per socket, so it would
  change the sibling partitions, which other sessions may use;
- when the declared cap is not a whole number of watts, which `amd-smi set`
  does not take;
- on multi-node;
- on a masked card `amd-smi` does not report, or whose current value cannot be
  read, since nothing could restore it;
- on a card that already holds more than 2 GB of VRAM before the session
  starts (someone else's resident model, which a card-wide cap would change
  too), or whose VRAM use cannot be read;
- when a set fails.

Only the declared settings are set, and only the settings they change are put
back. A perf-level apply is refused before anything is set on a card that
starts at a level a restore could not bring back: `determinism` (its clock
soft-max is set with `amd-smi set -d`, which a level restore does not carry) and
`manual` (its hand-tuned clock limits are reset by setting another level). A
setting a card already has is left alone instead: neither set nor put back, so
a card asked to stay at its level needs no refusal. A cap-only apply is allowed
on such cards. After every restore the cards are read
back, and that reading decides: a card not back at its recorded values is
reported and its record kept, so the next launch warns or recovers it, while
each card that reads back at its originals has its record cleared, even if a
restore command failed (an apply refused for lack of privilege changed
nothing).

The originals are written to a per-card record under
`/var/tmp/hyperloom/gpu_power_settings/` before anything is set. That path is
host-local and fixed, whatever `HYPERLOOM_RUNTIME_DIR` or `USER_DATA_PATH` say,
so every session on the host finds the same records. Inside a container it is
the container's own `/var/tmp`, so mount the host's `/var/tmp/hyperloom` into
every container that applies settings, or sessions in different containers will
not see each other's records. Each record is stamped with the host's boot id, so
one written before a reboot is ignored. The session holds an exclusive `flock`
on each record until it restores the card, which keeps two sessions from setting
the same card. A restore sets whatever a record says, as root, so the record
directory and its parent are writable only by root, and records are readable by
every account (for the warning below) but writable only by root; a launch with
the flag creates both directories at mode 0755, and sets an existing one root
owns (such as a mounted host directory) to 0755 unless other accounts can write
it, which is refused instead. A record that is a symlink or a hard link, or a
record or directory owned by another unprivileged account or writable by others,
is refused rather than read. Any local account can still make a flagged launch
refuse: by holding a lock on a card's record, which reads as a live session
holding the card, or by creating `/var/tmp/hyperloom` first, which is then not
root's and refused. Both fail safe, with nothing set, until an administrator
removes them.

The originals are restored when the session ends: at a stop reason, an operator
stop, an error exit, or a SIGTERM / SIGHUP at any point after the cards were
set, including startup and the close sequence. A stop signal during the restore
itself is ignored so it cannot cut it short. A SIGKILL (an OOM kill, or a
scheduler escalating after its grace period, such as Slurm's `KillWait`) cannot
be caught: the cards stay at the applied settings, and the record is left with
its lock free. The next launch on that host that passes the flag with those
cards in its mask restores them before setting anything; a launch without the
flag warns that the cards were left changed and names their original values.
Records are judged setting by setting: a setting that still reads at the applied
value is put back (a restore that put back the cap but not the level gets the
level restored), while a setting the card no longer reads at (an operator put it
back, or set a value of their own) is left alone. A setting whose reading is
missing is not taken as moved: the launch with the flag refuses with the record
kept and names the record file to remove once the card has been checked by hand,
and the warning reports the card as possibly still changed. A record none of
whose settings still reads as applied is stale: it is cleared, not restored, and
not warned about. Give a scheduler's grace period enough time for the session to
close. On resume the flag must be passed again, since the settings were restored
when the previous leg exited. A resume first stops the serving
processes its previous leg left running (a leg that was SIGKILLed leaves its
server up), so the check for other tenants' models does not refuse on the
session's own server.

### Power budget (constraint on KEEP)

Two budgets bound a round's GPU power, for the serving frameworks:

- `--max-power-w N` caps the serving GPUs' total: the sum of each serving
  card's mean power over the measured phase (for example 2800 W for four cards
  budgeted at 700 W each).
- `--max-per-gpu-power-w GPU_ID W`, repeated once per card, caps one card by
  its physical `amd-smi` index. A serving card with no entry is bound only by
  the total, and a listed card that did not serve a round is not judged on it.

The session refuses to start (exit 2) if a listed GPU is not one it can use
(its visible-device mask, or every card `amd-smi` reports when there is no
mask), or if the per-GPU limits add up to more than `--max-power-w`. Either
budget is refused, at launch and on resume, when no round could be measured: on
a multi-node session (the sampler reads only its own host's cards), on a
scriptable framework (`xdit`, `custom`: the sampler follows a server's measured
phase, and a scriptable round runs no server), with
`HYPERLOOM_GPU_POWER_SAMPLING=0`, or without `amd-smi` on `PATH`. Resuming with
different values is refused.

Both ride the same verdict as `--max-latency-ms`: a candidate that would
otherwise KEEP is a REVERT with `veto_reason` `gpu_power_budget_exceeded` when a
card is over its own limit, `power_budget_exceeded` when the total is over, or
`power_unmeasured` when its round reported no per-GPU power (fail closed).
Every KEEP decision point reads that verdict, so a lane reverts its own
over-budget change; the correctness-fix route, which can KEEP a verified fix
after that verdict reverted it, applies the same budgets itself. If the baseline itself breaks a budget, or reported no
power, the run stops with `baseline_over_power_budget`. The reading is the
measured-phase per-GPU figure described above, carried as `gpu_power_by_gpu_w`;
only Hyperloom's own sampler reads every serving card, so a round it did not
sample counts as unmeasured. So does a round in which a card that may be
serving could not be read: one holding a model with no power reading (`N/A`, or
a 0 W or negative socket power, which no working sensor reports), or one in
scope (the visible-device mask, or every card `amd-smi` reported) whose VRAM was
never read. Such cards are listed as `unread_gpus` in `gpu_power.json`, since a
card missing from the readings would drop out of the total and escape its own
limit. A GEAK replay boots its own server once per
replica, so its phases are read from the server logs the replay writes, with
the same markers, and each new log counts as a fresh boot.

The budgets are on average draw at the deployed cap, the shape of a
power-provisioned rack where cards keep their full cap for headroom: the total
is the rack's share, and a per-GPU limit covers a card with less headroom, such
as one in a hotter slot. To tune
for a lower hardware cap instead, set that cap with `amd-smi set` and assert it
with `--gpu-power-cap-w`; no ceiling is needed.

### Runnable gate (earned KEEP)

A verified build does not KEEP on artifact verification alone. After a
build's artifacts verify, the Coordinator runs a launch probe: it boots the
actual model with the built runtime through the same runnable-decision gate
the authored-patch lane uses. Only a runtime that actually launches — and
passes minimal correctness — earns KEEP. For an eval-origin trigger the gate
additionally re-runs the accuracy eval against the captured contract and a
KEEP requires the accuracy to meet the floor; the KEEP is finalized only after
a genuine baseline revalidates it. Otherwise the build reverts, or, if the
boot advanced past the original failure to a new or deeper gap, the loop
advances to the next round to repair that gap.

### Discovery-driven build refs

Candidate PR refs discovered by the framework-agent feed directly into the
build: a PR reference becomes a checkout of that PR's head, not just a
released-tag autoselect. This makes support that exists only in an
unreleased PR or branch reachable. When a discovered PR ref drives the
build, the source PR URL is recorded as build provenance in the session
breakdown's `installed_versions` map (`source_pr_url`); see the
[`enablement` section](../reference/session-breakdown.md#enablement--admission-round-lifecycle-builds--attempt-runtimes).

### Novelty and crash safety

- **A build answers no round.** A build runs beside the authoring round
  that asked for it, and that round is settled by its own integrate. A
  failed build boots nothing, so it settles no round and stacks nothing; its
  failure reaches the next round's mandate. An identical build — same
  component, ref, GPU arch, and build command — is never run twice.
- **Crash / resume.** An in-flight build is tracked by a durable sentinel,
  so a crash or resume reclaims the running build or cleans up the orphaned
  one rather than leaking it.

## FRAMEWORK_AGENT — the optimisation phase

One phase, two levers worked in parallel:

- **Configuration** — the `explore` action runs server-argument and
  environment grids through the canonical `explore_search` ledger. Nothing on
  disk changes, so a revert is a non-composition.
- **Source and upstream** — `integrate_patch` lands every patch, and
  `patch_source` says where the diff came from: `specialist_authored` for one
  a specialist wrote, `upstream_pr` for a diff fetched from an upstream PR
  that a `candidate_discovery_specialist` found, ranked and judged. Both
  serialise on the `workspace_mutation` lane: one landing at a time,
  independent of how fast proposals arrive. A candidate whose every patch
  reverse-applies cleanly, and that changes nothing else, is not benchmarked:
  it ends `already_present`, and a candidate with an upstream PR identity is
  recorded so in the framework KB.

`specialist` serves both levers — investigation, patch authoring, and
candidate discovery are all dispatches of the one specialist action. An
operator can add one domain of their own, `custom_specialist`, whose focus is
the text they supply at launch. The Coordinator alone dispatches it, once per
macro cycle on entering this phase; it and the grid benching its proposals run
ahead of all other queued work, and a plateau exit waits for both within the
phase budget.

With a [PrimaTune](../reference/primatune-predictor.md) endpoint configured,
the phase also asks that predictor for a launch recipe once per decision point:
the macro cycle, the stack depth, and the number of profiles taken. At entry it
asks after the phase's own first dispatch and, in `active` mode, waits up to
five minutes for the answer before Orchestration's first turn, so the first
grid can carry it. Every tick then files an answer that has arrived and asks
again once a KEEP or a new profile has moved the decision point. The answer
joins the untested-proposal queue ahead of the specialists' rows and counts as
the phase's work; a source change it proposes is offered as a mandate a patch
specialist takes by id. `shadow` asks and only logs; without an endpoint, or in
`off`, nothing is asked and nothing waits.

The phase advances to KERNEL_AGENT only when **both** levers are dry
(`optimize_no_more_leverage`). Either arm going quiet raises
`switch_bottleneck` so the next macro-cycle steers off this bottleneck,
without abandoning the lever that is still paying. It also exits when its
phase budget is spent.

After each KEEP the runtime revalidates the full stack end to end, so the
reported `cumulative_gain_validated` always comes from a measurement taken
with every accepted change applied.

## KERNEL_AGENT

The `KERNEL_AGENT` phase is the bridge to kernel-agent work. Orchestration might
send kernel requests, but the Coordinator owns the request handlers and safety
gates.

What the phase actually does depends on the kernel backend, and the branches
look very different from Orchestration's side:

- **Default (`geak`)**: entering the phase enqueues one Coordinator-owned
  `kernel_agent` task, which holds `server_lifecycle`, `workspace_mutation` and
  `benchmark_lane` for the whole pipeline. Under GEAK it runs a single
  whole-pipeline GEAK e2e run, which then sets the
  `kernel_no_more_leverage` exit. When the run produces no win, the phase machine
  exits immediately and the phase closes without Orchestration ever
  taking a turn in it.
- **On a GEAK win**, the same `kernel_agent` task re-measures the candidate on
  the orchestrator's own harness under the lanes it already holds, and writes
  the headline from that measurement before it returns. A candidate whose
  launch config, applied over `current_best`, launches what `current_best`
  already runs, and that carries no loadable overlay, is settled as
  `no_material` without being measured. KERNEL leaves once the task settles, or
  when its phase budget runs out.
- **Forge (`KERNEL_OPT_BACKEND_ORDER=forge`)**: the phase runs the deterministic
  KERNEL-entry ladder — GEMM tuning, then the fusion lane, then the kernel
  rewrite controller, which selects its own operators and publishes patches
  Hyperloom re-measures end to end.

See [Kernel optimization execution path](../reference/kernel-execution-path.md) for the
entry-hook branch order.

The LLM-proposable actions for KERNEL_AGENT phase (`allowed_actions_for(KERNEL_AGENT)`)
are:

- `integrate`
- `roofline`
- `profile`

`kernel_agent` is Coordinator-internal; PolicyGate rejects it with
`rule="coordinator_managed_action"` if an LLM proposes it.

Within the kernel-agent request channel, the handler dispatches request kinds
such as `trace_analyze` and `integrate`
(`request_handlers.py`); these are handler kinds, not phase actions.

Kernel-owned results are recorded separately from non-kernel action
attempts. A KEEP must be integrated before the run can proceed to final
reporting, and hot reusable kernels above the configured threshold must
be attempted or explicitly rejected before report can close the run.

## SWEEP

SWEEP measures the optimized stack against the baseline across a
concurrency ladder, one arm each, and produces the throughput-vs-
interactivity curve. The ladder is sized for the workload: powers of two
down from 256 for a synthetic run, `1,4,8,10,14,20,28` for an agentic one,
where a request carries orders of magnitude more prompt and the same card
saturates far lower. Override either with `--conc-sweep-concs`. Under
native Magpie AgentX, concurrency is fixed for the session and
`--enable-conc-sweep` is rejected at startup. Start a separate session for each
CONC. The independent MLPerf backend keeps its optional sweep, disabled by
default. See the [AgentX workflow](../how-to/optimize.md#optimize-an-agentx-workload)
for supported workload controls.

Each arm runs the session's own CONC first, then the rest of the ladder
cheapest first: low to high for a synthetic workload, whose rungs send requests
in proportion to CONC, and high to low for an agentic one, whose rungs replay a
fixed set of trajectories on more lanes the higher CONC goes.
A pair needs both arms, so the baseline arm runs first: its first rung may use
the whole budget, and each later rung must fit in half of what was left when
the arm started. The optimized arm then runs only the concurrencies the
baseline reached. A budget that runs out therefore drops whole pairs, and the
anchor pair is the last one it drops. A rung is not started once the time left
is shorter than the last benchmark measured after the anchor, since none past
it runs faster. SWEEP dispatches the sweep only when the time it can grant holds
one pair: two of the session's measured baseline rounds.

Results update `last_conc_sweep` and feed the final report and breakdown.
A ladder that fails closes the phase as `sweep_failed`; one that finishes
(`sweep_done`) loops back or closes on the reason below. While the sweep it
enqueued is still running, SWEEP's own budget and cap wait for the budget the
sweep was granted (plus its lease grace); the session deadline does not. A
sweep the orchestrator cancels stops its ladder at once and records
`cancelled`, so the phase closes on whatever ordered the cancel rather than on
`sweep_failed`.

When SWEEP does not loop back, the close names what refused the next
macro-cycle: `target_reached`, `global_converged` (consecutive cycles without
gain, or every direction saturated) or `time_exhausted` (less session time
left than the reloop floor). The refusing rule is kept in the transition
evidence as `reloop_blocked`. A run without `FRAMEWORK_AGENT` has no next
cycle to refuse, so it closes on the ladder's own reason unless the target was
reached.

## CLOSE

CLOSE drains the final artifacts:

1. `report` renders the operator-facing final report.
2. `session_breakdown` writes the downstream JSON contract.
3. The CLI finally-block writes a safety-net breakdown if the close
   sequencer did not already finish cleanly.

The close path must be idempotent because sessions can end through a
normal phase transition, a wall-clock deadline, an operator interrupt, or
a resumed run.

## Orchestration prompt model

Every Orchestration turn is stateless: the Coordinator composes one
unconditional full state projection (mission, `SharedState`, gaps,
warm-start, scores, and the inbox events since the last turn), so the
turn never depends on what an earlier turn happened to remember.

- **Cycle directive**: While SWEEP is open and another macro-cycle is
  still feasible, the Coordinator appends one handoff request to an ordinary
  Orchestration turn. The reply, plain text and no intent required, is stored
  as `orchestration_memory` (`next_cycle_directive`, `for_cycle`,
  `parse_error`) and reseeds the next cycle's system prompt once FRAMEWORK is
  entered; a turn that produced nothing usable leaves the directive empty.
- **Context tools**: A read-only MCP surface lets the agent pull what the
  projection leaves out — finished outcomes (`get_recent_outcomes`),
  in-flight work (`get_running_tasks`), failure packets (`get_failure` /
  `get_variant_failures`), reference docs (`read_reference`), the raw
  `analysis.md` (`show_analysis_md`), denial history (`why_denied`) — plus
  Transports without MCP tools get the projection only.
- **Resume**: On resume the projection is rebuilt from
  `orchestration_memory` plus the authoritative `SharedState` facts —
  not by replaying a non-deterministic transcript.
- **Write path**: All write actions flow through `emit_intent` → the
  Coordinator's intent handler, so Critic review, the accuracy gate,
  and PolicyGate's invariants (path sandbox, resource leases,
  data dependencies, single-writer rules) apply to every turn.
  Repetition is checked against state — the tested-variant ledger and
  the action-failure log — not against agent recall.

Critic is likewise reactive and stateless per tick. Runtime RCA and automatic
supervision are not roles in this loop. Stopped sessions require an explicit
operator `--resume-from` decision; `recover` reconstructs artifacts offline
or, with `--confirm-scope-gone` / `--confirm-stopped`, records the cleanup
confirmation a refused resume asks for (see
[Scenario A](../reference/operations.md#scenario-a-pod-was-oom-killed-mid-session)).

## Feedback loops

The loop adapts through facts, not through retired score tables:

- `SharedState` carries current best, stack entries, phase history,
  action attempts, kernel attempts, framework-agent progress, and warnings.
- `RecipeKB` records durable lessons and pitfalls for future sessions.
- Critic verdicts gate risky patches and framework candidates.
- PolicyGate blocks unsafe paths, invalid envelopes, and actions whose
  structural requirements are not met before they mutate runtime state.
  Phase fit is enforced at admission: a proposal or delegate whose action
  the current phase does not offer is refused (`phase_incompatible`)
  before a task exists, and the phase transition drops incompatible tasks
  still queued from the previous phase.
- The baseline anchor every gain is measured against moves only to a
  higher re-baseline (or a hot pass correcting a cold anchor), and only
  until a cumulative gain has been validated
  against it; after that a re-baseline is recorded (`anchor_held`) but the
  anchor stays, except for an ENABLEMENT revalidation, whose patch changed
  what the anchor measures.

## What is retired

These names shouldn't appear as live positive instructions in prompts
or docs:

- `setup`
- `classify`
- `backends`
- `params`
- `validate_stack`
- `select_kernels`

They can remain only in migration readers, archived breakdown aliases,
or explicit rejection tests.

## Artifacts to inspect

For a finished or interrupted session, start with:

- `manifest.json`
- `state.json`
- `storage/coordinator.db`
- `runs/<action>/<task_id>/`
- `reports/`
- `session_breakdown.json`

For reports and dashboards, `session_breakdown.json` is the external
contract. Its producer code lives under
`src/hyperloom/inference_optimizer/breakdown/`,
and its consumer-facing shape is documented in
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).
