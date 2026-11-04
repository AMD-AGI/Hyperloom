# The dossier: what it is for, and the rules every part follows

The dossier plugin's skills share this page:
- **`dossier:read`** finds a dossier and reads it;
- **`dossier:discover`**, **`dossier:measure`**, **`dossier:reference`** and **`dossier:directions`** write its parts;
- **`dossier:update`** rewrites them in rounds.

The dossier's skeleton is [template.md](template.md), beside this page. Its checker is
`python3 -m arbor.dossier.check <dossier dir>` (run from the Arbor tree).

## Why a dossier

Audience: an agent (or a person) about to optimize a model on a GPU target,
or asked what to optimize. The dossier answers, in order and with sources,
what the campaign would otherwise rediscover. In one DeepSeek V4.1 Flash
campaign, each of these cost a plan its result or its time:

- a knob aimed at a fused kernel that never ran;
- a kernel-boundary win that graph capture had already amortized;
- "C8" turned out to put 1.1 to 1.5 requests in flight, at 6 to 18 rows per
  step, not 48;
- every C32 profile across several plans captured the client's warmup;
- one plan's best possible gain sat an order of magnitude below its noise floor;
- AITER's tuned MoE table had no row for the model's shapes, so every token
  tier ran the untuned fallback.

## Ask the user first

Some inputs only a person has. When you write or refresh a part that needs
one, ask for all of them in one message at the start, then continue without
waiting; a part that needs an answer stays pending. In a run with no one to ask (a campaign), write the questions in the
dossier's index under "Open asks" and go on.

1. **Reference-platform traces** (optional; they help a lot): traces of the
   same model and workload on B200 or another NVIDIA GPU, or access to a node
   to take them. Part 9 lists what is needed.
2. **Earlier optimizations, where no public record exists.** For training,
   or an internal framework, ask where AMD's earlier work on this model or
   workload is recorded: pull requests, branches, MLPerf submission notes,
   internal reports. For inference, part 10 reads InferenceX first and asks
   only for what is missing. If nothing is available, part 10 says so and is
   skipped.
3. **The target**, if no one has stated it: the metric, the operating points,
   what counts as a win (part 4).

## Rules

- **Parts in order.** Each states its question, what it contains, where the
  evidence comes from, and when it is done. Parts 1 to 5 and 10 need no GPU
  if the evidence exists; 6 and 7 need a node; 8 needs 3 and 7; 9 needs the
  user; 11 is written last, from the others only.
- **The profile is taken while the dossier is written.** Where the target
  hardware is available (a campaign holds a node), part 7 is a TraceLens profile
  (the `profiler` skill) on the served tree, taken right after parts 1 to 6
  and before parts 8 and 11 are written. A part 7 with a plan and no results
  leaves the dossier unwritten: parts 8 and 11 are drafts until it lands, and
  no optimization starts from them. The profiler's checks run with it; a
  check not run is the next work, not a footnote. Model M campaign B wrote all twelve parts before any capture and ranked
  part 11 by evidence instead of ceilings; its first profile, one iteration
  later, re-ranked it.
- **Re-profile when the tree or the deployment moves.** After a KEEP, and
  after a change of flags or parallelism (expert parallelism, say), part 7 is
  stale; profile again before part 11 is re-ranked from it.
- **Every number has a source and a label**: node or host, GPU, seed,
  concurrency, n. A number without a source is removed.
- **Every part has a status**: written, partial, missing or not available,
  with what would fill it, its date, and the pins and workload it depends on.
  When one of those moves, the part is stale, and it is refreshed before a
  campaign relies on it.
- **One GPU type per claim.** MI350 results never stand in for MI355X, and a
  B200 result is a lead for an AMD claim, never evidence for one.
- **The dossier informs; it never gates.** Gates are the operator's.
- **Where it lives**: in the project's repository, `docs/dossier/<model>-<workload>-<gpu>/`,
  from [template.md](template.md). A campaign gets a digest of it as pack
  knowledge (`packs/<pack>/knowledge/`, which Arbor renders into the agent's
  instructions), citing the dossier; parts 8 and 11 feed the pack's roadmap.
  The campaign's knowledge collection shares it (`--dossier`), which is how
  campaigns run from other repositories find it.
