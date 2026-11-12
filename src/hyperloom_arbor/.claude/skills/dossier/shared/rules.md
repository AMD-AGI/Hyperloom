# The dossier: what it is for, and the rules every part follows

The dossier plugin's skills share this page:
- **`dossier:read`** finds a dossier and reads it;
- **`dossier:discover`**, **`dossier:measure`** and **`dossier:directions`** write its parts;
- **`dossier:update`** rewrites them in rounds.

The dossier's skeleton is [template.md](template.md), beside this page. Its checker is
`python3 -m arbor.dossier.check <dossier dir>` (run from the Arbor tree).

## Why a dossier

Audience: an agent (or a person) about to optimize a model on a GPU target,
or asked what to optimize. The dossier answers, in order and with sources,
what the campaign would otherwise rediscover. Each of these can cost a plan
its result or its time:

- a knob aimed at a fused kernel that never runs;
- a kernel-boundary win that graph capture has already amortized;
- a concurrency setting that puts only one or two requests in flight, at a
  handful of rows per step, not the client's cap;
- a profile that captures the client's warmup instead of the measured load;
- a plan whose best possible gain sits far below its noise floor;
- a tuned kernel table with no row for the model's shapes, so every token
  tier runs the untuned fallback.

## Ask the user first

Some inputs only a person has. When you write or refresh a part that needs
one, ask for all of them in one message at the start, then continue without
waiting; a part that needs an answer stays pending. In a run with no one to ask (a campaign), write the questions in the
dossier's index under "Open asks" and go on.

1. **Earlier optimizations, where no public record exists.** For training,
   or an internal framework, ask where AMD's earlier work on this model or
   workload is recorded: pull requests, branches, MLPerf submission notes,
   internal reports. For inference, part 10 reads InferenceX first and asks
   only for what is missing. If nothing is available, part 10 says so and is
   skipped.
2. **The target**, if no one has stated it: the metric, the operating points,
   what counts as a win (part 4).

## Rules

- **Parts in order.** Each states its question, what it contains, where the
  evidence comes from, and when it is done. Parts 1 to 5 and 10 need no GPU
  if the evidence exists; 6 and 7 need a node; 8 needs 3 and 7; 11 is
  written last, from the others only.
- **The profile is taken while the dossier is written.** Where the target
  hardware is available (a campaign holds a node), part 7 is a TraceLens profile
  (the `profiler` skill) on the served tree, taken right after parts 1 to 6
  and before parts 8 and 11 are written. A part 7 with a plan and no results
  leaves the dossier unwritten: parts 8 and 11 are drafts until it lands, and
  no optimization starts from them. The profiler's checks run with it; a
  check not run is the next work, not a footnote. Parts written before any
  capture rank part 11 by evidence instead of ceilings, and the first profile
  re-ranks it.
- **Re-profile when the tree or the deployment moves.** After a KEEP, and
  after a change of flags or parallelism (expert parallelism, say), part 7 is
  stale; profile again before part 11 is re-ranked from it.
- **Every number has a source and a label**: node or host, GPU, seed,
  concurrency, n. A number without a source is removed.
- **A fact from knowledge search keeps its citation**, where the part uses it, exactly as `find`, `get` or `ask` gave
  it: `<source>@<pin>:<path>#L<start>-<end>`, the pinned version that was read, never a page's path alone.
  `arbor.dossier.check` resolves each against the registry's pins and fails one that does not resolve. In a
  campaign that loads `knowledge:search`, it also fails a page behind search named by its path.
- **Every part has a status**: written, partial, missing or not available,
  with what would fill it, its date, and the pins and workload it depends on.
  When one of those moves, the part is stale, and it is refreshed before a
  campaign relies on it.
- **One GPU type per claim.** MI350 results never stand in for MI355X.
- **The dossier informs; it never gates.** Gates are the operator's.
- **Where it lives**: in the project's repository, `docs/dossier/<model>-<workload>-<gpu>/`,
  from [template.md](template.md). A campaign gets a digest of it as pack
  knowledge (`packs/<pack>/knowledge/`, which Arbor renders into the agent's
  instructions), citing the dossier; parts 8 and 11 feed the pack's roadmap.
  The campaign's knowledge collection shares it (`--dossier`), which is how
  campaigns run from other repositories find it.
