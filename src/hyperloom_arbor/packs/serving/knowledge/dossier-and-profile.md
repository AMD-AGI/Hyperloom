# The dossier and the profiler in this campaign

This campaign runs two skills by default: `dossier` and `profiler`. A pack derived from this one
may add `hardware`, `serving-playbook` and `kernels`. Read this before your first iteration.

Whether a dossier exists depends on the model, the workload and the GPU: look it up (below)
before writing anything. If the lookup finds one, read its index, refresh only the parts that are
stale for the tree you serve (a dossier written for another tree says which), and work from its
part 11. If it finds none, writing it is the campaign's first work, in the order below.

**If the campaign was given its directions** (a host's plan, or an operator's map), the map is set:
work those directions, and write or refresh only the dossier parts they need.

## Where things are

| What | Where |
| --- | --- |
| Dossiers | `docs/dossier/<model>-<workload>-<gpu>/` in this Arbor checkout (your working directory); `README.md` is each one's index |
| Lookup | `bash .claude/skills/dossier/find-dossier.sh` |
| Template | `.claude/skills/dossier/template.md` |
| The served tree | the config's `workspace_root` (the framework's installed package, made a git repository) |
| TraceLens | installed with the environment; reads PyTorch-profiler traces (the `profiler` skill) |
| Profiles | `<session>/profiles/<name>/PROFILE.md` and their small files |
| The operating point | `knowledge/operating-point.md` |

## Order, when the dossier is missing

Each step is a LEARN iteration ending with LEDGER records (pack.yaml §3). Write the parts that
need no GPU first; one iteration can write several.

1. **Parts 1–3** (model, system, code reading): the model's `config.json` and code in the
   container, the launch flags in the config, and the server log. Part 3 is the served path
   through the model's code, method by method.
2. **Parts 4–6** (target, workload and its CPU work, baseline): pack.yaml §5,
   `knowledge/operating-point.md`, the bench's own output, and the baseline.
3. **Part 7** (profile) with the `profiler` skill. The host's profile comes first, when there is one;
   then a TraceLens profile of prefill and decode steps at the operating point, graphs on, with an eager
   trace (labelled eager) when op attribution is needed. Stop any traced server before DONE.
4. **Part 8** (kernel usage) from part 7's kernels, AITER's source in the container, and the CUDA
   path if the model has one.
5. **Parts 9 and 10** need people: a reference on another GPU, and earlier AMD optimizations
   (an InferenceX recipe for this model, if one exists). Record what is missing as an open ask,
   not a blocker.
6. **Part 11** (directions), last, from parts 1–10 only. Each direction gets a map ID and becomes
   the campaign's work.

## Checks this campaign runs first

- The rendered program you read lists the skills under "Skills". If it does not, say so in your
  first DONE line: the skills did not load.
- TraceLens answers (`TraceLens_generate_perf_report_pytorch_inference --help`). If it does not, record why as
  an UNKNOWN and profile nothing until it is fixed.
