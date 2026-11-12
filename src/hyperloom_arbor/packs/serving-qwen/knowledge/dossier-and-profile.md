# The dossier and the profiler in this campaign

This campaign runs two skills by default: `dossier-v1` and `profiler`. A pack derived from this one
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
| Lookup | `bash .claude/skills/dossier-v1/find-dossier.sh` |
| Template | `.claude/skills/dossier-v1/template.md` |
| The served tree | the config's `workspace_root` (the framework's installed package, made a git repository) |
| How profiles are taken | the `profiler` skill: the host's profile first, then TraceLens on a PyTorch-profiler trace |
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
3. **Part 7** (profile) with the `profiler` skill: the host's profile first when there is one, then
   TraceLens on a PyTorch-profiler trace of prefill and decode steps at the operating point. Stop
   any traced server before DONE.
4. **Part 8** (kernel usage) from part 7's kernels, AITER's source in the container, and the CUDA
   path if the model has one.
5. **Part 10** needs people: earlier AMD optimizations (an InferenceX recipe for this model, if
   one exists). Record what is missing as an open ask, not a blocker.
6. **Part 11** (directions), last, from parts 1–10 only. Each direction gets a map ID and becomes
   the campaign's work.

## Checks this campaign runs first

- The rendered program you read lists the skills under "Skills". If it does not, say so in your
  first DONE line: the skills did not load.
- TraceLens is installed where the profile is read (`TraceLens_generate_perf_report_pytorch_inference`).
  If it is not, record why as an UNKNOWN and profile nothing until it is.
