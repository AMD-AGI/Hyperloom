---
name: reference
description: "Write the dossier's part 9, the reference platform (B200, or another NVIDIA GPU): what a faster platform does differently on the same model and workload, at matched operating points, from traces or access the user supplies. Optional, and opt-in: without the user's traces the part is 'not available'. Run by Arbor in the map.measure chapter when a pack lists it, or by hand."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: map.measure
    after: ["dossier:measure"]
    produces: ["{dossier}/09-reference.md"]
    reuse_key: [identity]
    gpu: false
---

# Reference: the dossier's part 9

**In a campaign, Arbor runs this skill in a turn of its own.** Its argument is the path of a brief: read the brief
first. **By hand,** the directory is `docs/dossier/<model>-<workload>-<gpu>/`.

**Its inputs are the user's:** reference traces, or access to take them. [The rules](../../shared/rules.md)' asks,
relative to this skill's directory, say how to ask. In a campaign with no one to ask, write the part `not available`,
with what would fill it, and put the ask in the index's "Open asks". Then end your reply with a `LEDGER:` FACT naming
the file.

### 9. Reference platform (B200): optional, with the user

- **Answers:** what a faster platform does differently on the same model and
  workload.
- **Needs the user** (asked up front). Without traces or access, the part is
  "not available", and parts 8 and 11 say what they could not compare.
- **What to get:** the reference recipe (for inference, InferenceX's script
  for that GPU: image, model revision, flags); the same workload settings
  (dataset snapshot, seed, concurrency, duration); a PyTorch-profiler trace on CUDA (TraceLens
  reads it there too) or an Nsight Systems trace; the client's
  metrics.
- **Compare:** at matched operating points (rows per step, context, requests
  in flight), because one concurrency puts a different load on each platform; time per step by module; which
  kernel serves each op (into part 8); framework choices (graph mode, capture
  sizes, backends, where the KV cache and other tables live, warmup).
- **Hardware gaps:** name the gap before porting an idea: native number
  formats, matrix instruction shapes, wavefront width (64 on gfx950), on-chip
  memory per compute unit, asynchronous copy engines, interconnect.

## Done when

- **Each part you wrote meets its own "Done when",** above, or is marked partial or not available, with what would fill
  it.
- **Its file holds real content:** `python3 -m arbor.dossier.check <dossier dir> --json` lists it `ok`. That means at
  least 300 words and 10 numbers outside headings, template labels and placeholders. Parts the other dossier skills
  write may fail until they run.
- **Every number has its source and label.**
- **The index lists each part you wrote** with its status, sources, date and staleness key, and the open asks.
