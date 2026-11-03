# Hardware counters with rocprofv3

ruppell records when each kernel ran and what launched it, not what it did
inside (its docs/limitations.md: "No hardware counters"). To learn why a
kernel is slow (bytes fetched and written, cache behaviour, occupancy), use
`rocprofv3`'s counter collection on the kernels and shapes ruppell's census
named. Take times from ruppell or an untraced bench, and counts from
`rocprofv3`: a counter pass changes kernel timing.

## Where to collect

- **A kernel bench at the served shapes**, by default: one process, one GPU,
  the op called with the shapes and dtypes the census recorded, warmed up,
  with the server's tuning tables and environment (for example an
  `AITER_CONFIG_FMOE` overlay). The runner's MoE tier study and its DLRMv4
  weight-gradient study collected counters this way (`tests/p30/` and
  `tests/d4/` in the ruppell repository, not vendored).
- **Under the full workload** only when the shapes cannot be reproduced
  outside it. A server under `rocprofv3` must be stopped process by process,
  with its output on local disk, or the GPU processes end without writing
  their files (ruppell's docs/limitations.md, "Stopping a served model under
  `rocprofv3` takes care").

## Rules, each learned the hard way

1. **One counter per pass, unless you have seen the group collected
   together.** On gfx950, `FETCH_SIZE` and `WRITE_SIZE` could not be
   collected in one pass, and `rocprofv3` then hung (ruppell's DLRMv4 study,
   job 59733, 2026-10-06). `rocprofv3 -L` lists the counters an arch offers.
2. **Every pass under a timeout** (`timeout -k 20 <seconds>`), or a hung pass
   holds the node.
3. **On the system ROCm runtime.** `rocprofv3` 7.2.1 could not run against
   the profiler SDK bundled in a torch wheel, with or without ruppell
   (`ruppell/docs/milestones/dlrmv4.md`, jobs 57971 to 58090). Run counter
   passes on the image's own torch, with any overlay removed from
   `PYTHONPATH` and `LD_PRELOAD`.
4. **A writable temp directory** on node-local disk: `ROCPROF_TMPDIR`.
5. **Permissions.** `rocprofv3` needs ptrace: `ptrace_scope=0` on the host, or
   a privileged container (`arbor/knowledge/tracelens-usage.md`).
6. **Only the kernels you study**: `--kernel-include-regex`, which also keeps
   a pass short.
7. **Keep the evidence**: the raw CSVs, the exact script and its arguments,
   the counter list from `-L`, and the versions (GPU and arch, ROCm,
   `rocprofv3`, image digest).

## A pass

```bash
NATIVE="cd /tmp && unset PYTHONPATH LD_PRELOAD &&"      # the image's own torch and runtime (rule 3)
docker exec $C bash -lc "$NATIVE rocprofv3 -L" > counters_available.txt
for CN in FETCH_SIZE WRITE_SIZE; do                     # one counter per pass (rule 1)
    docker exec $C bash -lc "$NATIVE ROCPROF_TMPDIR=$W/rptmp timeout -k 20 300 \
        rocprofv3 --pmc $CN --kernel-include-regex '<kernel regex>' \
        --output-format csv -d $W/counters/$CN -o %pid% -- python3 bench.py <shape arguments>" \
        > counters_$CN.log 2>&1
    echo "$CN rc=$?"                                    # 124: the pass timed out
done
```

## Reading them

- **Bytes moved per call**: `FETCH_SIZE` and `WRITE_SIZE` are in kilobytes
  per dispatch. Set them beside the least the op must move (its weights,
  activations and outputs at the served dtypes); a large excess is
  over-fetch or poor reuse.
- **Achieved bandwidth**: bytes over the kernel's time at the same shape,
  with the time from ruppell or the untraced bench, never from the counter
  pass. Compare it with what the device achieves (the hardware skill's
  measured table, `.claude/skills/hardware/SKILL.md`), not the datasheet
  peak, and say where the number comes from.
- **Per shape and per kernel variant**, never pooled across them: a tier or
  tile change moves both the bytes and the time.
