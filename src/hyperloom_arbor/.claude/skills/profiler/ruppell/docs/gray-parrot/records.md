# gray-parrot's records: the contract both platforms write

What the ROCm tool (runner plan 03) and the CUDA tool (runner plan 04) write when counters are
on, and what the engine reads. Settled on 2026-10-21, before either tool writes a record (the
operator: a worktree of its own for the shared parts).
- **Change it in one place:** the `gray-parrot-shared` worktree, then `main`. Both platforms'
  work rebases on it.
- **Keeping C and Python in step:** `tests/test_gp_records_offline.py` reads the structs out of
  `runtime/src/records.h` and checks them against the engine's dtypes.

**With counters off** (`RUPPELL_COUNTERS` and `RUPPELL_DEVICE_COUNTERS` unset):
- none of these files exists;
- `meta.json` has `"counters": "off"`.

## Files

| File | Record | Written when |
| --- | --- | --- |
| `counters.bin` | `ruppell_counter_rec`, 48 bytes; engine `COUNTER_DT` | per counted kernel and counter, on counter steps (`RUPPELL_COUNTERS`, scope `kernel`) |
| `device_counters.bin` | `ruppell_device_counter_rec`, 40 bytes; engine `DEVICE_COUNTER_DT` | per device reading and counter: periodic samples (`RUPPELL_DEVICE_COUNTERS`), or one reading per counter step (`RUPPELL_COUNTER_SCOPE=step`) |
| `counters.tsv` | text | at start, once the groups are built; one line per group and counter |
| `meta.json` | `counters` | as now, at each flush and at the end |
| `steps.tsv` | the step's fields | a counter step carries `"counters": <group>` |

## `counters.bin`

| Field | Type | ROCm | CUDA |
| --- | --- | --- | --- |
| `key0` | u64 | `dispatch_id`; joins `dispatch.bin` | correlation id; a replayed kernel's is its graph launch's |
| `key1` | u64 | 0 | the instantiated graph node, as in `ruppell_kernel_rec.graph_node_id`; 0 for an eager launch |
| `seq` | u64 | 0 | the range's index within the counter step (auto range) |
| `agent` | u64 | the agent handle, as `ruppell_dispatch_rec.agent_id` | the device id |
| `step` | u32 | the counter step's id, as the adapter numbered it | the same |
| `group` | u16 | the group collected on this step (`counters.tsv`) | the same |
| `counter` | u16 | the counter (`counters.tsv`) | the same |
| `value` | f64 | the SDK's value, reduced over dimension instances as `counters.tsv` says | CUPTI's evaluated value for the range |

**How the engine joins them:**
- **ROCm:** on `dispatch_id` to the dispatch record, which gives the kernel, its step, scope and
  graph position.
- **CUDA:**
  - an eager kernel on its correlation id;
  - a replayed kernel on the graph launch's correlation id and the instantiated node, through
    the clone records, as kernel records join today.
- **No kernel record:** the counter step's kernel records may be missing on CUDA. Then `seq` and
  the mapping plan 04 settles give the kernel's place, and the step comes from `step`.

## `device_counters.bin`

| Field | Type | Meaning |
| --- | --- | --- |
| `t0`, `t1` | u64 | ns, the tool's timestamp domain: the interval the value covers |
| `agent` | u64 | as above |
| `step` | u32 | a step reading: the counter step's id; a periodic sample: 0, placed on the step timeline by time |
| `group`, `counter` | u16 | `counters.tsv` |
| `value` | f64 | over `[t0, t1]` |

## `counters.tsv`

Tab-separated, with a header line, one line per (group, counter):

| Column | Meaning |
| --- | --- |
| `group` | the group's number, as in the records |
| `counter` | the counter's number, unique across groups; a counter in several groups has one number |
| `name` | the vendor's name: a rocprofiler counter (`FETCH_SIZE`), a CUPTI metric (`dram__bytes_read.sum`) |
| `set` | the portable set the group belongs to (`memory`, `compute`, `occupancy`, `roofline`), or `raw` |
| `kind` | `base` or `derived` |
| `source` | a base counter's block (`TCC`), or a derived counter's formula as the vendor defines it in this run's version |
| `reduce` | how dimension instances became one value: `sum`, `max`, or `none` for a counter split by dimension |
| `unit` | as the vendor gives it (`KB` for `FETCH_SIZE`; `count`, `cycles`, `percent`, `bytes`) |

- **A counter split by dimension** (`RUPPELL_COUNTER_DIMS=xcc`, say) gets a number per
  instance, with the instance in its name (`TCC_HIT[xcc=3]`) and `reduce` `none`. The records
  keep their layout.
- **Derived metrics are not recorded.** Bytes from KB, hit rates, FLOPs from MFMA operations and
  the rest are computed by the analysis from these values (scope, §1.7).

## `meta.json`'s `counters`

`"off"`, or an object:

| Key | Meaning |
| --- | --- |
| `version` | this contract's version: 1 |
| `platform`, `arch` | `rocm` or `cuda`; `gfx950`, `sm_100`, ... |
| `collector` | the SDK's version (ROCm), or CUPTI's API version (CUDA) |
| `scope` | `kernel` or `step` |
| `sets`, `groups` | the sets asked for; each group's counters and its pass count (always 1) |
| `steps` | `RUPPELL_COUNTER_STEPS` as given |
| `counter_steps` | steps counted |
| `records`, `dropped` | records written; records or ranges lost, with the reason |
| `refused` | null, or why counters were refused for this run (tracing carries on) |
| `device` | null, or the device readings' set, rate and samples |
| `notes` | what the platform reports that a reader needs, such as ROCm's Peak TOPS Limiter state when it is known |

## Versioning

A change to either struct, or to a column's meaning, bumps `version`, changes this page,
`records.h`, the engine's dtype and the test in one commit, and says so in both plans.
