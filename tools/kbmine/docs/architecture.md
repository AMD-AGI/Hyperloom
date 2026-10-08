<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->
# Architecture

The tool answers one question — *is this target worth a session?* — from
evidence that prior sessions already produced. No GPU is touched at any point.

## Design in one idea

Two services hold complementary halves of the answer, and neither is
sufficient:

* The **Recipe KB** is scoped to *replay material*: configuration, patches and
  the minimum metadata needed to re-apply them. It is the only place an
  accepted parallelism layout can be read from, and it carries no roofline
  anywhere in its 401 distinct key paths.
* **Pulse** is the fleet index of `session_breakdown.json`, holding *execution
  evidence*: roofline ceilings, both throughput arms, token spend and elapsed
  time. It carries no accepted server args, so no layout can be read from it.

So a source is a **projector**, not a fork in the aggregation. Each projector
flattens one source's document into the same canonical row, and everything
downstream — scoping, pooling, warnings, ranking — runs once, source-agnostic.
Adding an evidence source means writing one function, not a second estimator.

## Architecture

```mermaid
flowchart LR
    subgraph src["Evidence sources"]
        direction TB
        KB["<b>Recipe KB</b><br/>POST /v1/kb/search<br/>GET /v1/kb/id/sessions/sid<br/><i>165 session docs, 82 identities</i>"]
        PU["<b>Pulse</b><br/>GET /v1/session-breakdowns<br/><i>~15k rows, 12,655 with roofline</i>"]
        FI["<b>Offline JSON</b><br/>saved envelopes<br/><i>no network, used by tests</i>"]
    end

    subgraph rd["Readers - stdlib urllib, bearer token from env"]
        direction TB
        KBC["<b>KBStoreClient</b><br/>kb_store_client.py<br/>paged search, rollup, session"]
        PUC["<b>PulseClient</b><br/>pulse.py<br/>limit/offset paging<br/>explicit CA bundle"]
    end

    subgraph pj["Projectors - one per source"]
        direction TB
        PS["<b>project_session</b><br/>reads 3 replay fields only"]
        PP["<b>project_pulse_row</b><br/>computes capture ratio"]
    end

    ROW["<b>canonical row</b><br/>identity: model, hardware, framework, version, precision<br/>scope: tp, conc, isl, osl<br/>gain, throughput, capture, layout"]

    subgraph core["estimate_from_sessions - mine.py, source-agnostic"]
        direction TB
        SC["scope filter<br/>matches_shape"]
        PW["pool warnings<br/>identity + shape mixing"]
        BS["bucket by replay scope<br/>by_shape"]
        SW["sharding_whatif<br/>rank layouts at fixed GPU count"]
        PWI["parallelism_whatif<br/>TP within a workload family"]
        ST["percentiles<br/>gain p50/p90, capture p50/p90"]
        RK["recipe_knobs<br/>every accepted flag and env,<br/>by family, scope-coupling flagged"]
        LN["learnings<br/>prose tagged with the scope<br/>it was learned at"]
    end

    REP["<b>JSON report</b><br/>historical, capture, by_shape,<br/>sharding_whatif, parallelism_whatif,<br/>recipe_knobs, learnings,<br/>pool_warnings, coverage, limitations"]

    MAIDAS["<b>MAIDAS roofline</b><br/><i>planned: cross-check the ceiling</i>"]

    KB --> KBC --> PS
    PU --> PUC --> PP
    FI --> PS
    PS -->|"gain + layout,<br/>capture unmeasured"| ROW
    PP -->|"gain + capture,<br/>layout unknown"| ROW
    ROW --> SC --> PW --> BS --> ST
    BS --> SW
    BS --> PWI
    ROW -->|"bypasses the scope filter,<br/>labels its own scope"| RK
    ROW --> LN
    ST --> REP
    SW --> REP
    PWI --> REP
    RK --> REP
    LN --> REP
    REP -.->|"ceiling + capture prior"| MAIDAS

    classDef planned stroke-dasharray: 5 5
    class MAIDAS planned
```

The asymmetry in the two projector edges is the whole boundary. On the KB path
`capture` is reported as **unmeasured**, never as zero, because an absent
ceiling is not a closed gap — treating it as zero is precisely what dragged the
earlier forecast toward nothing. On the Pulse path the layout is labelled
`layout_unknown` rather than `framework-default`, because calling an unknown
layout "default" would invent an evidence arm that no session published.

## Run flow

```mermaid
flowchart TD
    START(["kbmine run"]) --> ARGS["parse args<br/>resolve token: flag, then file, then KB_STORE_TOKEN"]
    ARGS --> PICK{"which source?"}

    PICK -->|"input"| OFF["load a JSON list<br/>or a sessions array"]

    PICK -->|"pulse-url"| P1["page /v1/session-breakdowns<br/>limit 200, offset walks"]
    P1 --> P2["filter identity client-side<br/>server-side prec is ignored,<br/>gpu_type under-matches"]
    P2 --> P3["project_pulse_row per row<br/>pick ceiling by bound kind,<br/>capture = closed / total gap,<br/>rescale per-GPU to total"]

    PICK -->|"neither, so KB"| K0{"URL configured?"}
    K0 -->|"no"| K0X(["exit 2, no network call"])
    K0 -->|"yes"| K1["search_inference_identities<br/>paged, match or hardware_in"]
    K1 --> K2["per identity: get_rollup<br/>collect session ids + champion"]
    K2 --> K3["per session: get_session<br/>failures collected, never fatal"]
    K3 --> K4["project_session per envelope<br/>gain, workload_shape,<br/>layout from extra_server_args"]

    OFF --> K4
    P3 --> AGG
    K4 --> AGG["estimate_from_sessions"]

    AGG --> S1["drop rows outside requested tp/conc/isl/osl<br/>count them in sessions_dropped_by_shape_filter"]
    S1 --> S2["collect gains; collect captures<br/>None means unmeasured, excluded"]
    S2 --> S3["build identity mix<br/>warn per dimension the pool spans"]
    S3 --> S4["bucket rows by tp/conc/isl/osl"]
    S4 --> S5["sharding_whatif: group by<br/>model/precision/framework/shape,<br/>rank layouts with 2+ arms"]
    S4 --> S6["parallelism_whatif: group by<br/>workload_family, compare TP,<br/>project target_tp if asked"]
    S4 --> S7["percentiles p50/p90<br/>for gain and capture"]
    S1 --> S8["recipe_knobs and learnings<br/>over the UNFILTERED pool,<br/>each item tagged in or out of scope,<br/>items naming the requested conc lead"]

    S5 --> OUT
    S6 --> OUT
    S7 --> OUT
    S8 --> OUT["assemble report"]
    OUT --> PATCH{"Pulse path?"}
    PATCH -->|"yes"| PX["sharding_whatif := unavailable<br/>evidence_source := pulse"]
    PATCH -->|"no"| PY["keep layout ranking"]
    PX --> FIN
    PY --> FIN(["stdout, or a file when output is given<br/>URL echoed, token never"])
```

The three entry points are `--input` for an offline pool, `--pulse-url` for
fleet evidence, and the default Recipe KB path when neither is given.

Note that `recipe_knobs` and `learnings` hang off `S1` rather than the buckets:
they read the pool *before* the shape filter. Everything statistical stays
strictly in-scope, while the recipe and the prose cross the boundary carrying a
label, because a target at an unseen scope has no in-scope evidence by
definition and an empty report is the least useful true answer available.

Two properties worth noting in that flow. A missing store URL exits `2`
*before* any network call, so a misconfiguration is never a partial fetch. And
per-identity fetch failures accumulate into `fetch_errors` instead of aborting
the run, because one unreadable rollup out of eighty should still yield a
prior — the report says how much it stood on via `coverage` and
`sessions_scored`.

## The capture ratio

Gain says how far a session moved. Capture says how much of the *available*
distance it covered, which is the part that transfers to a target that has not
run:

```mermaid
flowchart LR
    B["<b>baseline</b><br/>tok/s/GPU"] -->|"closed by the session"| O["<b>optimized</b>"]
    O -->|"headroom left"| C["<b>ceiling</b><br/>memory- or compute-bound,<br/>whichever binds"]
```

$$\text{capture} = \frac{\text{optimized} - \text{baseline}}{\text{ceiling} - \text{baseline}}$$

A no-run forecast then reads
`baseline + p50_capture x (ceiling - baseline)`, which is why the ceiling has
to come from the same session as the two arms. The ceiling is selected by the
session's own `roofline_bound_kind`; when that label is absent the lower of the
two ceilings is used, since the lower one is what actually binds.

Capture is `None` — not `0` — whenever the snapshot or the ceiling is missing,
and a test pins that. Three of 5,008 sampled rows exceeded 100%, meaning the
analytic ceiling was too low; Pulse flags those with
`roofline_ceiling_exceeded` and excluding them is a known to-do.

## Why the scoping is not optional

Pooling every session a search returns makes the median meaningless, so rows
are grouped along two different axes depending on the question:

| grouping | holds fixed | answers |
| --- | --- | --- |
| `shape_key` | tp, conc, isl, osl | what gain to expect at this scope |
| `replay_scope_key` | model, precision, framework, **and** the full shape | which layout wins inside a fixed GPU count |
| `workload_family` | model, precision, conc, isl, osl — TP left free | how throughput scales with TP |

The last two differ by exactly one dimension, and that difference is the
point. Layouts must be compared at a *fixed* GPU count, since `tp=1 dp=2` on
two GPUs versus `tp=2` on two GPUs is a real choice while the same labels on
eight GPUs is a different experiment. TP scaling needs the opposite: GPU count
must vary while everything else holds, or the comparison is confounded — ISL
changes arithmetic intensity and model size dictates the TP that fits at all,
so a 0.6B at TP1 against a 70B at TP8 reads as catastrophic scaling when
nothing scaled.

Live data keeps making the case: on 103 MI355X sessions a 23.4% pooled median
hid per-family medians ranging from 6.2% to 311%, and a 6,000-row Pulse pull
gave a p50 gain of 41.9% for mi300x/sglang/fp8/tp1 against 11.6% for
mi355x/vllm/bf16/tp1. `pool_warnings` names every dimension a pool spans so a
reader cannot mistake one for the other.
