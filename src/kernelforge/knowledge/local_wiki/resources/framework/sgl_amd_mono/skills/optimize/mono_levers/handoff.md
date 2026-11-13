---
title: hand-offs — making waits between CTAs and ranks cheaper without changing a value
kind: lever
lever: monokernel_handoff
gens: [gfx950]
bottleneck: latency-bound (the critical chain is waits)
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/common/sync.py
  - https://github.com/sgl-project/sgl-amd-mono/pull/19
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
  - https://github.com/sgl-project/sgl-amd-mono/pull/62
  - https://github.com/sgl-project/sgl-amd-mono/pull/68
---

# Hand-offs

## Route here when
The stage timeline's critical chain is consumers waiting on producers (`skills/profile/stage_timeline.md`):
a consumer starts well after the producer's last mark, or a launch's time barely moves with the bytes
it reads.

## What to change
| Change | Why it is cheaper | Evidence |
|---|---|---|
| **Tag the values instead of draining and flagging.** Write each value as a tagged pair and let the consumer poll the values themselves; drop the ready flag. | drain → flag → poll flag → read is two round trips in series; polling tagged values is one | glm53_flash #62 and glm53 #68 (selected tokens): the region doubles, the flag goes |
| **Uncached scratch** for in-launch mailboxes: allocate it as a one-rank `common.peer.PeerBuffer` (`hipDeviceMallocUncached`). | a consumer's load is served at the line's home memory instead of through the producer XCD's L2; the kernel is unchanged | #62, #68 |
| **Fewer vector memory instructions ahead of a poll.** Merge small loads (8 contiguous bf16 weights in one) and stage partials two pairs a load. | each vector memory instruction holds the CU's in-order queue about the same time, whatever its size, so loads issued ahead of a poll delay it | #62 |
| **Poll in batches**: every row's key in one poll, not a round trip a row. | one wait instead of S | #68 (the cache task) |
| **Publish a smaller value earlier.** The norm publishes each token's `rstd` before it writes the row; the router normalizes its own columns from it. | the consumer no longer waits for the whole row | #59 |
| **Keep the guarded wait loop single-exit and wave-uniform.** Fail from inside the loop, keep poll counters in SGPRs, keep the failure path out of line. | a `stop` value carried as loop state made LLVM copy the polled words every poll: 256 VGPRs and 754 spilled vs 228 and none | #19 (`common/sync.py::_spin`) |

A poll batch is bounded by `sync.POLL_MAX = 12` specs; past that, live registers grow with the batch.

## Verify
- **Bitwise equal to `main`** at every width and TP size: a hand-off change moves no value (#62 and #68
  compared SHA-256 digests of every output and write, with a negative control).
- The fault case still passes and the number of guarded wait sites is unchanged (or the cases are
  updated with it).
- Kernel-only time per launch at each served width, and the stage timeline's chain before and after.

## Expected magnitude
glm53_flash, three hand-off changes (#62): median TPOT −2.5% to −5.8% across TP2–TP8 at 1–4 requests.
glm53, hand-offs with placement (#68): −5.7% per MTP cycle at TP8, one request; kernel-only −7.7%
(TP8, 2 rows).

## Failure modes
| Symptom | Cause |
|---|---|
| passes at small scale, stops at the deadline under load | a drain or fence removed that still ordered a plain store before its publication |
| stops after many launches | a tag narrower than the launches it must tell apart |
| check tools fail on the new layout | the scratch layout moved; update the tools in the same change (#62 did) |
