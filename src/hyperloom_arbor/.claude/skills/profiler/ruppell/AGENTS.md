# AGENTS.md: working on ruppell

Audience: anyone, person or agent, changing this repository.

## What a result must carry

- **The build.** Every number `ruppell` produces for a report names the
  `ruppell` commit and the digest of the built runtime library.
- **A witness.** Before a version is used for a claim, its per-kernel counts
  and GPU time are checked against an independent collector (`rocprofv3`,
  Kineto) on the same work. A tracer that has not been checked against
  something else is not evidence.
- **Its coverage.** Every output states the share of GPU time it attributed
  and lists what it could not attribute, by kernel name.
- **Its cost.** Overhead is measured, never assumed: a paired A/B with the
  tracer off and on, same node, and the tracer's own intervals recorded as
  events.

## How to change it

- One change per commit. Layers stay separate: the runtime knows nothing of
  vLLM, an adapter knows nothing of a workload.
- Refuse rather than guess. A join whose preconditions do not hold on the
  trace in front of it (for example, a graph whose replays differ) refuses and
  says why; it never produces a plausible number.
- Count steps by distinct step id, never by matching names in text.
- Each trace has its own time base; align through clock pairs and refuse on
  mismatch.
- Keep adapters thin, and propose their hooks upstream so they stop being
  patches.
- Supported platform versions are pinned. An unknown ROCm version fails
  loudly at load time.

## Hygiene

- LF line endings (`.gitattributes`). Never commit traces, binaries or
  secrets; scan for tokens before committing.
- Parse large traces on the GPU node, not on a shared login host.

## Using ruppell to profile

Profiling a server with it (running, checking, reconciling its limitations,
reading it with TraceLens, ranking sites) is a separate job with its own
instructions: [`.claude/skills/ruppell-profile/SKILL.md`](.claude/skills/ruppell-profile/SKILL.md)
and [`reference.md`](.claude/skills/ruppell-profile/reference.md). Claude
Code loads it as the `ruppell-profile` skill; other agents read the files.
