# Contributing

Issues and pull requests are welcome. The maintainer is @ThePie88; every change is read and
run before it lands, so a pull request may come back with questions, or be merged together
with a follow-up commit that completes it.

## Before you start

- Install with [SETUP.md](SETUP.md). The working rules in [CLAUDE.md](CLAUDE.md) apply to people
  as much as to AI agents.
- For anything bigger than a fix, open an issue first so we agree on the direction.
- One topic per pull request; keep unrelated cleanups out of it.

## What a pull request needs

- What changed and why, linking the issue if there is one.
- How you validated it by running it: the exact commands, the GPU and its gfx arch, and the
  torch / ROCm / Triton versions. A change that was only read or grepped is not validated.
- For performance work, numbers before and after on the same machine, measured with nothing
  else on the GPU (on Windows a leftover process makes VRAM spill into shared memory, and
  every timing taken like that is meaningless).
- For engine-behaviour changes, an executable check (CLAUDE.md, "Behaviour-changing work").
- Docs updated in the same pull request: README.md, SETUP.md, run/README.md and
  patches/README.md, as relevant.
- Changes to the gitignored `vllm/` clone go in as a regenerated `patches/vllm/*.patch`
  (rules in patches/README.md).
- Plugin code logs through `logging`, never `print` to stdout (issue #17), and contains no
  machine-specific paths: use `tools/winrocm_paths.py` or an environment variable.

## AI-assisted contributions

Welcome. Say so in the pull request description, and make sure a person ran what the pull
request claims. Changes to CLAUDE.md get the same review as code, because AI agents follow
that file as instructions.

## Branches

Name the branches on your fork as you like; a numbered `NNN/type/slug` scheme keeps many open
pull requests apart. Do not push unrelated commits to a branch that is the head of an open pull
request: they join that pull request.

## License

Contributions are licensed under the Apache License 2.0, like the rest of the repository
(section 5 of the license), with no separate agreement. Keep the SPDX header on every file;
on files you create, add your own copyright line, and leave the existing ones in place.
