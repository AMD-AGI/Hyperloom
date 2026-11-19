---
title: contributing to sgl-amd-mono — PR shape, review gates and the evidence a perf PR carries
kind: technique
gens: [gfx950]
updated: 2026-11-17
sources:
  - sgl-project/sgl-amd-mono@219f40be:.github/pull_request_template.md
  - sgl-project/sgl-amd-mono@219f40be:.github/workflows/pr-title.yml
  - sgl-project/sgl-amd-mono@219f40be:.github/workflows/lint.yml
  - sgl-project/sgl-amd-mono@b504de5:.github/CODEOWNERS
  - sgl-project/sgl-amd-mono@b504de5:scripts/ci/README.md
  - sgl-project/sgl-amd-mono@b504de5:scripts/ci/models.py
  - sgl-project/sgl-amd-mono@b504de5:.github/workflows/pr-test-amd.yml
  - https://github.com/sgl-project/sgl-amd-mono/pull/56
  - https://github.com/sgl-project/sgl-amd-mono/pull/59
  - https://github.com/sgl-project/sgl-amd-mono/pull/62
  - https://github.com/sgl-project/sgl-amd-mono/pull/68
---

# Contributing upstream

A forge KEEP is a candidate, not a PR. The PR has to carry the evidence below, measured on hardware
the spec serves, against the SGLang commit it names.

## The gates a PR must pass
- **Title** (`.github/workflows/pr-title.yml`): `[Type] scope: title` or `[Type] [Hardware] scope: title`.
  Type ∈ Feat, Fix, Refactor, Perf, Doc, Test, CI, Chore; scope is a module or model
  (`minimax_m31`, `kimi_k3/kimi_k31`, `common`); hardware is one uppercase tag (`[MI355X]`,
  `[MI300X]`). Example: `[Perf] [MI355X] glm53_flash: Score a verify request's index tiles once for its rows`.
- **Lint** (`.github/workflows/lint.yml`): `pre-commit run --all-files` (the pre-commit-hooks checks,
  isort, ruff, ruff-format and codespell).
- **Review** by a code owner from `.github/CODEOWNERS`; a new model's PR adds its owner line there.
- **Squash and merge**: the title becomes the commit title on `main`.

## The GPU CI (`scripts/ci/README.md`)
Two MI355X nodes run the review guideline's checks (#107). The **PR test** runs only when a maintainer
adds the `run-ci` label (`run-ci-extra` adds GPQA, for attention, KV-cache or long-sequence changes);
the run removes the label when it starts, so a later push needs a maintainer again (#130 lets fork PRs
run). It sets the `AMD PR test` status and keeps one report comment on the PR.
- **Scope** (`scripts/ci/scope.py`): a changed model package (or its tests) runs that model's full
  tier; `common/`, `sgl_amd_mono/__init__.py` or `pyproject.toml` runs every other model's smoke tier;
  a PR that changes no model runs the unit and common kernel tests.
- **What a model runs** (`scripts/ci/models.py`): its kernel and e2e tests, microbenchmarks, GSM8K
  (sgl-eval, 1319 questions) on and off, and `sglang.benchmark.serving` (random 8k in / 1k out, range
  ratio 0.8) on the base branch and the PR at the model's concurrencies. A model whose `gpus` the nodes
  do not have is skipped with a warning: `minimax_m31` lists `mi350x` only, so its PRs carry
  MI355X results by hand.
- **The gate** fails on a failed test or step; no routing logged by the monokernel server; a GSM8K loss
  with the monokernels on (exact one-sided McNemar on the questions the two runs answer differently,
  p < 0.05); median TPOT more than 2% worse than the base at a covered concurrency
  (`gate_concurrency`); **a `[Perf]` or `[Feat]` PR none of whose models gains 5% or more in median
  TPOT at a covered concurrency**; or a change to existing lines of `.github/CODEOWNERS`.
- The **nightly** (main, newest image) also checks the sgl-amd-mono commit the SGLang image pins
  (`/sgl-workspace/sgl-amd-mono`) against that SGLang: `test_contract.py` and a short serving run.

## The body (`.github/pull_request_template.md`)
| Section | What upstream PRs put there |
|---|---|
| Motivation | where the time goes, measured: per-layer chain time on the standard path (#56: about 76 µs a sparse layer, half of an 8.66 ms step), per-CTA stage timings, ISA statistics (#19: VGPRs, spills, instruction and branch counts) |
| Modifications | one commit per change, each reviewable on its own and each with its own measured gain (#62, #68) |
| Accuracy Tests | monokernels on and off on the same server; `test/unit`, `test/kernels`, `test/e2e` counts; check-mode results against `NUMERICS`; chat GSM8K past the largest width |
| Speed Tests and Profiling | per-user decode with the monokernels on and off, including past the largest width, plus kernel-only time of the launch |
| Not covered | ISAs, TP sizes and widths not measured; what was tried and dropped, each measured (#59) |
| Checklist | pre-commit passes; tests cover the change; the SGLang commit is named; results attached |

## Evidence norms
- **Bitwise where values do not change.** A hand-off or placement change claims bitwise equality with
  `main` at every width and TP size the spec builds, from SHA-256 digests of every output and every
  write (`test/kernels/glm53/check_layer.py --dump` / `--compare`), with a negative control that the
  comparison can fail (`--perturb`).
- **Graphs replay the eager step** (`--replays N` in the GLM check tools).
- **Fail-stop still holds**: the e2e fault case — a rank stalled by `SGLANG_AMD_MONO_FAULT_INJECT`
  stops every GPU at the null page and no token is sent — passes on the branch.
- **Honest kernel timing.** Time a graph of the launches; rotate the weights through several copies so
  they stream from HBM as a server's layers do — one copy stays in the 256 MB MALL (#59).
- **Honest serving timing.** Alternate the arms or run two rounds with the second in reverse order;
  report medians; hold speculative acceptance fixed with `SGLANG_SIMULATE_ACC_LEN` for speed tests and
  measure real acceptance separately.
- **Name unrelated failures.** A contract failure caused by the SGLang branch rather than the change is
  stated as such (#56, #59).

## From a forge campaign to a PR
1. Rebase the KEEP onto upstream `main`; split independent changes into commits, each re-measured.
2. Re-run on the spec's hardware and TP sizes; a gfx942-only measurement cannot support a gfx950 claim.
3. Carry the campaign's reverted attempts into "Not covered" when each was measured — they save the
   reviewer the same experiments.
4. Measure on the image the CI runs (the newest `rocm/sgl-dev:v*-rocm10-mi35x-*`). A campaign on an
   older FlyDSL / AITER carries the direction of a change, not its size: MiniMax-M3.1's six FFN changes
   measured 2-7% on a ROCm 7.2 / FlyDSL 0.3.4 stack and 5-10% on the CI image (#145).
5. Prove each commit bitwise with a comparison that can fail: `main`'s launch (a copy inside the
   `sgl_amd_mono` package, whose JIT cache key traces only that package) and the commit's, in one
   process on the same inputs, every output compared by bits; then break a constant once and show
   the comparison catches it.
6. Clean what the search left: parameters nothing reads, constants defined away from the others,
   comments that only make sense inside the campaign. Run `pre-commit` on every commit
   (`git rebase --exec`), not only on the tip.
7. Before titling a PR `[Perf]`, check the CI's 5% median-TPOT gate at a covered width; when the model's
   CI entry cannot run on the nodes (`gpus`), attach the serving tests and per-user decode by hand.
8. A repository test that fails on `main` too is named as such, and fixed on its own when the fix is
   clear (#146: the o_proj parity test held to the GEMM SGLang dispatches for the shape).
