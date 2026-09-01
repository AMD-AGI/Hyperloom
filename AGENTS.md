<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AGENTS.md — authoring rules of engagement

Rules for anyone (human or AI) writing code in this repo. This file is **pointers, not
restatements**: it says *what to hold to* and *where the detail lives*, so there is one
source of truth per topic.

- **Workflow & PR process** → [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Formatting, naming, testing, REUSE conventions** → [`docs/contributing/style-guide.md`](docs/contributing/style-guide.md)
- **What a review will flag** → [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) and the Copilot instructions under [`.github/`](.github/)

## One concern per change
A PR fixes one issue or adds one capability. If you must bundle, say why in the PR
description. Split unrelated refactors out; don't ride them in on a fix.

## Size budget
Prefer small, reviewable diffs. A large diff is a signal to stop and split, not to push
harder. Drive-by cleanup in an unrelated file is a separate PR.

## Fix upstream, not around it
Hyperloom is glue over external components (GEAK, Magpie, TraceLens, and the frameworks
resolved in `src/hyperloom/orchestrator/framework/paths.py`). When the root cause is in one
of those, fix it there and pin the fix — do not paper over it with a local workaround.
Record the upstream ticket in the PR.

## Don't grow the debt we're paying down
- **No new broad `except Exception` / bare `except`.** Catch the specific error, or let it
  raise. There is a backlog being ratcheted down; don't add to it.
- **No new lint/type suppressions** (`# noqa`, `# type: ignore`, `# pylint: disable`,
  `# nosec`) without a one-line reason and, where possible, a narrower fix.
- **No new feature flags / env toggles** to route around a design problem. A flag is a
  decision deferred; make the decision.

## Delete, don't comment out
Dead code goes. Version control is the archive. Commented-out blocks and `# removed …`
tombstones rot and mislead.

## Clean design preferences
- One boundary rule per concern, owned by one module (e.g. path containment lives in
  `framework/paths.py`; don't re-derive it in callers).
- Derive over hardcode: prefer a single computed source over duplicated constants.
- Minimal typed interfaces; push validation to the system boundary, trust internal callers.

## New framework or platform
Building a new framework (e.g. AgentX) or platform (e.g. world models)? Work bring-up
through the owning-component leads **before** integration code lands — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) § *Proposing a new framework or platform*.
