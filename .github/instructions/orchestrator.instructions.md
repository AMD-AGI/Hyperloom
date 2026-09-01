<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->
---
applyTo: "src/hyperloom/orchestrator/**"
---

# Orchestrator invariants (path-scoped review)

These fire only on changes under `src/hyperloom/orchestrator/`. Flag a diff that
breaks one of them; otherwise stay within the repo-wide instructions.

## Path containment has one owner
Framework source-root resolution and the containment test live in
`orchestrator/framework/paths.py` (`resolved_within` + the `_DEFAULT_SOURCE_ROOTS`
probe order). PolicyGate, AST discovery, and patch application all go through it.
Flag any new code that re-derives a source root, re-implements a containment/prefix
check, or hardcodes a container path (`/sgl-workspace/...`, `/app/ATOM/...`,
`/app/xDiT/...`) instead of calling this module.

## Backend transport surface
Only the tools transport mounts `emit_intent`; the structured-output backend has no
tool surface (see `orchestrator/prompts/transport.py`). Flag a prompt block or
backend that assumes `emit_intent` exists unconditionally, or that documents a tool
for a backend whose transport is `TRANSPORT_STRUCTURED_OUTPUT`.

## PolicyGate is the boundary
Patch targets and trace paths are gated in `orchestrator/policy/gate.py` against
allowlists resolved from `framework/paths.py`. Flag a new file-write or patch path
that reaches the framework tree without going through the gate's `resolved_within`
allowlist check.
