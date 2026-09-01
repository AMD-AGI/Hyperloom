# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prompt fragments shared between kernel backends."""

from __future__ import annotations


def context_sections_block(*, knowledge_content: str) -> str:
    """Render the knowledge tail as a multi-line XML block.

    Does not append a trailing newline; the caller provides it via the
    closing triple-quote boundary.
    """
    return f"\n<knowledge>\n{knowledge_content}\n</knowledge>"


# Always-resident pointer to the two shared method cards under
# ``local_knowledge/common_methodology/optimization/``. The knowledge tree is
# Read-on-demand, so a card nobody opens teaches nothing: the rules an agent
# must not have to go looking for are restated here, and the detail stays in
# the card. Every kernel backend carries this block -- the two campaigns that lost these
# moves ran on a kernel backend whose own prompt never mentioned them.
EDIT_SURFACE_AND_SWEEPS_PROMPT = """\
## Edit surface & cheap sweeps (shared cards — read before pricing a direction)

Both live under `common_methodology/optimization/` in the knowledge tree. Open
them with the `Read` tool; what follows is the part you must not have to look up.

- **`lever_edit_surface.md` — `editable_sources` is a FLOOR, not a ceiling, and
  it never bounds WHAT YOU CHANGE.** The planning context lists the campaign's
  declared source set there (entry 0 is the primary kernel path; data and config
  files count exactly as much as `.py` sources); every other tracked,
  non-protected implementation file is editable too, and you may add new files.
  From a permitted file you can rebind a symbol in an installed package before
  the framework consumes it; carry device-side source in through the framework's
  own hook (`import_source`, `pragma_import_c`, an intrinsic, inline asm) and
  call it on the extern-call path; change a module-level constant that another
  module's dispatch reads; or append a row to a permitted data/config file that
  a lookup consults. Those are instances of one move — find the last point,
  reachable from a file you may edit, at which the behaviour is still mutable,
  and change it there — not a list of four routes to enumerate and close.
  A constant whose default comes from `os.environ` is an ordinary constant in an
  editable file: an `os.environ.get(...)` default says NOTHING about the edit
  surface. Writing "that would mean patching the framework/library, not this
  file" has already cost a campaign its largest available win; before you write
  it, work out what actually runs first from the files you were given.
- **`lever_cheap_sweeps.md` — to time one constant, do not edit-and-gate.** Read it on
  the host as `FORGE_SWEEP_<NAME>` defaulting to today's value, echo
  `sweep_const: <NAME> <value>` on every read (a point with no echo fails and
  carries no time), parse a BOOLEAN knob against an explicit token set rather
  than with `bool(value)` (`bool("0")` is `True`, so the OFF point would time the
  ON kernel and the echo would still confirm it), and take one data point per
  command:
  `python3 -m kernelforge.mcp_server.tools.bench --driver <driver command> --case <CASE_ID> --set <NAME>=<value>`
  Pass `--driver` exactly the command you were told to run the driver with (name
  the WRAPPER when your session names one). Sweep coupled constants JOINTLY, and
  sweep every inherited literal in BOTH directions. Sweep numbers are
  exploratory; the canonical gate still decides what survives.
  **KEEP the knobs, defaulted to the winning literals, for the whole search.** A
  knob collapsed mid-campaign is an axis the next session would have to
  re-author before it can even ask the question, which means it never asks.
  Strip them only at final submission, and only if the deliverable must be
  knob-free — then re-run the gate to prove the collapse changed nothing."""
