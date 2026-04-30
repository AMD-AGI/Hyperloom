# Actions — Subskill Index (Kernel Agent)

This directory holds **per-request playbooks** the kernel agent reads on
demand when it sees a matching `request{kind=...}` event in its inbox.
SKILL.md links to these by kind; this INDEX is a flat lookup.

| File | Trigger (`request{kind=...}`) | Response kind on success |
|---|---|---|
| `select_kernels.md` | `select_kernels` | `select_kernels_done` |
| `run_optimization.md` | `run_optimization` | `optimization_done` |
| `apply_patch.md` | `apply_patch` | `patch_applied` |

Every subskill follows the same outline:

1. **Inputs** — what fields the request payload must carry
2. **Procedure** — bash steps + Read invocations
3. **Output** — RESPONSE payload schema (kind + status + result)
4. **Failure modes** — when + how to emit `response{status=failed}`
5. **Soft rules** — IR pointers (warnings only)

## Adding a new request kind

1. Drop a new `actions/<kind>.md` here describing inputs/procedure/output
2. Add a row to the table above
3. Add a row to `../SKILL.md`'s "What you do" table
4. (Optional) extend `policy.py::REQUEST_ROUTING` if the new kind has
   different source-role requirements
