# Install

Drop this `geko-hipblaslt-tuning/` folder into a Claude Code skills directory:

- Personal (all projects):  `~/.claude/skills/geko-hipblaslt-tuning/`
- Project (committed):      `<repo>/.claude/skills/geko-hipblaslt-tuning/`

Then in a Claude Code session: run `/skill-doctor` (or `/skills`) to confirm it loaded, and invoke with
`/geko-hipblaslt-tuning` — or just ask (e.g. "install geko and run a hipBLASLt tune"); the description
auto-triggers it.

Before running the bundled scripts, export `WS` to your geko_agentx workspace root (now required — the
scripts abort if it is unset), and run them INSIDE the geko container. The trace verifier also honors an
optional `TORCHPROF_DIR` (defaults to `$WS/experiments/torchprof`). Validate with:
`claude plugin validate .claude/skills/geko-hipblaslt-tuning`
