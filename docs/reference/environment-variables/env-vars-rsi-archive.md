---
myst:
    html_meta:
        "description": "Multi-branch RSI archive variables for Hyperloom: archive directory path, as-of timestamp for replay experiments, and write-only mode for runs beside the archive."
        "keywords": "Hyperloom, environment variables, RSI archive, HYPERLOOM_RSI_ARCHIVE, HYPERLOOM_RSI_AS_OF, HYPERLOOM_RSI_WRITE_ONLY, multi-branch"
---
# Hyperloom Multi-branch RSI archive

A multi-branch session reads its predecessors' search trees, lessons, scheduling playbooks and outcomes from this
archive, and writes its own into it at CLOSE. `python -m hyperloom_multibranch.blackboard.rsi.lessons` (`list`, `show`,
`revise`, `retire`, `merge`, `purge`) is a person's look at, and change to, the lesson pool beside it; its `--archive`
names the archive in place of `HYPERLOOM_RSI_ARCHIVE`. Without `--archive` the command reads the pool as a session
does, so under `HYPERLOOM_RSI_WRITE_ONLY=1` it finds none and `HYPERLOOM_RSI_AS_OF` hides what was written after it.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_RSI_ARCHIVE` | None; required | The archive directory sessions read and write. A session refuses to start when it is unset or this account cannot create and remove a file in it (a real write, named with its errno, owner and mode). An archived search tree is made readable by the archive folder's group (`640`) wherever that folder is setgid and its group may read it, whatever the writing account's umask, also where others can write it; never by others. Operator configuration. |
| `HYPERLOOM_RSI_AS_OF` | Unset (everything) | An ISO time the archive is read as of, for sessions run as repeats or as pairs: lessons and playbooks by when they were written, a search tree by its session's start plus budget, not by when CLOSE wrote it. A tree that ended by then but is written after it is read by a session that loads the archive after the write and missed by one that loads it before. For experiments. |
| `HYPERLOOM_RSI_WRITE_ONLY` | Unset (read and write) | `1` reads nothing of the archive, for a run beside the archive runs, while CLOSE still writes the session's outcomes into it. For experiments. |
