---
myst:
    html_meta:
        "description": "Hyperloom Inspector: a read-only web view of a live or finished Hyperloom session. How to run it locally or in a pod, and what it shows."
        "keywords": "Hyperloom, Inspector, session, observability, Kubernetes, port-forward, AMD GPU, ROCm"
---
# Hyperloom Inspector

The Inspector is a read-only web view of one Hyperloom session directory, live or finished. It answers "what is the
run doing now, and what went wrong?" in one place, which otherwise takes `kubectl exec`, `sqlite3` and `jq` on the
pod. It only reads files and opens the session's coordinator database read-only, so it is safe to run next to a live
optimizer.

The server is the Python package
[`src/hyperloom/inspector`](https://github.com/AMD-AGI/Hyperloom/tree/main/src/hyperloom/inspector). It needs the
Hyperloom it ships with, and no Python package the `hyperloom` command does not already need.

## Run

```bash
hyperloom inspect --session-dir "$SESSION_DIR"
# the same, needing only the inspector package:
python3 -m hyperloom.inspector --session-dir "$SESSION_DIR"
```

Then open <http://127.0.0.1:8765/api/snapshot>.

Options:

- `--session-dir DIR`: a session directory, or a directory above sessions (`$USER_DATA_PATH/<model>`), where the
  newest session is used.
- `--host`, `--port`: bind address and port. Default `127.0.0.1:8765`.
- `--verbose`, `-v`: `hyperloom inspect` only. `-v` logs INFO messages; `-vv` adds DEBUG, with a line per HTTP
  request.

### In a pod

Run the Inspector in a pod from a source tree copied in from your machine. The copy carries the parts of Hyperloom
the Inspector imports, so the Hyperloom version the image has does not matter; the pod's Python needs PyYAML, which
any image with Hyperloom installed has.
Copy the tree in, start the Inspector and forward its port:

```bash
tar -c --exclude=__pycache__ --exclude='.env*' \
    src/hyperloom/__init__.py src/hyperloom/inspector src/hyperloom/common src/hyperloom_common \
  | kubectl -n <namespace> exec -i deploy/<deployment> -- \
      sh -c 'mkdir -p /tmp/hl-insp && tar -x -C /tmp/hl-insp'
kubectl -n <namespace> exec deploy/<deployment> -- sh -c \
  'cd /tmp/hl-insp && PYTHONPATH=src nohup python3 -m hyperloom.inspector \
     --session-dir <session-dir> > inspect.log 2>&1 &'
kubectl -n <namespace> port-forward deploy/<deployment> 8765:8765
```

`src/hyperloom/common` holds the secret masking the Inspector applies before it serves anything, and
`src/hyperloom_common` what that masking imports.

A session whose operator pins hold a credential keeps its coordinator database outside the session directory, in the
running account's private ledger: below `HYPERLOOM_LEDGER_DIR`, else below `TMPDIR`. To see its tasks, run the
Inspector as that account, in the same container or with the same `HYPERLOOM_LEDGER_DIR`. Otherwise the Inspector
reports an error badge saying that the tasks cannot be read, rather than an empty task list.

## Features

- **Run information**: model, GPU, TP/EP, workload, framework and stack fingerprint, budget, phase, baseline and best
  tok/s, start and resume history, and a badge when the tasks cannot be read because the coordinator database is in
  another account's private ledger.
- **Artifacts**: every file grouped by the node and task that produced it, read as JSON, JSONL, text and
  live-following logs, and a side-by-side diff of two text files. Profiler traces show metadata only.

## Security

- It binds to `127.0.0.1` by default and accepts requests only for a loopback address or the bound host.
- `.env*` files are never listed or served, and paths can't leave the session directory.
- Everything it serves is redacted: secret-shaped keys and tokens are masked.
- Treat the Inspector as operator-only all the same: prompts and logs can hold sensitive content that redaction doesn't
  recognise.
