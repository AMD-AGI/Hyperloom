---
myst:
    html_meta:
        "description": "Hyperloom Inspector: a read-only web page for a live or finished Hyperloom session. How to build the UI, run it locally or in a pod, and what it shows."
        "keywords": "Hyperloom, Inspector, session, observability, timeline, replay, Kubernetes, port-forward, AMD GPU, ROCm"
---
# Hyperloom Inspector

The Inspector is a read-only web page for one Hyperloom session directory, live or finished. It answers "what is the
run doing now, and what went wrong?" in one place, which otherwise takes `kubectl exec`, `sqlite3` and `jq` on the
pod. It only reads files and opens the session's coordinator database read-only, so it is safe to run next to a live
optimizer.

The server is the Python package
[`src/hyperloom/inspector`](https://github.com/AMD-AGI/Hyperloom/tree/main/src/hyperloom/inspector). It needs the
Hyperloom it ships with, and no Python package the `hyperloom` command does not already need. The source of
its UI (Svelte 5, Vite 6) is in
[`frontend/inspector`](https://github.com/AMD-AGI/Hyperloom/tree/main/frontend/inspector).

## Build

The server serves the UI at `/`, compiled into `src/hyperloom/inspector/static/modern/` (gitignored). In a tree or
an install without that build, `/` answers with an error that says how to get it, and `hyperloom inspect` says so
when it starts.

Build the UI with Node 18.19 or later:

```bash
cd frontend/inspector
npm ci
npm run build    # writes ../../src/hyperloom/inspector/static/modern/
```

Other scripts:

- `npm run dev`: the Vite dev server; `/api` goes to the Inspector at `$INSPECTOR_URL`, default
  `http://127.0.0.1:8798`.
- `npm run check`: svelte-check.
- `npm test`: vitest.
- `npm run format:check`: prettier.

Wheels and sdists run `npm ci && npm run build` themselves, through the build backend in `build_backend/`:

- If npm is missing or its build fails, an existing `static/modern/` build ships as is, with a warning. With
  neither, the package ships without the UI, with a warning, and the Inspector says how to get it. `pip install`
  shows the warning only with `-v`.
- `HYPERLOOM_REQUIRE_UI=1`, which the release and packaging CI set, fails the build instead of shipping without
  the UI.
- `HYPERLOOM_SKIP_UI_BUILD=1` skips the frontend build, and whatever `static/modern/` holds ships as is.

Editable installs never build the UI: run `npm run build` by hand.

## Run

```bash
hyperloom inspect --session-dir "$SESSION_DIR"
# the same, needing only the inspector package:
python3 -m hyperloom.inspector --session-dir "$SESSION_DIR"
```

Then open <http://127.0.0.1:8765/>.

Options:

- `--session-dir DIR`: a session directory, or a directory above sessions (`$USER_DATA_PATH/<model>`), where the
  newest session is used.
- `--host`, `--port`: bind address and port. Default `127.0.0.1:8765`.
- `--replay [--speed N]`: replay the session on a virtual clock from its first record, with a scrubber. Default
  speed 1.0.
- `--verbose`, `-v`: `hyperloom inspect` only. `-v` logs INFO messages; `-vv` adds DEBUG, with a line per HTTP
  request.

### In a pod

Run the Inspector in a pod from a source tree copied in from your machine. The copy carries the parts of Hyperloom
the Inspector imports, so the Hyperloom version the image has does not matter; the pod's Python needs PyYAML, which
any image with Hyperloom installed has.
Don't run it from the pod's own checkout: an editable install never builds the UI, so the page only
shows the build error. Build the UI first (the copy includes the gitignored build), then copy the tree in, start the
Inspector and forward its port:

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
`src/hyperloom_common` what that masking imports. For a replay
next to the live view, start a second Inspector with `--port 8766 --replay --speed 600` and forward `8766:8766` as
well.

A session whose operator pins hold a credential keeps its coordinator database outside the session directory, in the
running account's private ledger: below `HYPERLOOM_LEDGER_DIR`, else below `TMPDIR`. To see its tasks, run the
Inspector as that account, in the same container or with the same `HYPERLOOM_LEDGER_DIR`. Otherwise the page shows
an error badge saying that the tasks cannot be read, rather than an empty task list.

## Features

- **Run information**: model, GPU, TP/EP, workload, framework and stack fingerprint, budget, phase, baseline and best
  tok/s, start and resume history, and a badge when the tasks cannot be read because the coordinator database is in
  another account's private ledger.
- **Harness graph**: the Coordinator, the LLM roles, executors and specialists as nodes; bus traffic, dispatch and
  LLM calls as edges. A layered or force layout. Clicking a node filters the other views.
- **Timeline**: a lane per node with task bars by health, phase bands, bus, LLM and lifecycle markers, and resumes.
  Idle stretches are compressed.
- **Artifacts**: every file grouped by the node and task that produced it, with a viewer for JSON, JSONL, text and
  live-following logs, and a side-by-side diff of two text files. Profiler traces show metadata only. The list
  stops at 20,000 files, and the page says so when a session holds more.
- **Drawer**: graph nodes, timeline bars, tasks, bus events and LLM calls open the raw record, related records,
  artifacts and, for LLM calls, the prompt and response.
- **Tabs** for tasks, bus events, LLM calls and the throughput journal.
- **Replay** with a scrubber and speed control.

## Security

- It binds to `127.0.0.1` by default and accepts requests only for a loopback address or the bound host.
- `.env*` files are never listed or served, and paths can't leave the session directory.
- Everything it serves is redacted: secret-shaped keys and tokens are masked.
- Treat the page as operator-only all the same: prompts and logs can hold sensitive content that redaction doesn't
  recognise.
