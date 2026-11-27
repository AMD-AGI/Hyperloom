---
myst:
    html_meta:
        "description": "Hyperloom Inspector: a read-only web page for a live or finished Hyperloom session. How to build the UI, run it locally or in a pod, configure the LLM narrator, and what it shows."
        "keywords": "Hyperloom, Inspector, session, observability, timeline, replay, narrator, Kubernetes, port-forward, AMD GPU, ROCm"
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
- `--history-dir DIR`: where memory and GPU samples are kept, so a restarted Inspector backfills its chart; never
  inside the session. Default `$XDG_STATE_HOME/hyperloom-inspector` or `~/.local/state/hyperloom-inspector`.
- `--no-history`: keep resource samples in memory only.
- `--token-dir DIR`: where the token timeline (every LLM call's tokens and cost over time, as JSON) is kept; never
  inside the session. Default `--history-dir`, else its default. `--no-token-file` keeps it in memory only. It is
  `<session>-<hash>.tokens.json`.
- `--narrate`: turn the narrator on, so it calls the LLM. Off by default, even when LLM credentials are in the
  environment (see [Narrator](#narrator)).
- `--llm-env-file FILE`: with `--narrate`, a dotenv file to take the narrator's LLM settings from. Without
  `--narrate` it is not read.
- `--no-llm`: never call an LLM. Contradicts `--narrate`, and the Inspector refuses to start with both.
- `--verbose`, `-v`: `hyperloom inspect` only. `-v` logs INFO messages; `-vv` adds DEBUG, with a line per HTTP
  request.

Memory and GPU samples are taken only while an Inspector runs live, and only on the optimizer's host: when the
session's lock names another host, the Inspector samples nothing, as its own machine's figures would not be the run's.

### In a pod

Run the Inspector in a pod from a source tree copied in from your machine. The copy carries the parts of Hyperloom
the Inspector imports, so the Hyperloom version the image has does not matter; the pod's Python needs PyYAML, which
any image with Hyperloom installed has, and for the narrator the LLM client (`httpx` or `openai`) Hyperloom calls
the model with. Don't run it from the pod's own checkout: an editable install never builds the UI, so the page only
shows the build error. Build the UI first (the copy includes the gitignored build), then copy the tree in, start the
Inspector and forward its port:

```bash
tar -c --exclude=__pycache__ --exclude='.env*' \
    src/hyperloom/__init__.py src/hyperloom/inspector src/hyperloom/common src/hyperloom_common \
  | kubectl -n <namespace> exec -i deploy/<deployment> -- \
      sh -c 'mkdir -p /tmp/hl-insp && tar -x -C /tmp/hl-insp'
kubectl -n <namespace> exec deploy/<deployment> -- sh -c \
  'cd /tmp/hl-insp && PYTHONPATH=src nohup python3 -m hyperloom.inspector \
     --session-dir <session-dir> --narrate --llm-env-file <env-file> > inspect.log 2>&1 &'
kubectl -n <namespace> port-forward deploy/<deployment> 8765:8765
```

`src/hyperloom/common` holds the secret masking the Inspector applies before it serves anything, and
`src/hyperloom_common` the price sheet the Tokens tab is priced by and the narrator's LLM client. For a replay
next to the live view, start a second Inspector with `--port 8766 --replay --speed 600` and forward `8766:8766` as
well.

A session whose operator pins hold a credential keeps its coordinator database outside the session directory, in the
running account's private ledger: below `HYPERLOOM_LEDGER_DIR`, else below `TMPDIR`. To see its tasks, run the
Inspector as that account, in the same container or with the same `HYPERLOOM_LEDGER_DIR`. Otherwise the page shows
an error badge saying that the tasks cannot be read, rather than an empty task list.

### Narrator

The narrator writes a short summary of what the session is doing at the current clock, live or in a replay, with the
LLM Hyperloom is configured with. You can ask it questions, and pin one as a focus for later summaries.

It is opt-in. Start the Inspector with `--narrate` to enable it; LLM credentials alone do nothing. Without `--narrate`
the panel says the narrator is off, and the Inspector makes no LLM call.

- Credentials come from the Inspector's environment. Processes started with `kubectl exec` don't have the pod's LLM
  variables, so pass `--llm-env-file` with `--narrate`. Only `ANTHROPIC_*`, `OPENAI_*`, `CLAUDE_MODEL`,
  `CLAUDE_CODE_OAUTH_TOKEN` and `HYPERLOOM_INSPECTOR_LLM_*` are read from it, plus variables that `*_CUSTOM_HEADERS`
  refers to as `${NAME}`. The environment wins, and values are never logged or served.
- `HYPERLOOM_INSPECTOR_LLM_BACKEND`: `auto` (default; Anthropic when it has a usable credential, else an
  OpenAI-compatible endpoint when `OPENAI_BASE_URL` or `OPENAI_API_KEY` is set), `anthropic`, `openai` or `none`.
- `HYPERLOOM_INSPECTOR_LLM_MODEL` (else `CLAUDE_MODEL`, else Hyperloom's default Claude model) and
  `HYPERLOOM_INSPECTOR_LLM_MAX_TOKENS` (default 1600; models that think count their thinking against it). A call
  times out after 90 s. An answer cut off at the limit says so.
- How often it narrates: with the panel open and auto on, the page asks for a narration on open, on a phase change,
  and every 120 s. The server writes a new one at most every 20 s, and only when the session changed.
- What each call sends: the session digest, at most 24,000 characters, plus the last 5 narrations and answers. The
  digest is redacted. Explain adds at most 6,000 characters of the item's own records (see below).
- One call runs at a time. History is kept in memory (200 entries) and is lost on restart.

Without an LLM the panel says why, and the rest of the page is unchanged.

### Explain this

Point the narrator at one thing: press **Explain** in the drawer, or right-click an item and pick **Explain**. It
works on graph nodes, tasks (rows and timeline bars), bus events, LLM calls, journal entries (rows and chart points)
and explore variants. The answer lands in the narrator's thread under "Explain: <item>", and a follow-up question
typed afterwards is about the same item until you remove its chip.

- The server adds that item's own records to what the model sees, so items older than the summary's recent window
  are covered too: a task's history, params, result and log tail; a variant's recipe, provenance, outcome and error;
  a journal entry with the best tok/s before it and the keep threshold; a call's prompt and response. Redacted, at
  most 6,000 characters, and in a replay only what happened before the clock.
- Each kind has a fixed question ("Explain task roofline: what it did, how it went and why."). Pressing Explain again
  on the same item at about the same clock reuses the answer instead of calling the model.
- `POST /api/ask` takes an optional `item` (`{"kind": "task", "id": "…"}`; kinds `node`, `task`, `event`, `call`,
  `journal`, `variant`); a malformed one is 400, an unknown or future one 404.

## Features

- **Run information**: model, GPU, TP/EP, workload, framework and stack fingerprint, budget, phase, baseline and best
  tok/s, start and resume history, and health badges, such as a task that crashed but was recorded as succeeded, or
  tasks that cannot be read because the coordinator database is in another account's private ledger.
- **Harness graph**: the Coordinator, the LLM roles, executors and specialists as nodes; bus traffic, dispatch and
  LLM calls as edges. A layered or force layout. Clicking a node filters the other views.
- **Timeline**: a lane per node with task bars by health, phase bands, bus, LLM and lifecycle markers, resumes, and a
  memory and GPU strip. Idle stretches are compressed.
- **Artifacts**: every file grouped by the node and task that produced it, with a viewer for JSON, JSONL, text and
  live-following logs, and a side-by-side diff of two text files. Profiler traces show metadata only. The list
  stops at 20,000 files, and the page says so when a session holds more.
- **Drawer**: graph nodes, timeline bars, tasks, bus events and LLM calls open the raw record, related records,
  artifacts and, for LLM calls, the prompt and response.
- **Tabs** for tasks, bus events, LLM calls and the throughput journal.
- **Throughput versus latency and uplift**: the journal tab plots each measured benchmark run's throughput against
  a latency you pick (TTFT, TPOT or end-to-end latency; mean, median or p99), and a foldable uplift panel on the
  timeline's time axis shows each explored variant's throughput gain over the baseline, its mean TTFT and, for the
  baseline and each kept variant, the accuracy state.json records.
- **Tokens and cost**: a Tokens tab with every LLM call's tokens and their cost in USD, priced by the built-in price
  sheet (the one multi-branch spends by). A model the sheet lacks shows its tokens without a cost (see
  [Tokens and cost](#tokens-and-cost)).
- **Replay** with a scrubber and speed control.
- **Narrator**: a summary at the clock, questions and a pinned focus; graph node names in the text are clickable.
- **Explain this**: an Explain button in the drawer and a right-click menu on nodes, tasks, events, calls, journal
  entries and variants; the answer is grounded in the item's own records.

## Tokens and cost

The Tokens tab counts every LLM call's tokens and estimates their cost in US dollars with Hyperloom's price sheet,
`src/hyperloom_common/llm/llm_prices.json`, the one multi-branch spends by. A price the sheet lacks is never guessed:
the call is left unpriced, and the tab says why.

Where it shows:

- **Tokens tab**: totals by token type, cache hit rate, token and cost charts over the run, totals by agent and by
  model, and a price table with each model's prices and what the run used and spent at them.
- `GET /api/tokens` and **Download receipt in JSON** serve the same data as JSON, as does the token timeline file
  (see `--token-dir`).

What is counted:

- **Calls**: the rows of `reports/trace/llm_calls.jsonl`. While a Claude CLI specialist runs, its turns are read from
  its `process.log` and replaced by the ledger's rows once the runner writes them; nothing is counted twice.
- **Side calls**: what a specialist's closing `result` line counts beyond its turns (WebSearch, WebFetch and the
  like) is added when the specialist ends.
- **Token types**: `input` (uncached), `cache_write`, `cache_read`, `output` and `reasoning`. Calls over the OpenAI
  protocol report cached input inside input and reasoning inside output; the Inspector takes them out before pricing.
  Which calls those are follows from what wrote the row: the proposal scorer always, the critic when it reports
  reasoning, and roles and specialists on a Codex backend (models `gpt-`, `o1`, `o3`, `o4`, `codex`). A model such
  as Kimi counts either way, depending on which of them called it.
- **Model**: each call is priced as the model that answered it, so a sub-agent's turn is priced at its own model's
  rates.
- **Time**: each call is placed when its request was sent.

How a call is priced:

```text
cost = ( input       × input price
       + cache_write × cache write price
       + cache_read  × cache read price
       + output      × output price
       + reasoning   × output price ) / 1,000,000
```

- **Prices**: the sheet's, in US dollars per 1M tokens.
- **Model names**: a call's model is looked up as multi-branch looks it up: in lower case, with dots as dashes and
  without a date (`-20261001`) or `[…]` suffix, so `Claude-Opus-5.5` and `claude-opus-5-5-20261001` are one model. A
  name with a provider route or region (`bedrock/…`, `us.anthropic.…`) is not matched.
- **Cache prices**: the sheet lists each model's cache read price; a cache write it lists no price for is billed at
  the input price. The price table's cache price basis says which.
- **Unpriced calls**: a missing price is never counted as $0. A call is unpriced when no model is recorded for it,
  when the sheet does not list its model, or when it ran web searches. The tab counts unpriced calls by reason and
  leaves them out of every part of the cost; a total none of whose calls is priced shows no cost.
- **Not modelled**: long-prompt tiers, batch or priority rates, image and audio prices, and the 1-hour cache write
  rate (every cache write is priced at the 5-minute one). Every call is priced at the sheet's one rate per token
  type, so the cost is an estimate, not an invoice.
- **Cross-check**: for finished Claude CLI specialists, the tab compares the Inspector's cost with the cost the CLI
  reported, which uses Anthropic's list prices. The two differ where the sheet's prices differ from those.

Limits:

- The sheet has no price per web search, so a call that searched is unpriced.
- A model the sheet does not list, such as one served on premises, shows its tokens without a cost.
- To price such a model, add it to `llm_prices.json` under its name as the sheet looks it up, with `input`, `output`
  and `cache_read` in US dollars per 1M tokens, and `cache_write` (and `cache_write_1h`) where it has them. The sheet
  is Hyperloom's, so this also changes what multi-branch records as that model's spend.

## Security

- It binds to `127.0.0.1` by default and accepts requests only for a loopback address or the bound host.
- `.env*` files are never listed or served, and paths can't leave the session directory.
- Everything it serves, and everything it sends to the LLM, is redacted: secret-shaped keys and tokens are masked.
- The narrator's POST endpoints need a JSON body of at most 4 KB, so another web page can't trigger a paid call.
- Treat the page as operator-only all the same: prompts and logs can hold sensitive content that redaction doesn't
  recognise.
