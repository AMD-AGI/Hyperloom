---
myst:
    html_meta:
        "description": "Critic and knowledge base variables for Hyperloom: Recipe backend mode, local and framework KB roots, KB Store endpoint and token, draft and warm-start handoffs, and critic token cap."
        "keywords": "Hyperloom, environment variables, critic, knowledge base, KB_STORE_URL, KB_STORE_TOKEN, KNOWLEDGE_STORE_MODE, CRITIC_AGENT_MAX_COMPLETION_TOKENS, recipe KB"
---
# Critic / knowledge base (KB)

The following variables configure the Critic and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `KNOWLEDGE_STORE_MODE`                | `local`                | Exclusive Recipe backend: `local` or `remote`. Ambient KB Store or GBrain credentials do not select remote mode. |
| `KNOWLEDGE_LOCAL_ROOT`                | `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` | Local Recipe/KG root. It is not used for Recipe data in remote mode. |
| `INFERENCE_OPTIMIZER_`<br>`FA_KB_PATH` | `$USER_DATA_PATH/framework-kb`, otherwise `/workspace/hyperloom/framework-kb` | Framework KB root, holding the lessons ledger the FRAMEWORK phase reads and writes. The orchestrator's KB reader and writeback both resolve through it. |
| `KB_STORE_URL`                        | Local Recipe mode: `https://global.primus-safe.amd.com/knowledge-base`; remote mode: unset | Unified KB Service endpoint. Local Recipe mode uses the default only for co-hosted PR Monitor access; IR-3 disables PR Monitor without disabling local Recipe storage when the endpoint is unreachable. Remote Recipe mode requires an explicit value, selects the current Recipe View, replays its Config, Patch, and Kernel columns, then writes one final session at CLOSE. PR Monitor is available at `${KB_STORE_URL}/pr-monitor/v1` (REST/IR-3/Framework/KernelForge) and `${KB_STORE_URL}/pr-monitor/mcp/` (specialist MCP). |
| `KB_STORE_TOKEN`                      | Unset                  | KB Store bearer token. Required when `KNOWLEDGE_STORE_MODE=remote`; transport failures during the final write are non-fatal. |
| `KB_DRAFT_DIR`                        | Runtime-generated      | Internal remote-mode handoff where each Recipe column (`config`, `patch`, `kernel`) stages its knowledge and files before CLOSE merges them. Hyperloom creates and exports it; operators must not set it. A Recipe cannot be built without it. |
| `KB_WARM_START_DIR`                   | Runtime-generated      | Internal remote-mode handoff pointing agents at the downloaded `recipe.json + files/` selected Recipe View. Hyperloom creates and exports it; operators must not set it. |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `CRITIC_AGENT_`<br>`MAX_COMPLETION_TOKENS` | `32000`           | Output-token cap for one critic review call. A reply cut off at the cap is retried once at twice this value and then fails the turn, so the cap is a ceiling rather than a budget: unused headroom is never billed, while a truncated reply bills the whole call and yields nothing. Lower it for a model whose own output limit is smaller. A non-positive or unparseable value logs a warning and falls back to the default. |
