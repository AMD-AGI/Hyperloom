# classify — model class detection

**Family**: `prep` · **Cost**: ~1‑2 min · **Risk**: zero

Detects the model architecture class so the scheduler can pick correct
priors (DESIGN §9.2). Read‑only; no GPU work.

Heuristics (see `score_priors.classify_model`):

- `gpt-oss` / `llama` / `qwen-dense` → DENSE
- `deepseek` / model with MLA layers → MOE_MLA
- `mixtral` / SWA‑style → MOE_SWA
- `kimi` / NSA‑bearing → MOE_MLA_NSA
- otherwise → UNKNOWN (default 5.0 priors across the board)

Outputs:

- `update_state` `model_class=...`
- `send_message` topic=`event` summarizing the class chosen.
