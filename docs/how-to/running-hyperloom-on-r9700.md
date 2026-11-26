---
myst:
    html_meta:
        "description": "Run a single-GPU SGLang session on Radeon AI PRO R9700 or R9600D using Hyperloom's gfx1201 default launch."
---
# Run SGLang on Radeon AI PRO R9700 or R9600D

Both boards use the `gfx1201` SGLang launch. Use the board's product name for
`--gpu-type`: `r9700` or `r9600d`, not `gfx1201`. The example runs Qwen3.5-9B
on one GPU, with text requests only.

## Prepare the serving environment

Use a fresh `rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0`
container with the target GPU exposed and the model weights mounted. This
build includes gfx1201; an MI-only SGLang image is not interchangeable.
Follow [Install Hyperloom](../install/install.md) in that container using its
`/opt/venv/bin/python3` interpreter. For the GEAK kernel phase, use a GEAK
checkout that includes [R9700 SGLang support](https://github.com/AMD-AGI/GEAK/pull/551)
or [R9600D SGLang support](https://github.com/AMD-AGI/GEAK/pull/637), respectively.

Set `MODEL_PATH` to the weights **inside** the container. Prefer local storage:
each configuration change can restart the server and reload the model.

## Optional warm-up

Start a stock server in the container on a free port:

```bash
SGLANG_USE_AITER=0 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --port 8399 --context-length 4096 \
  --language-only --dtype bfloat16 --max-running-requests 64 \
  --attention-backend triton --mem-fraction-static 0.9 --disable-radix-cache
```

Once `curl -fsS http://localhost:8399/health` succeeds, run the client from a
second shell in the same container with the same `MODEL_PATH`:

```bash
/opt/venv/bin/python3 -m sglang.bench_serving --backend sglang --port 8399 \
  --model "$MODEL_PATH" --dataset-name random-ids --num-prompts 150 \
  --random-input-len 512 --random-output-len 128 --random-range-ratio 1.0 \
  --max-concurrency 15
```

Stop the warm-up server and check that GPU memory is free before launching
Hyperloom. This benchmark is a smoke check, not the session baseline.

## Start the session

```bash
HYPERLOOM_EVAL_LIMIT=400 /opt/venv/bin/python3 -m hyperloom optimize \
  --model "$MODEL_PATH" --framework sglang --gpu-type r9700 \
  --tp 1 --conc 15 --isl 512 --osl 128 --precision bf16 --max-model-len 4096 \
  --max-hours 10 --server-args="--language-only --dtype bfloat16 --max-running-requests 64"
```

On R9600D change only `--gpu-type r9700` to `--gpu-type r9600d`. The server
flags select Qwen's language weights and bound the number of concurrent
requests for its Mamba cache. When no recipe serves the workload,
`inference_suite` selects Triton attention, static memory fraction `0.9`,
no radix cache and `SGLANG_USE_AITER=0`. No Magpie launcher is required.

The image above predates the Triton EAGLE/MTP verify-mask sizing fix. Before
benchmarking speculative decoding, check that the serving build includes the
[SGLang fix](https://github.com/sgl-project/sglang/pull/32920); a successful
non-speculative warm-up does not exercise that path.
