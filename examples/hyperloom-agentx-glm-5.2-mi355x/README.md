# Native Magpie AgentX: GLM-5.2 on MI355X

This example uses Magpie commit
3642ce66ae46ca4dc125340b3d14a3f4640c369b and InferenceX commit
3d5581562f643f9bdeb8410cd924e2c70906c966. The YAML omits
`inferencex_path`, so preflight reuses or clones that tested pin. An optional
`benchmark.inferencex_path` only nominates a preferred writable checkout:
preflight replaces a missing or wrong-revision checkout with its pinned clone.
Pinned Magpie currently interpolates that path into an unquoted local
`bash -c`, so a resolved path containing whitespace or shell metacharacters is
rejected before launch.

Run Hyperloom from inside the exact recipe image:

    python -m hyperloom.inference_optimizer.cli optimize \
      --benchmark-config examples/hyperloom-agentx-glm-5.2-mi355x/benchmark.yaml \
      --max-hours 3

The explicit three-hour budget is sized for one canonical baseline. The normal
two-hour default is commonly shorter than model load, warmup, drain, and the
3600-second measurement together; increase it if additional rounds are intended.

As written, the source model id is fetched through the normal Hugging Face
path, and the native InferenceX subprocess receives no local `MODEL_PATH`
override. To use a mounted checkpoint, keep the canonical recipe id in
`benchmark.model` and add `--model /absolute/path/to/checkpoint`.

The YAML source switch automatically enables the session-wide AgentX contract;
HYPERLOOM_AGENTX is only needed by the legacy environment-driven entry point.
At `CONC=8`, the recipe has one GLM-5.2 arm: TP4/EP4 with DRAM+HiCache.
`envs.TP` is not an arm selector and is intentionally absent. If a concurrency
belongs to multiple arms, change `agentx: enable` into the object form shown in
the YAML comments and select the arm under `agentx.selector`; keep
`envs.CONC` as the fixed measurement point. Magpie resolves and persists the
recipe-owned TP/PP/PCP/EP, model prefix, HiCache, DRAM, CPU-memory, and
fingerprint fields.

The YAML `docker_image` overrides/pins the effective recipe image and is folded
into its fingerprint; an optional pre-existing `HYPERLOOM_IMAGE` must match it.
Neither field starts a container or proves which outer image is actually
running, so launch Hyperloom inside that image before invoking the command.

This integration is measurement-only until the pinned InferenceX launcher
exposes a fingerprinted optimizer-argv hook. Hyperloom therefore rejects
server-argument candidates, GEAK kernel optimization, and the post-optimization
concurrency sweep instead of measuring an unchanged server under a candidate
label. Native measurement does not collect a PyTorch trace. If PRELUDE schedules
roofline/profile analysis, Hyperloom uses a generic-server compatibility path;
that trace is diagnostic and is not recipe-identical to the native AgentX run.
It skips TraceLens/CK framework source patches to preserve subsequent native
measurements, so trace annotation coverage may be lower.
