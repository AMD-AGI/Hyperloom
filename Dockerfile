# syntax=docker/dockerfile:1
# Hyperloom vLLM local-mode image — multimodal (VL) benchmarking on MI355X.
#
# This image is built on the vLLM ROCm base and bakes in all Hyperloom runtime
# dependencies (Magpie, InferenceX, TraceLens, Ray, bench deps). It is designed
# to be used with the feat/vl-model-support branch mounted at runtime:
#
#   -v $PWD:/workspace/Hyperloom
#
# Requirements:
#   - SSH key authorized for AMD-AGI repos on GitHub loaded in the host agent.
#   - docker buildx (BuildKit) for --ssh forwarding.
#
# Pre-build checks:
#   ssh-add $HOME/.ssh/id_amd       # load key into agent
#   ssh -T git@github.com           # verify agent access
#
# Build:
#   docker buildx build --ssh default -t hyperloom-vl-vllm-local-$USER .
#
#   Override defaults:
#     --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:v0.24.0
#     --build-arg MAGPIE_REF=<commit-sha>   # pin for reproducibility
#
# Run:
#   docker run -d \
#     --name hyperloom-vl-vllm-local-$USER \
#     --shm-size 64g \
#     --device /dev/kfd \
#     --device /dev/dri \
#     --group-add video \
#     -v $PWD:/workspace/Hyperloom \
#     -v /data2/hf_hub_cache:/models \
#     -v "$SSH_AUTH_SOCK:/ssh-agent" \
#     -e SSH_AUTH_SOCK=/ssh-agent \
#     -e USER_DATA_PATH=/workspace/hyperloom \
#     -e SAFE_API_KEY="$SAFE_API_KEY" \
#     -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
#     -e ANTHROPIC_CUSTOM_HEADERS="$ANTHROPIC_CUSTOM_HEADERS" \
#     hyperloom-vllm-local-$USER \
#     tail -f /dev/null

ARG BASE_IMAGE=amdsiloai/vllm-private:mlperf6.1-q3vl-r72-w4a4-fusemoe-20260620
FROM $BASE_IMAGE

# MAGPIE_REF defaults to main so the baked clone always contains
# vllm_mi355x_mm.sh (added after the default Hyperloom pin b1d4dcd).
# Override with --build-arg MAGPIE_REF=<sha> for reproducible builds.
ARG MAGPIE_REF=main

# Hyperloom branch to clone for the build-time dep install. Defaults to the
# dev branch which carries the Docker tooling + kernel-agent .env fixes.
ARG HYPERLOOM_BRANCH=feat/vl-model-support-dev

ENV HYPERLOOM_PATH=/workspace/Hyperloom \
    USER_DATA_PATH=/workspace/hyperloom

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates openssh-client curl \
    && rm -rf /var/lib/apt/lists/*

# --- §1: Clone Hyperloom (feat/vl-model-support) for build-time dep install ---
# The working tree is NOT kept in the image; it is mounted at runtime via
# -v $PWD:/workspace/Hyperloom. The clone here is used only to drive
# install.sh so ALL deps (Magpie, InferenceX, inference_optimizer, plus the
# kernel-agent stack: Ray, TraceLens, GEAK code) are baked into the image.
# The GEAK RAG index is the one thing NOT baked (needs the GPU; see below) —
# it builds on first container start. framework-agent (the `fa` CLI) is still
# skipped — it is only needed for the FRAMEWORK_PR phase and pulls extra deps.
#
# install.sh requires SAFE_API_KEY + OPENAI_BASE_URL to pass its credential
# preflight. The heavy work (git clone + pip install) never calls the gateway,
# so dummy values are safe here. Real keys must be supplied at runtime via -e.
#
# Build-time gotchas handled here:
#   * KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0 — GEAK is AMD/ROCm-native and its
#     semantic RAG index builds in ~1 min on an AMD GPU (torch "cuda" == HIP on
#     ROCm). But `docker build` never attaches the GPU (/dev/kfd + /dev/dri; no
#     --gpus flag, runc runtime), so at build time the only options are a crash
#     (cuda with no device node) or a ~1.5h CPU embedding run. We therefore bake
#     GEAK's CODE (clone + pip) here and DEFER the index build to first container
#     start, where the GPU IS attached and the index builds fast. install.sh /
#     the CLI preflight builds it on first run.
#   * ray start is fail-soft during build (no live head persists across layers);
#     the ray head is (re)started at container runtime by install.sh / the CLI
#     preflight.
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh \
    && ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts \
    && git clone --depth 1 \
        --branch "${HYPERLOOM_BRANCH}" \
        git@github.com:AMD-AGI/Hyperloom.git \
        /opt/hyperloom-build \
    && MAGPIE_REF="${MAGPIE_REF}" \
       SAFE_API_KEY=build \
       OPENAI_BASE_URL=https://invalid.local/v1 \
       USER_DATA_PATH=/workspace/hyperloom \
       KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0 \
       bash /opt/hyperloom-build/inference_optimizer/scripts/install.sh \
           --skip-framework-agent \
    && python3 -m pip install --quiet --break-system-packages \
           /opt/hyperloom-build[test] \
    && rm -rf /opt/hyperloom-build


# --- §2: Claude Code CLI ---
# IS_SANDBOX=1 allows --dangerously-skip-permissions inside the container.
WORKDIR /root
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && chmod -R a+rX /root/.local \
    && chmod a+x /root /root/.local /root/.local/bin /root/.local/share 2>/dev/null || true
ENV PATH="/root/.local/bin:${PATH}"

ENV IS_SANDBOX=1 \
    ANTHROPIC_API_KEY="dummy" \
    ANTHROPIC_BASE_URL="https://llm-api.amd.com/Anthropic" \
    ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: YOUR_LLM_KEY" \
    ANTHROPIC_MODEL="Claude-Opus-4.8" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="Claude-Opus-4.8" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="Claude-Sonnet-4.6" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="Claude-Haiku-4.5-20251001" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1


# --- §3: Runtime env defaults (all overridable via docker run -e) ---
# SAFE_API_KEY / OPENAI_BASE_URL / ANTHROPIC_CUSTOM_HEADERS must be set at
# runtime; placeholders here so the image starts without errors.
ENV SAFE_API_KEY=YOUR_API_KEY \
    OPENAI_BASE_URL=https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1 \
    TRACELENS_ROOT="" \
    TRACELENS_INTERNAL_ROOT=""

WORKDIR ${HYPERLOOM_PATH}
