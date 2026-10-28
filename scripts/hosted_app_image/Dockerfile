# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Application image of a hosted Hyperloom run: the framework (ROCm) image of the optimize flow
# with no Hyperloom in it, plus the bootstrap that builds the wheel of HYPERLOOM_SOURCE_SHA when
# the run starts. Build with scripts/hosted_app_image/build.sh (the build context is the repo root).
ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.22-rocm10-mi30x-20261022
FROM ${BASE_IMAGE}

# The account HYPERLOOM_SPECIALIST_USER names. Its own uid and primary group, no supplementary
# group (so it is never in group 0), a home it can write.
ARG SPECIALIST_USER=hl-specialist
ARG SPECIALIST_ID=2000

USER root
# git fetches the source, setpriv (util-linux) starts a specialist as another user.
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates git util-linux; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --gid "${SPECIALIST_ID}" "${SPECIALIST_USER}"; \
    useradd --create-home --uid "${SPECIALIST_ID}" --gid "${SPECIALIST_ID}" --shell /bin/bash "${SPECIALIST_USER}"; \
    command -v git setpriv

# The claude CLI the specialists spawn from PATH, installed the way install_kernel_tools.sh does
# (Claude Code's native installer, HYPERLOOM_CLAUDE_CODE_VERSION, latest when unset), but under a
# directory every account can read. The codex CLI comes with the wheel's openai-codex-cli-bin.
ARG CLAUDE_CODE_VERSION=latest
RUN set -eux; \
    command -v curl; \
    mkdir -p /opt/claude-code; \
    curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | HOME=/opt/claude-code bash -s -- "${CLAUDE_CODE_VERSION}"; \
    chmod -R a+rX /opt/claude-code; \
    ln -s /opt/claude-code/.local/bin/claude /usr/local/bin/claude; \
    claude --version

COPY src/hyperloom/hosted/app_bootstrap.sh /opt/hyperloom-host/app_bootstrap.sh
RUN chmod 0755 /opt/hyperloom-host/app_bootstrap.sh

CMD ["/opt/hyperloom-host/app_bootstrap.sh"]
