# syntax=docker/dockerfile:1.4
FROM primussafe/sglang:v0.5.11-rocm720-mi30x-profilerfix


#install AMD self-signed certificate
RUN curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh | bash


RUN mkdir -p /workspace/hyperloom

#clone Hyperloom
RUN --mount=type=ssh mkdir -p -m 700 /root/.ssh \
    && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts \
    && git clone git@github.com:AMD-AGI/Hyperloom.git /opt/Hyperloom

RUN --mount=type=ssh export KERNEL_FORGE_REPO="git@github.com:AMD-AGI/KernelForge.git" && \
    bash /opt/Hyperloom/inference_optimizer/scripts/local_setup.sh --deps-root /opt --session-dir /workspace/hyperloom

RUN cp /opt/Hyperloom/.env.template /opt/Hyperloom/.env

COPY *.sh /workspace/hyperloom/
RUN chmod +x /workspace/hyperloom/*.sh


ARG HYPERLOOM_REFRESH=0
#tmp patch
RUN --mount=type=ssh cd /opt/Hyperloom && git pull && git checkout feature/percy/quick-start && git log -2

ENTRYPOINT ["/workspace/hyperloom/setup_env.sh"]

