# syntax=docker/dockerfile:1.4
FROM primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix


#install AMD self-signed certificate
RUN curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh | bash


RUN mkdir -p /workspace/hyperloom

#clone Hyperloom
RUN --mount=type=ssh mkdir -p -m 700 /root/.ssh \
    && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /root/.ssh/known_hosts \
    && git clone git@github.com:AMD-AGI/Hyperloom.git /opt/Hyperloom

RUN --mount=type=ssh export KERNEL_FORGE_REPO="git@github.com:AMD-AGI/KernelForge.git" && \
    bash /opt/Hyperloom/src/hyperloom/inference_optimizer/assets/local_setup.sh --deps-root /opt --session-dir /workspace/hyperloom

COPY *.sh /workspace/hyperloom/
RUN chmod +x /workspace/hyperloom/*.sh

RUN cp /opt/Hyperloom/.env.template /opt/Hyperloom/.env

RUN mkdir -p /workspace/hyperloom/logs && \
SKIP_RAY_START=1 KERNEL_AGENT_SKIP_MODEL_LOAD=1 bash -o pipefail -c  "source /workspace/hyperloom/runtime/local-setup.env.sh && /opt/Hyperloom/src/hyperloom/inference_optimizer/assets/install.sh 2>&1 | tee /workspace/hyperloom/logs/pre-install.log"

WORKDIR /opt/Hyperloom
ENTRYPOINT ["/workspace/hyperloom/setup_env.sh"]
