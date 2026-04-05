#!/usr/bin/bash

sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

HF_HUB_CACHE_MOUNT="/shareddata/hf_hub_cache_$(hostname)/"
PORT=8888

server_name="bmk-server"

TRACELENS_ARGS=()
if [[ -n "${TRACELENS_REPO:-}" && -d "${TRACELENS_REPO}" ]]; then
    TRACELENS_ARGS+=(-v "${TRACELENS_REPO}:/workspace/TraceLens-internal:ro")
    TRACELENS_ARGS+=(-e TRACELENS_REPO=/workspace/TraceLens-internal)
fi
# When TRACELENS_PATCHES=1 but no local repo, the container will clone it
TRACELENS_ARGS+=(-e TRACELENS_PATCHES -e TRACELENS_GIT_URL -e TRACELENS_GIT_REF -e TRACELENS_VLLM_VERSION)

set -x
docker run --rm --ipc=host --shm-size=16g --network=host --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v $HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
-v $GITHUB_WORKSPACE:/workspace/ -w /workspace/ \
-e HF_TOKEN -e HF_HUB_CACHE -e MODEL -e TP -e CONC -e MAX_MODEL_LEN -e PORT=$PORT \
-e ISL -e OSL -e PYTHONPYCACHEPREFIX=/tmp/pycache/ -e RANDOM_RANGE_RATIO -e RESULT_FILENAME -e RUN_EVAL -e RUNNER_TYPE \
-e PROFILE -e SGLANG_TORCH_PROFILER_DIR -e VLLM_TORCH_PROFILER_DIR -e VLLM_RPC_TIMEOUT \
-e FRAMEWORK \
"${TRACELENS_ARGS[@]}" \
--entrypoint=/bin/bash \
$IMAGE \
benchmarks/single_node/"${EXP_NAME%%_*}_${PRECISION}_mi300x.sh"
