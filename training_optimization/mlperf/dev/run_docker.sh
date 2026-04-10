# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

docker run -it \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt=seccomp=unconfined \
    --group-add video \
    --privileged \
    --volume $(pwd):/workspace/code/ \
    --volume /data/mlperf_llama31_8b/data:/data \
    --volume /data/mlperf_llama31_8b/model:/model \
    --volume=$(pwd)/../../AMD:/workspace/AMD \
    --volume=$(pwd)/../../utilities:/workspace/utilities \
    --name gpt-oss-20b-training-`whoami` rocm/mlperf:gpt-oss-20b-primus-turbo-test #rocm/mlperf:gpt-oss-20b-primus-2026-01-19-20-44-11