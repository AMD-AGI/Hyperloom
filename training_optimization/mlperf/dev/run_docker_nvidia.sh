# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

docker run -it --rm \
    --net=host --uts=host \
    --ipc=host --gpus all \
    --security-opt=seccomp=unconfined \
    --volume=/mnt/shared/mlperf_llama31_8b/data:/data \
    --volume=/mnt/shared/mlperf_llama31_8b/model:/model \
    --volume=/data/experiments/mlperf_llama31_8b/:/results \
    --volume $(pwd):/workspace/code/ \
    --name gpt-oss-20b-training-b200-`whoami` rocm/mlperf:gpt-oss-20b-training-b200_2026-01-30-19-37-52