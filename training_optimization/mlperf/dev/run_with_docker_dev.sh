#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euxo pipefail

# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODELDIR:?MODELDIR not set}"
: "${LOGDIR:?LOGDIR not set}"
: "${CONFIG_NAME:?CONFIG_NAME not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=0}"
: "${MLPERF_RULESET:=5.1.0}"
: "${UTILITIES:="$(pwd)/../../utilities"}"

#    Profiling settings
: "${CONT_NAME:=dev-${CUSTOM_TAG}}"
: "${PROFILER:=""}"
: "${ROCPROF_PROFILER:='none'}"
: "${ROCPROF:='/opt/rocm/bin/rocprofv3'}"
: "${NGPU:=1}"
: "${PROF_WARMUP_STEPS:=80}"
: "${PROF_ACTIVE_STEPS:=10}"
: "${HIPBLASLT_LOG:=0}"
: "${GEMM_OFFLINE_TUNING:=0}"
: "${GEMM_USE_TUNING_RESULTS:=0}"
: "${ENABLE_TRAINING_METRICS_COLLECTION:=0}"
: "${LOG_FREQ:=0}"
: "${RUN_OPTUNA:=0}"
: "${OPTUNA_CONFIG:=''}"
: "${ENABLE_MEMORY_PROFILING:=0}"
: "${POWER_PROFILING:=0}"
: "${PROF_OUTPUT_PATH:=$LOGDIR/artifacts}"
: "${TRACELENS_EXTENSION_FILE:=""}"
: "${HF_TOKEN:=""}"

# Other vars
readonly _config_file="${CONFIG_NAME}"
echo "CONFIG_NAME: ${CONFIG_NAME}"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${MODELDIR}:/model/" "--volume=$(pwd):/workspace/code" "--volume=$(pwd)/../../AMD:/workspace/AMD" "--volume=${UTILITIES}:/workspace/utilities" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"
mkdir -p "${PROF_OUTPUT_PATH}"


# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODELDIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(PROFILER)
_config_env+=(LOGDIR)
_config_env+=(HIPBLASLT_LOG)
_config_env+=(GEMM_OFFLINE_TUNING)
_config_env+=(GEMM_USE_TUNING_RESULTS)
_config_env+=(HF_TOKEN)
_config_env+=(SEED)
if [[ $PROFILER = "rpd" ]]; then
    _config_env+=(RPD_TRACE_PYTHON)
fi
if [[ $PROFILER = "torchprof" ]]; then
    _config_env+=(TORCHPROF_OUTPUT)
    _config_env+=(TORCHPROF_OUTPUT_DIR)
    _config_env+=(TORCHPROF_VERBOSE)
    _config_env+=(TORCHPROF_MAXROWS)
    _config_env+=(TORCHPROF_PROFILE_MEMORY)
    _config_env+=(TORCHPROF_WITH_STACK)
    _config_env+=(TORCHPROF_RECORD_SHAPES)
    _config_env+=(TORCHPROF_WITH_FLOPS)
    _config_env+=(PROF_WARMUP_STEPS)
    _config_env+=(PROF_ACTIVE_STEPS)
    _config_env+=(PROF_REPETITIONS)
fi
if [[ $ROCPROF_PROFILER != 'none' ]]; then
    _config_env+=(ROCPROF_PROFILER)
    _config_env+=(ROCPROF)
fi
if [[ $PROFILER != "" ]]; then
    _config_env+=(PROF_WARMUP_STEPS)
    _config_env+=(PROF_ACTIVE_STEPS)
    _config_env+=(PROF_OUTPUT_PATH)
fi

if [[ $ENABLE_TRAINING_METRICS_COLLECTION -eq 1 ]]; then
    _config_env+=(LOG_FREQ)
fi

if [[ $RUN_OPTUNA -eq 1 ]]; then
    _config_env+=(OPTUNA_CONFIG)
fi


if [[ $ENABLE_MEMORY_PROFILING -eq 1 ]]; then
    _config_env+=(ENABLE_MEMORY_PROFILING)
fi


echo "TEST"
echo ${_config_env[@]}
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
if [[ $POWER_PROFILING -eq 1 ]]; then
  docker run --rm --init --detach \
      --net=host --uts=host --ipc=host \
      --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
      --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
      --security-opt=seccomp=unconfined \
      --group-add video \
      --privileged \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
else
  docker run --rm --init --detach \
      --net=host --uts=host --ipc=host \
      --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
      --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
      --security-opt=seccomp=unconfined \
      --group-add video \
      --privileged \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity
fi

# Make sure container has time to finish initialization
sleep 5
docker exec "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    # Use existing SEED if set; otherwise use a new RANDOM value
    _config_env+=(--env=SEED="${SEED:-$RANDOM}")
    echo 'launching experiment using:'  ${_config_env[@]} ${_cont_name} /workspace/code/dev/run_and_time.sh
    docker exec ${_config_env[@]} ${_cont_name} bash /workspace/code/dev/run_and_time.sh
  ) | grep --line-buffered -v "connected peer ranks" | tee "${_logfile_base}_${_experiment_index}.log"

  if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log"
  fi

done

echo "Number of experiments $NEXP"

: "${BASE_DIR:=/workspace/AMD}"
: "${OUTPUT_DIR:=$LOGDIR}"
: "${CODE_DIR:=/workspace/code}"
: "${RESULTS_DIR:=${LOGDIR}/logs}"
: "${MODEL_NAME:=gpt-oss-20b}"
: "${SYSTEM:=MI355X_EPYC_9575F}"
: "${FRAMEWORK:=pytorch}"
: "${SYSTEM_JSON:=/workspace/AMD/systems/${SYSTEM}_${FRAMEWORK}.json}"

# Convert log file to the proper format for the RCP checker
# Package submission
if [[ "${NEXP}" -ge 10 ]]; then
  echo "Running RCP check on /results"
  mkdir -p "${LOGDIR}/logs" && mv ${LOGDIR}/*.log "${LOGDIR}/logs/"
  docker exec "${_config_env[@]}" "${_cont_name}" python /workspace/utilities/select_best_runs.py ${MODEL_NAME} /results/logs /results/logs

  # create submission directory and copy files accordingly
  docker exec "${_config_env[@]}" "${_cont_name}" bash /workspace/utilities/package_mlperf_submission.sh --base_dir ${BASE_DIR} --output_dir /results --model ${MODEL_NAME} --code ${CODE_DIR} --results /results/logs --system ${SYSTEM} --system_json ${SYSTEM_JSON} --framework ${FRAMEWORK}

fi

if [[ $PROFILER = "torchprof" ]] ; then
    docker exec "${_config_env[@]}" "${_cont_name}" bash -c "
      bash /workspace/utilities/metrics/profiling/tracelens_perf_report.sh  $TORCHPROF_OUTPUT_DIR  $TRACELENS_EXTENSION_FILE
    "
fi

# Run ROCm Profiler
if [[ $ROCPROF_PROFILER != 'none' ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" chmod +x ./dev/run_and_time.sh
fi
if [[ $ROCPROF_PROFILER == 'rocprof' ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" rocprofv3 --stats --hip-trace --hsa-trace --rccl-trace --kernel-trace --memory-copy-trace --hsa-amd-trace -d /results/rocprof -o mlperf -- ./dev/run_and_time.sh ${NGPU}
fi
if [[ $ROCPROF_PROFILER == 'rocprof-compute' ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" rocprof-compute profile -b 10 11 --no-roof -n mlperf -p /results/rocprof --verbose -- ./dev/run_and_time.sh ${NGPU}
fi
if [[ $ROCPROF_PROFILER == 'rocprof-compute-roofline-only' ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" rocprof-compute profile -b SQ SQC SPI --roof-only -n mlperf -p /results/rocprof-roofline --verbose -- ./dev/run_and_time.sh ${NGPU}
  mkdir -p "$LOGDIR/rocprof-roofline"
  mv $LOGDIR/rocprof/*.pdf "$LOGDIR/rocprof-roofline"
  mv $LOGDIR/rocprof/roofline.csv "$LOGDIR/rocprof-roofline"
fi
echo "ENABLE_TRAINING_METRICS_COLLECTION is set to $ENABLE_TRAINING_METRICS_COLLECTION" 
if [[ $ENABLE_TRAINING_METRICS_COLLECTION -eq 1 ]]; then
  echo "Parsing training metrics"
  mkdir -p "${LOGDIR}/logs" && mv ${LOGDIR}/*.log "${LOGDIR}/logs/"
  docker exec "${_config_env[@]}" "${_cont_name}" python3 /workspace/utilities/metrics/training/parse_training_metrics.py --model_name "gpt-oss-20b"
fi

if [[ $RUN_OPTUNA -eq 1 ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" bash /workspace/utilities/optuna/run_optuna.sh
fi


if [[ $POWER_PROFILING -eq 1  ]]; then
  docker exec "${_config_env[@]}" "${_cont_name}" bash -c '
  sudo pip3 install -i http://ppdtool.amd.com:6543/simple papi2 --trusted-host ppdtool.amd.com --upgrade;
  export AGT_FOLDER="/workspace/utilities/gfm_profiling/agt";
  mkdir -p "$AGT_FOLDER";
  wget -O "$AGT_FOLDER/AGT.tar.gz" \
      http://gpudiagnostics.amd.com/release/tools/releaseManagement/Tools/atitool/EngineeringBuild/4.1.89.0/AGT_Internal_4.1.89.0_Linux_64bit.tar.gz;
  tar -zxvf /workspace/utilities/gfm_profiling/agt/AGT.tar.gz -C "$AGT_FOLDER";
  /workspace/utilities/gfm_profiling/runme.sh
  '
fi
