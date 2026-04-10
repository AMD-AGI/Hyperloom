ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Update Primus-Turbo to target commit on main branch
RUN cd /workspace/Primus-Turbo && \
    rm -rf build/ primus_turbo.egg-info/ 3rdparty/composable_kernel/ -rf && \
    git checkout . && \
    git checkout main && \
    git pull origin main && \
    git checkout c79e0f359a0c0952c634f3b48312cd6341b5d843 && \
    git submodule update --init --recursive

# Copy patch file and apply it to Megatron-LM before installation
WORKDIR /workspace/code/patches
COPY patches/megatron_validation_consumed_samples.patch .

RUN cd /workspace/Primus/third_party/Megatron-LM && \
    git apply /workspace/code/patches/megatron_validation_consumed_samples.patch && \
    pip install -e . --no-deps --no-build-isolation

# Rebuild Primus-Turbo: uninstall aiter, clean build artifacts, reinstall
RUN pip uninstall amd-aiter aiter -y && \
    cd /workspace/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" pip3 install --no-build-isolation . -v

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.1-py3-none-any.whl /tmp/primus_mllog-0.1.1-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.1-py3-none-any.whl && rm /tmp/primus_mllog-0.1.1-py3-none-any.whl
