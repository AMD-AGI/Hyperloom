ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    uuid-runtime \
    sqlite3 libsqlite3-dev \
    libfmt-dev

# Install library dependencies
WORKDIR /workspace/deps

# RPD
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData \
    && cd rocmProfileData \
    && cd rocpd_python \
    && python3 setup.py bdist_wheel \
    && pip install dist/*.whl \
    && cd .. \
    && cd rpd_tracer \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl \
    && cd .. \
    && sed -i 's/pip install --user \./pip install ./g' rocpd_python/Makefile \
    && sed -i 's/pip install --user \./pip install ./g' rpd_tracer/Makefile \
    && sed -i '/strncpy(row\.completionSignal/d' rpd_tracer/CuptiDataSource.cpp \
    && make; make install

RUN pip install git+https://github.com/AMD-AGI/TraceLens.git
# RUN pip install openpyxl==3.1.5

WORKDIR /workspace/code
