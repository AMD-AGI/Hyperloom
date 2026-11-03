#!/bin/bash
# Install geko + its deps into the geko_build container (python3.12), from our own
# rebuilt hipBLASLt source tree (v1-style: no pre-existing geko_build to copy from).
# Runs inside the container. Idempotent-ish; logs each step.
set -uo pipefail
HB="${WS:?set WS to your geko_agentx workspace root}"/codebases/rocm-libraries/projects/hipblaslt
PIP="pip install --break-system-packages -q"

echo "=== [1/5] rocisa (C-ext, must match container py3.12) ==="
# Build ONE geko job at a time — the source tree is shared NFS. CRITICAL: clean build/ first: a prior
# container's root-owned build/ artifacts on NFS break a later build (geko: "[Errno 17] File exists:
# build/.../geko-0.1.0.dist-info"). Verify rocisa's symbol from /tmp (the source dir's local ./rocisa package
# shadows the installed one and lacks the compiled .so).
cd "$HB/tensilelite/rocisa" && rm -rf build dist ./*.egg-info 2>/dev/null; $PIP . && (cd /tmp && python3 -c "from rocisa import rocIsa; print('rocisa OK')")

echo "=== [2/5] geko ==="
cd "$HB/utilities/geko" && rm -rf build dist ./*.egg-info 2>/dev/null; $PIP . && (cd /tmp && python3 -c "import geko; print('geko OK')")

echo "=== [3/5] invoke ==="
$PIP invoke && invoke --version

echo "=== [4/5] tensilelite requirements ==="
$PIP -r "$HB/tensilelite/requirements.txt" && echo "tensile reqs OK"

echo "=== [5/5] libomp (tensilelite-client runtime dep) ==="
# DPkg lock timeout so this coexists with a concurrent invoke build --install-deps apt run
APT="apt-get install -y -qq -o DPkg::Lock::Timeout=600"
$APT libomp5 2>/dev/null || $APT libomp5-14 2>/dev/null || echo "libomp apt failed"
ldconfig
if [ ! -f /usr/lib/x86_64-linux-gnu/libomp.so ]; then
  src=$(find /usr/lib -name 'libomp.so.*' 2>/dev/null | head -1)
  [ -n "$src" ] && ln -sf "$src" /usr/lib/x86_64-linux-gnu/libomp.so && echo "libomp.so symlinked -> $src"
fi
ls -la /usr/lib/x86_64-linux-gnu/libomp.so 2>/dev/null || echo "libomp.so still missing"

echo "GEKO_INSTALL_DONE"
