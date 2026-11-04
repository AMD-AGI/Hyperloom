# geko setup — install in the container (Steps 1–3)

## Environment

| Item | v1 (original) | v2 (new) |
|---|---|---|
| Docker image | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | `vllm/vllm-openai-rocm:nightly-*` (or any ROCm 7.2.x image) |
| ROCm | 7.2.0 | 7.2.3 |
| Python | 3.10 | 3.12 |
| `/opt/venv` | yes | no |
| hipblaslt pre-installed | 1.5.0 (built in-container) | 1.2.2 (needs upgrade) |
| Build from source | ~60–90 min | **~5 min** (copy artifacts) |

---

## Critical: Container must be started with `--ipc=host`

Without `--ipc=host`, the Tensile benchmark subprocess triggers a GPU fault that **kills the entire container** mid-tune.
This was the root blocker on ROCm 7.2.3 images; it does not occur on ROCm 7.2.0 sglang images.

```bash
# IMAGE: fill in your actual image tag, e.g. vllm/vllm-openai-rocm:nightly-<commit>
IMAGE=<your-image-tag>

docker run -d --name geko_vllm \
  --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=0 \
  --entrypoint /bin/bash \
  $IMAGE \
  -c "sleep infinity"
```

> **`--entrypoint /bin/bash`** is required for vllm images — they have a custom entrypoint
> that interprets all arguments as `vllm serve` flags, causing immediate exit otherwise.

---

## Prerequisites

An existing `geko_build` container with hipblaslt 1.5.0 built and geko installed
(see `geko_tune_setup_guide.md` for how to create it). All artifacts are copied from there.

---

## Step 1: Export artifacts from `geko_build`

Run on the **host**. Assumes `geko_build` mounts a host directory to `/work` — replace
`/work/tmp_debs` with any path accessible to both containers or to the host directly.

```bash
# WORKDIR: host path shared with geko_build via -v mount (its /work)
WORKDIR=<your-host-workdir>   # e.g. /home/user/mywork/agentx

docker exec geko_build bash -c "
mkdir -p /work/tmp_debs

# hipblaslt 1.5.0 deb packages
# The git hash suffix (e.g. a3eb50b9ea) comes from your geko_build; list the files first:
#   ls /tmp/rocm-libraries/projects/hipblaslt/build/release/*.deb
cp /tmp/rocm-libraries/projects/hipblaslt/build/release/hipblaslt_*.deb                 /work/tmp_debs/
cp /tmp/rocm-libraries/projects/hipblaslt/build/release/hipblaslt-dev_*.deb             /work/tmp_debs/
cp /tmp/rocm-libraries/projects/hipblaslt/build/release/hipblaslt-benchmarks_*.deb      /work/tmp_debs/
cp /tmp/rocm-libraries/projects/hipblaslt/build/release/hipblaslt-clients-common_*.deb  /work/tmp_debs/

# tensilelite source + geko source
tar czf /work/tmp_debs/tensilelite_geko.tar.gz \
  -C /tmp/rocm-libraries/projects/hipblaslt \
  --exclude='tensilelite/__pycache__' --exclude='tensilelite/*.pyc' \
  tensilelite utilities/geko

# cmake modules + shared deps (needed to build rocisa for new Python version)
tar czf /work/tmp_debs/rocm_lib_deps.tar.gz \
  -C /tmp/rocm-libraries cmake shared

# hipblaslt root CMake files
tar czf /work/tmp_debs/hipblaslt_cmake.tar.gz \
  -C /tmp/rocm-libraries/projects/hipblaslt \
  CMakePresets.json CMakeLists.txt cmake/

# Minimal build/release artifacts (device-library + bench binary — 2 MB)
tar czf /work/tmp_debs/build_release_minimal.tar.gz \
  -C /tmp/rocm-libraries/projects/hipblaslt/build/release \
  device-library/MatchTable.yaml clients/hipblaslt-bench

# Pre-built tensilelite-client binary (~30 MB, saves LLVM build inside new container)
# build_tmp/ lives wherever geko previously ran a tune — check: find / -name tensilelite-client 2>/dev/null
# Common locations: /sgl-workspace/build_tmp  or  /app/build_tmp  or  /tmp/rocm-libraries/.../tensilelite/build_tmp
tar czf /work/tmp_debs/build_tmp.tar.gz \
  -C <parent-dir-containing-build_tmp> build_tmp

# gfx950 Tensile logic yaml files (needed for full_libs creation)
tar czf /work/tmp_debs/hipblaslt_logic_gfx950.tar.gz \
  -C /tmp/rocm-libraries/projects/hipblaslt \
  library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950
"
```

---

## Step 2: Install everything inside the new container

```bash
TARGET=geko_vllm   # name of the new container (set above)
DEBS=$WORKDIR/tmp_debs

docker cp $DEBS/. $TARGET:/tmp/hipblaslt_debs/

docker exec $TARGET bash -c '
set -e

### 1. hipblaslt 1.5.0 deb packages
# dpkg may complain about version conflicts; use --force-all to configure stubs
dpkg -i /tmp/hipblaslt_debs/hipblaslt-clients-common_*.deb || true
dpkg -i /tmp/hipblaslt_debs/hipblaslt_1.5.0-*.deb
dpkg -i /tmp/hipblaslt_debs/hipblaslt-dev_*.deb
dpkg -i /tmp/hipblaslt_debs/hipblaslt-benchmarks_*.deb || true
dpkg --configure --force-all hipblaslt-clients-common hipblaslt-benchmarks
# Note: rocm-libs meta-package will be removed due to version conflict — this is OK
hipblaslt-bench --version   # expect: hipBLASLt version: 100500

### 2. Unpack rocm-libraries tree
mkdir -p /tmp/rocm-libraries && cd /tmp/rocm-libraries
tar xzf /tmp/hipblaslt_debs/rocm_lib_deps.tar.gz            # cmake/ shared/
mkdir -p projects/hipblaslt && cd projects/hipblaslt
tar xzf /tmp/hipblaslt_debs/tensilelite_geko.tar.gz         # tensilelite/ utilities/geko/
tar xzf /tmp/hipblaslt_debs/hipblaslt_cmake.tar.gz          # CMakePresets.json CMakeLists.txt cmake/
tar xzf /tmp/hipblaslt_debs/hipblaslt_logic_gfx950.tar.gz  # library/src/.../gfx950/
mkdir -p build/release && cd build/release
tar xzf /tmp/hipblaslt_debs/build_release_minimal.tar.gz   # device-library/ clients/hipblaslt-bench

### 3. Build rocisa for the container Python version
# rocisa is a C extension that must match the Python ABI — cannot copy from geko_build
cd /tmp/rocm-libraries/projects/hipblaslt/tensilelite/rocisa
pip install . -q

### 4. Install geko + dependencies
cd /tmp/rocm-libraries/projects/hipblaslt/utilities/geko
pip install . -q
pip install invoke -q
pip install -r /tmp/rocm-libraries/projects/hipblaslt/tensilelite/requirements.txt -q

### 5. libomp (required by tensilelite-client at runtime)
# libomp-dev package name varies by Ubuntu/LLVM version; find the right one first:
#   apt-cache search libomp | grep -v doc
# On Ubuntu 22.04 with LLVM 14: libomp5-14
# On Ubuntu 20.04: libomp5
apt --fix-broken install -y -qq   # removes conflicting rocm-libs stub first
apt-get install -y -qq libomp5    # adjust package name if not found
ldconfig
# tensilelite-client needs unversioned libomp.so; Ubuntu only ships the versioned one
[ -f /usr/lib/x86_64-linux-gnu/libomp.so ] || \
  ln -s "$(find /usr/lib -name 'libomp.so.*' | head -1)" /usr/lib/x86_64-linux-gnu/libomp.so

### 6. Pre-built tensilelite-client (avoids LLVM rebuild inside container)
mkdir -p /app && cd /app
tar xzf /tmp/hipblaslt_debs/build_tmp.tar.gz
# hash.txt must match get_git_revision_hash() output; with no .git dir it returns today YYYY-MM-DD
echo "$(date +%Y-%m-%d)" > /app/build_tmp/hash.txt

### 7. Symlink build/release/clients/* to system versions
# geko hardcodes build/release/clients/hipblaslt-bench — must point to installed 1.5.0 binary
CLIENTS=/tmp/rocm-libraries/projects/hipblaslt/build/release/clients
mv $CLIENTS/hipblaslt-bench $CLIENTS/hipblaslt-bench.bak
ln -s /opt/rocm/bin/hipblaslt-bench      $CLIENTS/hipblaslt-bench
ln -s /opt/rocm/bin/hipblaslt_gentest.py $CLIENTS/hipblaslt_gentest.py
# template yaml path may include versioned rocm dir
TMPL=$(find /opt/rocm* -name hipblaslt_template.yaml 2>/dev/null | head -1)
ln -s "$TMPL" $CLIENTS/hipblaslt_template.yaml
'
```

---

## Step 3: three critical `geko --tune` patches for hipBLASLt 1.5.0

Without these, `geko --tune` fails end-to-end. All three are the "fix geko --tune e2e errors" changes; they make
tensilelite emit **msgpack** device libraries (fixes 1–2) and make geko's bench input carry **`scale_type`**
(fix 3) — both required by hipBLASLt 1.5.0. Each snippet is **idempotent** (prints `OK`/`SKIP`). Resolve the two
source roots first (they move with ROCm/Python versions):

```bash
docker exec $TARGET bash -c '
set -e
TENSILE=/tmp/rocm-libraries/projects/hipblaslt/tensilelite/Tensile     # tensilelite Tensile dir
GEKO=$(python3 -c "import geko, os; print(os.path.dirname(geko.__file__))")   # installed geko package dir

# ── Fix 1 — Tensile/Tensile.py: --library-format default yaml → msgpack ─────────────────────────
#    WHY: hipBLASLt 1.5.0 LOADS msgpack device libraries. The yaml default makes TensileCreateLibrary
#    emit the wrong artifact and the rebuilt host .so faults at load ("No library mapping found ...").
#    (stock: ~line 246  action="store", default="yaml", help="select which library format to use")
grep -q "default=\"msgpack\", help=\"select which library format" "$TENSILE/Tensile.py" \
  && echo "Fix1 SKIP" \
  || { sed -i "s/default=\"yaml\", help=\"select which library format to use\"/default=\"msgpack\", help=\"select which library format to use\"/" "$TENSILE/Tensile.py"; echo "Fix1 OK"; }

# ── Fix 2 — Tensile/GenerateSummations.py: hardcoded --library-format=yaml → msgpack ─────────────
#    WHY: same artifact-format issue; this path hardcodes the flag (stock: ~line 57).
#    Belt-and-suspenders: the module default in Common/GlobalParameters.py is also yaml (~line 155).
grep -q -- "--library-format=msgpack" "$TENSILE/GenerateSummations.py" \
  && echo "Fix2 SKIP" \
  || { sed -i "s/--library-format=yaml/--library-format=msgpack/" "$TENSILE/GenerateSummations.py"; echo "Fix2 OK"; }
sed -i "s/globalParameters\[\"LibraryFormat\"\] = \"yaml\"/globalParameters[\"LibraryFormat\"] = \"msgpack\"/" "$TENSILE/Common/GlobalParameters.py" || true

# ── Fix 3 — geko/library/library.py: scale_type in create_bench_input() ──────────────────────────
#    WHY: hipBLASLt 1.5.0 hipblaslt-bench REQUIRES a scale_type field; without it geko'\''s generated
#    bench input is rejected. In this geko snapshot it is ALREADY present — create_bench_input() sets
#    `common["scale_type"] = compute_type` right after deriving compute_type in the ComputeDataType
#    branch (~line 427), which is the path geko'\''s decode GEMMs take. Verify; add only if your geko
#    predates it:
python3 - "$GEKO/library/library.py" << "PY"
import sys
p = sys.argv[1]; c = open(p).read()
if "scale_type" in c:
    print("Fix3 SKIP (scale_type already in create_bench_input, ~line 427)")
else:
    anchor = "            common[\"compute_type\"] = \"c_\" + compute_type\n"
    assert anchor in c, "anchor not found — geko layout changed; add common[\"scale_type\"]=compute_type by hand in create_bench_input()"
    open(p, "w").write(c.replace(anchor, "            common[\"scale_type\"] = compute_type\n" + anchor))
    print("Fix3 OK (added scale_type)")
PY
'
```

**Auxiliary** (only if tensilelite'\''s `hipblaslt_gentest.py` is invoked — Python-3 text-mode stdout fix):
```bash
docker exec $TARGET sed -i "s/default=sys.stdout\b/default=sys.stdout.buffer/" /opt/rocm/bin/hipblaslt_gentest.py
```

> Self-check after patching: `grep -c msgpack $TENSILE/Tensile.py $TENSILE/GenerateSummations.py` (each ≥1) and
> `grep -c scale_type $GEKO/library/library.py` (≥1). All three must be satisfied before `geko --tune`.

---

