#!/usr/bin/env bash
# probe-image.sh: which kernel tooling this container ships, and whether each piece imports.
#
# Run it INSIDE the container, from the host:
#   docker exec -i "$C" bash -s < probe-image.sh
#   docker cp probe-image.sh "$C":/tmp/ && docker exec "$C" bash /tmp/probe-image.sh
# Every probe prints OK or MISSING with the error; details are indented under it.
# It never fails: the exit status is 0 whatever it finds. PYTHON and PROBE_TIMEOUT override defaults.
# The image digest is only visible from the host: docker inspect --format '{{.Image}}' "$C".

PY=${PYTHON:-$(command -v python3 || command -v python)}
TO=${PROBE_TIMEOUT:-180}

say() { printf '%-8s %-14s %s\n' "$1" "$2" "$3"; }
indent() { sed 's/^/                        /'; }
# Every child reads /dev/null: under `bash -s` stdin is this script, and a child would eat it.
run() { if command -v timeout >/dev/null 2>&1; then timeout "$TO" "$@" </dev/null; else "$@" </dev/null; fi; }

# pyprobe NAME CODE: the code prints result lines prefixed "@@"; the first is the status line.
pyprobe() {
    local name=$1 code=$2 out rc res
    if [ -z "$PY" ]; then say MISSING "$name" "no python3 or python on PATH"; return; fi
    out=$(run "$PY" -c "$code" 2>&1); rc=$?
    res=$(printf '%s\n' "$out" | sed -n 's/^@@//p')
    if [ "$rc" -eq 0 ] && [ -n "$res" ]; then
        say OK "$name" "$(printf '%s\n' "$res" | head -n 1)"
        printf '%s\n' "$res" | tail -n +2 | indent
    else
        say MISSING "$name" "$(printf '%s\n' "$out" | grep -v '^@@' | grep -v '^[[:space:]]*$' | tail -n 1) (rc=$rc)"
        [ -n "$res" ] && printf '%s\n' "$res" | indent
    fi
}

shprobe() {  # shprobe NAME COMMAND...: OK with the first output lines, or MISSING
    local name=$1 out rc
    shift
    out=$(run "$@" 2>&1); rc=$?
    if [ "$rc" -eq 0 ] && [ -n "$out" ]; then
        say OK "$name" "$(printf '%s\n' "$out" | head -n 1)"
        printf '%s\n' "$out" | sed -n '2,4p' | indent
    else
        say MISSING "$name" "$(printf '%s\n' "$out" | tail -n 1) (rc=$rc)"
    fi
}

echo "== probe-image.sh $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname 2>/dev/null) $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-}")"

# --- ROCm, compilers, the GPU -------------------------------------------------------------
if [ -r /opt/rocm/.info/version ]; then say OK rocm "$(cat /opt/rocm/.info/version) ($(readlink -f /opt/rocm))"
else say MISSING rocm "no /opt/rocm/.info/version; ls: $(ls -d /opt/rocm* 2>/dev/null | tr '\n' ' ')"; fi
shprobe hipcc hipcc --version
shprobe hipconfig hipconfig --version
if command -v rocminfo >/dev/null 2>&1; then
    ri=$(run rocminfo 2>&1)
    gfx=$(printf '%s\n' "$ri" | grep -m1 -oE 'gfx[0-9a-f]+')
    if [ -n "$gfx" ]; then
        say OK rocminfo "$gfx, $(printf '%s\n' "$ri" | grep -c 'Device Type:[[:space:]]*GPU') GPU agents"
        printf '%s\n' "$ri" | grep -m1 'Compute Unit:' | sed 's/^[[:space:]]*//' | indent
        printf '%s\n' "$ri" | awk '/Segment:[[:space:]]*GROUP/{getline; sub(/^[[:space:]]*/, ""); print "LDS (GROUP segment) " $0; exit}' | indent
    else say MISSING rocminfo "no gfx agent: $(printf '%s\n' "$ri" | tail -n 1)"; fi
else say MISSING rocminfo "not on PATH"; fi
for d in /opt/rocm/include/ck /opt/rocm/include/ck_tile; do
    if [ -d "$d" ]; then say OK "ck headers" "$d"; else say MISSING "ck headers" "$d"; fi
done
lib=$(ls /opt/rocm/lib/libhipblaslt.so.* 2>/dev/null | head -n 1)
if [ -n "$lib" ]; then say OK hipblaslt "$lib"; else say MISSING hipblaslt "no /opt/rocm/lib/libhipblaslt.so.*"; fi
for t in ninja rocprofv3; do
    if command -v "$t" >/dev/null 2>&1; then say OK "$t" "$(command -v "$t")"; else say MISSING "$t" "not on PATH"; fi
done

# --- Python packages ----------------------------------------------------------------------
pyprobe python 'import sys; print("@@" + sys.version.split()[0] + " " + sys.executable)'

pyprobe torch '
import torch
print("@@%s hip=%s gpu=%s devices=%d" % (torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count()))
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("@@device 0: %s arch=%s CUs=%d memory=%.0f GiB" % (p.name, getattr(p, "gcnArchName", "?"), p.multi_processor_count, p.total_memory / 2**30))
print("@@dtypes: " + " ".join("%s=%s" % (n, hasattr(torch, n)) for n in
      ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn_x2")))
'

pyprobe triton '
import os, triton, triton.language as tl
print("@@%s %s" % (triton.__version__, os.path.dirname(triton.__file__)))
try:
    t = triton.runtime.driver.active.get_current_target()
    print("@@target: backend=%s arch=%s warp_size=%s" % (t.backend, t.arch, getattr(t, "warp_size", "?")))
except Exception as e:
    print("@@target: unknown (%s: %s)" % (type(e).__name__, e))
print("@@tl.dot_scaled: %s" % hasattr(tl, "dot_scaled"))
try:
    import triton.experimental.gluon
    print("@@gluon: importable")
except Exception as e:
    print("@@gluon: not importable (%s)" % type(e).__name__)
'

# FlyDSL is a separate package; without it every AITER FlyDSL path is inert
# (KernelForge local_knowledge/framework/aiter/skills/optimize/aiter_levers/flydsl_path.md).
pyprobe flydsl '
import os, importlib.metadata as md
import flydsl
v = getattr(flydsl, "__version__", None)
if v is None:
    try:
        v = md.version("flydsl")
    except Exception:
        v = "version unknown"
print("@@%s %s" % (v, os.path.dirname(flydsl.__file__)))
import flydsl.compiler, flydsl.expr
print("@@flydsl.compiler, flydsl.expr: import OK")
try:
    from flydsl.runtime.device import get_rocm_arch
    print("@@arch: %s" % get_rocm_arch())
except Exception as e:
    print("@@arch: unknown (%s)" % type(e).__name__)
cache = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", os.path.expanduser("~/.flydsl/cache"))
print("@@cache: %s (%s)" % (cache, "exists" if os.path.isdir(cache) else "absent"))
'

pyprobe aiter '
import glob, os, importlib.metadata as md
v = "version unknown"
for n in ("amd-aiter", "aiter", "aiter-amd"):
    try:
        v = "%s (%s)" % (md.version(n), n)
        break
    except Exception:
        pass
import aiter
pkg = os.path.dirname(aiter.__file__)
print("@@%s %s" % (v, pkg))
meta = asm = None
try:
    from aiter.jit import core
    meta, asm = getattr(core, "AITER_META_DIR", None), getattr(core, "AITER_ASM_DIR", None)
except Exception as e:
    print("@@aiter.jit.core: %s: %s" % (type(e).__name__, e))
up = os.path.dirname(pkg)
bases = [b for b in (meta, up, os.path.join(up, "aiter_meta"), os.path.join(pkg, "aiter_meta"), pkg) if b]
hsa = [asm, os.environ.get("AITER_ASM_DIR")] + [os.path.join(b, "hsa") for b in bases]
hsa = next((h for h in hsa if h and os.path.isdir(h)), None)
if hsa:
    gdirs = [hsa.rstrip("/")] if os.path.basename(hsa.rstrip("/")).startswith("gfx") else sorted(glob.glob(os.path.join(hsa, "gfx*")))
    for g in gdirs:
        ops = sorted(d for d in os.listdir(g) if os.path.isdir(os.path.join(g, d)))
        cos = {d: len(glob.glob(os.path.join(g, d, "**", "*.co"), recursive=True)) for d in ops}
        print("@@asm %s: %d .co in %d ops: %s" % (g, sum(cos.values()), len(ops), " ".join("%s(%d)" % kv for kv in cos.items())))
else:
    print("@@asm: no hsa/ dir found (looked in %s)" % ", ".join(bases))
fly = os.path.join(pkg, "ops", "flydsl")
if os.path.isdir(fly):
    files = sorted(os.path.relpath(f, fly) for f in glob.glob(os.path.join(fly, "**", "*.py"), recursive=True))
    print("@@aiter/ops/flydsl: %d files: %s" % (len(files), " ".join(files)))
    try:
        from aiter.ops.flydsl.utils import is_flydsl_available
        print("@@is_flydsl_available(): %s" % is_flydsl_available())
    except Exception as e:
        print("@@is_flydsl_available: %s" % type(e).__name__)
else:
    print("@@aiter/ops/flydsl: absent (no FlyDSL kernels in this AITER)")
tri = glob.glob(os.path.join(pkg, "ops", "triton", "**", "*.py"), recursive=True)
print("@@aiter/ops/triton: %d .py files" % len(tri))
ck = next((os.path.join(b, "3rdparty", "composable_kernel", "include") for b in bases
           if os.path.isdir(os.path.join(b, "3rdparty", "composable_kernel", "include"))), None)
print("@@ck source for JIT: %s" % (ck or "not found (CK ops need their prebuilt modules)"))
so = glob.glob(os.path.join(pkg, "jit", "*.so"))
print("@@prebuilt modules in aiter/jit: %d; user JIT dir %s" % (len(so), os.environ.get("AITER_JIT_DIR", os.path.expanduser("~/.aiter/jit"))))
cfg = os.path.join(pkg, "configs")
csvs = sorted(glob.glob(os.path.join(cfg, "*.csv")))
print("@@configs: %d CSVs in %s; model_configs: %d" % (len(csvs), cfg, len(glob.glob(os.path.join(cfg, "model_configs", "*")))))
for f in csvs:
    if "tuned" in os.path.basename(f) and "untuned" not in os.path.basename(f):
        rows = open(f, errors="replace").read().splitlines()[1:]
        fl = sum("flydsl" in r.lower() for r in rows)
        print("@@  %s: %d rows, %d naming flydsl" % (os.path.basename(f), len(rows), fl))
'

pyprobe serving '
import importlib.metadata as md
got = []
for n in ("vllm", "sglang", "atom", "mori"):
    try:
        got.append("%s %s" % (n, md.version(n)))
    except Exception:
        pass
print("@@" + (", ".join(got) or "none of vllm, sglang, atom, mori"))
'

# --- Environment that changes kernel selection --------------------------------------------
envs=$(env | grep -E '^(AITER_|FLYDSL_|TRITON_|AMDGCN_|VLLM_ROCM_|HSA_|HIP_|ROCR_|GPU_MAX_HW_QUEUES)' | sort)
if [ -n "$envs" ]; then say OK env "$(printf '%s\n' "$envs" | wc -l) kernel-related variables set"; printf '%s\n' "$envs" | indent
else say OK env "no AITER_/FLYDSL_/TRITON_/VLLM_ROCM_/HSA_/HIP_ variables set"; fi
exit 0
