# geko — known issues + reference run

## New Issues Discovered (v2 additions)

### Issue 1: GPU fault kills container without `--ipc=host`

**Symptom:** Process exits with code 137 mid-tune; container stops entirely.  
**Root cause:** Same `_SAV_UserArgs_` kernel GPU fault documented in v1, but on ROCm 7.2.3
the fault propagates to the container rather than just killing the subprocess.  
**Fix:** `--ipc=host` at container creation. Cannot be added after container is created.

### Issue 2: `build/release/clients/hipblaslt-bench` is wrong binary

**Symptom:** `hipblaslt-bench failed to run` during the analyze phase.  
**Root cause:** geko hardcodes the path `build/release/clients/hipblaslt-bench`. The binary
copied from `geko_build` (ROCm 7.2.0) is incompatible with ROCm 7.2.3 runtime.  
**Fix:** Symlink to `/opt/rocm/bin/hipblaslt-bench`.

### Issue 3: `hipblaslt_gentest.py` not found in `build/release/clients/`

**Symptom:** `sh: hipblaslt_gentest.py: not found` during benchmark phase.  
**Root cause:** geko calls `build/release/clients/hipblaslt_gentest.py` (also `hipblaslt_template.yaml`).
The minimal archive only contains the bench binary.  
**Fix:** Symlink both from `/opt/rocm/bin/`.

### Issue 4: `rocisa` cannot be copied between Python versions

**Symptom:** `ImportError: cannot import name 'rocIsa' from 'rocisa'` or pip build error.  
**Root cause:** rocisa is a compiled Python C extension (`.cpython-310-*.so`). Copying
from a Python 3.10 container to a Python 3.12 container breaks ABI compatibility.  
**Fix:** Always build rocisa from source in the target container. Requires copying
`cmake/` and `shared/` from rocm-libraries (60 KB + 15 MB, already in the archive).

### Issue 5: `tensilelite-client` needs `libomp.so` (unversioned)

**Symptom:** `libomp.so: cannot open shared object file: No such file or directory`.  
**Root cause:** The pre-built binary links against `libomp.so`. Ubuntu provides
`libomp.so.5` but not the unversioned symlink by default.  
**Fix:** `apt install libomp5-14` then `ln -s libomp.so.5 libomp.so`.

### Issue 6: tensilelite-client cache hash mismatch

**Symptom:** geko triggers a full LLVM rebuild of tensilelite-client on first run
despite the binary already being present.  
**Root cause:** geko caches the client by git commit hash. Without a `.git` directory
it falls back to today's date (YYYY-MM-DD). The `build_tmp/hash.txt` from `geko_build`
contains a git SHA, which doesn't match.  
**Fix:** After extracting `build_tmp.tar.gz`, overwrite: `echo "$(date +%Y-%m-%d)" > /app/build_tmp/hash.txt`.
Must be re-applied each day if the container is restarted after midnight.

### Issue 7: Logic yaml files needed for `full_libs` creation

**Symptom:** `FileNotFoundError: .../Origami_ntb4/gfx950_*.yaml`.  
**Root cause:** `from_full_dataframe` reads reference Tensile logic files from the
hipblaslt source tree. The deb package does not include these yaml files (only `.co`/`.dat.zlib`).  
**Fix:** Copy `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/`
from `geko_build` (compresses to ~18 MB).

---

## Reference Run (decode shapes, dsv41flash vllm container, 2026-09-16)

| Metric | Value |
|---|---|
| Shapes | 24 decode (3 layers × M=1…8) |
| Valid / Kept (≥3%) | 24 / 22 |
| Median uplift | **+40.9%** |
| Mean uplift | +40.0% |
| Max uplift | +76.4% (M=3, N=5120, K=288) |
| Weighted uplift | +27.0% |
| Regressed (excluded) | 2 — M=1,8 for N=5120,K=288 |
