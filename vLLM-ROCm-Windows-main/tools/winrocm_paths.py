# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Machine-independent path and toolchain resolution for the native Windows + ROCm build scripts.

Everything that used to be hardcoded to one machine (HIP SDK location, repo checkout, scratch build
directories, GPU architecture) is resolved here, so a fresh clone builds without editing any file.
Each value can still be pinned with an environment variable:

    HIP_PATH / ROCM_PATH / ROCM_HOME   HIP SDK install root (else auto-probed)
    VLLM_WIN_GFX_ARCH                  target arch, e.g. gfx1200 (else taken from the live GPU)
    VLLM_WIN_BUILD_ROOT                parent of the scratch build dirs (else C:\\, else LOCALAPPDATA)
    VLLM_WIN_BUILD_CLEAN=0             keep the scratch build dir instead of wiping it (incremental)
    VLLM_CSRC                          vLLM csrc tree (else <repo>/vllm/csrc)
    CK_ROOT / CK_FMHA_GEN              Composable Kernel checkout / generated FMHA instances

Scripts find this module without knowing their own depth:

    _D = os.path.dirname(os.path.abspath(__file__))
    while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
        _D = os.path.dirname(_D)
    sys.path.insert(0, os.path.join(_D, "tools"))
"""
import glob
import os
import shutil
import tempfile


def repo_root() -> str:
    """Repo checkout root (this file lives in <repo>/tools/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hip_root() -> str:
    """HIP SDK / ROCm install root. Honours HIP_PATH first, then probes the usual locations."""
    for var in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        v = (os.environ.get(var) or "").strip().rstrip("\\/")
        if v and os.path.isdir(v):
            return v
    cands = [r"C:\HIP-SDK"]
    for base in {os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramW6432", r"C:\Program Files")}:
        cands += sorted(glob.glob(os.path.join(base, "AMD", "ROCm", "*")), reverse=True)
    for c in cands:
        if os.path.isfile(os.path.join(c, "bin", "hipcc.exe")) or os.path.isfile(
                os.path.join(c, "bin", "hipcc.bat")):
            return c
    for c in cands:
        if os.path.isdir(os.path.join(c, "bin")):
            return c
    raise RuntimeError(
        "HIP SDK not found. Install the AMD HIP SDK, then set HIP_PATH to it, for example:\n"
        '    set HIP_PATH=C:\\Program Files\\AMD\\ROCm\\7.2')


def pip_sdk_roots() -> list[str]:
    """ROCm SDK roots pip-installed in the current environment (TheRock wheels).

    The `_rocm_sdk_devel` wheel ships the full device bitcode set for its gfx family
    (e.g. gfx110X-all), which the installed Windows HIP SDK may not (7.2 ships none for
    gfx110x on some installs)."""
    try:
        import rocm_sdk
    except ImportError:
        return []
    roots = []
    try:
        out = rocm_sdk.find_libraries("amdhip64")
        # roots look like .../_rocm_sdk_devel/bin/amdhip64_7.dll -> walk up to the package dir
        for lib in out or []:
            d = os.path.dirname(str(lib))
            while d and os.path.basename(d) not in ("_rocm_sdk_devel", "_rocm_sdk_core"):
                nd = os.path.dirname(d)
                if nd == d:
                    break
                d = nd
            if os.path.basename(d) in ("_rocm_sdk_devel", "_rocm_sdk_core") and d not in roots:
                roots.append(d)
    except Exception:  # noqa: BLE001
        pass
    return roots


def _bitcode_has_arch(bc_dir: str, arch: str) -> bool:
    """True if the bitcode dir contains ISA objects for the target gfx arch.

    gfx1101 bitcode ships as oclc_isa_version_1101.bc; check both the plain isa-version
    name and a direct gfx prefix."""
    isa = "".join(ch for ch in arch if ch.isdigit())
    try:
        names = os.listdir(bc_dir)
    except OSError:
        return False
    return any(arch in n or f"isa_version_{isa}" in n for n in names if n.endswith(".bc"))


def hip_root_for_arch(arch: str) -> str:
    """hip_root(), but preferring a root that actually has device bitcode for `arch`.

    The installed HIP SDK (e.g. 7.2 under Program Files) can lack gfx110x bitcode entirely;
    compiling for a detected gfx1101 then fails with 'cannot find ROCm device library'.
    If the env-pinned root lacks the bitcode but a pip SDK has it, use the pip SDK and say
    so (an explicit HIP_PATH/ROCM_HOME always wins silently — the user asked for it)."""
    pinned = None
    for var in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        v = (os.environ.get(var) or "").strip().rstrip("\\/")
        if v and os.path.isdir(v):
            pinned = v
            break
    root = hip_root()
    bc = _bitcode_dir_of(root)
    if bc and _bitcode_has_arch(bc, arch):
        return root
    for pip_root in pip_sdk_roots():
        pbc = _bitcode_dir_of(pip_root)
        if pbc and _bitcode_has_arch(pbc, arch):
            if pip_root != root:
                print(f"winrocm: '{root}' has no {arch} device bitcode; "
                      f"using pip SDK {pip_root} (set HIP_PATH to override)")
            return pip_root
    if pinned:
        return pinned
    raise RuntimeError(
        f"No ROCm device bitcode for {arch} found under '{root}'"
        + (f" or the pip SDK roots {pip_sdk_roots()}" if pip_sdk_roots() else "")
        + ". Point HIP_PATH at an SDK that ships it, e.g. the pip-installed "
        "_rocm_sdk_devel wheel from the same TheRock index as torch.")


def _bitcode_dir_of(root: str) -> str | None:
    for rel in (("lib", "llvm", "amdgcn", "bitcode"), ("amdgcn", "bitcode"), ("lib", "bitcode")):
        p = os.path.join(root, *rel)
        if os.path.isdir(p):
            return p
    return None


def hip_bin() -> str:
    return os.path.join(hip_root(), "bin")


def hip_lib() -> str:
    return os.path.join(hip_root(), "lib")


def hip_include() -> str:
    return os.path.join(hip_root(), "include")


def device_lib() -> str:
    """amdgcn device bitcode for --rocm-device-lib-path (layout moved between SDK versions).

    Resolved through hip_root_for_arch(): if the primary SDK has no bitcode for the build's
    target arch (e.g. an installed HIP SDK 7.2 without gfx110x objects), the pip-installed
    SDK wheel is used instead, so the build succeeds on cards the installed SDK predates."""
    try:
        root = hip_root_for_arch(offload_arch())
    except Exception:  # noqa: BLE001  (arch unknown at import time in some callers)
        root = hip_root()
    for rel in (("lib", "llvm", "amdgcn", "bitcode"), ("amdgcn", "bitcode"), ("lib", "bitcode")):
        p = os.path.join(root, *rel)
        if os.path.isdir(p):
            return p
    return os.path.join(root, "lib", "llvm", "amdgcn", "bitcode")


def build_root() -> str:
    """Parent of the scratch build dirs. Kept short: ninja/MSVC hit the Windows 260-char path limit
    on deep trees, which is why these live at the drive root rather than inside the repo.

    Default here is C:\\AI\\build (ASCII-first location) when it exists, falling back to the
    drive root; winrocm_env.bat sets the same default for the .bat wrappers. Override freely
    with VLLM_WIN_BUILD_ROOT."""
    v = (os.environ.get("VLLM_WIN_BUILD_ROOT") or "").strip()
    if v:
        os.makedirs(v, exist_ok=True)
        return v
    cands = ["C:\\", os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()]
    if os.path.isdir("C:\\AI"):
        # Adopt the ASCII-first layout only when C:\AI already exists (matching the .bat
        # guard): the makedirs probe below is recursive and would otherwise silently create
        # C:\AI\build on every machine and migrate its future builds there.
        cands.insert(0, "C:\\AI\\build")
    for cand in cands:
        probe = os.path.join(cand, "vw_probe_tmp")
        try:
            os.makedirs(probe, exist_ok=True)
            os.rmdir(probe)
            return cand
        except OSError:
            continue
    return tempfile.gettempdir()


def build_dir(name: str, env_var: str | None = None, clean: bool = False) -> str:
    """Scratch dir for one native extension, e.g. build_dir("vw_cext_build", "VLLM_WIN_C_DIR").

    `clean` reproduces what the .bat wrappers used to do (wipe before building); set
    VLLM_WIN_BUILD_CLEAN=0 to keep it and let cpp_extension rebuild incrementally."""
    d = (os.environ.get(env_var) if env_var else None) or os.path.join(build_root(), name)
    if clean and os.environ.get("VLLM_WIN_BUILD_CLEAN", "1") == "1":
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


def offload_arch() -> str:
    """gfx target to compile for, taken from the live GPU, so an RDNA4 box builds gfx1200 by itself.
    gcnArchName can carry feature suffixes (gfx1100:xnack-) which --offload-arch does not want."""
    v = (os.environ.get("VLLM_WIN_GFX_ARCH") or "").strip()
    if v:
        return v.split(":")[0]
    import torch
    try:
        name = torch.cuda.get_device_properties(0).gcnArchName
        if name:
            return name.split(":")[0]
    except Exception:  # noqa: BLE001  (no GPU visible at build time)
        pass
    lst = [a for a in (torch.cuda.get_arch_list() or []) if a.startswith("gfx")]
    if lst:
        return lst[0].split(":")[0]
    raise RuntimeError("could not determine the GPU arch; set VLLM_WIN_GFX_ARCH, e.g. gfx1100")


def vllm_csrc() -> str:
    """The cloned vLLM's csrc tree (the clone is gitignored, see tools/patch_vllm.py)."""
    v = (os.environ.get("VLLM_CSRC") or "").strip()
    if v:
        return v
    p = os.path.join(repo_root(), "vllm", "csrc")
    if not os.path.isdir(p):
        raise RuntimeError(
            f"vLLM csrc not found at {p}. Clone vLLM into <repo>/vllm (see README Setup), "
            "or set VLLM_CSRC to an existing checkout.")
    return p


def ck_root() -> str:
    """Composable Kernel checkout (used by the CK ck_tile FMHA prefill kernels)."""
    v = (os.environ.get("CK_ROOT") or "").strip()
    if v:
        return v
    for c in (os.path.join(os.path.dirname(repo_root()), "composable_kernel"),
              os.path.join(os.path.expanduser("~"), "composable_kernel")):
        if os.path.isdir(c):
            return c
    raise RuntimeError(
        "Composable Kernel checkout not found. Clone it next to this repo:\n"
        "    git clone --depth 1 https://github.com/ROCm/composable_kernel\n"
        "or set CK_ROOT to an existing checkout.")


def ck_gen_dir() -> str:
    """Where CK's example/ck_tile/01_fmha/generate.py emitted the FMHA instance sources."""
    v = (os.environ.get("CK_FMHA_GEN") or "").strip()
    if v:
        return v
    return os.path.join(os.path.dirname(ck_root()), "ckfmha_gen")


def add_dll_dirs(*extra: str) -> None:
    """Make the HIP runtime DLLs (and any extra build dirs) loadable before importing a .pyd."""
    dirs = []
    try:
        dirs = [hip_bin(), hip_lib()]
    except RuntimeError:
        pass
    for d in list(dirs) + list(extra):
        try:
            os.add_dll_directory(d)
        except Exception:  # noqa: BLE001  (missing dir is not fatal)
            pass
