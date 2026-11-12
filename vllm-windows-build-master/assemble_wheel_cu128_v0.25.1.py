"""Assemble and validate the vLLM v0.25.1 Windows cu128 wheel."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import sys
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLLM_WHEEL_SOURCE", ROOT / "vllm-source-v0.25.1")).resolve()
OUT_DIR = Path(os.environ.get("VLLM_WHEEL_OUTPUT", ROOT / "dist-v9")).resolve()
FLASH_ATTN_SRC = Path(
    os.environ.get(
        "VLLM_FLASH_ATTN_SOURCE",
        SRC / ".deps" / "vllm-flash-attn-src",
    )
).resolve()

VERSION = "0.25.1+cu128"
PYTHON_TAG = "cp313"
ABI_TAG = "cp313"
PLATFORM_TAG = "win_amd64"
DIST_NAME = "vllm"
DIST_INFO = f"{DIST_NAME}-{VERSION}.dist-info"
WHEEL = OUT_DIR / (f"{DIST_NAME}-{VERSION}-{PYTHON_TAG}-{ABI_TAG}-{PLATFORM_TAG}.whl")

SOURCE_METADATA_DIR = SRC / "vllm.egg-info"
INSTALLED_METADATA_DIR = SRC / ".venv" / "Lib" / "site-packages" / DIST_INFO
METADATA_DIR = Path(
    os.environ.get(
        "VLLM_WHEEL_METADATA",
        SOURCE_METADATA_DIR if SOURCE_METADATA_DIR.is_dir() else INSTALLED_METADATA_DIR,
    )
).resolve()
METADATA_PATH = (
    METADATA_DIR / "METADATA"
    if (METADATA_DIR / "METADATA").is_file()
    else METADATA_DIR / "PKG-INFO"
)

SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

REQUIRED_FILES = [
    "vllm/_C_stable_libtorch.pyd",
    "vllm/_moe_C_stable_libtorch.pyd",
    "vllm/_rust_tool_parser.pyd",
    "vllm/cumem_allocator.pyd",
    "vllm/fs_io_C.pyd",
    "vllm/spinloop.pyd",
    "vllm/vllm_flash_attn/_vllm_fa2_C.pyd",
    "vllm/vllm-rs.exe",
    "vllm/third_party/triton_kernels",
    "vllm/third_party/fmha_sm100",
    "LICENSE",
]

REQUIRED_NONEMPTY_FILES = {
    "vllm/_C_stable_libtorch.pyd",
    "vllm/_moe_C_stable_libtorch.pyd",
    "vllm/_rust_tool_parser.pyd",
    "vllm/cumem_allocator.pyd",
    "vllm/fs_io_C.pyd",
    "vllm/spinloop.pyd",
    "vllm/vllm_flash_attn/_vllm_fa2_C.pyd",
    "vllm/vllm-rs.exe",
}

REQUIRED_ARCHIVE_FILES = {
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py",
    "vllm/v1/kv_offload/cpu/policies/arc.py",
    "vllm/v1/kv_offload/cpu/policies/lru.py",
    "vllm/v1/kv_offload/cpu/gpu_worker.py",
    "vllm/v1/kv_offload/cpu/shared_offload_region.py",
    "vllm/v1/kv_offload/file_mapper.py",
    "vllm/v1/kv_offload/cpu/spec.py",
    "vllm/v1/kv_offload/tiering/fs/io.py",
    "vllm/v1/kv_offload/tiering/fs/manager.py",
    "vllm/v1/kv_offload/tiering/spec.py",
    "vllm/v1/simple_kv_offload/cuda_mem_ops.py",
    "vllm/v1/worker/block_table.py",
    "vllm/v1/worker/gpu/sample/states.py",
    "vllm/vllm_flash_attn/layers/__init__.py",
    "vllm/vllm_flash_attn/layers/rotary.py",
    "vllm/vllm_flash_attn/ops/__init__.py",
    "vllm/vllm_flash_attn/ops/triton/__init__.py",
    "vllm/vllm_flash_attn/ops/triton/rotary.py",
    "vllm/vllm_flash_attn/cute/__init__.py",
    "vllm/vllm_flash_attn/cute/interface.py",
}

SAMPLING_STATES = "vllm/v1/worker/gpu/sample/states.py"
INT64_SEED_FIX = b"_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64"

SIMPLE_KV_DMA = "vllm/v1/simple_kv_offload/cuda_mem_ops.py"
SIMPLE_KV_WINDOWS_MARKERS = (
    b"def _copy_blocks_windows(",
    b"(err,) = cudart.cudaMemcpyAsync(",
    b'if sys.platform == "win32":',
)

BLOCK_TABLE = "vllm/v1/worker/block_table.py"
BLOCK_TABLE_WINDOWS_MARKERS = (
    b"from vllm.triton_utils import HAS_TRITON, tl, triton",
    b'if not HAS_TRITON and self.device.type != "cpu":',
    b"def _compute_slot_mapping_torch(",
)

SHARED_REGION = "vllm/v1/kv_offload/cpu/shared_offload_region.py"
SHARED_REGION_WINDOWS_MARKERS = (
    b'tempfile.gettempdir() if os.name == "nt" else "/dev/shm"',
    b"access=mmap.ACCESS_WRITE",
    b"def _wait_for_path_size(",
    b"_wait_for_path_size(self.mmap_path, self.total_size_bytes)",
)

GPU_WORKER = "vllm/v1/kv_offload/cpu/gpu_worker.py"
GPU_WORKER_WINDOWS_MARKERS = (
    b'if os.name == "nt" and uses_shared_mmap:',
    b"uses_shared_mmap=mmap_region is not None",
    b"Route mmap restores through native DMA",
)

FILE_MAPPER = "vllm/v1/kv_offload/file_mapper.py"
FILE_MAPPER_WINDOWS_MARKERS = (
    b"import ntpath",
    b"_INVALID_PATH_COMPONENT_CHARS",
    b"safe_model_name = ntpath.basename(model_name)",
    b"safe_model_name = safe_model_name[:_MAX_MODEL_COMPONENT_LEN]",
)

COMMON_REQUIREMENTS = "requirements/common.txt"
WINDOWS_REQUIRED_DEPENDENCIES = ("llguidance", "xgrammar")

FS_IO = "vllm/v1/kv_offload/tiering/fs/io.py"
FS_IO_WINDOWS_MARKERS = (
    b'if hasattr(os, "readv"):',
    b'O_BINARY = getattr(os, "O_BINARY", 0)',
    b"os.O_RDONLY | O_DIRECT | O_BINARY",
    b"while bytes_read < block_size:",
    b"data = os.read(fd, block_size - bytes_read)",
)

NATIVE_BATCH_SOURCE = "csrc/libtorch_stable/cache_kernels.cu"
NATIVE_BATCH_WINDOWS_MARKER = (
    b"!defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION)"
)

NATIVE_FS_SOURCE = "csrc/fs_io.cpp"
NATIVE_FS_WINDOWS_MARKERS = (
    b"#include <filesystem>",
    b"std::filesystem::u8path(paths[i])",
)


def digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def version_py() -> bytes:
    return f"""# Generated by assemble_wheel_cu128_v0.25.1.py
from __future__ import annotations

__all__ = [
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "__commit_id__",
    "commit_id",
]

version: str
__version__: str
__version_tuple__: tuple[int | str, ...]
version_tuple: tuple[int | str, ...]
commit_id: str | None
__commit_id__: str | None

__version__ = version = {VERSION!r}
__version_tuple__ = version_tuple = (0, 25, 1, "cu128")

__commit_id__ = commit_id = None
""".encode()


def wheel_file() -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: assemble_wheel_cu128_v0.25.1.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {PYTHON_TAG}-{ABI_TAG}-{PLATFORM_TAG}\n"
    ).encode()


def rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def package_files() -> list[Path]:
    files: list[Path] = []
    package_root = SRC / "vllm"
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(SRC).parts
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix().lower())


def generated_flash_attn_files() -> dict[str, Path | bytes]:
    """Return Python payloads generated by vllm-flash-attn's CMake install."""
    entries: dict[str, Path | bytes] = {}
    package_root = FLASH_ATTN_SRC / "vllm_flash_attn"
    if package_root.is_dir():
        for path in package_root.rglob("*.py"):
            relative = path.relative_to(package_root).as_posix()
            if relative in {"__init__.py", "flash_attn_interface.py"}:
                continue
            entries[f"vllm/vllm_flash_attn/{relative}"] = path

    cute_root = FLASH_ATTN_SRC / "flash_attn" / "cute"
    if cute_root.is_dir():
        for path in cute_root.rglob("*.py"):
            relative = path.relative_to(cute_root).as_posix()
            data = path.read_bytes().replace(
                b"flash_attn.cute", b"vllm.vllm_flash_attn.cute"
            )
            entries[f"vllm/vllm_flash_attn/cute/{relative}"] = data
    return entries


def archive_entries() -> dict[str, Path | bytes]:
    entries: dict[str, Path | bytes] = {}
    for path in package_files():
        archive_name = rel(path)
        entries[archive_name] = (
            version_py() if archive_name == "vllm/_version.py" else path
        )
    for archive_name, payload in generated_flash_attn_files().items():
        entries.setdefault(archive_name, payload)
    return entries


def require_markers(
    errors: list[str],
    relative_path: str,
    markers: tuple[bytes, ...],
    description: str,
) -> None:
    path = SRC / relative_path
    if not path.is_file():
        errors.append(f"missing {description}: {path}")
        return
    data = path.read_bytes()
    if missing := [marker for marker in markers if marker not in data]:
        errors.append(
            f"{description} is incomplete in {path}; missing markers: "
            + ", ".join(repr(marker.decode(errors="replace")) for marker in missing)
        )


def validate_source() -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        path = SRC / name
        if not path.exists():
            errors.append(f"missing required path: {path}")
        elif path.is_dir() and not any(item.is_file() for item in path.rglob("*")):
            errors.append(f"required directory is empty: {path}")
    for name in REQUIRED_NONEMPTY_FILES:
        path = SRC / name
        if path.is_file() and path.stat().st_size == 0:
            errors.append(f"required binary is empty: {path}")

    for path in (
        METADATA_PATH,
        METADATA_DIR / "entry_points.txt",
        METADATA_DIR / "top_level.txt",
    ):
        if not path.is_file():
            errors.append(f"missing required metadata file: {path}")

    sampling_states = SRC / SAMPLING_STATES
    if not sampling_states.is_file():
        errors.append(f"missing sampling state source: {sampling_states}")
    elif INT64_SEED_FIX not in sampling_states.read_bytes():
        errors.append("sampling seed generation does not request np.int64")

    require_markers(
        errors,
        SIMPLE_KV_DMA,
        SIMPLE_KV_WINDOWS_MARKERS,
        "simple KV-offload Windows DMA fallback",
    )
    require_markers(
        errors,
        BLOCK_TABLE,
        BLOCK_TABLE_WINDOWS_MARKERS,
        "non-Triton CUDA slot-mapping fallback",
    )
    require_markers(
        errors,
        SHARED_REGION,
        SHARED_REGION_WINDOWS_MARKERS,
        "Windows shared CPU-offload mmap support",
    )
    require_markers(
        errors,
        GPU_WORKER,
        GPU_WORKER_WINDOWS_MARKERS,
        "Windows mmap CPU-to-GPU DMA fallback",
    )
    require_markers(
        errors,
        FILE_MAPPER,
        FILE_MAPPER_WINDOWS_MARKERS,
        "Windows-safe filesystem-tier model namespace",
    )
    require_markers(
        errors,
        FS_IO,
        FS_IO_WINDOWS_MARKERS,
        "Windows filesystem-tier read support",
    )
    require_markers(
        errors,
        NATIVE_BATCH_SOURCE,
        (NATIVE_BATCH_WINDOWS_MARKER,),
        "native Windows KV batch-copy fallback",
    )
    require_markers(
        errors,
        NATIVE_FS_SOURCE,
        NATIVE_FS_WINDOWS_MARKERS,
        "native Windows filesystem path support",
    )

    requirements_path = SRC / COMMON_REQUIREMENTS
    if not requirements_path.is_file():
        errors.append(f"missing common requirements: {requirements_path}")
    else:
        requirement_lines = requirements_path.read_text(encoding="utf-8").splitlines()
        for dependency in WINDOWS_REQUIRED_DEPENDENCIES:
            matching = [
                line
                for line in requirement_lines
                if line.lower().startswith(dependency)
            ]
            if not matching or not any(
                'platform_machine == "AMD64"' in line for line in matching
            ):
                errors.append(
                    f"{dependency} requirement does not include Windows AMD64"
                )

    if METADATA_PATH.is_file():
        metadata = BytesParser().parsebytes(METADATA_PATH.read_bytes())
        if metadata.get("Name", "").lower() != DIST_NAME:
            errors.append(
                f"metadata Name is {metadata.get('Name')!r}, expected {DIST_NAME!r}"
            )
        if metadata.get("Version") != VERSION:
            errors.append(
                f"metadata Version is {metadata.get('Version')!r}, expected {VERSION!r}"
            )
        requires_dist = metadata.get_all("Requires-Dist", [])
        for dependency in WINDOWS_REQUIRED_DEPENDENCIES:
            matching = [
                requirement
                for requirement in requires_dist
                if requirement.lower().startswith(dependency)
            ]
            if not matching or not any(
                'platform_machine == "AMD64"' in requirement for requirement in matching
            ):
                errors.append(
                    f"metadata {dependency} requirement does not include Windows AMD64"
                )
    return errors


def validate_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        if bad_member := wheel.testzip():
            raise ValueError(f"ZIP CRC check failed: {bad_member}")

        names = wheel.namelist()
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate archive members: {sorted(duplicates)}")

        if missing := REQUIRED_ARCHIVE_FILES - set(names):
            raise ValueError(f"missing required wheel payloads: {sorted(missing)}")

        if INT64_SEED_FIX not in wheel.read(SAMPLING_STATES):
            raise ValueError("wheel is missing the Windows int64 sampling-seed fix")

        for archive_name, markers, description in (
            (SIMPLE_KV_DMA, SIMPLE_KV_WINDOWS_MARKERS, "simple KV-offload DMA"),
            (
                BLOCK_TABLE,
                BLOCK_TABLE_WINDOWS_MARKERS,
                "non-Triton CUDA slot mapping",
            ),
            (SHARED_REGION, SHARED_REGION_WINDOWS_MARKERS, "shared mmap support"),
            (
                GPU_WORKER,
                GPU_WORKER_WINDOWS_MARKERS,
                "mmap CPU-to-GPU DMA fallback",
            ),
            (
                FILE_MAPPER,
                FILE_MAPPER_WINDOWS_MARKERS,
                "filesystem-tier model namespace",
            ),
            (FS_IO, FS_IO_WINDOWS_MARKERS, "filesystem-tier reads"),
        ):
            data = wheel.read(archive_name)
            if missing_markers := [marker for marker in markers if marker not in data]:
                raise ValueError(
                    f"wheel has incomplete Windows {description}: {missing_markers!r}"
                )

        cute_files = [
            name
            for name in names
            if name.startswith("vllm/vllm_flash_attn/cute/") and name.endswith(".py")
        ]
        if len(cute_files) < 40:
            raise ValueError(f"incomplete CuteDSL payload: {len(cute_files)} files")
        for name in cute_files:
            data = wheel.read(name).replace(b"vllm.vllm_flash_attn.cute", b"")
            if b"flash_attn.cute" in data:
                raise ValueError(f"unrewritten CuteDSL import: {name}")

        metadata_name = f"{DIST_INFO}/METADATA"
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))
        if metadata.get("Name", "").lower() != DIST_NAME:
            raise ValueError(
                f"wheel metadata has invalid Name: {metadata.get('Name')!r}"
            )
        if metadata.get("Version") != VERSION:
            raise ValueError(
                f"wheel metadata has invalid Version: {metadata.get('Version')!r}"
            )
        requires_dist = metadata.get_all("Requires-Dist", [])
        for dependency in WINDOWS_REQUIRED_DEPENDENCIES:
            matching = [
                requirement
                for requirement in requires_dist
                if requirement.lower().startswith(dependency)
            ]
            if not matching or not any(
                'platform_machine == "AMD64"' in requirement for requirement in matching
            ):
                raise ValueError(
                    f"wheel metadata {dependency} requirement omits Windows AMD64"
                )

        record_name = f"{DIST_INFO}/RECORD"
        rows = {
            row[0]: (row[1], row[2])
            for row in csv.reader(io.StringIO(wheel.read(record_name).decode()))
        }
        if set(rows) != set(names):
            raise ValueError("RECORD does not cover every archive member")
        for name in names:
            file_digest, size = rows[name]
            if name == record_name:
                if file_digest or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            data = wheel.read(name)
            if file_digest != digest(data) or size != str(len(data)):
                raise ValueError(f"invalid RECORD entry: {name}")


def main() -> int:
    source_errors = validate_source()
    if source_errors:
        for error in source_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    entries = archive_entries()
    if missing := sorted(REQUIRED_ARCHIVE_FILES - entries.keys()):
        for name in missing:
            print(f"ERROR: missing required wheel payload: {name}", file=sys.stderr)
        print(f"FlashAttention source checked at: {FLASH_ATTN_SRC}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WHEEL.unlink(missing_ok=True)
    records: list[tuple[str, str, int]] = []

    def write(zout: zipfile.ZipFile, archive_name: str, data: bytes) -> None:
        zout.writestr(archive_name, data)
        records.append((archive_name, digest(data), len(data)))

    with zipfile.ZipFile(WHEEL, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for archive_name, payload in sorted(entries.items()):
            data = payload.read_bytes() if isinstance(payload, Path) else payload
            write(zout, archive_name, data)

        write(
            zout,
            f"{DIST_INFO}/METADATA",
            METADATA_PATH.read_bytes(),
        )
        write(zout, f"{DIST_INFO}/WHEEL", wheel_file())
        write(
            zout,
            f"{DIST_INFO}/entry_points.txt",
            (METADATA_DIR / "entry_points.txt").read_bytes(),
        )
        write(
            zout,
            f"{DIST_INFO}/top_level.txt",
            (METADATA_DIR / "top_level.txt").read_bytes(),
        )
        write(zout, f"{DIST_INFO}/licenses/LICENSE", (SRC / "LICENSE").read_bytes())

        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for archive_name, file_digest, size in records:
            writer.writerow([archive_name, file_digest, str(size)])
        writer.writerow([f"{DIST_INFO}/RECORD", "", ""])
        zout.writestr(f"{DIST_INFO}/RECORD", record.getvalue().encode())

    try:
        validate_wheel(WHEEL)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: assembled wheel validation failed: {exc}", file=sys.stderr)
        WHEEL.unlink(missing_ok=True)
        return 1

    print(f"package files : {len(entries)}")
    print(f"flash-attn src: {FLASH_ATTN_SRC}")
    print(f"metadata src  : {METADATA_DIR}")
    print(f"wheel         : {WHEEL}")
    print(f"size          : {WHEEL.stat().st_size / 1e6:.1f} MB")
    print("validation    : passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
