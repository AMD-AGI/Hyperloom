"""Validate the assembled Windows wheel before publishing it."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path


REQUIRED_FLASH_ATTN_FILES = {
    "vllm/vllm_flash_attn/_vllm_fa2_C.pyd",
    "vllm/vllm_flash_attn/layers/rotary.py",
    "vllm/vllm_flash_attn/ops/triton/rotary.py",
    "vllm/vllm_flash_attn/cute/interface.py",
}

REQUIRED_RELEASE_FILES = REQUIRED_FLASH_ATTN_FILES | {
    "vllm/_C_stable_libtorch.pyd",
    "vllm/_moe_C_stable_libtorch.pyd",
    "vllm/cumem_allocator.pyd",
    "vllm/fs_io_C.pyd",
    "vllm/spinloop.pyd",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "vllm/v1/kv_offload/cpu/gpu_worker.py",
    "vllm/v1/kv_offload/cpu/policies/arc.py",
    "vllm/v1/kv_offload/cpu/policies/lru.py",
    "vllm/v1/kv_offload/cpu/shared_offload_region.py",
    "vllm/v1/kv_offload/file_mapper.py",
    "vllm/v1/kv_offload/tiering/fs/io.py",
    "vllm/v1/kv_offload/tiering/fs/manager.py",
    "vllm/v1/kv_offload/tiering/spec.py",
    "vllm/v1/simple_kv_offload/cuda_mem_ops.py",
    "vllm/v1/worker/block_table.py",
    "vllm/v1/worker/gpu/sample/states.py",
    "vllm/vllm-rs.exe",
}

# Namespace-package initializers may be omitted by setuptools; all functional
# FlashAttention payloads remain required above.
REQUIRED_NONEMPTY_FILES = REQUIRED_RELEASE_FILES

SAMPLING_STATES = "vllm/v1/worker/gpu/sample/states.py"
INT64_SEED_FIX = b"_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64"
KV_OFFLOAD_DMA = "vllm/v1/simple_kv_offload/cuda_mem_ops.py"
WINDOWS_KV_OFFLOAD_FIX = b'if sys.platform == "win32":\n        _copy_blocks_windows'
WINDOWS_KV_OFFLOAD_COPY = b"(err,) = cudart.cudaMemcpyAsync("
GPU_WORKER = "vllm/v1/kv_offload/cpu/gpu_worker.py"
SHARED_REGION = "vllm/v1/kv_offload/cpu/shared_offload_region.py"
FILE_MAPPER = "vllm/v1/kv_offload/file_mapper.py"
FS_IO = "vllm/v1/kv_offload/tiering/fs/io.py"
BLOCK_TABLE = "vllm/v1/worker/block_table.py"


def sha256_record(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def validate_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        bad_member = wheel.testzip()
        assert bad_member is None, f"ZIP CRC check failed: {bad_member}"
        names = wheel.namelist()
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        assert not duplicates, f"duplicate archive members: {duplicates}"

        missing = sorted(REQUIRED_RELEASE_FILES - set(names))
        assert not missing, f"missing release payloads: {missing}"
        for name in REQUIRED_NONEMPTY_FILES:
            assert wheel.getinfo(name).file_size > 0, f"empty release payload: {name}"

        assert INT64_SEED_FIX in wheel.read(SAMPLING_STATES), (
            "wheel is missing the Windows int64 sampling-seed fix"
        )
        kv_offload_data = wheel.read(KV_OFFLOAD_DMA).replace(b"\r\n", b"\n")
        assert WINDOWS_KV_OFFLOAD_FIX in kv_offload_data, (
            "wheel is missing the Windows KV-offload DMA fallback"
        )
        assert WINDOWS_KV_OFFLOAD_COPY in kv_offload_data, (
            "wheel is missing the Windows cudaMemcpyAsync implementation"
        )
        assert b'if os.name == "nt" and uses_shared_mmap:' in wheel.read(GPU_WORKER)
        shared_region = wheel.read(SHARED_REGION)
        assert b'tempfile.gettempdir() if os.name == "nt" else "/dev/shm"' in shared_region
        assert b"access=mmap.ACCESS_WRITE" in shared_region
        assert b"def _wait_for_path_size(" in shared_region
        file_mapper = wheel.read(FILE_MAPPER)
        assert b"safe_model_name = ntpath.basename(model_name)" in file_mapper
        fs_io = wheel.read(FS_IO)
        assert b'O_BINARY = getattr(os, "O_BINARY", 0)' in fs_io
        assert b'if hasattr(os, "readv"):' in fs_io
        assert b'cumem allocator is not supported on Windows' in wheel.read(
            "vllm/device_allocator/cumem.py"
        ), "wheel is missing the Windows cumem import guard"
        assert b'_find_windows_cudart()' in wheel.read(
            "vllm/distributed/device_communicators/cuda_wrapper.py"
        ), "wheel is missing the Windows CUDA runtime lookup"
        block_table = wheel.read(BLOCK_TABLE)
        assert b"def _compute_slot_mapping_torch(" in block_table
        assert b'if not HAS_TRITON and self.device.type != "cpu":' in block_table

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, f"expected one METADATA, found {metadata_names}"
        metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
        assert metadata.get("Name", "").lower() == "vllm"
        assert metadata.get("Version") == "0.33.0"
        metadata_data = wheel.read(metadata_names[0])
        assert b'platform_machine == "AMD64"' in metadata_data, (
            "wheel metadata does not install AMD64 structured-output dependencies"
        )

        cute_files = [
            name
            for name in names
            if name.startswith("vllm/vllm_flash_attn/cute/") and name.endswith(".py")
        ]
        assert len(cute_files) >= 40, f"incomplete CuteDSL payload: {len(cute_files)} files"
        for name in cute_files:
            data = wheel.read(name).replace(b"vllm.vllm_flash_attn.cute", b"")
            assert b"flash_attn.cute" not in data, (
                f"unrewritten CuteDSL import in {name}"
            )

        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        assert len(record_names) == 1, f"expected one RECORD, found {record_names}"
        record_name = record_names[0]
        rows = {
            row[0]: (row[1], row[2])
            for row in csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8")))
        }
        assert set(rows) == set(names), "RECORD does not cover every archive member"

        for name in names:
            digest, size = rows[name]
            if name == record_name:
                assert digest == "" and size == "", "RECORD must hash neither itself nor its size"
                continue
            data = wheel.read(name)
            assert digest == sha256_record(data), f"RECORD hash mismatch: {name}"
            assert size == str(len(data)), f"RECORD size mismatch: {name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    validate_wheel(args.wheel.resolve())
    print(f"wheel content validation passed: {args.wheel.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
