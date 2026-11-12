from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLLM_SHA256 = "28C45D24E246D515D1AE87E8CA286CF7D3B1DADBD8AEC4FEB685B01B93627321"
MTQ_SHA256 = "5B310E05904B588539D9A8E3374DFA6C160F025F9C2099BA5C7877C79B2FA149"
PATCH_SHA256 = "CB0D7041F0A3B1C126C1FC89930A861296D62066590D6037EC72FF079EAF4206"


def batch_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    pattern = re.compile(r'^set "([^=]+)=(.*)"$', re.IGNORECASE)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            settings[match.group(1).upper()] = match.group(2)
    return settings


def markdown_section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


class ReleaseContractTests(unittest.TestCase):
    def test_installer_and_launcher_use_same_artifact_contract(self) -> None:
        install = batch_settings(ROOT / "install.bat")
        launch = batch_settings(ROOT / "launch.bat")
        self.assertEqual(install["WHEEL_SHA256"], VLLM_SHA256)
        self.assertEqual(install["MTQ_SHA256"], MTQ_SHA256)
        self.assertEqual(launch["EXPECTED_WHEEL_SHA256"], VLLM_SHA256)
        self.assertEqual(launch["EXPECTED_MTQ_SHA256"], MTQ_SHA256)
        self.assertEqual(install["WHEEL_SIZE"], "243466236")
        self.assertEqual(install["MTQ_SIZE"], "136429")
        self.assertIn("v0.33.0-win-cu130", install["WHEEL_URL"])
        self.assertIn("vllm-0.33.0-cp313-cp313-win_amd64.whl", install["WHEEL_URL"])
        self.assertIn("dist-v0.33.0", install["WHEEL_FILE"])

        verifier = (ROOT / "verify_install.py").read_text(encoding="utf-8")
        candidate = (ROOT / "docs" / "v0.33.0-build-candidate.md").read_text(
            encoding="utf-8"
        )
        self.assertIn('EXPECTED_VLLM_VERSION = "0.33.0"', verifier)
        self.assertIn("vllm-0.33.0-cp313-cp313-win_amd64.whl", candidate)
        self.assertIn("28c45d24e246d515d1ae87e8ca286cf7d3b1dadbd8aec4feb685b01b93627321", candidate)

    def test_verifier_distinguishes_wheel_and_runtime_versions(self) -> None:
        from verify_install import validate_vllm_versions

        validate_vllm_versions("0.33.0", "0.33.0")
        validate_vllm_versions("0.33.0+local", "0.33.0")

        with self.assertRaisesRegex(RuntimeError, "distribution version"):
            validate_vllm_versions("0.33.0", "0.33.0+cu130")
        with self.assertRaisesRegex(RuntimeError, "runtime version"):
            validate_vllm_versions("0.29.0", "0.33.0")

    def test_installer_is_atomic_and_does_not_parse_hash_stdout(self) -> None:
        script = (ROOT / "install.bat").read_text(encoding="utf-8")
        self.assertIn("verify_artifact.py", script)
        self.assertIn("verify_bootstrap.ps1", script)
        self.assertIn("expand_zip.ps1", script)
        self.assertIn("PowerShell 3 or newer", script)
        download_lines = [
            line for line in script.splitlines() if "Invoke-WebRequest" in line
        ]
        self.assertEqual(len(download_lines), 5)
        self.assertTrue(
            all("Invoke-WebRequest -UseBasicParsing" in line for line in download_lines)
        )
        self.assertIn(
            "pypa/get-pip/5e84c8360eaf92009551b3eec69d734137f31cec/",
            script,
        )
        self.assertNotIn("bootstrap.pypa.io/get-pip.py", script)
        self.assertIn("%WHEEL_NAME%.part", script)
        self.assertIn("%MTQ_NAME%.part", script)
        self.assertIn('"%~dp0python.part"', script)
        self.assertIn('move /Y "%~dp0python.part" "%~dp0python"', script)
        self.assertNotIn("Get-FileHash", script)
        self.assertNotIn("Expand-Archive", script)
        bootstrap_helpers = (
            (ROOT / "verify_bootstrap.ps1").read_text(encoding="utf-8")
            + (ROOT / "expand_zip.ps1").read_text(encoding="utf-8")
        )
        self.assertNotIn("Get-FileHash", bootstrap_helpers)
        self.assertNotIn("Expand-Archive", bootstrap_helpers)
        self.assertLess(script.index("--cuda"), script.rindex("WHEEL_SHA256=%WHEEL_SHA256%"))

    def test_build_fails_closed(self) -> None:
        script = (ROOT / "build.bat").read_text(encoding="utf-8")
        self.assertIn("git apply --reverse --check", script)
        self.assertIn("Source is not based on upstream vLLM v0.25.1", script)
        self.assertIn("call :requireArtifact", script)
        self.assertIn('call :requireArtifact "vllm\\fs_io_C.pyd"', script)
        self.assertNotIn("Continuing anyway", script)
        self.assertNotIn("xcopy", script.lower())

    def test_patch_digest(self) -> None:
        from verify_artifact import sha256_file

        self.assertEqual(sha256_file(ROOT / "vllm-windows-v12.patch"), PATCH_SHA256)

    def test_patch_forces_int64_sampling_seed(self) -> None:
        patch = (ROOT / "vllm-windows-v12.patch").read_text(encoding="utf-8")
        self.assertIn(
            "+            seed = np.random.randint("
            "_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64)",
            patch,
        )

    def test_patch_uses_safe_windows_kv_offload_dma(self) -> None:
        patch = (ROOT / "vllm-windows-v12.patch").read_text(encoding="utf-8")
        self.assertIn('+    if sys.platform == "win32":', patch)
        self.assertIn("+        _copy_blocks_windows", patch)
        self.assertIn("+        (err,) = cudart.cudaMemcpyAsync(", patch)

    def test_patch_contains_windows_tiered_kv_cache_fixes(self) -> None:
        patch = (ROOT / "vllm-windows-v12.patch").read_text(encoding="utf-8")
        markers = (
            '+    if os.name == "nt" and uses_shared_mmap:',
            "+def _wait_for_path_size(",
            '+O_BINARY = getattr(os, "O_BINARY", 0)',
            "+        safe_model_name = ntpath.basename(model_name) or \"model\"",
            "+def _compute_slot_mapping_torch(",
            'platform_machine == "AMD64"',
        )
        for marker in markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, patch)

    def test_launcher_keeps_kv_offload_opt_in(self) -> None:
        launcher = (ROOT / "vllm_launcher.py").read_text(encoding="utf-8")
        self.assertIn('choices=("disabled", "cpu-lru", "cpu-arc", "fs-lru", "fs-arc")', launcher)
        self.assertIn('default="disabled"', launcher)
        self.assertIn('"spec_name": "TieringOffloadingSpec"', launcher)
        self.assertIn('"type": "fs"', launcher)
        self.assertIn('"offload_prompt_only": True', launcher)
        self.assertIn('--kv-offload-fs-root is required for fs-lru/fs-arc', launcher)
        self.assertIn("has no automatic size quota", launcher)
        launch_batch = (ROOT / "launch.bat").read_text(encoding="utf-8")
        self.assertIn(
            'if not defined PYTHONHASHSEED set "PYTHONHASHSEED=0"',
            launch_batch,
        )

    def test_turing_compatibility_controls_are_forwarded(self) -> None:
        launcher = (ROOT / "vllm_launcher.py").read_text(encoding="utf-8")
        for flag in (
            'parser.add_argument("--attention-backend", default=None,',
            'parser.add_argument("--kv-cache-dtype", default=None,',
            'parser.add_argument("--block-size", type=int, default=None,',
            'parser.add_argument("--turing-compat", action="store_true",',
        ):
            self.assertIn(flag, launcher)
        self.assertIn('args.attention_backend = args.attention_backend or "TRITON_ATTN"', launcher)
        self.assertIn('args.kv_cache_dtype = args.kv_cache_dtype or "float16"', launcher)
        self.assertIn('args.block_size = args.block_size or 32', launcher)
        self.assertIn(
            'args.gpu_memory_utilization = TURING_GPU_MEMORY_UTILIZATION',
            launcher,
        )
        self.assertIn('args.max_num_seqs = TURING_MAX_NUM_SEQS', launcher)
        self.assertIn(
            'args.max_num_batched_tokens = TURING_MAX_NUM_BATCHED_TOKENS',
            launcher,
        )
        self.assertIn('AttentionConfig(backend=args.attention_backend)', launcher)
        self.assertIn('llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype', launcher)
        self.assertIn('llm_kwargs["block_size"] = args.block_size', launcher)
        self.assertIn('llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size', launcher)
        self.assertIn('set "PYTHONUTF8=1"', (ROOT / "launch.bat").read_text(encoding="utf-8"))

    def test_launcher_rejects_direct_gguf_with_actionable_message(self) -> None:
        launcher = (ROOT / "vllm_launcher.py").read_text(encoding="utf-8")
        self.assertIn('Path(model).suffix.lower() == ".gguf"', launcher)
        self.assertIn("validate_model_input(args.model)", launcher)
        self.assertIn("Direct GGUF files are not supported", launcher)
        self.assertIn("config.json and safetensors/AWQ/GPTQ weights", launcher)
        self.assertNotIn('"model_type": "gguf"', launcher)
        self.assertNotIn("HuggingFace directory or GGUF file", launcher)

    def test_fastapi_request_annotation_is_globally_resolvable(self) -> None:
        launcher = (ROOT / "vllm_launcher.py").read_text(encoding="utf-8")
        self.assertIn("from fastapi import FastAPI, Request", launcher)
        self.assertNotIn("    from fastapi import FastAPI, Request", launcher)
        self.assertIn("async def embeddings(request: Request):", launcher)
        self.assertIn("async def chat_completions(request: Request):", launcher)

    def test_quickstarts_use_fast_reproducible_baseline(self) -> None:
        sections = {
            "README.md": markdown_section(
                (ROOT / "README.md").read_text(encoding="utf-8"),
                "## Hello world",
            ),
            "VLLM.md": markdown_section(
                (ROOT / "VLLM.md").read_text(encoding="utf-8"),
                "## Hello World",
            ),
            "docs/usage.md": markdown_section(
                (ROOT / "docs" / "usage.md").read_text(encoding="utf-8"),
                "## (A) Python embedding",
            ),
        }
        for name, section in sections.items():
            with self.subTest(name=name):
                snippet = section.split("```python", 1)[1].split("```", 1)[0]
                ast.parse(snippet)
                self.assertIn('kv_cache_dtype="auto"', section)
                self.assertNotIn('kv_cache_dtype="isoquant4"', section)
                self.assertIn("max_tokens=32", section)
                self.assertIn("seed=0", section)
                self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", section)

    def test_launcher_defaults_are_documented_exactly(self) -> None:
        launcher = (ROOT / "vllm_launcher.py").read_text(encoding="utf-8")
        usage = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
        expected = (
            (
                'parser.add_argument("--port", type=int, default=8100)',
                "| `--port` | 8100 |",
            ),
            (
                'parser.add_argument("--max-model-len", type=int, default=8192,',
                "| `--max-model-len` | 8192 |",
            ),
            (
                "DEFAULT_GPU_MEMORY_UTILIZATION = 0.6",
                "| `--gpu-memory-utilization` | 0.6 (0.89 with `--turing-compat`) |",
            ),
            (
                "DEFAULT_MAX_NUM_SEQS = 64",
                "| `--max-num-seqs` | 64 (1 with `--turing-compat`) |",
            ),
            (
                "TURING_MAX_NUM_BATCHED_TOKENS = 2048",
                "| `--max-num-batched-tokens` | auto (2048 with `--turing-compat`) |",
            ),
            (
                'parser.add_argument("--gpu-id", type=int, default=None,',
                "| `--gpu-id` | (none) |",
            ),
        )
        for code_default, documented_default in expected:
            self.assertIn(code_default, launcher)
            self.assertIn(documented_default, usage)

    def test_windows_docs_do_not_enable_unsupported_expandable_segments(
        self,
    ) -> None:
        paths = (
            ROOT / "README.md",
            ROOT / "VLLM.md",
            ROOT / "build.bat",
            ROOT / "docs" / "benchmarks.md",
            ROOT / "docs" / "install.md",
            ROOT / "docs" / "usage.md",
            ROOT / "tests" / "README.md",
            ROOT / "tests" / "test_tq_real.py",
            ROOT / "tests" / "test_tq_thorough.py",
            ROOT / "tests" / "test_v19.py",
        )
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    text,
                )
                self.assertNotIn(
                    'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", '
                    '"expandable_segments:True")',
                    text,
                )

        troubleshooting = (ROOT / "docs" / "troubleshooting.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "expandable_segments not supported on this platform",
            troubleshooting,
        )


if __name__ == "__main__":
    unittest.main()
