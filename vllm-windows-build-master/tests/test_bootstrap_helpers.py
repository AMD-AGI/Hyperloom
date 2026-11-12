from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class BootstrapHelperTests(unittest.TestCase):
    def run_script(self, script: str, *args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / script),
                *(str(arg) for arg in args),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_hash_verifier_accepts_exact_file_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vllm bootstrap ") as directory:
            artifact = Path(directory) / "python archive.zip"
            data = b"bootstrap-test"
            artifact.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()

            valid = self.run_script(
                "verify_bootstrap.ps1", artifact, digest, len(data)
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertIn(digest.upper(), valid.stdout)

            invalid = self.run_script(
                "verify_bootstrap.ps1", artifact, "0" * 64, len(data)
            )
            self.assertNotEqual(invalid.returncode, 0)

    def test_zip_extractor_overwrites_and_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vllm extract ") as directory:
            root = Path(directory)
            archive = root / "safe archive.zip"
            destination = root / "destination with spaces"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("nested/payload.txt", "first")

            first = self.run_script("expand_zip.ps1", archive, destination)
            self.assertEqual(first.returncode, 0, first.stderr)
            payload = destination / "nested" / "payload.txt"
            self.assertEqual(payload.read_text(encoding="utf-8"), "first")

            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("nested/payload.txt", "second")
            second = self.run_script("expand_zip.ps1", archive, destination)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(payload.read_text(encoding="utf-8"), "second")

            malicious = root / "malicious.zip"
            with zipfile.ZipFile(malicious, "w") as output:
                output.writestr("../escape.txt", "blocked")
            rejected = self.run_script("expand_zip.ps1", malicious, destination)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
