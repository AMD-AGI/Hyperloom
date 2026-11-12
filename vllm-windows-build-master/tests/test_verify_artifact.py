from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from verify_artifact import verify_artifact


class VerifyArtifactTests(unittest.TestCase):
    def test_accepts_exact_artifact_in_path_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vllm artifact ") as directory:
            path = Path(directory) / "release wheel.whl"
            data = b"known-good-artifact"
            path.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            self.assertTrue(verify_artifact(path, expected, len(data)))

    def test_rejects_bad_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.whl"
            path.write_bytes(b"bad")
            self.assertFalse(verify_artifact(path, "0" * 64, 3))

    def test_rejects_bad_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.whl"
            data = b"data"
            path.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            self.assertFalse(verify_artifact(path, expected, len(data) + 1))

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                verify_artifact(Path(directory) / "missing.whl", "0" * 64, 0)
            )


if __name__ == "__main__":
    unittest.main()
