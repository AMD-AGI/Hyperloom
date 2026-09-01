"""Unit tests for seed_prebuilt_modules symlink/skip/error/empty branches.

seed_prebuilt_modules symlinks the package's warm ``.so`` into a pristine
baseline shard so the preflight skips the multi-minute cold CK compile. The
warm source is resolved from FORGE_AITER_WARM_JIT_DIR (or the installed aiter
package), so we point it at a tmp dir here and exercise every branch.
"""

from __future__ import annotations

from kernelforge.loop import aiter_cache


def _make_warm(tmp_path, names):
    warm = tmp_path / "warm"
    warm.mkdir()
    for n in names:
        (warm / n).write_text("stub")
    return warm


def test_seed_symlinks_all_modules(monkeypatch, tmp_path):
    warm = _make_warm(tmp_path, ["module_a.so", "module_b.so"])
    monkeypatch.setenv("FORGE_AITER_WARM_JIT_DIR", str(warm))
    shard = tmp_path / "shard"

    stats = aiter_cache.seed_prebuilt_modules(shard)

    assert stats["seeded"] == 2
    assert stats["skipped"] == 0
    assert stats["errors"] == 0
    assert (shard / "module_a.so").is_symlink()
    assert (shard / "module_b.so").is_symlink()


def test_seed_skips_existing_dest(monkeypatch, tmp_path):
    warm = _make_warm(tmp_path, ["module_a.so", "module_b.so"])
    monkeypatch.setenv("FORGE_AITER_WARM_JIT_DIR", str(warm))
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "module_a.so").write_text("already here")

    stats = aiter_cache.seed_prebuilt_modules(shard)

    assert stats["seeded"] == 1  # only module_b
    assert stats["skipped"] == 1  # module_a pre-existed
    assert stats["errors"] == 0


def test_seed_empty_warm_dir_warns(monkeypatch, tmp_path, caplog):
    warm = _make_warm(tmp_path, [])  # no .so at all
    monkeypatch.setenv("FORGE_AITER_WARM_JIT_DIR", str(warm))
    shard = tmp_path / "shard"

    with caplog.at_level("WARNING"):
        stats = aiter_cache.seed_prebuilt_modules(shard)

    assert stats["seeded"] == 0
    assert any("seeded 0 modules" in r.message for r in caplog.records)


def test_seed_missing_source_returns_early(monkeypatch, tmp_path):
    # Override points at a nonexistent dir -> _global_aiter_jit_dir is None ->
    # early return, no crash, nothing seeded.
    monkeypatch.setenv("FORGE_AITER_WARM_JIT_DIR", str(tmp_path / "nope"))
    stats = aiter_cache.seed_prebuilt_modules(tmp_path / "shard")
    assert stats["seeded"] == 0
    assert stats["src"] == ""


def test_seed_symlink_error_counted(monkeypatch, tmp_path):
    warm = _make_warm(tmp_path, ["module_a.so"])
    monkeypatch.setenv("FORGE_AITER_WARM_JIT_DIR", str(warm))
    shard = tmp_path / "shard"

    def _boom(*a, **k):
        raise OSError("read-only fs")

    monkeypatch.setattr(aiter_cache.os, "symlink", _boom)

    stats = aiter_cache.seed_prebuilt_modules(shard)
    assert stats["seeded"] == 0
    assert stats["errors"] == 1
