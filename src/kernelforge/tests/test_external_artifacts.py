"""Tests for transactional external task-preparer artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kernelforge.loop.external_artifacts import (
    ExternalArtifactError,
    ExternalArtifactTransaction,
)


def _make_tree(tmp_path: Path):
    root = tmp_path / "attempt"
    workspace = root / "workspace"
    root.mkdir()
    workspace.mkdir()
    driver = root / "driver.py"
    helper = root / "helper.py"
    obsolete = root / "obsolete.py"
    program = root / "program.md"
    driver.write_text("ORIGINAL_DRIVER\n", encoding="utf-8")
    helper.write_text("ORIGINAL_HELPER\n", encoding="utf-8")
    obsolete.write_text("OBSOLETE\n", encoding="utf-8")
    program.write_text("READ_ONLY_PROGRAM\n", encoding="utf-8")
    (workspace / "kernel.py").write_text("ORIGINAL_KERNEL\n", encoding="utf-8")
    return root, workspace, driver, helper, obsolete, program


def test_publish_applies_complete_helper_change_set(tmp_path):
    root, workspace, driver, helper, obsolete, program = _make_tree(tmp_path)
    audit_dir = root / "artifacts" / "task_preparation"
    audit_dir.mkdir(parents=True)
    audit_file = audit_dir / "evidence.json"
    audit_file.write_text('{"preserved": true}\n', encoding="utf-8")
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace, audit_dir],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        staged = transaction.stage_root
        assert (staged / "workspace").is_symlink()
        assert (staged / "workspace" / "kernel.py").read_text() == "ORIGINAL_KERNEL\n"

        (staged / "driver.py").write_text("PREPARED_DRIVER\n", encoding="utf-8")
        (staged / "helper.py").write_text("PREPARED_HELPER\n", encoding="utf-8")
        (staged / "new_helper.py").write_text("NEW_HELPER\n", encoding="utf-8")
        (staged / "obsolete.py").unlink()
        cache = staged / "__pycache__"
        cache.mkdir()
        (cache / "helper.pyc").write_bytes(b"generated")
        staged_audit = staged / "artifacts" / "task_preparation"
        staged_audit.mkdir(parents=True)
        (staged_audit / "evidence.json").write_text(
            '{"preserved": false}\n',
            encoding="utf-8",
        )

        changes = transaction.publish()

        assert driver.read_text(encoding="utf-8") == "PREPARED_DRIVER\n"
        assert helper.read_text(encoding="utf-8") == "PREPARED_HELPER\n"
        assert (root / "new_helper.py").read_text(encoding="utf-8") == "NEW_HELPER\n"
        assert not obsolete.exists()
        assert not (root / "__pycache__").exists()
        assert (workspace / "kernel.py").read_text() == "ORIGINAL_KERNEL\n"
        assert audit_file.read_text(encoding="utf-8") == '{"preserved": true}\n'
        assert set(changes.wrote_files) == {
            str(driver),
            str(helper),
            str(root / "new_helper.py"),
            str(obsolete),
        }
        assert changes.created_files == (str(root / "new_helper.py"),)
    finally:
        transaction.close()


def test_discard_detects_but_does_not_overwrite_out_of_band_changes(tmp_path):
    root, workspace, driver, helper, _obsolete, _program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    try:
        driver.write_text("ESCAPED_DRIVER_EDIT\n", encoding="utf-8")
        helper.unlink()
        (root / "escaped_helper.py").write_text("ESCAPED_HELPER\n", encoding="utf-8")

        with pytest.raises(ExternalArtifactError, match="left untouched"):
            transaction.rollback()

        assert driver.read_text(encoding="utf-8") == "ESCAPED_DRIVER_EDIT\n"
        assert not helper.exists()
        assert (root / "escaped_helper.py").read_text() == "ESCAPED_HELPER\n"
    finally:
        transaction.close()


def test_publish_rejects_and_restores_out_of_band_changes(tmp_path):
    _root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    try:
        (transaction.stage_root / "driver.py").write_text(
            "PREPARED_DRIVER\n",
            encoding="utf-8",
        )
        driver.write_text("ESCAPED_DRIVER_EDIT\n", encoding="utf-8")

        with pytest.raises(ExternalArtifactError, match="outside the staging"):
            transaction.publish()

        assert driver.read_text(encoding="utf-8") == "ESCAPED_DRIVER_EDIT\n"
        assert transaction.published is False
    finally:
        transaction.close()


def test_publish_rejects_read_only_input_changes(tmp_path):
    _root, workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        (transaction.stage_root / "program.md").write_text(
            "TAMPERED_PROGRAM\n",
            encoding="utf-8",
        )

        with pytest.raises(ExternalArtifactError, match="read-only"):
            transaction.publish()

        assert program.read_text(encoding="utf-8") == "READ_ONLY_PROGRAM\n"
        assert transaction.published is False
    finally:
        transaction.close()


def test_restore_passthroughs_discards_staged_workspace_replacement(tmp_path):
    _root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    try:
        staged_workspace = transaction.stage_root / "workspace"
        staged_workspace.unlink()
        staged_workspace.mkdir()
        (staged_workspace / "kernel.py").write_text(
            "FAKE_STAGED_KERNEL\n",
            encoding="utf-8",
        )

        transaction.restore_passthroughs()

        assert staged_workspace.is_symlink()
        assert (staged_workspace / "kernel.py").read_text() == "ORIGINAL_KERNEL\n"
    finally:
        transaction.close()


def test_external_symlink_escape_is_rejected(tmp_path):
    root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE\n", encoding="utf-8")
    (root / "escaped_link.py").symlink_to(outside)

    with pytest.raises(ExternalArtifactError, match="symlink"):
        ExternalArtifactTransaction(
            driver_path=driver,
            excluded_paths=[workspace],
            passthrough_paths=[workspace],
        )


def test_staged_absolute_symlink_is_not_published(tmp_path):
    root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE\n", encoding="utf-8")
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    try:
        (transaction.stage_root / "new_link.py").symlink_to(outside)

        with pytest.raises(ExternalArtifactError, match="absolute symlink"):
            transaction.publish()

        assert not (root / "new_link.py").exists()
        assert outside.read_text(encoding="utf-8") == "OUTSIDE\n"
    finally:
        transaction.close()


def test_only_one_transaction_can_own_an_external_directory(tmp_path):
    _root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    try:
        with pytest.raises(ExternalArtifactError, match="another.*transaction"):
            ExternalArtifactTransaction(
                driver_path=driver,
                excluded_paths=[workspace],
                passthrough_paths=[workspace],
            )
    finally:
        transaction.close()


def _jit_cache(root: Path) -> Path:
    """The aiter JIT cache a real external driver bundle sits next to."""
    cache = root / "aiter" / "jit" / "flydsl_cache" / "launch_gemm_deadbeef"
    cache.mkdir(parents=True)
    (cache / "0.pkl").write_bytes(b"compiled-kernel-blob")
    return cache


def test_jit_cache_written_during_the_attempt_does_not_block_publish(tmp_path):
    """A compile during preparation must not invalidate the transaction.

    The driver itself writes the JIT cache every time it compiles a kernel, so
    treating that cache as transaction state made publish() fail with "external
    artifact directory changed outside the staging transaction" and discard a
    perfectly good driver — nondeterministically, depending on whether anything
    got compiled during the attempt.
    """
    root, workspace, driver, helper, _, program = _make_tree(tmp_path)
    cache = _jit_cache(root)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        (transaction.stage_root / "driver.py").write_text("REPAIRED\n", encoding="utf-8")
        # The driver compiles a kernel mid-attempt: new blob + rewritten blob.
        (cache / "0.pkl").write_bytes(b"recompiled-kernel-blob")
        (cache / "1.pkl").write_bytes(b"another-kernel-blob")

        changes = transaction.publish()
    finally:
        transaction.close()

    assert driver.read_text(encoding="utf-8") == "REPAIRED\n"
    # The cache is neither staged nor reported, and is left exactly as the
    # compile left it — publish must not roll it back either.
    assert all("flydsl_cache" not in path for path in changes.wrote_files)
    assert (cache / "1.pkl").read_bytes() == b"another-kernel-blob"
    assert (cache / "0.pkl").read_bytes() == b"recompiled-kernel-blob"


def test_jit_cache_is_not_copied_into_staging(tmp_path):
    root, workspace, driver, _, _, program = _make_tree(tmp_path)
    _jit_cache(root)
    (root / "build").mkdir()
    (root / "build" / "module.so").write_bytes(b"\x7fELF")
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        assert not (transaction.stage_root / "aiter" / "jit" / "flydsl_cache").exists()
        assert not (transaction.stage_root / "build").exists()
        # The real payload still stages.
        assert (transaction.stage_root / "driver.py").is_file()
    finally:
        transaction.close()


def test_extra_ignored_dirs_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_EXTERNAL_IGNORE_DIRS", "vendor_cache, other_cache")
    root, workspace, driver, _, _, program = _make_tree(tmp_path)
    (root / "vendor_cache").mkdir()
    (root / "vendor_cache" / "blob.bin").write_bytes(b"x")
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        assert not (transaction.stage_root / "vendor_cache").exists()
        (root / "vendor_cache" / "blob.bin").write_bytes(b"changed")
        changes = transaction.publish()
    finally:
        transaction.close()

    assert all("vendor_cache" not in path for path in changes.wrote_files)


def test_symlinked_driver_is_rejected_before_the_directory_is_locked(tmp_path):
    """A symlinked driver would publish through to a file outside the root."""
    root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    link = root / "driver_link.py"
    link.symlink_to("driver.py")

    with pytest.raises(ExternalArtifactError, match="cannot be a symlink"):
        ExternalArtifactTransaction(
            driver_path=link,
            excluded_paths=[workspace],
            passthrough_paths=[workspace],
        )

    # The rejected transaction must not have taken the directory lock.
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
    )
    transaction.close()


def test_unusable_artifact_directories_are_rejected(tmp_path):
    missing = tmp_path / "nowhere" / "driver.py"
    with pytest.raises(ExternalArtifactError, match="does not exist"):
        ExternalArtifactTransaction(driver_path=missing)

    with pytest.raises(ExternalArtifactError, match="filesystem root"):
        ExternalArtifactTransaction(driver_path=Path("/driver.py"))


def test_driver_inside_an_excluded_path_is_rejected(tmp_path):
    """A driver under the kernel workspace could never be staged or published."""
    _root, workspace, _driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    inner_driver = workspace / "driver.py"
    inner_driver.write_text("ORIGINAL_DRIVER\n", encoding="utf-8")

    with pytest.raises(ExternalArtifactError, match="inside an excluded path"):
        ExternalArtifactTransaction(
            driver_path=inner_driver,
            excluded_paths=[workspace],
            passthrough_paths=[workspace],
        )


def test_a_published_transaction_refuses_to_publish_or_roll_back_again(tmp_path):
    _root, workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        (transaction.stage_root / "driver.py").write_text(
            "PREPARED_DRIVER\n",
            encoding="utf-8",
        )
        transaction.publish()
        assert transaction.published is True

        with pytest.raises(ExternalArtifactError, match="already published"):
            transaction.publish()
        with pytest.raises(ExternalArtifactError, match="cannot roll back published"):
            transaction.rollback()

        # Neither rejected call touched the published result.
        assert driver.read_text(encoding="utf-8") == "PREPARED_DRIVER\n"
    finally:
        transaction.close()


def test_original_symlink_escaping_the_artifact_root_is_rejected(tmp_path):
    """Relative escapes are as dangerous as the absolute ones."""
    root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE\n", encoding="utf-8")
    (root / "escaped_link.py").symlink_to(Path("..") / "outside.py")

    with pytest.raises(ExternalArtifactError, match="escapes its staging root"):
        ExternalArtifactTransaction(
            driver_path=driver,
            excluded_paths=[workspace],
            passthrough_paths=[workspace],
        )


def test_original_symlink_into_excluded_state_is_rejected(tmp_path):
    root, workspace, driver, _helper, _obsolete, _program = _make_tree(tmp_path)
    (root / "workspace_link").symlink_to("workspace")

    with pytest.raises(ExternalArtifactError, match="targets protected state"):
        ExternalArtifactTransaction(
            driver_path=driver,
            excluded_paths=[workspace],
            passthrough_paths=[workspace],
        )


def test_relative_symlinks_survive_a_round_trip_through_staging(tmp_path):
    """Internal relative links are staged as links and published as links."""
    root, workspace, driver, helper, _obsolete, program = _make_tree(tmp_path)
    package = root / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("MODULE\n", encoding="utf-8")
    (root / "alias.py").symlink_to("helper.py")

    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        staged = transaction.stage_root
        assert (staged / "alias.py").is_symlink()
        assert (staged / "pkg" / "mod.py").read_text(encoding="utf-8") == "MODULE\n"

        (staged / "pkg" / "alias_mod.py").symlink_to(Path("..") / "helper.py")

        changes = transaction.publish()
    finally:
        transaction.close()

    published = root / "pkg" / "alias_mod.py"
    assert published.is_symlink()
    assert os.readlink(published) == str(Path("..") / "helper.py")
    assert published.read_text(encoding="utf-8") == "ORIGINAL_HELPER\n"
    assert changes.created_files == (str(published),)
    # The pre-existing link was unchanged, so it is not reported as written.
    assert str(root / "alias.py") not in changes.wrote_files


def test_staged_symlink_into_read_only_input_is_not_published(tmp_path):
    root, workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        (transaction.stage_root / "program_link.md").symlink_to("program.md")

        with pytest.raises(ExternalArtifactError, match="targets protected state"):
            transaction.publish()

        assert not (root / "program_link.md").exists()
        assert transaction.published is False
    finally:
        transaction.close()


def test_staged_symlink_escaping_the_transaction_is_not_published(tmp_path):
    root, workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE\n", encoding="utf-8")
    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        (transaction.stage_root / "escaped_link.py").symlink_to(Path("..") / ".." / "outside.py")

        with pytest.raises(ExternalArtifactError, match="escapes the transaction"):
            transaction.publish()

        assert not (root / "escaped_link.py").exists()
    finally:
        transaction.close()


def test_replacing_an_ancestor_of_excluded_state_is_rejected(tmp_path):
    """Publishing a file over the parent of the kernel workspace would delete it."""
    root, _workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    workspace = nested / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("NESTED_KERNEL\n", encoding="utf-8")

    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        staged_nested = transaction.stage_root / "nested"
        (staged_nested / "workspace").unlink()
        staged_nested.rmdir()
        staged_nested.write_text("NESTED_IS_NOW_A_FILE\n", encoding="utf-8")

        with pytest.raises(ExternalArtifactError, match="ancestor of excluded"):
            transaction.publish()

        assert (workspace / "kernel.py").read_text(encoding="utf-8") == "NESTED_KERNEL\n"
    finally:
        transaction.close()


def test_staged_file_replaces_an_original_directory(tmp_path):
    root, workspace, driver, _helper, _obsolete, program = _make_tree(tmp_path)
    package = root / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("MODULE\n", encoding="utf-8")

    transaction = ExternalArtifactTransaction(
        driver_path=driver,
        excluded_paths=[workspace],
        passthrough_paths=[workspace],
        read_only_paths=[program],
    )
    try:
        staged_package = transaction.stage_root / "pkg"
        (staged_package / "mod.py").unlink()
        staged_package.rmdir()
        staged_package.write_text("PKG_IS_NOW_A_FILE\n", encoding="utf-8")

        changes = transaction.publish()
    finally:
        transaction.close()

    assert package.is_file()
    assert package.read_text(encoding="utf-8") == "PKG_IS_NOW_A_FILE\n"
    assert set(changes.wrote_files) == {str(package), str(package / "mod.py")}
    assert changes.created_files == (str(package),)
