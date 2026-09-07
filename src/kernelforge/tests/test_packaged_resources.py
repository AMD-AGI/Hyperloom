"""Packaging resource smoke tests."""

from kernelforge.resources import resource_path


def test_packaged_resource_paths_are_available():
    # No knowledge_base assertion: that packaged tree was removed once an audit found nothing read it, and
    # `Config.knowledge_dir` went with it.
    assert (resource_path("local_knowledge") / "hardware").is_dir()
    assert (resource_path("examples") / "flydsl-softmax-forge-loop").is_dir()
