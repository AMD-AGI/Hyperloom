"""Tests for the learning module (postmortem, skills, sources)."""

import tempfile

from kernelforge.tracker.schema import Experiment
from kernelforge.learning.postmortem import PostMortem


# ─── PostMortem tests ───


def test_postmortem_finds_regressions():
    exp = Experiment(experiment_id="test", backend="ck")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0, config={"BLOCK_M": 128})
    exp.add_iteration(
        snr_db=33.0, wall_ms=2.5, config={"BLOCK_M": 64}, decision="tried smaller block"
    )  # 25% regression

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PostMortem(tmpdir)
        lessons = pm.analyze(exp)

        pitfalls = [l for l in lessons if l.category == "pitfall"]
        assert len(pitfalls) >= 1
        assert "regression" in pitfalls[0].title.lower() or "regression" in pitfalls[0].description.lower()


def test_postmortem_finds_improvements():
    exp = Experiment(experiment_id="test", backend="flydsl")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0, config={"wpe": 3})
    exp.add_iteration(snr_db=34.0, wall_ms=1.5, config={"wpe": 2}, decision="reduced waves per EU")  # 25% improvement

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PostMortem(tmpdir)
        lessons = pm.analyze(exp)

        opts = [l for l in lessons if l.category == "optimization"]
        assert len(opts) >= 1


def test_postmortem_detects_occupancy_cliff():
    exp = Experiment(experiment_id="test", backend="ck")
    exp.add_iteration(snr_db=35.0, wall_ms=1.0, vgpr=240)
    exp.add_iteration(snr_db=34.0, wall_ms=1.8, vgpr=280)  # crossed 256

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PostMortem(tmpdir)
        lessons = pm.analyze(exp)

        occupancy = [l for l in lessons if "occupancy" in l.title.lower()]
        assert len(occupancy) >= 1


def test_postmortem_saves_lessons():
    exp = Experiment(experiment_id="test", backend="ck")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0)
    exp.add_iteration(snr_db=33.0, wall_ms=2.5)  # regression

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PostMortem(tmpdir)
        lessons = pm.analyze(exp)
        saved = pm.save_lessons(lessons)

        for path in saved:
            assert path.exists()
            content = path.read_text()
            assert "## What Happened" in content
