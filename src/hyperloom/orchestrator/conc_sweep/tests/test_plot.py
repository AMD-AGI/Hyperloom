# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from hyperloom.common.perf_metric import graded_metric_key


def test_render_conc_sweep_curve_no_data_returns_none(tmp_path: Path):
    from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

    payload = {
        "baseline": {"points": []},
        "optimized": {"points": []},
    }
    result = render_conc_sweep_curve(
        payload, tmp_path / "out.png", ConcCurveOptions(model_label="M", gpu_label="GPU", tp=1)
    )
    assert result is None


def test_render_conc_sweep_curve_writes_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

    payload = {
        "baseline": {
            "points": [
                {"conc": 16, "output_throughput": 1600.0},
                {"conc": 4, "output_throughput": 600.0},
            ]
        },
        "optimized": {
            "points": [
                {"conc": 16, "output_throughput": 2000.0},
                {"conc": 4, "output_throughput": 750.0},
            ]
        },
        "roofline_ceiling": {
            "rows": [
                {"conc": 4, "t_peak_tok_s": 800.0},
                {"conc": 16, "t_peak_tok_s": 2500.0},
            ]
        },
    }
    out_path = tmp_path / "plots" / "curve.png"
    result = render_conc_sweep_curve(
        payload, out_path, ConcCurveOptions(model_label="TestModel", gpu_label="MI300X", tp=8)
    )
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


class TestTheChartFollowsTheGradedAxis:
    """The chart has to show the ranking the session was scored by."""

    def _payload(self, mode: str, metric: str) -> dict[str, Any]:
        return {
            "benchmark_mode": mode,
            "summary": {"metric": metric},
            "baseline": {
                "points": [
                    {
                        "conc": 8,
                        "output_throughput": 183.44,
                        "total_token_throughput": 25984.8,
                        "e2e_norm_intvty_p90": 447.2,
                    }
                ]
            },
            "optimized": {"points": []},
        }

    def test_the_total_axis_reads_interactivity_and_total(self):
        from hyperloom.orchestrator.conc_sweep import plot

        axes = plot._axes_for_metric("total_token_throughput", tp_eff=8.0)
        xs, ys, _concs = plot._arm_series(
            self._payload("agentx", "total_token_throughput")["baseline"]["points"], 8.0, axes
        )
        assert xs == [pytest.approx(447.2)]
        assert ys == [pytest.approx(25984.8 / 8.0)]
        assert "P90 Interactivity" in axes.x_label
        assert "per Chip" in axes.y_label

    def test_the_output_axis_keeps_the_output_pair(self):
        from hyperloom.orchestrator.conc_sweep import plot

        axes = plot._axes_for_metric("output_throughput", tp_eff=8.0)
        xs, ys, _concs = plot._arm_series(
            self._payload("synthetic", "output_throughput")["baseline"]["points"], 8.0, axes
        )
        assert xs == [pytest.approx(183.44 / 8)]
        assert ys == [pytest.approx(183.44 / 8.0)]

    def test_a_summary_naming_no_metric_reads_as_the_comparison_default(self):
        from hyperloom.orchestrator.conc_sweep import plot

        assert plot._graded_metric_of({}) == "output_throughput"
        assert plot._graded_metric_of({"summary": {"metric": ""}}) == "output_throughput"
        assert plot._graded_metric_of({"summary": {"metric": "total_token_throughput"}}) == "total_token_throughput"

    @pytest.mark.parametrize(
        ("mode", "env"),
        [
            ("agentx", {}),
            ("synthetic", {}),
            ("agentx", {"HYPERLOOM_PERF_METRIC": "output_throughput"}),
            ("synthetic", {"HYPERLOOM_PERF_METRIC": "intvty_v1"}),
            ("", {"HYPERLOOM_AGENTX": "1"}),
        ],
    )
    def test_the_chart_plots_the_field_the_summary_graded(self, monkeypatch, mode: str, env: dict[str, str]):
        """Binds both sides: whichever way the session resolved its axis, the curve reads the same point field the summary took its speedups on."""
        from hyperloom.orchestrator.conc_sweep import plot

        monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
        monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        metric = graded_metric_key(benchmark_mode=mode)
        payload = self._payload(mode, metric)
        chart_metric = plot._graded_metric_of(payload)
        axes = plot._axes_for_metric(chart_metric, tp_eff=8.0)
        _xs, ys, _concs = plot._arm_series(payload["baseline"]["points"], 8.0, axes)

        # AgentX chart: x = interactivity, y = total_token_throughput / tp.
        # Synthetic chart: x = output_throughput / conc, y = output_throughput / tp.
        from hyperloom.common.perf_metric import GRADED_OUTPUT

        if chart_metric != GRADED_OUTPUT:
            # AgentX: y is total throughput per chip
            assert ys == [pytest.approx(25984.8 / 8.0)]
        else:
            # Synthetic: y is output throughput per TP
            assert ys == [pytest.approx(183.44 / 8.0)]

    def test_a_rung_missing_its_axis_is_dropped_not_zeroed(self):
        from hyperloom.orchestrator.conc_sweep import plot

        axes = plot._axes_for_metric("total_token_throughput", tp_eff=1.0)
        points = [
            {"conc": 8, "total_token_throughput": 25984.8},
            {"conc": 4, "total_token_throughput": 20000.0, "e2e_norm_intvty_p90": 500.0},
        ]
        xs, ys, _concs = plot._arm_series(points, 1.0, axes)
        assert xs == [pytest.approx(500.0)]
        assert ys == [pytest.approx(20000.0)]

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("e2e_norm_intvty_p90", ([447.2], [25984.8 / 4.0])),
            ("output_throughput", ([183.44 / 8.0], [183.44 / 4.0])),
        ],
    )
    def test_a_swept_rung_lands_on_the_pair_its_axis_is_read_on(
        self, metric: str, expected: tuple[list[float], list[float]]
    ):
        """The row a round becomes carries both pairs, so either chart draws it rather than dropping it."""
        from hyperloom.orchestrator.actions.executors.grid._grid_runner import VariantResult
        from hyperloom.orchestrator.conc_sweep import plot
        from hyperloom.orchestrator.conc_sweep.report import point_from_variant

        point = point_from_variant(
            VariantResult(
                name="optimized_conc8",
                extra_server_args="",
                extra_envs={"CONC": "8"},
                status="succeeded",
                output_throughput=183.44,
                total_token_throughput=25984.8,
                intvty_p90=447.2,
            ),
            arm="optimized",
        )
        xs, ys, concs = plot._arm_series([point], 4.0, plot._axes_for_metric(metric, tp_eff=4.0))
        assert (xs, ys, concs) == (*expected, [8.0])


class TestTheCurveFollowsTheLadder:
    """A ladder is joined in the order it was walked, and a rung only one arm measured says so."""

    # One real sweep: baseline skipped conc 4 for budget, and optimized conc 4 reads lower interactivity than conc 8.
    _BASELINE = {8: (72.87, 47001.8), 14: (59.93, 72138.7), 28: (39.36, 153283.4)}
    _OPTIMIZED = {4: (75.09, 25519.5), 8: (76.30, 47001.8), 14: (61.68, 71828.6), 28: (42.28, 157345.0)}

    @staticmethod
    def _arm(rows: dict[int, tuple[float, float]]) -> dict[str, Any]:
        return {
            "points": [{"conc": c, "e2e_norm_intvty_p90": x, "total_token_throughput": y} for c, (x, y) in rows.items()]
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "benchmark_mode": "agentx",
            "summary": {"metric": "total_token_throughput"},
            "baseline": self._arm(self._BASELINE),
            "optimized": self._arm(self._OPTIMIZED),
        }

    def test_a_series_runs_in_concurrency_order_even_where_interactivity_does_not(self):
        from hyperloom.orchestrator.conc_sweep import plot

        axes = plot._axes_for_metric("total_token_throughput", tp_eff=8.0)
        xs, _ys, concs = plot._arm_series(self._payload()["optimized"]["points"], 8.0, axes)
        assert concs == [4.0, 8.0, 14.0, 28.0]
        # conc 4 reads lower interactivity than conc 8, so an x-sorted series would have put it second.
        assert xs == [pytest.approx(75.09), pytest.approx(76.30), pytest.approx(61.68), pytest.approx(42.28)]

    def test_the_rendered_line_joins_the_rungs_in_concurrency_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

        calls = _install_fake_matplotlib(monkeypatch)
        out_path = tmp_path / "curve.png"
        assert render_conc_sweep_curve(self._payload(), out_path, ConcCurveOptions(tp=8)) == out_path

        lines = {
            kwargs["label"]: args[0]
            for args, kwargs in calls["plots"]
            if kwargs.get("label") in {"baseline", "optimized"}
        }
        assert lines["optimized"] == [pytest.approx(x) for x, _y in map(self._OPTIMIZED.get, (4, 8, 14, 28))]
        assert lines["baseline"] == [pytest.approx(x) for x, _y in map(self._BASELINE.get, (8, 14, 28))]
        assert sorted(text for (text, *_rest), _kw in calls["annotations"] if text.startswith("c=")) == sorted(
            [f"c={c} {y / 8:.0f}" for c, (_x, y) in self._BASELINE.items()]
            + [f"c={c} {y / 8:.0f}" for c, (_x, y) in self._OPTIMIZED.items()]
        )

    def test_a_rung_only_one_arm_measured_is_drawn_hollow_and_explained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

        calls = _install_fake_matplotlib(monkeypatch)
        render_conc_sweep_curve(self._payload(), tmp_path / "curve.png", ConcCurveOptions(tp=8))

        hollow = [(args, kwargs) for args, kwargs in calls["plots"] if "markerfacecolor" in kwargs and args[0]]
        assert [args[0] for args, _kwargs in hollow] == [[pytest.approx(75.09)]]
        assert [args[1] for args, _kwargs in hollow] == [[pytest.approx(25519.5 / 8)]]
        assert any("one arm only" in str(kwargs.get("label", "")) for _args, kwargs in calls["plots"])

    def test_matched_rungs_carry_no_hollow_marker_or_note(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

        calls = _install_fake_matplotlib(monkeypatch)
        payload = self._payload()
        payload["optimized"] = self._arm({c: self._OPTIMIZED[c] for c in self._BASELINE})
        render_conc_sweep_curve(payload, tmp_path / "curve.png", ConcCurveOptions(tp=8))

        assert not any("markerfacecolor" in kwargs for _args, kwargs in calls["plots"])

    def test_an_arm_against_an_empty_arm_has_nothing_to_be_unpaired_with(self):
        from hyperloom.orchestrator.conc_sweep import plot

        assert plot._unpaired([4.0, 8.0], []) == [False, False]
        assert plot._unpaired([4.0, 8.0], [8.0]) == [True, False]


def test_render_conc_sweep_curve_missing_matplotlib_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When matplotlib is missing, render_conc_sweep_curve returns None gracefully."""
    import builtins

    _real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("no module named matplotlib (mocked)")
        return _real_import(name, *args, **kwargs)

    from hyperloom.orchestrator.conc_sweep.plot import ConcCurveOptions, render_conc_sweep_curve

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    result = render_conc_sweep_curve(
        {"baseline": {"points": [{"conc": 4, "output_throughput": 400.0}]}, "optimized": {"points": []}},
        tmp_path / "out.png",
        ConcCurveOptions(model_label="M", gpu_label="G", tp=1),
    )
    assert result is None


def test_conc_sweep_plot_series_helpers_filter_and_sort_points_by_conc():
    from hyperloom.orchestrator.conc_sweep import plot as conc_sweep_plot

    axes = conc_sweep_plot._axes_for_metric("output_throughput", 2.0)
    xs, ys, concs = conc_sweep_plot._arm_series(
        [
            {"conc": 4, "output_throughput": 800.0},
            {"conc": 2, "output_throughput": 300.0},
            {"conc": 0, "output_throughput": 1000.0},
            {"conc": 1, "output_throughput": None},
            {"conc": 8, "output_throughput": -1},
        ],
        2.0,
        axes,
    )
    assert xs == [150.0, 200.0]
    assert ys == [150.0, 400.0]
    assert concs == [2.0, 4.0]
    assert conc_sweep_plot._arm_series([{"conc": 0, "output_throughput": 0}], 1.0, axes) == ([], [], [])


def _install_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Render through a stand-in pyplot and return what the chart asked it to do."""
    calls: dict[str, Any] = {"plots": [], "annotations": [], "labels": [], "titles": [], "closed": False}

    class _FakePatch:
        def set_facecolor(self, value):
            calls["fig_facecolor"] = value

    class _FakeFig:
        patch = _FakePatch()

        def tight_layout(self):
            calls["tight_layout"] = True

        def savefig(self, path, **kwargs):
            calls["saved"] = (path, kwargs)
            Path(path).write_bytes(b"fake-png")

    class _FakeSpine:
        def set_color(self, value):
            calls.setdefault("spines", []).append(value)

    class _FakeLegendText:
        def set_color(self, value):
            calls.setdefault("legend_text_colors", []).append(value)

    class _FakeLegend:
        def get_texts(self):
            return [_FakeLegendText(), _FakeLegendText()]

    class _FakeAx:
        spines = {"left": _FakeSpine(), "right": _FakeSpine()}

        def set_facecolor(self, value):
            calls["ax_facecolor"] = value

        def plot(self, *args, **kwargs):
            calls["plots"].append((args, kwargs))

        def annotate(self, *args, **kwargs):
            calls["annotations"].append((args, kwargs))

        def set_xlabel(self, value, **kwargs):
            calls["labels"].append(("x", value, kwargs))

        def set_ylabel(self, value, **kwargs):
            calls["labels"].append(("y", value, kwargs))

        def set_title(self, value, **kwargs):
            calls["titles"].append((value, kwargs))

        def grid(self, *args, **kwargs):
            calls["grid"] = (args, kwargs)

        def tick_params(self, **kwargs):
            calls["ticks"] = kwargs

        def legend(self, **kwargs):
            calls["legend"] = kwargs
            return _FakeLegend()

    fake_matplotlib = ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: calls.setdefault("backend", backend)
    fake_pyplot = ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda figsize: (_FakeFig(), _FakeAx())
    fake_pyplot.close = lambda fig: calls.update({"closed": fig is not None})
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
    return calls
