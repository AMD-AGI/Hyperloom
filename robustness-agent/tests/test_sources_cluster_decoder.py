# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``sources/cluster_decoder.py`` (M2.5)."""

from __future__ import annotations

from robustness_agent.sources.cluster_decoder import (
    decode_gpu_snapshot,
    merge_gpu_snapshots,
)


def _series(labels, values):
    return {"labels": labels, "values": values}


def _result(name, *, category="gpu", unit="C", series=()):
    return {
        "name": name,
        "category": category,
        "unit": unit,
        "series": list(series),
    }


def _pod(ns, name, *, results=()):
    return {"namespace": ns, "name": name, "results": list(results)}


def _response(*, pods=()):
    return {"data": {"pods": list(pods)}}


def test_decode_returns_empty_on_garbage():
    assert decode_gpu_snapshot(None) == {}
    assert decode_gpu_snapshot({}) == {}
    assert decode_gpu_snapshot({"data": {}}) == {}
    assert decode_gpu_snapshot({"data": {"pods": "wat"}}) == {}


def test_decode_picks_latest_value_per_metric():
    """Multiple samples per series -> only the latest one wins."""

    response = _response(
        pods=[
            _pod(
                "ns1",
                "podA",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [
                                    {"timestamp": 1, "value": 70.0},
                                    {"timestamp": 5, "value": 95.0},
                                    {"timestamp": 3, "value": 80.0},
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    out = decode_gpu_snapshot(response)
    assert out["tool"] == "robust-api"
    assert len(out["gpus"]) == 1
    snap = out["gpus"][0]
    assert snap["gpu_id"] == 0
    assert snap["temperature_c"] == 95.0
    assert snap["pod_namespace"] == "ns1"
    assert snap["pod_name"] == "podA"


def test_decode_merges_metrics_per_gpu():
    """Same gpu_id, multiple metric kinds -> single snapshot row."""

    response = _response(
        pods=[
            _pod(
                "ns1",
                "podA",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [{"timestamp": 10, "value": 92.0}],
                            )
                        ],
                    ),
                    _result(
                        "rocm_gpu_utilization",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [{"timestamp": 10, "value": 87.5}],
                            )
                        ],
                    ),
                    _result(
                        "rocm_power_average_watts",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [{"timestamp": 10, "value": 250.0}],
                            )
                        ],
                    ),
                ],
            )
        ]
    )
    out = decode_gpu_snapshot(response)
    assert len(out["gpus"]) == 1
    snap = out["gpus"][0]
    assert snap["temperature_c"] == 92.0
    assert snap["util_gpu_pct"] == 87.5
    assert snap["power_watts"] == 250.0


def test_decode_handles_dcgm_metric_names():
    response = _response(
        pods=[
            _pod(
                "ns1",
                "gpu-pod",
                results=[
                    _result(
                        "DCGM_FI_DEV_GPU_TEMP",
                        series=[
                            _series(
                                {"DCGM_FI_DRIVER_DEVICE_ID": "3"},
                                [{"timestamp": 50, "value": 88.0}],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    out = decode_gpu_snapshot(response)
    assert out["gpus"][0]["gpu_id"] == 3
    assert out["gpus"][0]["temperature_c"] == 88.0


def test_decode_keeps_string_id_when_label_is_not_numeric():
    response = _response(
        pods=[
            _pod(
                "ns1",
                "p1",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[
                            _series(
                                {"device": "amdgpu0"},
                                [{"timestamp": 1, "value": 60.0}],
                            )
                        ],
                    )
                ],
            )
        ]
    )
    out = decode_gpu_snapshot(response)
    assert out["gpus"][0]["gpu_id"] == "amdgpu0"


def test_decode_distinguishes_pods_with_overlapping_gpu_ids():
    response = _response(
        pods=[
            _pod(
                "ns1",
                "podA",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [{"timestamp": 1, "value": 80.0}],
                            )
                        ],
                    )
                ],
            ),
            _pod(
                "ns1",
                "podB",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[
                            _series(
                                {"gpu": "0"},
                                [{"timestamp": 1, "value": 95.0}],
                            )
                        ],
                    )
                ],
            ),
        ]
    )
    out = decode_gpu_snapshot(response)
    assert len(out["gpus"]) == 2
    by_pod = {r["pod_name"]: r["temperature_c"] for r in out["gpus"]}
    assert by_pod == {"podA": 80.0, "podB": 95.0}


def test_decode_skips_unknown_metrics():
    response = _response(
        pods=[
            _pod(
                "ns1",
                "podA",
                results=[
                    _result(
                        "vendor_specific_unknown_metric",
                        series=[
                            _series({"gpu": "0"}, [{"timestamp": 1, "value": 1.0}])
                        ],
                    )
                ],
            )
        ]
    )
    assert decode_gpu_snapshot(response) == {}


def test_decode_skips_series_without_values():
    response = _response(
        pods=[
            _pod(
                "ns1",
                "podA",
                results=[
                    _result(
                        "rocm_temperature_celsius",
                        series=[_series({"gpu": "0"}, [])],
                    )
                ],
            )
        ]
    )
    assert decode_gpu_snapshot(response) == {}


def test_merge_combines_multiple_pod_snapshots_in_stable_order():
    snap1 = decode_gpu_snapshot(
        _response(
            pods=[
                _pod(
                    "ns2",
                    "podZ",
                    results=[
                        _result(
                            "rocm_temperature_celsius",
                            series=[
                                _series(
                                    {"gpu": "0"},
                                    [{"timestamp": 1, "value": 80.0}],
                                )
                            ],
                        )
                    ],
                )
            ]
        )
    )
    snap2 = decode_gpu_snapshot(
        _response(
            pods=[
                _pod(
                    "ns1",
                    "podA",
                    results=[
                        _result(
                            "rocm_temperature_celsius",
                            series=[
                                _series(
                                    {"gpu": "0"},
                                    [{"timestamp": 1, "value": 90.0}],
                                )
                            ],
                        )
                    ],
                )
            ]
        )
    )
    merged = merge_gpu_snapshots([snap1, snap2])
    assert merged["tool"] == "robust-api"
    keys = [(g["pod_namespace"], g["pod_name"], g["gpu_id"]) for g in merged["gpus"]]
    # ns1 / podA should sort before ns2 / podZ even though it was decoded second.
    assert keys == [("ns1", "podA", 0), ("ns2", "podZ", 0)]


def test_merge_returns_empty_when_no_snapshots_have_gpus():
    assert merge_gpu_snapshots([]) == {}
    assert merge_gpu_snapshots([{}, {"tool": "x"}, {"gpus": []}]) == {}
