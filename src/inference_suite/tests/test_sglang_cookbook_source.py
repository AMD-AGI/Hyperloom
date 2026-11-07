# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sglang cookbook as a recipe source, against a snapshot of sglang main.

``c843608d39.json.gz`` holds, at sglang c843608d39, the deployment configs of GLM-5.3 and Kimi-K3.1-Pro and the
code-built one of Muse Spark, the AgentX launches of Kimi-K3.1-Pro on MI355X and of GLM-5.3 on B300, and the
GLM-5.3 page with its MI355X AgentX launch blocks.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from functools import cache
from pathlib import Path

from inference_suite.argv import argv_flag_occurrences
from inference_suite.contracts import LaunchQuery, SourceKind, Topology, Workload
from inference_suite.recipe.model import RepoBundle, SourceLaunch, SourceMiss
from inference_suite.recipe.sglang_cookbook import SGLANG_REPO, sglang_cookbook_launches

_FIXTURE = Path(__file__).parent / "fixtures" / "sglang" / "c843608d39.json.gz"
_GLM53_CONFIG = "docs/src/snippets/configs/zai-org/glm-5.3.jsx"
_KIMI_CONFIG = "docs/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx"
_KIMI_AGENTX = "docs/src/snippets/agentx/moonshotai/kimi-k3.1-pro-mi355x.jsx"
_GLM53_PAGE = "docs/cookbook/autoregressive/GLM/GLM-5.3.mdx"
_KIMI = "moonshotai/Kimi-K3.1-Pro"
_GLM53_MXFP4 = "amd/GLM-5.3-Quark-MXFP4-AttnFP8"


@cache
def _main() -> RepoBundle:
    return RepoBundle.from_document(json.loads(gzip.decompress(_FIXTURE.read_bytes())))


def _fixed(model: str, *, board: str = "mi355x", concurrencies=(64,), **topology) -> LaunchQuery:
    return LaunchQuery(model, "sglang", board, Workload.FIXED, tuple(concurrencies), isl=1024, osl=1024, **topology)


def _agentx(model: str, *, concurrencies, board: str = "mi355x", **topology) -> LaunchQuery:
    return LaunchQuery(model, "sglang", board, Workload.AGENTX, tuple(concurrencies), **topology)


def _launches(query: LaunchQuery, bundle: RepoBundle | None = None) -> tuple[SourceLaunch, ...]:
    found = sglang_cookbook_launches(query, bundle or _main())
    assert not isinstance(found, SourceMiss), found
    return found


def _miss(query: LaunchQuery, bundle: RepoBundle | None = None) -> str:
    found = sglang_cookbook_launches(query, bundle or _main())
    assert isinstance(found, SourceMiss), found
    assert found.kind is SourceKind.SGLANG_COOKBOOK
    return found.reason


def _bundle(files: dict[str, str]) -> RepoBundle:
    return RepoBundle(SGLANG_REPO, "c843608d390ac416791e49e0252d2f17fafdc489", files)


def test_fixed_glm53_takes_the_balanced_cell_with_its_image():
    (launch,) = _launches(_fixed("zai-org/GLM-5.3"))

    assert launch.source.kind is SourceKind.SGLANG_COOKBOOK
    assert (launch.source.repo, launch.source.commit) == (SGLANG_REPO, "c843608d390ac416791e49e0252d2f17fafdc489")
    assert (launch.source.path, launch.source.locator, launch.source.strategy) == (
        _GLM53_CONFIG,
        "cells[28]",
        "balanced",
    )
    assert launch.source.files == {_GLM53_CONFIG: hashlib.sha256(_main().text(_GLM53_CONFIG).encode()).hexdigest()}
    assert launch.args == (
        *("--tp", "8", "--dsa-prefill-backend", "tilelang", "--dsa-decode-backend", "tilelang"),
        *("--chunked-prefill-size", "32768", "--mem-fraction-static", "0.85", "--cuda-graph-max-bs-decode", "128"),
        *("--max-running-requests", "80", "--watchdog-timeout", "1200"),
    )
    assert launch.stripped == ("--model-path", "--host", "--port")
    assert launch.image == "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260920"
    assert (launch.framework, launch.entrypoint) == ("sglang", ("python3", "-m", "sglang.launch_server"))
    assert (launch.topology, launch.conc, launch.exact, launch.env, launch.golden) == (Topology(8), 64, True, {}, None)
    assert any("overlay options ['pdMode', 'parsers']" in warning for warning in launch.warnings)


def test_a_like_board_takes_the_cell_tuned_on_its_twin_and_says_so():
    (launch,) = _launches(_fixed("zai-org/GLM-5.3", board="mi350x"))

    assert launch.source.locator == "cells[28]"
    assert any("on mi355x, not mi350x" in warning for warning in launch.warnings)
    assert _launches(_fixed("zai-org/GLM-5.3", board="mi325x"))[0].image.endswith("mi30x-20260920")


def test_without_a_balanced_cell_the_first_listed_strategy_serves_and_the_exact_board_wins():
    (launch,) = _launches(_fixed(_KIMI, concurrencies=(4,)))

    assert (launch.source.path, launch.source.locator, launch.source.strategy) == (
        _KIMI_CONFIG,
        "cells[21]",
        "low-latency",
    )
    assert any("no balanced cell" in warning for warning in launch.warnings)
    flag = launch.args.index("--cuda-graph-bs-prefill")
    assert launch.args[flag + 1 : flag + 9] == ("256", "512", "768", "1024", "1536", "2048", "3072", "4096")
    assert launch.env["SGLANG_ROCM_RUNTIME_RESERVE_GB"] == "6"
    assert launch.image == "lmsysorg/sglang:dev-kimi-k31-pro-mi35x"


def test_a_strategy_capping_running_requests_below_the_concurrency_yields_to_one_that_admits_it():
    (launch,) = _launches(_fixed(_KIMI, concurrencies=(48,)))
    assert (launch.source.locator, launch.source.strategy, launch.exact) == ("cells[22]", "high-throughput", True)

    (launch,) = _launches(_fixed("zai-org/GLM-5.3", concurrencies=(256,)))
    assert (launch.source.locator, launch.source.strategy, launch.exact) == ("cells[29]", "high-throughput", True)


def test_when_every_cell_caps_running_requests_below_the_concurrency_the_nearest_serves_inexactly():
    (launch,) = _launches(_fixed(_KIMI, concurrencies=(64,)))

    assert (launch.source.locator, launch.source.strategy, launch.conc, launch.exact) == (
        "cells[22]",
        "high-throughput",
        64,
        False,
    )
    assert any("cells[22] runs at most 48 requests at once, below 64" in warning for warning in launch.warnings)


def test_above_every_cap_a_strategy_that_sets_none_does_not_win_for_stating_none():
    for conc in (300, 512):
        (launch,) = _launches(_fixed("zai-org/GLM-5.3", concurrencies=(conc,)))

        assert (launch.source.locator, launch.source.strategy, launch.conc, launch.exact) == (
            "cells[29]",
            "high-throughput",
            conc,
            False,
        )
        assert any(f"cells[29] runs at most 256 requests at once, below {conc}" in w for w in launch.warnings)


def test_cells_built_by_code_are_refused_not_evaluated():
    reason = _miss(_fixed("amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"))

    assert "meta-models/muse-spark.jsx: its cells are computed by code" in reason


def test_a_cell_whose_flags_are_code_is_refused_and_a_speculative_cell_skipped_for_fixed():
    config = """export const config = {
      modelNames: { default: "org/model" },
      dockerImages: { mi355x: "lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x" },
      cells: [
        { match: { hw: "mi355x", strategy: "balanced" }, env: [], flags: FLAGS },
        { match: { hw: "mi355x", strategy: "low-latency" }, env: [],
          flags: ["--model-path {{MODEL_NAME}}", "--tp 8", "--speculative-algorithm EAGLE"] },
      ],
    };"""
    reason = _miss(_fixed("org/model"), _bundle({"docs/src/snippets/configs/org/model.jsx": config}))

    assert "cells[0]: its match, flags or env is computed by code" in reason
    assert "cells[1]: it drafts tokens, and a fixed workload runs no speculative decoding" in reason


def test_agentx_kimi_applies_the_point_patch_and_reads_a_placeholder_image_as_none():
    (launch,) = _launches(_agentx(_KIMI, concurrencies=(32,)))

    assert (launch.source.path, launch.source.locator) == (
        _KIMI_AGENTX,
        "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8#0 points[1]",
    )
    assert launch.source.entry == "kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8"
    assert (launch.conc, launch.exact, launch.kv_offloading, launch.topology) == (32, True, "hicache", Topology(8))
    occurrences = dict(argv_flag_occurrences(launch.args))
    assert occurrences["max-running-requests"] == occurrences["cuda-graph-max-bs-decode"] == "32"
    assert occurrences["speculative-algorithm"] == "DFLASH"
    assert launch.stripped == ("--model-path", "--served-model-name", "--host", "--port", "--enable-metrics")
    assert launch.image is None
    assert any("placeholder image <K31_MI35X_IMAGE>" in warning for warning in launch.warnings)
    assert launch.env["SGLANG_HICACHE_MAMBA_HOST_RATIO"] == "12"


def test_agentx_takes_each_cells_nearest_larger_point():
    launches = {launch.source.entry: launch for launch in _launches(_agentx(_KIMI, concurrencies=(4, 8)))}

    mono = launches["kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8-mono"]
    ladder = launches["kimik3.1-fp4-mi355x-sglang-agentic-dflash-agg-dcp8"]
    assert (mono.conc, mono.exact) == (8, True)
    assert (ladder.conc, ladder.exact) == (16, False)
    assert any("no point at concurrency 8; took 16" in warning for warning in ladder.warnings)


def test_agentx_refuses_a_disaggregated_launch_and_an_operator_topology_it_lacks():
    assert "not one aggregated single-node worker" in _miss(_agentx(_KIMI, concurrencies=(92,)))
    assert "it runs tp 8, not 4" in _miss(_agentx(_KIMI, concurrencies=(8,), tp=4))


def test_agentx_launches_of_another_board_do_not_serve():
    assert "no AgentX launch serves it here" in _miss(_agentx("nvidia/GLM-5.3-NVFP4", concurrencies=(4,)))


def test_agentx_page_blocks_give_env_topology_image_and_the_unapplied_scaling_rule():
    (launch,) = _launches(_agentx(_GLM53_MXFP4, concurrencies=(8,)))

    assert (launch.source.path, launch.source.locator) == (_GLM53_PAGE, "line 142 'TP4·EP4 (concurrency 4–16)'")
    assert launch.env == {"SGLANG_OPT_USE_TOPK_V2": "true", "SGLANG_TIMEOUT_KEEP_ALIVE": "900"}
    assert (launch.topology, launch.conc, launch.exact, launch.kv_offloading) == (Topology(4, 4), 8, True, "hicache")
    assert launch.image == "lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261025"
    assert launch.args[:4] == ("--tp-size", "4", "--ep-size", "4")
    assert "--speculative-algorithm" in launch.args
    assert any(
        "adjusts --cuda-graph-max-bs-decode, --hicache-size, --max-running-requests in its text" in warning
        for warning in launch.warnings
    )


def test_page_blocks_cover_their_title_range_and_only_their_workload():
    by_tp = {launch.topology.tp: launch for launch in _launches(_agentx(_GLM53_MXFP4, concurrencies=(2,)))}

    assert (by_tp[8].conc, by_tp[8].exact) == (2, True)
    assert (by_tp[4].conc, by_tp[4].exact) == (4, False)
    assert "tuned for concurrency 4-16, below 20" in _miss(_agentx(_GLM53_MXFP4, concurrencies=(20,)))
    assert "no page launch block serves it here" in _miss(_fixed(_GLM53_MXFP4))


def test_a_page_block_for_another_board_is_skipped_not_given_as_the_reason():
    for board in ("mi300x", "mi325x"):
        assert _miss(_agentx(_GLM53_MXFP4, concurrencies=(16,), board=board)) == (
            "no AgentX launch serves it here; no page launch block serves it here"
        )


def test_a_page_block_using_shell_variables_is_refused():
    page = """## Agentic deployment

### MI355X launch

```bash Serve (concurrency 1–8)
python -m sglang.launch_server --model-path org/model --tp-size $TP
```
"""
    reason = _miss(_agentx("org/model", concurrencies=(4,)), _bundle({"docs/cookbook/org/Model.mdx": page}))

    assert "Model.mdx:5: it uses shell syntax beyond one literal command" in reason


def test_only_verified_agentx_launches_serve():
    data = json.loads(gzip.decompress(_FIXTURE.read_bytes()))["files"][_KIMI_AGENTX]
    unverified = data.replace('"status": "verified"', '"status": "in-progress"')

    reason = _miss(_agentx(_KIMI, concurrencies=(8,)), _bundle({_KIMI_AGENTX: unverified}))
    assert "its provenance is 'in-progress', not verified" in reason


def test_the_cookbook_serves_sglang_only():
    query = LaunchQuery("zai-org/GLM-5.3", "vllm", "mi355x", Workload.FIXED, (8,), isl=1024, osl=1024)

    assert _miss(query) == "the sglang cookbook launches sglang, not vllm"
