"""M7 demo: the M1 demo's model with NVTX (ROCTX on ROCm) annotations, for ruppell's ranges.

    RUPPELL_NVTX=1 RUPPELL_CPU_OPS=1 python -m ruppell run --out RUN -- python examples/nvtx_demo.py [STEPS] [--no-scopes]

PyTorch's emit_nvtx() puts a range around every aten op during warmup and while the graph is
captured; the replays then run as one graph launch, whose kernels take the ranges open at
capture. Around those, explicit ranges: "demo.step" per step, "demo.side" around the side
stream's work, a start/end range "demo.loop" around the replay loop, and a mark per replay.
With --no-scopes ruppell's module scopes are not installed, so the ranges are all there is.
"""

import sys
import time

import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn

import ruppell


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


def main(steps: int, scopes: bool) -> None:
    mode = ruppell.init()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.cuda.init()
    main_stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()

    step = 0
    ruppell.step_begin(step, phase="setup")
    model = nn.Sequential(Block(1024), Block(1024)).to(dev, torch.float16).eval()
    x = torch.randn(16, 1024, device=dev, dtype=torch.float16)
    w = torch.randn(512, 512, device=dev, dtype=torch.float16)
    ruppell.step_end(step)
    step += 1
    if scopes:
        ruppell.instrument(model, prefix="model")

    def side_work():
        nvtx.range_push("demo.side")
        side.wait_stream(main_stream)
        with torch.cuda.stream(side):
            z = w @ w
        main_stream.wait_stream(side)
        nvtx.range_pop()
        return z

    with torch.no_grad(), torch.autograd.profiler.emit_nvtx():
        for _ in range(3):
            ruppell.step_begin(step, phase="warmup")
            nvtx.range_push("demo.step")
            model(x)
            side_work()
            nvtx.range_pop()
            ruppell.step_end(step)
            step += 1
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        ruppell.step_begin(step, phase="capture")
        nvtx.range_push("demo.capture")
        with ruppell.capture("demo.b16", batch=16):
            with torch.cuda.graph(graph):
                model(x)
        nvtx.range_pop()
        ruppell.step_end(step)
        step += 1

    torch.cuda.synchronize()
    loop0 = time.perf_counter()
    loop = nvtx.range_start("demo.loop")
    with torch.no_grad():
        for _ in range(steps):
            ruppell.step_begin(step, phase="replay")
            nvtx.range_push("demo.step")
            nvtx.mark("demo.replay")
            graph.replay()
            side_work()
            nvtx.range_pop()
            ruppell.step_end(step)
            step += 1
    torch.cuda.synchronize()
    nvtx.range_end(loop)
    loop_ms = (time.perf_counter() - loop0) * 1e3   # the replay steps alone, for overhead A/B
    print(f"nvtx demo: mode={mode} scopes={scopes} steps={step} torch={torch.__version__} hip={torch.version.hip} "
          f"cuda={torch.version.cuda} loop_ms {loop_ms:.3f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 200, "--no-scopes" not in sys.argv)
