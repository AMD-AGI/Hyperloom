"""M1 demo: a scoped PyTorch model, captured as a HIP or CUDA graph and replayed, with
eager work on a second stream inside every step.

    python -m ruppell run --out RUN -- python examples/torch_graph_demo.py [STEPS]
    python -m ruppell export RUN -o RUN/trace.json

Every kernel should come out attributed: replayed kernels through the graph
registry (scopes recorded at capture by the module hooks), eager kernels
through the scope active when they were launched. Warmup and capture happen
inside steps too, so every kernel lands in exactly one step.

Hardware counters (gray-parrot): with RUPPELL_COUNTERS set, the replay steps
that RUPPELL_COUNTER_STEPS picks (default 1/100; 1/1000 on CUDA) are counter steps, counting
on before the step's begin marker and off after its end marker.
"""

import sys
import time

import torch
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


def main(steps: int) -> None:
    mode = ruppell.init()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.cuda.init()
    main_stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()

    # step 0: setup (weights to the GPU, inputs), so every kernel lands in a step
    step = 0
    ruppell.step_begin(step, phase="setup")
    with ruppell.scope("setup.weights", op="to_device"):
        model = nn.Sequential(Block(1024), Block(1024)).to(dev, torch.float16).eval()
    with ruppell.scope("setup.inputs", op="randn"):
        x = torch.randn(16, 1024, device=dev, dtype=torch.float16)
        w = torch.randn(512, 512, device=dev, dtype=torch.float16)
    ruppell.step_end(step)
    step += 1
    ruppell.instrument(model, prefix="model")

    def side_work():
        side.wait_stream(main_stream)
        with torch.cuda.stream(side):
            with ruppell.scope("side.matmul", op="matmul", shape=[512, 512]):
                z = w @ w
        main_stream.wait_stream(side)
        return z

    with torch.no_grad():
        for _ in range(3):
            ruppell.step_begin(step, phase="warmup")
            model(x)
            side_work()
            ruppell.step_end(step)
            step += 1
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        ruppell.step_begin(step, phase="capture")
        with ruppell.capture("demo.b16", batch=16):
            with torch.cuda.graph(graph):
                model(x)
        ruppell.step_end(step)
        step += 1

        torch.cuda.synchronize()
        counting = mode == "active" and ruppell.counter_steps() is not None
        loop0 = time.perf_counter()
        for _ in range(steps):
            group = ruppell.counters(True, step=step) if counting and ruppell.is_counter_step(step) else None
            fields = {"counters": group} if group is not None and group >= 0 else {}
            ruppell.step_begin(step, phase="replay", **fields)
            graph.replay()
            side_work()
            ruppell.step_end(step)
            if group is not None:
                ruppell.counters(False)
            step += 1
    torch.cuda.synchronize()
    loop_ms = (time.perf_counter() - loop0) * 1e3   # the replay steps alone, for overhead A/B
    print(f"demo: mode={mode} steps={step} torch={torch.__version__} hip={torch.version.hip} cuda={torch.version.cuda} "
          f"loop_ms {loop_ms:.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
