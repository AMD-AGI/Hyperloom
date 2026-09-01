---
myst:
  html_meta:
    "description": "Learn what Hyperloom is: an autonomous agentic system that optimizes LLM inference workloads on AMD GPUs through profiling, kernel optimization, and iterative benchmarking."
    "keywords": "Hyperloom, what is Hyperloom, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, IntelliKit, Magpie, kernel optimization, optimization loop"
---

# What is Hyperloom?

ROCm Hyperloom is an autonomous agentic system designed to optimize end-to-end inference workloads
(targeting both host code and GPU kernels) on AMD GPUs. Using advanced AI agents and profiling tools,
Hyperloom analyzes your workload, identifies performance bottlenecks, implements targeted optimizations,
and validates the performance and correctness of the optimizations without requiring manual intervention.

The system operates through a sophisticated multi-stage pipeline. First TraceLens, the profiling brain of
the workload understanding stage, consumes traces collected by Magpie (which in turn relies on IntelliKit
for some low-level GPU profiling tools), captures bottlenecks, and derives the roofline targets that seed
the optimization search tree.

Next, Hyperloom employs a self-evolving code optimization engine following an iterative agentic loop (Think
→ Decide → Implement → Benchmark). Arbor intelligently explores the optimization space using a Dynamic
Specialist Agent and Knowledge Base. In parallel to Arbor, GEAK, a multi-agent GPU performance optimizer,
optimizes "hot" kernels (kernels that have been identified as good candidates for optimization). Once optimizations are identified and validated, Hyperloom prepares
the optimized code and generates a report with all proposed changes and expected performance improvements.
This end-to-end automation enables developers to achieve significant performance improvements while
maintaining code quality and reducing the manual effort traditionally required for GPU optimization.

Provide your workload, and the agent works toward an optimized configuration: profiling against peak
hardware potential, identifying bottlenecks, and iteratively rewriting code to maximize throughput on
AMD GPUs.

## The optimization loop

The following diagram describes how Hyperloom processes a workload from submission to validated delivery.

```{image} images/Hyperloom_architecture.png
:alt: Hyperloom architecture diagram showing the multi-stage optimization pipeline from workload profiling through kernel optimization to validated delivery.
:class: hl-lightbox-trigger
```

```{raw} html
<style>
  img.hl-lightbox-trigger { cursor: zoom-in; }
  dialog.hl-lightbox-overlay {
    max-width: 90vw; max-height: 90vh; padding: 0; border: none;
    background: transparent; overflow: visible;
  }
  dialog.hl-lightbox-overlay::backdrop { background: rgba(0,0,0,0.75); }
  dialog.hl-lightbox-overlay img { max-width: 90vw; max-height: 90vh; display: block; cursor: zoom-out; }
</style>
<script>
  document.addEventListener('DOMContentLoaded', function() {
    if (window._hlLightboxInit) return;
    window._hlLightboxInit = true;
    document.querySelectorAll('img.hl-lightbox-trigger').forEach(function(img) {
      img.addEventListener('click', function() {
        var dialog = document.createElement('dialog');
        dialog.className = 'hl-lightbox-overlay';
        var clone = new Image();
        clone.src = img.src;
        clone.alt = img.alt;
        dialog.appendChild(clone);
        dialog.addEventListener('click', function() { dialog.close(); dialog.remove(); });
        document.body.appendChild(dialog);
        dialog.showModal();
      });
    });
  });
</script>
```


Hyperloom combines:

- Trace analysis, identifying kernel bottlenecks and bridge planning through
  [TraceLens](https://github.com/AMD-AGI/TraceLens) Agent (backend support
   from [Magpie](https://github.com/AMD-AGI/Magpie) and
   [IntelliKit](https://github.com/AMDResearch/intellikit)).
- Kernel optimization through the
  [GEAK](https://github.com/AMD-AGI/GEAK) backend.
- Agentic search space exploration through
  [Arbor](https://arxiv.org/abs/2606.12563), a tree-based cognition layer
  with dynamic agents, long-horizon campaigns, and self-evolving optimization
  guided by a curated knowledge base of hardware learnings, pitfalls, and
  prior campaign artifacts.
