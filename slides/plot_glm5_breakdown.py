"""Render the GLM-5 FP8 per-GPU throughput slide figure.

Builds a bar chart comparing MI355X baseline, NVIDIA B200, and Hyperloom
MI355X tokens/sec per GPU, then saves it as
``docs/figs/glm5_optimization_breakdown.png`` for use in the slide deck. Uses
the non-interactive ``Agg`` backend so it can run headless.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#8b949e',
    'xtick.color': '#8b949e',
    'ytick.color': '#484f58',
    'grid.color': '#21262d',
    'text.color': '#e6edf3',
    'font.family': 'monospace',
    'font.size': 11,
})

fig, ax = plt.subplots(figsize=(8, 5.5))

categories = ['MI355X Baseline', 'NVIDIA B200', 'Hyperloom MI355X']
values = [174, 448, 510]
colors = ['#8b949e', '#58a6ff', '#ff6e40']

x = np.arange(len(categories))
bars = ax.bar(x, values, color=colors, width=0.45, zorder=3,
              edgecolor='#0d1117', linewidth=1.5)

for bar, val, c in zip(bars, values, colors):
    ax.text(bar.get_x() + bar.get_width()/2, val + 12, str(val),
            ha='center', va='bottom', fontsize=18, fontweight='bold', color=c)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11.5, color='#c9d1d9')
ax.set_ylabel('tok/s / GPU', fontsize=13, fontweight='bold', color='#8b949e', labelpad=10)
ax.set_ylim(0, 600)
ax.set_xlim(-0.55, 2.55)
ax.grid(axis='y', alpha=0.4, zorder=0)
ax.xaxis.grid(False)
ax.tick_params(axis='y', labelsize=10)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.set_title('GLM-5 FP8 — Per-GPU Throughput\n', fontsize=15, fontweight='bold',
             color='#ff6e40', pad=28)
ax.text(0.5, 1.08, '2.93x baseline  ·  1.14x B200',
        transform=ax.transAxes, ha='center', fontsize=13, fontweight='bold', color='#c9d1d9')
ax.text(0.5, 1.01, 'CONC=64  ·  ISL/OSL=1024',
        transform=ax.transAxes, ha='center', fontsize=9, color='#484f58')

fig.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figs', 'glm5_optimization_breakdown.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved to {out}')
