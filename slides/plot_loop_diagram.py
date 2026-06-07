# Copyright Advanced Micro Devices, Inc. All rights reserved.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'text.color': '#e6edf3',
    'font.family': 'monospace',
    'font.size': 11,
})

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0.8, 6.8)
ax.axis('off')

box_kw = dict(boxstyle='round,pad=0.4', linewidth=1.8)
txt_kw = dict(ha='center', va='center', fontsize=11, fontweight='bold')

# Phase boxes (top row)
phases = [
    (1.2, 5.8, 'Classify\nModel', '#8b949e'),
    (3.2, 5.8, 'Baseline\nMeasure', '#8b949e'),
    (5.2, 5.8, 'Profile\nGPU', '#8b949e'),
    (7.2, 5.8, 'Build\nStack', '#58a6ff'),
]
for x, y, label, color in phases:
    ax.text(x, y, label, **txt_kw, color='#e6edf3',
            bbox=dict(**box_kw, facecolor='#161b22', edgecolor=color))

# Arrows between phases
for i in range(len(phases) - 1):
    ax.annotate('', xy=(phases[i+1][0]-0.6, phases[i+1][1]),
                xytext=(phases[i][0]+0.6, phases[i][1]),
                arrowprops=dict(arrowstyle='->', color='#484f58', lw=1.5))

# DFS loop boxes — shifted down slightly for more room
loop_boxes = [
    (7.2, 3.6, 'Pop Highest\nScored Action', '#ff6e40'),
    (7.2, 2.0, 'Execute\n+ Benchmark', '#ff6e40'),
    (3.8, 2.0, 'Re-score All\n+ Push New', '#ff6e40'),
    (3.8, 3.6, 'Scores\n< 1.0 ?', '#58a6ff'),
]
for x, y, label, color in loop_boxes:
    ax.text(x, y, label, **txt_kw, color='#e6edf3',
            bbox=dict(**box_kw, facecolor='#161b22', edgecolor=color))

# Arrow: Build Stack → Pop
ax.annotate('', xy=(7.2, 4.05), xytext=(7.2, 5.35),
            arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=2))

# Arrow: Pop → Execute
ax.annotate('', xy=(7.2, 2.45), xytext=(7.2, 3.15),
            arrowprops=dict(arrowstyle='->', color='#ff6e40', lw=2))

# Arrow: Execute → Re-score
ax.annotate('', xy=(4.55, 2.0), xytext=(6.45, 2.0),
            arrowprops=dict(arrowstyle='->', color='#ff6e40', lw=2))

# Arrow: Re-score → Scores check
ax.annotate('', xy=(3.8, 3.15), xytext=(3.8, 2.45),
            arrowprops=dict(arrowstyle='->', color='#ff6e40', lw=2))

# Arrow: Scores check → Pop (loop back, "No")
ax.annotate('', xy=(6.45, 3.6), xytext=(4.55, 3.6),
            arrowprops=dict(arrowstyle='->', color='#ff6e40', lw=2, linestyle='--'))
ax.text(5.5, 3.85, 'No', fontsize=10, color='#ff6e40', ha='center', fontweight='bold')

# Arrow: Scores check → Sweep (exit, "Yes")
ax.annotate('', xy=(1.8, 3.6), xytext=(3.05, 3.6),
            arrowprops=dict(arrowstyle='->', color='#3fb950', lw=2))
ax.text(2.4, 3.85, 'Yes', fontsize=10, color='#3fb950', ha='center', fontweight='bold')

# Sweep + Report box
ax.text(1.2, 3.6, 'Sweep\n+ Report', **txt_kw, color='#e6edf3',
        bbox=dict(**box_kw, facecolor='#161b22', edgecolor='#3fb950'))

# Loop outline — enough clearance below boxes
loop_rect = mpatches.FancyBboxPatch((2.9, 1.15), 5.2, 3.35, boxstyle='round,pad=0.3',
                                     linewidth=1.5, edgecolor='#ff6e40', facecolor='none',
                                     alpha=0.3, linestyle='--')
ax.add_patch(loop_rect)

# "DFS Loop" label — inside rectangle, top-left corner, clear of dashed line
ax.text(3.55, 4.25, 'DFS LOOP', fontsize=10, fontweight='bold', color='#ff6e40',
        ha='left', va='center', alpha=0.5)

fig.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figs', 'optimization_loop.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved to {out}')
