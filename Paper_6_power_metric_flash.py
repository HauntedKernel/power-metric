"""
FlashAttention + Power Metric Combined Efficiency Stack
========================================================
Paper 6: "Combining FlashAttention Memory Bandwidth Optimization
          with Power Metric Compute Allocation"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Computes combined efficiency from two independent optimizations:
  1. FlashAttention (Dao et al. 2022/2023) — memory bandwidth efficiency
  2. Power Metric — adaptive compute allocation (Papers 1 and 2)

Independence argument:
  FlashAttention targets memory bandwidth per attention operation.
  Power metric targets which operations occur (training) or how many
  samples are generated (inference). These address distinct bottlenecks
  and compound multiplicatively.

Formula:
  combined_savings = 1 - (1 - FA_savings) × (1 - PM_savings)

Inputs:
  FA savings:     from Dao et al. (2022, 2023) published benchmarks
                  — real empirical results from peer-reviewed papers
  PM training:    from Paper 1 — hypothesized 21-43% compute reduction
                  at θ=0.7 (conservative anchor: 29% average)
                  NOTE: this is a hypothesis, not an experimentally
                  proven savings figure
  PM inference:   from Paper 2 — 92.7% sampling reduction at θ=0.3
                  (stylized simulation calibrated to Brown et al. 2024)

Related papers:
  Paper 1:  power_metric_training.py
  Paper 2:  power_metric_inference.py
  Paper 14: https://zenodo.org/records/19685841
  Series:   https://github.com/HauntedKernel/power-metric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict

# ── FlashAttention savings (Dao et al. 2022/2023) ────────────────────────
# Source: FlashAttention / FlashAttention-2 paper benchmarks
# End-to-end wall-clock speedup on A100 GPU
FA_SAVINGS = {
    512:   0.25,   # ~25% savings at seq 512
    1024:  0.38,   # ~38% savings at seq 1K
    2048:  0.50,   # ~50% savings at seq 2K
    4096:  0.65,   # ~65% savings at seq 4K  ← headline
    8192:  0.72,   # ~72% savings at seq 8K
    16384: 0.78,   # ~78% savings at seq 16K (FlashAttention-2)
}

# ── Power metric savings ──────────────────────────────────────────────────
# Training (Paper 1): hypothesized 21-43% at θ=0.7 across Pythia models
# Conservative anchor: 29% (average across 160M/1.4B/6.9B at θ=0.7)
# IMPORTANT: this is a signal-analysis hypothesis, not experimentally proven
PM_TRAINING_SAVINGS  = 0.29    # conservative avg, Paper 1 θ=0.7

# Inference (Paper 2): 92.7% sampling reduction at θ=0.3
# Stylized simulation calibrated to Brown et al. GSM8K/Llama-3-8B
PM_INFERENCE_SAVINGS = 0.927   # Paper 2 θ=0.3, 127 problems


def compute_combined(
    fa_savings: float,
    pm_savings: float,
) -> Dict[str, float]:
    """
    Compute combined savings under independence assumption.

    combined = 1 - (1 - FA) × (1 - PM)

    Independence holds because FA and PM target different bottlenecks:
    FA → memory bandwidth per operation
    PM → which/how many operations occur
    """
    combined = 1.0 - (1.0 - fa_savings) * (1.0 - pm_savings)
    multiplier = 1.0 / (1.0 - combined)
    return dict(combined=combined, multiplier=multiplier)


def build_table(pm_training: float = PM_TRAINING_SAVINGS,
                pm_inference: float = PM_INFERENCE_SAVINGS) -> dict:
    """Build full results table across sequence lengths."""
    rows = {}
    for seq, fa in FA_SAVINGS.items():
        train = compute_combined(fa, pm_training)
        infer = compute_combined(fa, pm_inference)
        rows[seq] = dict(
            fa_savings       = fa,
            pm_training      = pm_training,
            pm_inference     = pm_inference,
            train_combined   = train['combined'],
            train_multiplier = train['multiplier'],
            infer_combined   = infer['combined'],
            infer_multiplier = infer['multiplier'],
        )
    return rows


def print_summary(rows: dict):
    print("FlashAttention + Power Metric Combined Stack")
    print(f"PM training:  {PM_TRAINING_SAVINGS*100:.0f}% "
          f"(Paper 1, θ=0.7, hypothesized)")
    print(f"PM inference: {PM_INFERENCE_SAVINGS*100:.1f}% "
          f"(Paper 2, θ=0.3, simulation)")
    print()
    print(f"{'Seq':>8} {'FA':>6} {'PM Train':>10} "
          f"{'Train Comb':>12} {'Train Mult':>12} "
          f"{'Infer Comb':>12} {'Infer Mult':>12}")
    print("-" * 78)
    for seq, r in rows.items():
        print(f"{seq:>8,} {r['fa_savings']*100:>5.0f}% "
              f"{r['pm_training']*100:>9.0f}% "
              f"{r['train_combined']*100:>11.1f}% "
              f"{r['train_multiplier']:>10.1f}x "
              f"{r['infer_combined']*100:>11.1f}% "
              f"{r['infer_multiplier']:>10.1f}x")

    r4k = rows[4096]
    print(f"\nHeadline at seq 4K:")
    print(f"  Training: {r4k['train_combined']*100:.1f}% combined "
          f"({r4k['train_multiplier']:.1f}x)")
    print(f"  Inference: {r4k['infer_combined']*100:.1f}% combined "
          f"({r4k['infer_multiplier']:.1f}x)")
    print(f"\nNOTE: training savings are hypothesized (Paper 1 signal analysis).")
    print(f"      inference savings are from stylized simulation (Paper 2).")
    print(f"      FA savings are from published Dao et al. benchmarks.")


def plot_results(rows: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 9), facecolor='#050810')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    seqs       = list(rows.keys())
    seq_labels = [f'{s//1024}K' for s in seqs]
    fa_s       = [rows[s]['fa_savings']*100          for s in seqs]
    train_s    = [rows[s]['train_combined']*100       for s in seqs]
    infer_s    = [rows[s]['infer_combined']*100       for s in seqs]
    train_m    = [rows[s]['train_multiplier']         for s in seqs]
    infer_m    = [rows[s]['infer_multiplier']         for s in seqs]

    # Chart 1: Combined savings vs seq length
    ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
    ax1.plot(seq_labels, fa_s,    color=CGRAY, lw=1.5, ls='--',
             marker='s', ms=5, label='FA only')
    ax1.plot(seq_labels, train_s, color=CA,    lw=2,
             marker='o', ms=6, label='FA + PM Training (hyp.)')
    ax1.plot(seq_labels, infer_s, color=C,     lw=2,
             marker='o', ms=6, label='FA + PM Inference (sim.)')
    ax1.set_title('Combined Savings vs Sequence Length', color=C, fontsize=11)
    ax1.set_xlabel('Sequence Length', color=CGRAY)
    ax1.set_ylabel('Savings (%)', color=CGRAY)
    ax1.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 2: Multipliers
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    ax2.plot(seq_labels, train_m, color=CA, lw=2, marker='o', ms=6,
             label='Training multiplier')
    ax2.plot(seq_labels, infer_m, color=C,  lw=2, marker='o', ms=6,
             label='Inference multiplier')
    for i, (sl, tm, im) in enumerate(zip(seq_labels, train_m, infer_m)):
        if sl in ['2K', '4K', '16K']:
            ax2.annotate(f'{tm:.1f}x', (sl, tm),
                         textcoords='offset points', xytext=(5, 5),
                         color=CA, fontsize=8)
            ax2.annotate(f'{im:.1f}x', (sl, im),
                         textcoords='offset points', xytext=(5, -12),
                         color=C, fontsize=8)
    ax2.set_title('Combined Multiplier vs Sequence Length', color=C, fontsize=11)
    ax2.set_xlabel('Sequence Length', color=CGRAY)
    ax2.set_ylabel('Multiplier (x)', color=CGRAY)
    ax2.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 3: Per-layer contribution at 4K
    ax3 = fig.add_subplot(gs[1, 0]); style(ax3)
    r4k     = rows[4096]
    methods = ['FA only', 'PM Training\n(hypothesized)', 'PM Inference\n(simulation)',
               'FA+PM Train', 'FA+PM Infer']
    savings = [r4k['fa_savings']*100,
               r4k['pm_training']*100,
               r4k['pm_inference']*100,
               r4k['train_combined']*100,
               r4k['infer_combined']*100]
    colors  = [CGRAY, CA, C, CA, C]
    bars    = ax3.bar(methods, savings, color=colors, alpha=0.85)
    ax3.set_title('Savings at Seq 4K (upper bound)', color=C, fontsize=11)
    ax3.set_ylabel('Savings (%)', color=CGRAY)
    ax3.tick_params(axis='x', labelsize=7)
    for b, v in zip(bars, savings):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{v:.1f}%', ha='center', color=C, fontsize=8,
                 fontweight='bold')

    # Chart 4: Summary
    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(PAN); ax4.axis('off')
    y = 0.96
    ax4.text(0.05, y, 'Stack Summary @ Seq 4K', color=C, fontsize=10,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    stats = [
        ('FA savings (Dao 2023)',    f"{r4k['fa_savings']*100:.0f}%"),
        ('PM training (Paper 1)',    f"{r4k['pm_training']*100:.0f}% (hypothesized)"),
        ('PM inference (Paper 2)',   f"{r4k['pm_inference']*100:.1f}% (simulation)"),
        ('Combined training',        f"{r4k['train_combined']*100:.1f}% / {r4k['train_multiplier']:.1f}x"),
        ('Combined inference',       f"{r4k['infer_combined']*100:.1f}% / {r4k['infer_multiplier']:.1f}x"),
        ('Independence assumed',     'FA ⊥ PM (different bottlenecks)'),
        ('Upper bound',              'Yes — validation required'),
    ]
    for lbl, val in stats:
        ax4.text(0.05, y, lbl+':', color=CGRAY, fontsize=8,
                 transform=ax4.transAxes)
        ax4.text(0.05, y-0.055, val, color=C, fontsize=8,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.115

    fig.suptitle(
        'FlashAttention + Power Metric — Combined Efficiency Stack\n'
        'FA: Dao et al. (2023) empirical · PM: Papers 1 & 2 · '
        'Upper bound under independence assumption',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Training PM savings hypothesized (Paper 1). '
                'Inference PM savings from stylized simulation (Paper 2). '
                'Cantrell (2026) · Paper 6 · github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print("FlashAttention + Power Metric Stack — Paper 6\n")
    rows = build_table()
    print_summary(rows)
    print("\nGenerating charts...")
    plot_results(rows,
                 save_path='/mnt/user-data/outputs/paper6_simulation.png')
    print("Done.")
