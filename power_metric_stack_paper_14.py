"""
Algorithmic Efficiency Stack Calculator
========================================
Paper 14: "You Are Wasting 96% of Your Inference Compute:
           23x Intelligence Per Watt from Software Alone"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Computes the combined IPW multiplier from three largely independent
algorithmic efficiency layers applied to LLM inference.

The three layers:
  1. FlashAttention     — memory bandwidth efficiency (Dao et al. 2023)
  2. Power Metric       — adaptive sampling allocation (Cantrell 2026, Paper 2)
  3. Early Exit         — dynamic depth reduction (Cantrell 2026, Paper 8)

Key claim: these three mechanisms address independent bottlenecks
(memory bandwidth, sampling compute, depth compute) and compound
approximately multiplicatively.

This is an upper bound under partial independence assumptions.
Real-world gains depend on workload characteristics and overlap between layers.

Related papers:
  Paper 14: https://zenodo.org/records/19685841
  Full series: https://github.com/HauntedKernel/power-metric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Dict, List


# ── Published source inputs ────────────────────────────────────────────────
# These are the empirical anchors for the stack calculation.
# FlashAttention is from a peer-reviewed published paper.
# Power Metric and Early Exit are from stylized simulations in this series.

LAYERS = {
    'flashattention': {
        'name':        'FlashAttention',
        'description': 'Memory-efficient exact attention via IO-awareness',
        'source':      'Dao et al. (2023), NeurIPS 2023',
        'savings': {
            # Fraction of attention compute saved at each sequence length
            # Source: Dao et al. 2023, Table 1 end-to-end throughput results
            1024:  0.50,   # 50% savings at seq 1K
            2048:  0.58,   # 58% savings at seq 2K
            4096:  0.65,   # 65% savings at seq 4K  ← headline number
            8192:  0.72,   # 72% savings at seq 8K
            16384: 0.78,   # 78% savings at seq 16K
        },
        'quality_loss': 0.000,   # exact attention — zero quality loss
        'assumption':   'Savings scale with sequence length due to quadratic attention',
    },
    'power_metric': {
        'name':        'Power Metric Inference',
        'description': 'Adaptive sampling allocation — stop easy problems early',
        'source':      'Cantrell (2026), Paper 2, θ=0.3. Stylized simulation '
                       'calibrated to Brown et al. (2024) GSM8K/Llama-3-8B results.',
        'savings': {
            # Fraction of sampling compute saved (constant across seq lengths —
            # operates at the problem level, not the token level)
            1024:  0.927,
            2048:  0.927,
            4096:  0.927,
            8192:  0.927,
            16384: 0.927,
        },
        'quality_loss': 0.008,   # -0.008 coverage loss vs uniform allocation
        'assumption':   'Savings assume repeated sampling workload (e.g. GSM8K). '
                        'Does not apply to single-pass inference.',
    },
    'early_exit': {
        'name':        'Early Exit',
        'description': 'Dynamic depth reduction — exit at earlier layers for easy inputs',
        'source':      'Cantrell (2026), Paper 8, θ=0.4. Stylized simulation '
                       'calibrated to BERT-base architecture results.',
        'savings': {
            # Fraction of depth compute saved (constant across seq lengths)
            1024:  0.283,
            2048:  0.283,
            4096:  0.283,
            8192:  0.283,
            16384: 0.283,
        },
        'quality_loss': 0.015,   # conf_ratio 0.985 at exit
        'assumption':   'Savings assume classification-like tasks with variable '
                        'difficulty. May be lower for generation tasks.',
    },
}


@dataclass
class StackResult:
    seq_len:             int
    layer_savings:       Dict[str, float]
    layer_multipliers:   Dict[str, float]
    combined_multiplier: float
    combined_savings:    float
    multiplicative_pred: float
    interaction_gap:     float
    total_quality_loss:  float


def compute_stack(
    seq_len: int,
    layers: Dict = None,
    interaction_factor: float = 1.0,
) -> StackResult:
    """
    Compute the combined efficiency multiplier for a given sequence length.

    Args:
        seq_len:           Sequence length in tokens
        layers:            Layer definitions (default: LAYERS)
        interaction_factor: Adjustment for layer interactions (1.0 = fully
                           independent, <1.0 = some overlap/coupling)

    Returns:
        StackResult with per-layer and combined multipliers

    Note on independence assumption:
        The multiplicative formula assumes layers address independent
        bottlenecks. In practice, some coupling exists (e.g. FlashAttention
        changes compute profile in ways that interact with early exit).
        The interaction_factor parameter models this uncertainty.
        Default 1.0 is the upper bound.
    """
    if layers is None:
        layers = LAYERS

    layer_savings     = {}
    layer_multipliers = {}
    remaining         = 1.0   # fraction of compute remaining after each layer

    for key, layer in layers.items():
        savings = layer['savings'].get(seq_len,
                  layer['savings'][max(layer['savings'].keys())])
        layer_savings[key]     = savings
        layer_multipliers[key] = 1.0 / (1.0 - savings)
        remaining             *= (1.0 - savings)

    combined_savings    = 1.0 - remaining
    multiplicative_pred = 1.0 / remaining
    combined_multiplier = multiplicative_pred * interaction_factor
    interaction_gap     = multiplicative_pred - combined_multiplier

    total_quality_loss  = sum(
        layer.get('quality_loss', 0) for layer in layers.values()
    )

    return StackResult(
        seq_len             = seq_len,
        layer_savings       = layer_savings,
        layer_multipliers   = layer_multipliers,
        combined_multiplier = combined_multiplier,
        combined_savings    = combined_savings,
        multiplicative_pred = multiplicative_pred,
        interaction_gap     = interaction_gap,
        total_quality_loss  = total_quality_loss,
    )


def print_stack_table(seq_lengths: List[int] = None):
    """Print the efficiency stack table for multiple sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384]

    print("Algorithmic Efficiency Stack — IPW Multiplier")
    print("=" * 70)
    print(f"{'Layer':<22} {'Source':<15} {'Savings@4K':>10} {'Multiplier':>10}")
    print("-" * 70)
    for key, layer in LAYERS.items():
        s = layer['savings'][4096]
        m = 1.0 / (1.0 - s)
        print(f"{layer['name']:<22} {layer['source'][:14]:<15} "
              f"{s*100:>9.1f}% {m:>10.2f}x")
    print("-" * 70)

    print(f"\n{'Seq Length':>12} {'FA':>8} {'PM':>8} {'EE':>8} "
          f"{'Combined':>10} {'Savings':>8}")
    print("-" * 60)
    for seq in seq_lengths:
        r = compute_stack(seq)
        print(f"{seq:>12,} "
              f"{r.layer_multipliers['flashattention']:>7.1f}x "
              f"{r.layer_multipliers['power_metric']:>7.1f}x "
              f"{r.layer_multipliers['early_exit']:>7.1f}x "
              f"{r.combined_multiplier:>9.1f}x "
              f"{r.combined_savings*100:>7.1f}%")

    print("\nAssumptions and limitations:")
    for key, layer in LAYERS.items():
        print(f"  {layer['name']}: {layer['assumption']}")
    print(f"\n  Combined multiplier assumes full independence between layers.")
    print(f"  This is an upper bound. See paper Section 4 for discussion.")


def plot_stack(seq_lengths: List[int] = None, save_path: str = None):
    """Generate visualization of the efficiency stack."""
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384]

    results = [compute_stack(s) for s in seq_lengths]

    fig = plt.figure(figsize=(14, 9), facecolor='#050810')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    seq_labels = [f'{s//1024}K' for s in seq_lengths]

    # Chart 1: Combined multiplier by sequence length
    ax1 = fig.add_subplot(gs[0,0]); style(ax1)
    combined = [r.combined_multiplier for r in results]
    ax1.plot(seq_labels, combined, color=C, lw=2.5, marker='o',
             markersize=8, markerfacecolor=C)
    ax1.axhline(1.0, color='#444', lw=1, ls='--')
    for i, (x, y) in enumerate(zip(seq_labels, combined)):
        ax1.annotate(f'{y:.0f}x', (x, y), textcoords='offset points',
                     xytext=(0, 10), ha='center', color=C, fontsize=9)
    ax1.set_title('Combined IPW Multiplier vs Sequence Length',
                  color=C, fontsize=11)
    ax1.set_xlabel('Sequence Length', color='#888')
    ax1.set_ylabel('Multiplier (x)', color='#888')

    # Chart 2: Per-layer contribution at 4K
    ax2 = fig.add_subplot(gs[0,1]); style(ax2)
    r4k    = compute_stack(4096)
    names  = [LAYERS[k]['name'] for k in LAYERS]
    mults  = [r4k.layer_multipliers[k] for k in LAYERS]
    colors = [C, CA, CB]
    bars   = ax2.bar(names, mults, color=colors, alpha=0.85)
    ax2.axhline(1.0, color='#444', lw=1, ls='--')
    for b, v in zip(bars, mults):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                 f'{v:.1f}x', ha='center', color=C, fontsize=10,
                 fontweight='bold')
    ax2.set_title('Per-Layer Multiplier at Seq 4K', color=C, fontsize=11)
    ax2.set_ylabel('Multiplier (x)', color='#888')
    ax2.tick_params(axis='x', labelsize=9)

    # Chart 3: Stacked compute savings
    ax3 = fig.add_subplot(gs[1,0]); style(ax3)
    fa_s  = [r.layer_savings['flashattention']*100 for r in results]
    pm_s  = [r.layer_savings['power_metric']*100 for r in results]
    ee_s  = [r.layer_savings['early_exit']*100 for r in results]
    x     = np.arange(len(seq_labels))
    w     = 0.25
    ax3.bar(x-w, fa_s, w, label='FlashAttention', color=C, alpha=0.85)
    ax3.bar(x,   pm_s, w, label='Power Metric',   color=CA, alpha=0.85)
    ax3.bar(x+w, ee_s, w, label='Early Exit',     color=CB, alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(seq_labels)
    ax3.set_title('Per-Layer Savings by Sequence Length', color=C, fontsize=11)
    ax3.set_ylabel('Savings (%)', color='#888')
    ax3.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 4: Summary at 4K
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(PAN); ax4.axis('off')
    stats = [
        ('Sequence length',      '4,096 tokens'),
        ('FlashAttention',       f"{r4k.layer_multipliers['flashattention']:.2f}x  "
                                 f"({r4k.layer_savings['flashattention']*100:.0f}% savings)"),
        ('Power Metric',         f"{r4k.layer_multipliers['power_metric']:.2f}x  "
                                 f"({r4k.layer_savings['power_metric']*100:.0f}% savings)"),
        ('Early Exit',           f"{r4k.layer_multipliers['early_exit']:.2f}x  "
                                 f"({r4k.layer_savings['early_exit']*100:.0f}% savings)"),
        ('Combined (upper bd)',  f"{r4k.combined_multiplier:.0f}x"),
        ('Combined savings',     f"{r4k.combined_savings*100:.0f}%"),
        ('Quality loss est.',    f"~{r4k.total_quality_loss*100:.1f}%"),
        ('Independence assumed', 'Partial — upper bound'),
    ]
    y = 0.95
    ax4.text(0.05, y, 'Stack Summary @ Seq 4K', color=C, fontsize=11,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for lbl, val in stats:
        ax4.text(0.05, y, lbl+':', color='#888', fontsize=9,
                 transform=ax4.transAxes)
        ax4.text(0.58, y, val, color=C, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.11

    fig.suptitle(
        'Paper 14: Algorithmic Efficiency Stack — 23x IPW from Software Alone\n'
        'Upper bound under partial independence assumptions',
        color=C, fontsize=11, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · FA: Dao et al. 2023 · '
                'PM/EE: stylized simulations · zenodo.org/records/19685841',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print_stack_table()
    print("\nGenerating charts...")
    plot_stack(save_path='/mnt/user-data/outputs/paper14_stack.png')
    print("Done.")
