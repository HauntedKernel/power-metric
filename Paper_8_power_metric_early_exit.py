"""
Early Exit via Stochastic Power Metrics
=========================================
Paper 8: "Early Exit as LIF Threshold Firing: Adaptive Depth
          Allocation via Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Implements P(t) = E(t) × W(t) as the exit criterion for early exit
transformers, applied at the layer level.

The problem:
  Transformers run every input through all N layers regardless of difficulty.
  Easy inputs reach high confidence in early layers — continuing wastes compute.

The solution:
  Monitor per-layer confidence improvement using P(t). Exit when P(t) >= threshold
  for PATIENCE consecutive layers AND confidence >= MIN_CONF.

Accuracy metric:
  conf_ratio = confidence_at_exit / confidence_at_final_layer
  Measures how much of the final-layer confidence was achieved at exit.
  This is a proxy — it does NOT measure ground truth accuracy.
  Confidence calibration varies in real models; high confidence ≠ correct.

Signal: per-layer confidence (max softmax probability)
  Note: real transformer confidence trajectories are noisier and non-monotonic.
  This simulation assumes smoothed sigmoid convergence — an idealization.

E(t) = confidence(layer) / E[R](t-1)  [pre-update, no information leak]
W(t) = EWMA of [E(t) > 1.0], span=3
P(t) = exp(-λ)·P(t-1) + (1-exp(-λ))·[E(t)×W(t)]
Exit when: P(t) >= threshold AND conf >= MIN_CONF for PATIENCE=2 consecutive layers

Baselines:
  Confidence threshold: exit when conf >= 0.85
    — proxy for entropy thresholding, not an exact implementation
  Stability heuristic: exit when discretized prediction stable for p=3 layers
    — coarse approximation of BERT Loses Patience (Zhou et al. 2020),
      not a faithful reimplementation

Calibration:
  BERT-base (12 layers, 600 inputs, 4 difficulty tiers).
  Easy inputs start with high confidence (0.85+) from early layers.
  Hard inputs build confidence slowly, peak at layer 10-12.

Related papers:
  Paper 14: https://zenodo.org/records/19685841
  Series:   https://github.com/HauntedKernel/power-metric
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3

# ── Architecture (BERT-base calibration) ──────────────────────────────────
N_LAYERS  = 12
N_INPUTS  = 600    # 150 easy + 200 medium + 200 hard + 50 very hard
PATIENCE  = 2      # consecutive stable layers before exit
MIN_CONF  = 0.70   # minimum confidence required to exit

# ── Difficulty tiers ───────────────────────────────────────────────────────
# Easy: high confidence from early layers (realistic for simple inputs)
# Hard: confidence builds slowly, never peaks very high
TIERS = {
    'easy':      {'n': 150, 'start_conf': 0.85, 'final_conf': 0.97, 'rise_at': 2,  'noise': 0.03},
    'medium':    {'n': 200, 'start_conf': 0.45, 'final_conf': 0.90, 'rise_at': 6,  'noise': 0.05},
    'hard':      {'n': 200, 'start_conf': 0.30, 'final_conf': 0.83, 'rise_at': 10, 'noise': 0.05},
    'very_hard': {'n':  50, 'start_conf': 0.25, 'final_conf': 0.76, 'rise_at': 12, 'noise': 0.05},
}


def generate_confidence_curve(tier: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate per-layer confidence trajectory.

    Easy inputs start confident and stay confident.
    Hard inputs build slowly. All curves are stylized sigmoids —
    real transformers exhibit noisier, non-monotonic patterns.
    """
    t      = TIERS[tier]
    layers = np.arange(1, N_LAYERS + 1)
    span   = t['final_conf'] - t['start_conf']
    curve  = t['start_conf'] + span / (1 + np.exp(-1.5 * (layers - t['rise_at'])))
    noisy  = curve + rng.normal(0, t['noise'], N_LAYERS)
    return np.clip(noisy, 0.05, 0.999)


def power_metric_exit(curve: np.ndarray, threshold: float = 0.5) -> Tuple[int, float]:
    """
    Exit when P(t) >= threshold AND conf >= MIN_CONF for PATIENCE layers.

    The MIN_CONF gate ensures we don't exit with low-confidence predictions
    even when P(t) is technically stable.

    Returns: (exit_layer, exit_confidence)
    """
    expected_r = None; ewma_win = 0.0; power = 0.0
    consecutive_high = 0

    for i, conf in enumerate(curve):
        if expected_r is None:
            eff = 1.0; expected_r = max(conf, 1e-6)
        else:
            eff = conf / expected_r
            expected_r = (1 - ALPHA) * expected_r + ALPHA * conf

        win      = 1.0 if eff > 1.0 else 0.0
        a        = 2.0 / (EWMA_SPAN + 1)
        ewma_win = a * win + (1 - a) * ewma_win
        inst     = eff * ewma_win
        power    = np.exp(-LAMBDA) * power + (1 - np.exp(-LAMBDA)) * inst

        # Both conditions must hold: P(t) stable AND confidence sufficient
        if power >= threshold and conf >= MIN_CONF:
            consecutive_high += 1
        else:
            consecutive_high = 0

        if consecutive_high >= PATIENCE and i >= 2:
            return i + 1, conf

    return N_LAYERS, curve[-1]


def confidence_threshold_exit(curve: np.ndarray,
                               threshold: float = 0.85) -> Tuple[int, float]:
    """
    Baseline: exit when conf >= threshold.
    Proxy for entropy thresholding — not an exact implementation.
    Static signal: ignores whether confidence is still improving.
    """
    for i, conf in enumerate(curve):
        if conf >= threshold:
            return i + 1, conf
    return N_LAYERS, curve[-1]


def patience_exit(curve: np.ndarray, patience: int = 3) -> Tuple[int, float]:
    """
    Baseline: exit when discretized prediction stable for p layers.
    Coarse approximation of BERT Loses Patience (Zhou et al. 2020).
    Uses conf > 0.70 threshold for discretization — not a faithful
    reimplementation; does not track actual token-level predictions.
    """
    prev_pred = None; count = 0
    for i, conf in enumerate(curve):
        pred = 1 if conf > 0.70 else 0
        if pred == prev_pred:
            count += 1
        else:
            count = 0
        prev_pred = pred
        if count >= patience:
            return i + 1, conf
    return N_LAYERS, curve[-1]


def run_simulation(threshold: float = 0.5, seed: int = 42) -> dict:
    """
    Run simulation across all 600 inputs.

    Accuracy metric: conf_ratio = exit_conf / final_layer_conf
    Measures fraction of final-layer confidence achieved at exit.
    NOT ground truth accuracy.
    """
    rng     = np.random.default_rng(seed)
    results = {m: {'exit_layers': [], 'conf_ratio': [], 'tier': []}
               for m in ['pm', 'conf', 'patience']}

    for tier_name, tier_def in TIERS.items():
        for _ in range(tier_def['n']):
            curve      = generate_confidence_curve(tier_name, rng)
            final_conf = curve[-1]

            pm_layer,   pm_conf   = power_metric_exit(curve, threshold)
            conf_layer, conf_conf = confidence_threshold_exit(curve)
            pat_layer,  pat_conf  = patience_exit(curve)

            for method, layer, conf in [
                ('pm',      pm_layer,   pm_conf),
                ('conf',    conf_layer, conf_conf),
                ('patience',pat_layer,  pat_conf),
            ]:
                results[method]['exit_layers'].append(layer)
                results[method]['conf_ratio'].append(
                    conf / final_conf if final_conf > 1e-6 else 1.0)
                results[method]['tier'].append(tier_name)

    return results


def print_summary(threshold: float, results: dict):
    print(f"\nθ={threshold}  (MIN_CONF={MIN_CONF}, PATIENCE={PATIENCE}):")
    labels = {'pm': 'Power Metric', 'conf': 'Conf Threshold (0.85)',
              'patience': 'Patience (p=3)'}
    for method in ['pm', 'conf', 'patience']:
        layers = np.array(results[method]['exit_layers'])
        ratios = np.array(results[method]['conf_ratio'])
        savings = (1 - layers.mean() / N_LAYERS) * 100
        print(f"  {labels[method]:24}: savings={savings:.1f}%  "
              f"avg_layer={layers.mean():.1f}  conf_ratio={ratios.mean():.3f}")

    # Per-tier breakdown for PM
    pm_layers = np.array(results['pm']['exit_layers'])
    pm_ratios = np.array(results['pm']['conf_ratio'])
    pm_tiers  = np.array(results['pm']['tier'])
    for t in ['easy', 'medium', 'hard', 'very_hard']:
        mask = pm_tiers == t
        print(f"    PM {t:10}: {pm_layers[mask].mean():.1f} layers  "
              f"conf_ratio={pm_ratios[mask].mean():.3f}")


def plot_results(results: dict, threshold: float, save_path: str = None):
    fig = plt.figure(figsize=(14, 9), facecolor='#050810')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; BG='#050810'; PAN='#0d1117'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    method_meta = [
        ('pm',      f'Power Metric (θ={threshold})', C),
        ('conf',    'Conf Threshold (0.85)',         CB),
        ('patience','Patience (p=3)',                CA),
    ]

    # Chart 1: Exit layer distribution
    ax1 = fig.add_subplot(gs[0,0]); style(ax1)
    bins = np.arange(0.5, N_LAYERS + 1.5)
    for key, label, color in method_meta:
        ax1.hist(results[key]['exit_layers'], bins=bins,
                 alpha=0.65, color=color, label=label)
    ax1.set_title('Exit Layer Distribution', color=C, fontsize=10)
    ax1.set_xlabel('Exit Layer', color='#888')
    ax1.set_ylabel('Inputs', color='#888')
    ax1.legend(fontsize=7, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 2: Savings vs accuracy tradeoff
    ax2 = fig.add_subplot(gs[0,1]); style(ax2)
    for key, label, color in method_meta:
        sav  = (1 - np.mean(results[key]['exit_layers']) / N_LAYERS) * 100
        ratio = np.mean(results[key]['conf_ratio']) * 100
        ax2.scatter(sav, ratio, color=color, s=120, zorder=5)
        ax2.annotate(label, (sav, ratio),
                     textcoords='offset points', xytext=(5, 5),
                     color=color, fontsize=8)
    ax2.set_title('Savings vs Accuracy Tradeoff', color=C, fontsize=10)
    ax2.set_xlabel('Compute Savings (%)', color='#888')
    ax2.set_ylabel('Conf Ratio (%)', color='#888')
    ax2.set_xlim(0, 80); ax2.set_ylim(50, 105)

    # Chart 3: Confidence ratio by method
    ax3 = fig.add_subplot(gs[0,2]); style(ax3)
    labels  = [label for _, label, _ in method_meta]
    ratios  = [np.mean(results[k]['conf_ratio'])*100 for k,_,_ in method_meta]
    colors  = [color for _,_,color in method_meta]
    bars    = ax3.bar(range(len(labels)), ratios, color=colors, alpha=0.85)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(['PM', 'Conf\nTh.', 'Patience'], fontsize=8)
    ax3.set_ylim(50, 105)
    ax3.set_title('Conf Ratio vs Full Pass\n(NOT ground truth accuracy)',
                  color=C, fontsize=9)
    ax3.set_ylabel('Conf Ratio (%)', color='#888')
    for b, v in zip(bars, ratios):
        ax3.text(b.get_x()+b.get_width()/2, v+0.3,
                 f'{v:.1f}%', ha='center', color=C, fontsize=9, fontweight='bold')

    # Chart 4: PM exit by difficulty
    ax4 = fig.add_subplot(gs[1,0]); style(ax4)
    pm_layers = np.array(results['pm']['exit_layers'])
    pm_tiers  = np.array(results['pm']['tier'])
    tier_names = ['easy','medium','hard','very_hard']
    tier_avgs  = [pm_layers[pm_tiers==t].mean() for t in tier_names]
    bars4 = ax4.bar(tier_names, tier_avgs, color=C, alpha=0.85)
    ax4.set_title('PM: Avg Exit Layer by Difficulty', color=C, fontsize=10)
    ax4.set_ylabel('Avg Exit Layer', color='#888')
    ax4.set_ylim(0, N_LAYERS + 1)
    for b, v in zip(bars4, tier_avgs):
        ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                 f'{v:.1f}', ha='center', color=C, fontsize=9, fontweight='bold')

    # Chart 5: PM conf ratio by difficulty
    ax5 = fig.add_subplot(gs[1,1]); style(ax5)
    pm_ratios = np.array(results['pm']['conf_ratio'])
    tier_ratios = [pm_ratios[pm_tiers==t].mean()*100 for t in tier_names]
    bars5 = ax5.bar(tier_names, tier_ratios, color=C, alpha=0.85)
    ax5.set_ylim(50, 105)
    ax5.set_title('PM: Conf Ratio by Tier', color=C, fontsize=10)
    ax5.set_ylabel('Conf Ratio (%)', color='#888')
    for b, v in zip(bars5, tier_ratios):
        ax5.text(b.get_x()+b.get_width()/2, v+0.3,
                 f'{v:.1f}%', ha='center', color=C, fontsize=9, fontweight='bold')

    # Chart 6: Summary
    ax6 = fig.add_subplot(gs[1,2]); ax6.set_facecolor(PAN); ax6.axis('off')
    pm_sav   = (1 - pm_layers.mean()/N_LAYERS)*100
    pm_ratio = pm_ratios.mean()*100
    ct_l = np.array(results['conf']['exit_layers'])
    ct_r = np.array(results['conf']['conf_ratio'])
    pt_l = np.array(results['patience']['exit_layers'])
    pt_r = np.array(results['patience']['conf_ratio'])
    stats = [
        ('Architecture',    f'BERT-base, {N_LAYERS} layers'),
        ('Inputs',          f'{N_INPUTS} (4 tiers)'),
        ('PM savings',      f'{pm_sav:.1f}%'),
        ('PM conf ratio',   f'{pm_ratio:.1f}%'),
        ('PM avg layers',   f'{pm_layers.mean():.1f} / {N_LAYERS}'),
        ('Conf savings',    f'{(1-ct_l.mean()/N_LAYERS)*100:.1f}%'),
        ('Conf ratio',      f'{ct_r.mean()*100:.1f}%'),
        ('Pat savings',     f'{(1-pt_l.mean()/N_LAYERS)*100:.1f}%'),
        ('Pat conf ratio',  f'{pt_r.mean()*100:.1f}%'),
        ('Accuracy metric', 'exit_conf / final_conf'),
    ]
    y = 0.96
    ax6.text(0.05, y, 'Results Summary', color=C, fontsize=10,
             fontweight='bold', transform=ax6.transAxes); y -= 0.09
    for lbl, val in stats:
        ax6.text(0.05, y, lbl+':', color='#888', fontsize=8,
                 transform=ax6.transAxes)
        ax6.text(0.58, y, val, color=C, fontsize=8,
                 fontweight='bold', transform=ax6.transAxes); y -= 0.09

    fig.suptitle(
        f'Early Exit via Power Metric — P(t) = E(t)×W(t)  |  θ={threshold}\n'
        f'BERT-base calibration, stylized sigmoid curves  |  '
        f'Conf ratio = exit_conf / final_conf  (NOT ground truth accuracy)',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · Paper 8 · github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print("Early Exit via Power Metric — Paper 8")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print(f"Architecture: BERT-base, {N_LAYERS} layers, {N_INPUTS} inputs")
    print(f"MIN_CONF={MIN_CONF}, PATIENCE={PATIENCE}")
    print("Conf ratio = exit_conf / final_conf (NOT ground truth accuracy)")
    print("Confidence curves are stylized sigmoids — real transformers are noisier\n")

    for th in [0.4, 0.5, 0.6]:
        print_summary(th, run_simulation(threshold=th))

    print("\nGenerating charts (θ=0.5)...")
    plot_results(run_simulation(threshold=0.5), 0.5,
                 save_path='/mnt/user-data/outputs/paper8_simulation.png')
    print("Done.")
