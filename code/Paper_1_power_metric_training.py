"""
Training Compute Allocation via Stochastic Power Metrics
=========================================================
Paper 1: "Dynamic Training Compute Allocation via Stochastic
          Power Metrics: Preliminary Evidence from the Pythia Suite"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies P(t) = E(t) × W(t) to training checkpoint health monitoring,
using actual Pythia benchmark data (Biderman et al., 2023).

This is the only paper in the series using real public data.

Method:
  At each checkpoint, P(t) is computed from benchmark score trajectory.
  P(t) drives a continuous allocation coefficient C(t) ∈ [0,1]:

    C(t) = sigmoid(k · (P(t) − θ))

  where θ is the monitoring threshold and k=10 controls steepness.
  C(t) ≈ 1.0 when P(t) >> θ  (full GPU allocation)
  C(t) ≈ 0.0 when P(t) << θ  (monitoring mode — reduced allocation)

  Total dynamic compute = Σ C(t) across checkpoints
  Savings = 1 − dynamic_compute / static_compute

  Crucially: C(t) = 0 does not terminate the run. The system
  checkpoints and reduces allocation, resuming automatically
  when P(t) recovers. Final benchmark performance is preserved
  because the training trajectory is unchanged — only allocation timing.

  Hypothesis: reducing GPU allocation during low-signal phases
  (P(t) < threshold) would not degrade final benchmark performance,
  since the signal indicates benchmark improvement has stalled
  relative to adaptive expectation.

  Score delta = 0.000 by construction — this analysis identifies
  when allocation could be reduced, not a controlled experiment.
  Empirical validation requires training runs with active
  allocation control on real GPU infrastructure.

Data source:
  EleutherAI/pythia GitHub (public)
  evals/pythia-v1/{model}/zero-shot/
  https://github.com/EleutherAI/pythia

Usage:
  Place pythia-main/ in the same directory as this script, then:
    python power_metric_training.py

  Or specify path:
    python power_metric_training.py --data /path/to/pythia-main/evals/pythia-v1

Related papers:
  Paper 14: https://zenodo.org/records/19685841
  Series:   https://github.com/HauntedKernel/power-metric
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3   # adaptive expected R mean reversion
LAMBDA    = 0.5   # exponential decay for P(t)
EWMA_SPAN = 3     # win rate smoothing

# ── C(t) allocation parameters ────────────────────────────────────────────
CT_STEEPNESS = 10  # k: sigmoid steepness around threshold

# ── Data configuration ────────────────────────────────────────────────────
BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']
MODELS     = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']
MIN_STEP   = 1000   # exclude warmup (steps 0-512)


def load_pythia_scores(
    base_path: str,
    model: str,
) -> Tuple[List[int], List[float]]:
    """
    Load benchmark scores from real Pythia evaluation files.

    Composite score = unweighted average of acc_norm across 5 benchmarks.
    Post-warmup only (step >= MIN_STEP).

    Download data from: https://github.com/EleutherAI/pythia
    """
    folder = os.path.join(base_path, model, 'zero-shot')
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Pythia eval folder not found: {folder}\n"
            f"Download from: https://github.com/EleutherAI/pythia\n"
            f"Extract and place pythia-main/ next to this script."
        )

    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m:
            continue
        step = int(m.group(1))
        if step < MIN_STEP:
            continue

        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)

        results = data.get('results', {})
        scores  = []
        for b in BENCHMARKS:
            if b in results:
                r = results[b]
                s = r.get('acc_norm', r.get('acc', None))
                if s is not None:
                    scores.append(s)

        if len(scores) == len(BENCHMARKS):
            step_scores[step] = float(np.mean(scores))

    steps      = sorted(step_scores.keys())
    score_list = [step_scores[s] for s in steps]
    return steps, score_list


def compute_power_metric(scores: List[float]) -> Dict[str, List[float]]:
    """
    Compute E(t), W(t), P(t) on benchmark score trajectory.
    E(t) uses pre-update baseline — no information leak.
    """
    expected_r   = None
    ewma_win     = 0.0
    power        = 0.0
    efficiencies = []
    win_rates    = []
    powers       = []

    for score in scores:
        if expected_r is None:
            eff        = 1.0
            expected_r = max(score, 1e-6)
        else:
            eff        = score / expected_r
            expected_r = (1 - ALPHA) * expected_r + ALPHA * score

        win      = 1.0 if eff > 1.0 else 0.0
        a        = 2.0 / (EWMA_SPAN + 1)
        ewma_win = a * win + (1 - a) * ewma_win
        inst     = eff * ewma_win
        power    = np.exp(-LAMBDA) * power + (1 - np.exp(-LAMBDA)) * inst

        efficiencies.append(eff)
        win_rates.append(ewma_win)
        powers.append(power)

    return dict(efficiency=efficiencies, win_rate=win_rates, power=powers)


def compute_allocation(
    powers: List[float],
    threshold: float = 0.5,
    k: float = CT_STEEPNESS,
) -> List[float]:
    """
    Compute continuous allocation coefficient C(t) ∈ [0,1].

    C(t) = sigmoid(k · (P(t) − threshold))

    Full allocation (C≈1) when P(t) >> threshold.
    Monitoring mode (C≈0) when P(t) << threshold.
    Smooth transition avoids discontinuous allocation changes.
    """
    return [1.0 / (1.0 + np.exp(-k * (p - threshold))) for p in powers]


def run_analysis(base_path: str, threshold: float = 0.5) -> dict:
    """Run full analysis across all three Pythia models."""
    results = {}

    for model in MODELS:
        steps, scores     = load_pythia_scores(base_path, model)
        pm                = compute_power_metric(scores)
        ct                = compute_allocation(pm['power'], threshold)

        static_compute    = float(len(steps))
        dynamic_compute   = float(sum(ct))
        savings           = 1.0 - dynamic_compute / static_compute

        # Reconstruct adaptive expected R for plotting
        exp_r = None; exp_series = []
        for s in scores:
            if exp_r is None: exp_r = s
            else: exp_r = (1-ALPHA)*exp_r + ALPHA*s
            exp_series.append(exp_r)

        results[model] = dict(
            steps           = steps,
            scores          = scores,
            exp_series      = exp_series,
            efficiency      = pm['efficiency'],
            win_rate        = pm['win_rate'],
            power           = pm['power'],
            ct              = ct,
            static_compute  = static_compute,
            dynamic_compute = dynamic_compute,
            savings         = savings,
            final_score     = scores[-1],
            score_delta     = 0.000,  # by construction — allocation not simulated
            # hypothesis: reducing allocation here would not change outcome
        )

    return results


def print_summary(results: dict, threshold: float):
    print(f"\nθ={threshold}  (k={CT_STEEPNESS}, α={ALPHA}, λ={LAMBDA})")
    print(f"{'Model':<16} {'Final':>8} {'Static':>8} {'Dynamic':>10} "
          f"{'Candidate%':>12} {'Note':>10}")
    print("-" * 72)
    for model, r in results.items():
        print(f"{model:<16} {r['final_score']:>8.4f} "
              f"{r['static_compute']:>8.1f} {r['dynamic_compute']:>10.2f} "
              f"{r['savings']*100:>11.1f}%  (hypothesized)")


def plot_results(results: dict, threshold: float, save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'
    MODEL_COLORS = {'pythia-160m': C, 'pythia-1.4b': CA, 'pythia-6.9b': CG}

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    for col, (model, r) in enumerate(results.items()):
        color  = MODEL_COLORS[model]
        steps  = r['steps']
        x      = range(len(steps))

        # Row 1: Score + adaptive expected
        ax1 = fig.add_subplot(gs[0, col]); style(ax1)
        ax1.plot(x, r['scores'],     color=color,  lw=2,   label='Score')
        ax1.plot(x, r['exp_series'], color=CGRAY,  lw=1.2, ls='--',
                 label='Adaptive E[R]')
        # Shade low-allocation regions (C(t) < 0.5)
        for i, c in enumerate(r['ct']):
            if c < 0.5 and i < len(steps)-1:
                ax1.axvspan(i, i+1, alpha=0.15, color=CB)
        ax1.set_title(f'{model}', color=color, fontsize=9)
        ax1.set_xlabel('Checkpoint', color=CGRAY, fontsize=8)
        ax1.set_ylabel('Composite Score', color=CGRAY, fontsize=8)
        ax1.legend(fontsize=7, labelcolor='white', facecolor=PAN, edgecolor='#333')

        # Row 2: P(t) and C(t)
        ax2 = fig.add_subplot(gs[1, col]); style(ax2)
        ax2_twin = ax2.twinx()
        ax2_twin.set_facecolor(PAN)
        ax2.plot(x, r['power'], color=color, lw=2,   label='P(t)')
        ax2_twin.plot(x, r['ct'],   color=CG,    lw=1.5, ls='--',
                      label='C(t) allocation', alpha=0.8)
        ax2.axhline(threshold, color=CB, lw=1, ls=':',
                    label=f'θ={threshold}')
        ax2.set_title('P(t) and C(t)', color=color, fontsize=9)
        ax2.set_xlabel('Checkpoint', color=CGRAY, fontsize=8)
        ax2.set_ylabel('P(t)', color=CGRAY, fontsize=8)
        ax2_twin.set_ylabel('C(t)', color=CG, fontsize=8)
        ax2_twin.tick_params(colors='#666666')
        ax2.set_ylim(0, 1.1); ax2_twin.set_ylim(0, 1.1)
        ax2.legend(fontsize=7, labelcolor='white', facecolor=PAN, edgecolor='#333',
                   loc='lower right')

    # Row 3: Savings comparison + summary
    ax3 = fig.add_subplot(gs[2, :2]); style(ax3)
    model_names = [m.replace('pythia-', '') for m in results.keys()]
    savings     = [r['savings']*100 for r in results.values()]
    dynamic     = [r['dynamic_compute'] for r in results.values()]
    colors      = list(MODEL_COLORS.values())
    bars        = ax3.bar(model_names, savings, color=colors, alpha=0.85)
    ax3.set_title('Compute Savings via C(t) Allocation Coefficient',
                  color=C, fontsize=10)
    ax3.set_ylabel('Savings (%)', color=CGRAY)
    ax3.set_ylim(0, 60)
    for b, v, d in zip(bars, savings, dynamic):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{v:.1f}%\n({d:.1f}/16)',
                 ha='center', color=C, fontsize=9, fontweight='bold')

    ax4 = fig.add_subplot(gs[2, 2]); ax4.set_facecolor(PAN); ax4.axis('off')
    y = 0.96
    ax4.text(0.05, y, 'Results Summary', color=C, fontsize=10,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for model, r in results.items():
        short = model.replace('pythia-', '')
        ax4.text(0.05, y, f'{short}:', color=CA, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.09
        for lbl, val in [
            ('Savings', f"{r['savings']*100:.1f}%"),
            ('Dynamic compute', f"{r['dynamic_compute']:.2f}/16.0"),
            ('Final score', f"{r['final_score']:.4f}"),
            ('Score Δ', '+0.000'),
        ]:
            ax4.text(0.08, y, f'{lbl}: {val}', color=C, fontsize=8,
                     transform=ax4.transAxes); y -= 0.08
        y -= 0.02

    fig.suptitle(
        'Training Compute Allocation via P(t) → C(t)\n'
        'Real Pythia benchmark data (Biderman et al. 2023) — '
        f'threshold θ={threshold} — NOT a stylized simulation',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'C(t) = sigmoid(k·(P(t)−θ))  |  Cantrell (2026) · Paper 1  |  '
                'Data: EleutherAI/pythia  |  github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./pythia-main/evals/pythia-v1',
                        help='Path to pythia-v1 evals directory')
    args = parser.parse_args()

    print("Training Compute Allocation — Paper 1")
    print(f"Real Pythia data: {args.data}")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}, k={CT_STEEPNESS}")
    print(f"Benchmarks: {', '.join(BENCHMARKS)}\n")
    print("C(t) = sigmoid(k·(P(t)−θ))  —  continuous allocation coefficient\n")

    for th in [0.5, 0.6, 0.7]:
        print_summary(run_analysis(args.data, th), th)

    print("\nGenerating charts (θ=0.7)...")
    r = run_analysis(args.data, threshold=0.7)
    plot_results(r, 0.7,
                 save_path='/mnt/user-data/outputs/paper1_simulation.png')
    print("Done.")
