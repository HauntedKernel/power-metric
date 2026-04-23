"""
Scaling Law Reliability via Adaptive Benchmarks
=================================================
Paper 3: "A Tunable Solution to the 61% Unreliability Problem:
          Adaptive Benchmarks via Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Replicates and extends Hu et al. (2025) scaling law unreliability
finding using real Pythia benchmark data (Biderman et al., 2023).

The problem (Hu et al. 2025):
  Fixed power-law predictions of benchmark performance are unreliable
  61% of the time — they fail to predict when scaling laws break down
  due to emergent capabilities, data distribution shifts, or
  architectural interactions.

The solution:
  Replace the fixed power-law benchmark with the adaptive expected
  benchmark E[R](t) from the power metric framework:

    E[R](t) = (1-α) · E[R](t-1) + α · score(t-1)

  At α=0.3, unreliability drops from 96% to 38% (average).
  At α=0.5, unreliability reaches 0% for Pythia-160M and 1.4B.

Method:
  - Fit power law to first 8 post-warmup checkpoints
  - Predict remaining 8 checkpoints
  - Compare |prediction - actual| vs threshold (0.02)
  - Unreliable = fraction of steps where |error| > threshold

  E[R](t) uses pre-update baseline:
    eff = score(t) / E[R](t-1)  [computed before updating E[R]]
    E[R] then updated: E[R](t) = (1-α)·E[R](t-1) + α·score(t)

Data: EleutherAI/pythia GitHub (public)
  https://github.com/EleutherAI/pythia

Related papers:
  Paper 1:  power_metric_training.py (same dataset)
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
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA_DEFAULT        = 0.3    # adaptive expected R mean reversion
UNRELIABILITY_THRESH = 0.02   # |prediction - actual| > this = unreliable
SPLIT                = 8      # first 8 fit, last 8 predict (16 total post-warmup)

BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']
MODELS     = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']


def load_pythia_scores(base_path: str, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load real Pythia benchmark scores. See power_metric_training.py."""
    folder = os.path.join(base_path, model, 'zero-shot')
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Pythia eval folder not found: {folder}\n"
            f"Download from: https://github.com/EleutherAI/pythia"
        )
    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        r = data.get('results', {})
        sc = []
        for b in BENCHMARKS:
            if b in r:
                bd = r[b]
                s  = bd.get('acc_norm', bd.get('acc', None))
                if s is not None: sc.append(s)
        if len(sc) == len(BENCHMARKS):
            step_scores[step] = float(np.mean(sc))
    steps = sorted(step_scores.keys())
    return np.array(steps), np.array([step_scores[s] for s in steps])


def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Standard power law: y = a·x^b + c"""
    return a * np.power(x, b) + c


def fixed_power_law_prediction(
    scores: np.ndarray,
    split: int = SPLIT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit power law to first `split` checkpoints, predict remainder.

    Returns:
        y_pred:   predicted scores for post-split checkpoints
        y_actual: actual scores for post-split checkpoints
    """
    x_fit  = np.arange(1, split + 1, dtype=float)
    x_pred = np.arange(split + 1, len(scores) + 1, dtype=float)

    try:
        popt, _ = curve_fit(
            power_law, x_fit, scores[:split],
            p0=[0.1, 0.3, 0.3], maxfev=10000,
            bounds=([-np.inf, -2, 0], [np.inf, 2, 1])
        )
        y_pred = power_law(x_pred, *popt)
    except RuntimeError:
        # Fallback: flat prediction at last fitted value
        y_pred = np.full(len(scores) - split, scores[split - 1])

    return y_pred, scores[split:]


def adaptive_benchmark_prediction(
    scores: np.ndarray,
    alpha: float,
    split: int = SPLIT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive E[R](t) as prediction for post-split checkpoints.

    E[R](t) = (1-α)·E[R](t-1) + α·score(t-1)

    Uses pre-update baseline — E[R](t-1) before incorporating score(t).
    Predictions for post-split checkpoints use the E[R] state at
    the end of the fitting phase, then continue updating.

    Returns:
        y_pred:   E[R](t-1) predictions for post-split checkpoints
        y_actual: actual scores for post-split checkpoints
    """
    er = scores[0]
    er_series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        er_series.append(er)

    # Pre-update predictions: E[R](t-1) predicts score(t)
    # For post-split checkpoints, use E[R] at previous step
    y_pred   = np.array(er_series[split - 1:-1])
    y_actual = scores[split:]

    return y_pred, y_actual


def compute_reliability_metrics(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    threshold: float = UNRELIABILITY_THRESH,
) -> Dict[str, float]:
    """Compute MAE and unreliability fraction."""
    errors = np.abs(y_pred - y_actual)
    return dict(
        mae         = float(np.mean(errors)),
        unreliable  = float(np.mean(errors > threshold)),
        n           = len(errors),
    )


def run_analysis(
    base_path: str,
    alpha: float = ALPHA_DEFAULT,
) -> dict:
    """Run full analysis across all three Pythia models."""
    results = {}

    for model in MODELS:
        steps, scores = load_pythia_scores(base_path, model)

        # Fixed power law
        y_fixed, y_actual = fixed_power_law_prediction(scores)
        fixed_metrics     = compute_reliability_metrics(y_fixed, y_actual)

        # Adaptive benchmark
        y_adapt, _    = adaptive_benchmark_prediction(scores, alpha)
        adapt_metrics = compute_reliability_metrics(y_adapt, y_actual)

        mae_reduction = ((fixed_metrics['mae'] - adapt_metrics['mae'])
                         / fixed_metrics['mae'] * 100)

        results[model] = dict(
            steps         = steps,
            scores        = scores,
            y_actual      = y_actual,
            y_fixed       = y_fixed,
            y_adapt       = y_adapt,
            fixed_mae     = fixed_metrics['mae'],
            fixed_unrel   = fixed_metrics['unreliable'],
            adapt_mae     = adapt_metrics['mae'],
            adapt_unrel   = adapt_metrics['unreliable'],
            mae_reduction = mae_reduction,
            alpha         = alpha,
        )

    return results


def run_alpha_sweep(
    base_path: str,
    model: str = 'pythia-1.4b',
    alphas: List[float] = None,
) -> dict:
    """Alpha sensitivity analysis for a single model."""
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5]

    steps, scores = load_pythia_scores(base_path, model)
    y_fixed, y_actual = fixed_power_law_prediction(scores)
    fixed_metrics = compute_reliability_metrics(y_fixed, y_actual)

    sweep = {}
    for alpha in alphas:
        y_adapt, _ = adaptive_benchmark_prediction(scores, alpha)
        m          = compute_reliability_metrics(y_adapt, y_actual)
        sweep[alpha] = dict(
            adapt_mae     = m['mae'],
            adapt_unrel   = m['unreliable'],
            mae_reduction = (fixed_metrics['mae'] - m['mae']) / fixed_metrics['mae'] * 100,
        )

    return sweep, fixed_metrics


def print_summary(results: dict, alpha: float):
    print(f"\nα={alpha} | Unreliability threshold={UNRELIABILITY_THRESH}"
          f" | Split={SPLIT}/{SPLIT}")
    print(f"{'Model':<16} {'Fixed MAE':>10} {'Fixed%':>8} "
          f"{'Adapt MAE':>10} {'Adapt%':>8} {'MAE Red.':>10}")
    print("-" * 66)

    fixed_unrels = []; adapt_unrels = []; fixed_maes = []; adapt_maes = []
    for model, r in results.items():
        print(f"{model:<16} {r['fixed_mae']:>10.4f} {r['fixed_unrel']*100:>7.0f}% "
              f"{r['adapt_mae']:>10.4f} {r['adapt_unrel']*100:>7.0f}% "
              f"{r['mae_reduction']:>9.1f}%")
        fixed_unrels.append(r['fixed_unrel']); adapt_unrels.append(r['adapt_unrel'])
        fixed_maes.append(r['fixed_mae']); adapt_maes.append(r['adapt_mae'])

    avg_mr = (np.mean(fixed_maes) - np.mean(adapt_maes)) / np.mean(fixed_maes) * 100
    print(f"{'Average':<16} {np.mean(fixed_maes):>10.4f} "
          f"{np.mean(fixed_unrels)*100:>7.0f}% "
          f"{np.mean(adapt_maes):>10.4f} {np.mean(adapt_unrels)*100:>7.0f}% "
          f"{avg_mr:>9.1f}%")


def plot_results(
    results: dict,
    sweep: dict,
    fixed_metrics: dict,
    threshold: float = UNRELIABILITY_THRESH,
    save_path: str = None,
):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'
    MODEL_COLORS = {'pythia-160m': C, 'pythia-1.4b': CA, 'pythia-6.9b': CG}

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    for col, (model, r) in enumerate(results.items()):
        color = MODEL_COLORS[model]
        n_pred = len(r['y_actual'])
        x_pred = np.arange(SPLIT + 1, SPLIT + n_pred + 1)

        # Row 1: Actual vs predictions
        ax1 = fig.add_subplot(gs[0, col]); style(ax1)
        ax1.plot(range(1, SPLIT+1), r['scores'][:SPLIT],
                 color=CGRAY, lw=1.5, ls='--', label='Fit phase')
        ax1.plot(x_pred, r['y_actual'],
                 color=color, lw=2, marker='o', ms=5, label='Actual')
        ax1.plot(x_pred, r['y_fixed'],
                 color=CB, lw=1.5, ls='--', label='Fixed PL')
        ax1.plot(x_pred, r['y_adapt'],
                 color=CG, lw=1.5, ls='-', label='Adaptive E[R]')
        ax1.axvline(SPLIT + 0.5, color='#444', lw=1, ls=':')
        ax1.set_title(f'{model}', color=color, fontsize=9)
        ax1.set_xlabel('Checkpoint', color=CGRAY, fontsize=8)
        ax1.set_ylabel('Composite Score', color=CGRAY, fontsize=8)
        ax1.legend(fontsize=6, labelcolor='white', facecolor=PAN,
                   edgecolor='#333', loc='lower right')

        # Row 2: Prediction errors
        ax2 = fig.add_subplot(gs[1, col]); style(ax2)
        err_fixed = np.abs(r['y_fixed'] - r['y_actual'])
        err_adapt = np.abs(r['y_adapt'] - r['y_actual'])
        ax2.bar(x_pred - 0.2, err_fixed, 0.35, color=CB, alpha=0.7,
                label='Fixed PL error')
        ax2.bar(x_pred + 0.2, err_adapt, 0.35, color=CG, alpha=0.7,
                label='Adaptive error')
        ax2.axhline(threshold, color='white', lw=1, ls='--',
                    label=f'Threshold ({threshold})')
        ax2.set_title('Prediction Error', color=color, fontsize=9)
        ax2.set_xlabel('Checkpoint', color=CGRAY, fontsize=8)
        ax2.set_ylabel('|Error|', color=CGRAY, fontsize=8)
        ax2.legend(fontsize=6, labelcolor='white', facecolor=PAN,
                   edgecolor='#333')

    # Row 3: Alpha sweep (1.4B) + summary
    ax3 = fig.add_subplot(gs[2, :2]); style(ax3)
    alphas      = sorted(sweep.keys())
    unrel_fixed = [fixed_metrics['unreliable'] * 100] * len(alphas)
    unrel_adapt = [sweep[a]['adapt_unrel'] * 100 for a in alphas]
    mae_red     = [sweep[a]['mae_reduction'] for a in alphas]

    ax3_twin = ax3.twinx(); ax3_twin.set_facecolor(PAN)
    ax3.plot(alphas, unrel_fixed, color=CB, lw=1.5, ls='--',
             label='Fixed PL unreliable%')
    ax3.plot(alphas, unrel_adapt, color=CG, lw=2, marker='o', ms=6,
             label='Adaptive unreliable%')
    ax3_twin.plot(alphas, mae_red, color=CA, lw=1.5, ls=':',
                  label='MAE reduction%')
    ax3.set_title('Alpha Sensitivity — Pythia-1.4B', color=C, fontsize=10)
    ax3.set_xlabel('α (adaptation rate)', color=CGRAY)
    ax3.set_ylabel('Unreliable (%)', color=CGRAY)
    ax3_twin.set_ylabel('MAE Reduction (%)', color=CA, fontsize=8)
    ax3_twin.tick_params(colors='#666666')
    ax3.legend(fontsize=7, labelcolor='white', facecolor=PAN,
               edgecolor='#333', loc='upper right')

    ax4 = fig.add_subplot(gs[2, 2]); ax4.set_facecolor(PAN); ax4.axis('off')
    y = 0.96
    ax4.text(0.05, y, 'Results Summary (α=0.3)', color=C, fontsize=10,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for model, r in results.items():
        short = model.replace('pythia-', '')
        ax4.text(0.05, y, f'{short}:', color=CA, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.09
        for lbl, val in [
            ('Fixed unrel.', f"{r['fixed_unrel']*100:.0f}%"),
            ('Adapt unrel.', f"{r['adapt_unrel']*100:.0f}%"),
            ('MAE reduction', f"{r['mae_reduction']:.1f}%"),
        ]:
            ax4.text(0.08, y, f'{lbl}: {val}', color=C, fontsize=8,
                     transform=ax4.transAxes); y -= 0.08
        y -= 0.02

    fig.suptitle(
        'Scaling Law Reliability — Fixed Power Law vs Adaptive E[R](t)\n'
        'Real Pythia data (Biderman et al. 2023) — '
        'Replicates Hu et al. (2025) unreliability finding',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                f'Unreliability threshold={UNRELIABILITY_THRESH}  |  '
                f'Split={SPLIT}/{SPLIT} checkpoints  |  '
                f'Cantrell (2026) · Paper 3 · github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./pythia-main/evals/pythia-v1')
    args = parser.parse_args()

    print("Scaling Law Reliability — Paper 3")
    print(f"Real Pythia data: {args.data}")
    print(f"Unreliability threshold: {UNRELIABILITY_THRESH}")
    print(f"Split: first {SPLIT} fit / last {SPLIT} predict\n")

    # Primary results
    for alpha in [0.3, 0.4, 0.5]:
        print_summary(run_analysis(args.data, alpha), alpha)

    # Alpha sweep for 1.4B
    print("\nAlpha sweep — Pythia-1.4B:")
    sweep, fixed_m = run_alpha_sweep(args.data)
    print(f"  Fixed baseline: MAE={fixed_m['mae']:.4f}, "
          f"Unreliable={fixed_m['unreliable']*100:.0f}%")
    print(f"  {'α':>5} {'Adapt MAE':>12} {'Unreliable':>12} {'MAE Red.':>12}")
    for a, m in sweep.items():
        print(f"  {a:>5.1f} {m['adapt_mae']:>12.4f} "
              f"{m['adapt_unrel']*100:>11.0f}% {m['mae_reduction']:>11.1f}%")

    print("\nGenerating charts...")
    r = run_analysis(args.data, alpha=0.3)
    sw, fm = run_alpha_sweep(args.data)
    plot_results(r, sw, fm,
                 save_path='/mnt/user-data/outputs/paper3_simulation.png')
    print("Done.")
