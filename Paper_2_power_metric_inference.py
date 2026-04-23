"""
Adaptive Inference Sampling via Stochastic Power Metrics
=========================================================
Paper 2: "Adaptive Inference Compute Allocation via Stochastic
          Power Metrics: Repeated Sampling Efficiency"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Implements adaptive non-uniform sampling allocation for test-time
repeated sampling (Brown et al., 2024 — Large Language Monkeys).

The problem:
  Brown et al. showed coverage scales log-linearly with sample count.
  Uniform allocation treats every problem identically — easy problems
  with 90% per-sample solve probability get the same budget as hard
  ones with 5%. That wastes compute on the former, under-allocates
  to the latter.

The solution:
  Run P(t) = E(t) × W(t) on each problem's sampling health in real time.
  Stop when: P(t) >= threshold for PATIENCE consecutive samples AND solved.
  Stop when: P(t) stuck low for MAX_UNSOLVABLE samples (genuinely hard).

Stopping criterion (from Paper 2, Section 3):
  - Solved + P(t) stable high → problem solved, stop sampling
  - P(t) stuck near zero → problem unsolvable by this model, stop

Calibration:
  Problem difficulty distribution calibrated to Brown et al. (2024):
  - GSM8K/Llama-3-8B: coverage 0.60 at n=1, 0.96 at n=250
  - ~4% problems genuinely unsolvable (Brown et al. 2024, Section 4)

Note: This is a stylized simulation. Validation requires running on
the actual monkey_business dataset (Brown et al. 2024).

Related papers:
  Paper 14: https://zenodo.org/records/19685841
  Series:   https://github.com/HauntedKernel/power-metric
"""

import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3   # adaptive expected R mean reversion
LAMBDA    = 0.5   # exponential decay for P(t)
EWMA_SPAN = 3     # win rate smoothing

# ── Simulation parameters ─────────────────────────────────────────────────
MAX_SAMPLES      = 250    # Brown et al. budget
PATIENCE         = 8      # consecutive stable samples before stopping
MAX_UNSOLVABLE   = 20     # consecutive low P(t) before classifying as unsolvable
LOW_THRESHOLD    = 0.10   # P(t) below this = unsolvable signal

# ── Calibration to Brown et al. (2024) GSM8K/Llama-3-8B ──────────────────
COVERAGE_AT_N1   = 0.60
COVERAGE_AT_N250 = 0.96
FRAC_UNSOLVABLE  = 0.04   # ~4% problems unsolvable (Brown et al. 2024, Section 4)


def run_problem(p_solve: float, threshold: float,
                rng: np.random.Generator) -> dict:
    """
    Simulate adaptive sampling for one problem with solve probability p.

    Stopping logic:
      - Continue sampling until P(t) >= threshold for PATIENCE steps AND solved
      - OR until P(t) < LOW_THRESHOLD for MAX_UNSOLVABLE steps (unsolvable)
      - OR budget exhausted (MAX_SAMPLES)
    """
    expected_r       = None
    ewma_win         = 0.0
    power            = 0.0
    solved           = False
    stopped          = MAX_SAMPLES
    consecutive_high = 0
    consecutive_low  = 0

    for i in range(MAX_SAMPLES):
        attempt = rng.random() < p_solve
        if attempt:
            solved = True

        signal = 1.0 if attempt else 0.0

        # Layer 1: E(t) with pre-update baseline
        if expected_r is None:
            eff        = 1.0
            expected_r = max(signal, 1e-6)
        else:
            eff        = signal / expected_r
            expected_r = (1 - ALPHA) * expected_r + ALPHA * signal

        # Layer 2: W(t)
        win      = 1.0 if eff > 1.0 else 0.0
        a        = 2.0 / (EWMA_SPAN + 1)
        ewma_win = a * win + (1 - a) * ewma_win

        # Layer 3: P(t)
        inst  = eff * ewma_win
        power = np.exp(-LAMBDA) * power + (1 - np.exp(-LAMBDA)) * inst

        # Stopping decisions (after warmup)
        if i >= PATIENCE:
            if power >= threshold and solved:
                consecutive_high += 1
            else:
                consecutive_high = 0

            if power < LOW_THRESHOLD:
                consecutive_low += 1
            else:
                consecutive_low = 0

            if consecutive_high >= PATIENCE:
                stopped = i + 1
                break

            if consecutive_low >= MAX_UNSOLVABLE:
                stopped = i + 1
                break

    return dict(samples_used=stopped, solved=solved, p_solve=p_solve)


def run_simulation(threshold=0.5, n_problems=127, seed=42) -> dict:
    """
    Simulate across n_problems (default=127, GSM8K test set size).
    """
    rng = np.random.default_rng(seed)

    # Build calibrated problem difficulty distribution
    n_hard = int(n_problems * FRAC_UNSOLVABLE)
    n_easy = n_problems - n_hard

    p_easy = rng.beta(2.5, 1.2, n_easy)
    p_easy = np.clip(p_easy * 0.625 / p_easy.mean(), 0.05, 0.99)
    p_hard = rng.uniform(0.001, 0.01, n_hard)
    p_all  = np.concatenate([p_easy, p_hard])
    rng.shuffle(p_all)

    results = dict(samples_used=[], solved=[], uniform_solved=[], p_solve=[])

    for p in p_all:
        r = run_problem(p, threshold, rng)
        # Uniform baseline: solve with full budget
        p_any = 1.0 - (1.0 - p) ** MAX_SAMPLES
        uni   = rng.random() < p_any

        results['samples_used'].append(r['samples_used'])
        results['solved'].append(r['solved'])
        results['uniform_solved'].append(uni)
        results['p_solve'].append(p)

    return results


def print_summary(threshold, results):
    samples    = np.array(results['samples_used'])
    solved     = np.array(results['solved'])
    uni_solved = np.array(results['uniform_solved'])
    n          = len(samples)
    reduction  = (1 - samples.sum() / (MAX_SAMPLES * n)) * 100
    delta      = solved.mean() - uni_solved.mean()
    print(f"  θ={threshold}:  "
          f"reduction={reduction:.1f}%  "
          f"coverage={solved.mean():.3f}  "
          f"delta={delta:+.3f}  "
          f"avg_samples={samples.mean():.1f}")


def plot_results(results, threshold, save_path=None):
    fig = plt.figure(figsize=(12, 8), facecolor='#050810')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; BG='#050810'; PAN='#0d1117'

    samples    = np.array(results['samples_used'])
    solved     = np.array(results['solved'])
    uni_solved = np.array(results['uniform_solved'])
    p_solves   = np.array(results['p_solve'])
    n          = len(samples)
    reduction  = (1 - samples.sum() / (MAX_SAMPLES * n)) * 100
    delta      = solved.mean() - uni_solved.mean()

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    # Chart 1: Samples used vs problem difficulty
    ax1 = fig.add_subplot(gs[0,0]); style(ax1)
    ax1.scatter(p_solves, samples, color=C, alpha=0.4, s=12)
    ax1.axhline(MAX_SAMPLES, color=CB, lw=1.5, ls='--',
                label=f'Uniform ({MAX_SAMPLES})')
    ax1.set_title('Samples Used vs Problem Difficulty', color=C, fontsize=11)
    ax1.set_xlabel('Per-sample solve probability', color='#888')
    ax1.set_ylabel('Samples used', color='#888')
    ax1.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 2: Sample distribution
    ax2 = fig.add_subplot(gs[0,1]); style(ax2)
    ax2.hist(samples, bins=30, color=C, alpha=0.8, edgecolor='none')
    ax2.axvline(MAX_SAMPLES, color=CB, lw=1.5, ls='--',
                label=f'Uniform ({MAX_SAMPLES})')
    ax2.axvline(samples.mean(), color=CA, lw=1.5,
                label=f'Mean: {samples.mean():.0f}')
    ax2.set_title('Sample Distribution', color=C, fontsize=11)
    ax2.set_xlabel('Samples per problem', color='#888')
    ax2.set_ylabel('Problems', color='#888')
    ax2.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 3: Coverage comparison
    ax3 = fig.add_subplot(gs[1,0]); style(ax3)
    methods   = ['Uniform\n(Brown et al.)', f'Adaptive PM\n(θ={threshold})']
    coverages = [uni_solved.mean(), solved.mean()]
    bars      = ax3.bar(methods, coverages, color=[CB, C], alpha=0.85, width=0.4)
    ax3.set_ylim(0.85, 1.0)
    ax3.set_title('Coverage Comparison', color=C, fontsize=11)
    ax3.set_ylabel('Coverage', color='#888')
    for b, v in zip(bars, coverages):
        ax3.text(b.get_x()+b.get_width()/2, v+0.001,
                 f'{v:.3f}', ha='center', color=C, fontsize=10,
                 fontweight='bold')

    # Chart 4: Summary
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(PAN); ax4.axis('off')
    stats = [
        ('Threshold θ',          f'{threshold}'),
        ('Problems',             f'{n}'),
        ('Compute reduction',    f'{reduction:.1f}%'),
        ('Avg samples/problem',  f'{samples.mean():.1f} / {MAX_SAMPLES}'),
        ('Coverage (adaptive)',  f'{solved.mean():.3f}'),
        ('Coverage (uniform)',   f'{uni_solved.mean():.3f}'),
        ('Coverage delta',       f'{delta:+.3f}'),
        ('PATIENCE',             f'{PATIENCE} consecutive samples'),
    ]
    y = 0.95
    ax4.text(0.05, y, 'Simulation Summary', color=C, fontsize=11,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for lbl, val in stats:
        ax4.text(0.05, y, lbl+':', color='#888', fontsize=9,
                 transform=ax4.transAxes)
        ax4.text(0.60, y, val, color=C, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.11

    fig.suptitle(
        f'Adaptive Inference Sampling — P(t) = E(t)×W(t)  |  θ={threshold}\n'
        f'Calibrated to Brown et al. (2024) GSM8K/Llama-3-8B  |  '
        f'Stylized simulation — validation on monkey_business dataset required',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · Paper 2 · github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print("Adaptive Inference Sampling — Paper 2")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print(f"PATIENCE={PATIENCE}, MAX_UNSOLVABLE={MAX_UNSOLVABLE}")
    print(f"Calibrated: coverage {COVERAGE_AT_N1} at n=1, "
          f"{COVERAGE_AT_N250} at n={MAX_SAMPLES}\n")

    for th in [0.3, 0.5, 0.7]:
        print_summary(th, run_simulation(threshold=th, n_problems=127))

    print("\nGenerating charts (θ=0.5)...")
    plot_results(run_simulation(threshold=0.5, n_problems=127), 0.5,
                 save_path='/mnt/user-data/outputs/paper2_simulation.png')
    print("Done.")
