"""
Verifier Health Monitoring via Stochastic Power Metrics
========================================================
Paper 17: "Monitoring Verifier Health in Test-Time Scaling
           Using Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies P(t) = E(t) x W(t) as a real-time stopping criterion
for LLM-based verifiers during test-time scaling.

Implementation notes:
- E(t) = score(t) / E[R](t-1)  [PRE-UPDATE baseline — no information leak]
- E[R] updated AFTER E(t) computed
- W(t) EWMA span=3 consistent with paper and prior work
- P(t) initialized at 0.0 with explicit min_candidates warmup
- Quality = best_score_found / best_score_available (NOT task accuracy)
- Signal consistent with Papers 1-4 in series (score vs adaptive expectation)

Related papers: https://zenodo.org/records/19687346
"""

import numpy as np
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3   # adaptive expected R mean reversion
LAMBDA    = 0.5   # exponential decay for P(t) integral
EWMA_SPAN = 3     # win rate smoothing — span=3 as stated in paper


@dataclass
class VerifierUpdate:
    candidate_idx:      int
    score:              float
    efficiency:         float   # E(t) = score / E[R](t-1)
    win_rate:           float   # W(t)
    power:              float   # P(t)
    should_stop:        bool
    best_score:         float
    best_candidate_idx: int


class VerifierHealthMonitor:
    """
    Real-time verifier health monitor.

    E(t) = score(t) / E[R](t-1)
      Is this candidate scoring above the adaptive expected score?
      Uses PRE-UPDATE baseline to avoid information leak.

    W(t) = EWMA of [E(t) > 1.0], span=3
      Is the verifier consistently finding above-expectation candidates?

    P(t) = exp(-λ)·P(t-1) + (1-exp(-λ))·[E(t)×W(t)]
      Integrated health signal, robust to single-step noise.

    Stop when P(t) < threshold AND candidate_idx >= min_candidates.

    Note on quality metric:
      quality = best_score_found / best_score_available
      Measures search efficiency within the score stream.
      This is NOT downstream task accuracy — see paper Section 4.
    """

    def __init__(self, threshold: float = 0.5, min_candidates: int = 10):
        self.threshold      = threshold
        self.min_candidates = min_candidates

        # Power metric state
        self.expected_r = None  # initialized on first observation
        self.ewma_win   = 0.0   # neutral start
        self.power      = 0.0   # start at 0 — no artificial warmup bias

        # Best candidate tracking
        self.best_score = -np.inf
        self.best_idx   = 0

        # History
        self.scores:       List[float] = []
        self.efficiencies: List[float] = []
        self.win_rates:    List[float] = []
        self.powers:       List[float] = []

    def update(self, score: float) -> VerifierUpdate:
        """
        Process one new candidate's verifier score.

        Critical: E[R] is updated AFTER E(t) is computed so the
        current observation does not influence its own baseline.
        """
        self.scores.append(score)
        idx = len(self.scores) - 1

        if score > self.best_score:
            self.best_score = score
            self.best_idx   = idx

        # ── Layer 1: E(t) using PRE-UPDATE expected_r ─────────────────
        if self.expected_r is None:
            efficiency      = 1.0                    # neutral first step
            self.expected_r = max(score, 1e-6)       # initialize
        else:
            # Use PRE-UPDATE baseline — no information leak
            efficiency      = score / self.expected_r
            # NOW update expected_r
            self.expected_r = ((1 - ALPHA) * self.expected_r
                               + ALPHA * score)

        # ── Layer 2: W(t) ─────────────────────────────────────────────
        win           = 1.0 if efficiency > 1.0 else 0.0
        alpha_ewma    = 2.0 / (EWMA_SPAN + 1)        # span=3 → α=0.5
        self.ewma_win = alpha_ewma * win + (1 - alpha_ewma) * self.ewma_win

        # ── Layer 3: P(t) ─────────────────────────────────────────────
        inst       = efficiency * self.ewma_win
        self.power = np.exp(-LAMBDA) * self.power + (1 - np.exp(-LAMBDA)) * inst

        self.efficiencies.append(efficiency)
        self.win_rates.append(self.ewma_win)
        self.powers.append(self.power)

        should_stop = (idx >= self.min_candidates - 1
                       and self.power < self.threshold)

        return VerifierUpdate(
            candidate_idx      = idx,
            score              = score,
            efficiency         = efficiency,
            win_rate           = self.ewma_win,
            power              = self.power,
            should_stop        = should_stop,
            best_score         = self.best_score,
            best_candidate_idx = self.best_idx,
        )


def generate_scores(regime: str, n: int, seed: int) -> np.ndarray:
    """
    Generate synthetic verifier expected-rank scores for N candidates.

    Scores represent expected rank output from LLM-as-a-Verifier
    log-prob ranking (higher = better candidate).

    Regimes (stylized — not empirically calibrated to LLM-as-a-Verifier):
      active:      Diverse pool — high variance, good candidates throughout
      convergence: Good candidates exhaust early, later pool is mediocre
      degradation: Verifier near capability limit, flat undiscriminating scores

    Note: Real validation requires Terminal-Bench 2.0 infrastructure.
    """
    rng = np.random.default_rng(seed)

    if regime == 'active':
        scores = rng.beta(2.5, 1.5, n)

    elif regime == 'convergence':
        n1     = n // 2
        part1  = rng.beta(3.0, 1.2, n1)
        part2  = rng.beta(1.5, 3.5, n - n1)
        scores = np.concatenate([part1, part2])

    elif regime == 'degradation':
        scores = rng.normal(0.50, 0.03, n).clip(0.40, 0.60)

    else:
        raise ValueError(f"Unknown regime '{regime}'. "
                         f"Choose: active, convergence, degradation")

    return scores


def run_simulation(threshold=0.5, n_candidates=250,
                   n_problems=500, seed=42) -> dict:
    """
    Simulate across n_problems (40% active / 40% convergence / 20% degradation).

    Quality = best_score_found / best_score_available.
    NOT task accuracy. See paper Section 4.
    """
    rng     = np.random.default_rng(seed)
    regimes = rng.choice(['active','convergence','degradation'],
                         size=n_problems, p=[0.4, 0.4, 0.2])
    results = dict(candidates_used=[], quality=[], regime=[],
                   stopped_early=[])

    for i, regime in enumerate(regimes):
        scores     = generate_scores(regime, n_candidates, seed + i)
        monitor    = VerifierHealthMonitor(threshold=threshold)
        stopped_at = n_candidates
        last_upd   = None

        for j, score in enumerate(scores):
            upd      = monitor.update(score)
            last_upd = upd
            if upd.should_stop:
                stopped_at = j + 1
                break

        true_best = scores.max()
        quality   = (min(last_upd.best_score / true_best, 1.0)
                     if true_best > 1e-10 else 1.0)

        results['candidates_used'].append(stopped_at)
        results['quality'].append(quality)
        results['regime'].append(regime)
        results['stopped_early'].append(stopped_at < n_candidates)

    return results


def print_summary(threshold, results):
    cands   = np.array(results['candidates_used'])
    quality = np.array(results['quality'])
    stopped = sum(results['stopped_early'])
    n       = len(cands)
    print(f"  θ={threshold}:  "
          f"reduction={(1-cands.mean()/250)*100:.1f}%  "
          f"quality={quality.mean():.4f}  "
          f"stopped={stopped}/{n}  "
          f"avg_candidates={cands.mean():.0f}")


def plot_results(results, threshold, save_path=None):
    fig = plt.figure(figsize=(12, 8), facecolor='#050810')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; BG='#050810'; PAN='#0d1117'

    cands   = np.array(results['candidates_used'])
    quality = np.array(results['quality'])
    regimes = np.array(results['regime'])

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    ax1 = fig.add_subplot(gs[0,0]); style(ax1)
    ax1.hist(cands, bins=40, color=C, alpha=0.8, edgecolor='none')
    ax1.axvline(250, color=CB, lw=1.5, ls='--', label='Budget (250)')
    ax1.axvline(cands.mean(), color=CA, lw=1.5, label=f'Mean: {cands.mean():.0f}')
    ax1.set_title('Candidates Used per Problem', color=C, fontsize=11)
    ax1.set_xlabel('Candidates', color='#888'); ax1.set_ylabel('Problems', color='#888')
    ax1.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    ax2 = fig.add_subplot(gs[0,1]); style(ax2)
    ax2.hist(quality, bins=30, color=CA, alpha=0.8, edgecolor='none')
    ax2.axvline(quality.mean(), color=C, lw=1.5, label=f'Mean: {quality.mean():.4f}')
    ax2.set_title('Search Quality vs Full Budget\n(best found / best available)',
                  color=C, fontsize=11)
    ax2.set_xlabel('Quality Score', color='#888'); ax2.set_ylabel('Problems', color='#888')
    ax2.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    ax3 = fig.add_subplot(gs[1,0]); style(ax3)
    reg_names = ['active','convergence','degradation']
    reg_reds  = [(1-cands[regimes==r].mean()/250)*100
                 if (regimes==r).sum()>0 else 0 for r in reg_names]
    bars = ax3.bar(reg_names, reg_reds, color=[C, CA, CB], alpha=0.8)
    ax3.set_title('Compute Reduction by Regime', color=C, fontsize=11)
    ax3.set_ylabel('Reduction (%)', color='#888')
    for b,v in zip(bars, reg_reds):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f'{v:.0f}%', ha='center', color=C, fontsize=9)

    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(PAN); ax4.axis('off')
    reduction = (1-cands.mean()/250)*100
    stats = [
        ('Threshold θ',       f'{threshold}'),
        ('Problems',          f'{len(cands)}'),
        ('Compute reduction', f'{reduction:.1f}%'),
        ('Avg candidates',    f'{cands.mean():.0f} / 250'),
        ('Avg quality',       f'{quality.mean():.4f}'),
        ('Min quality',       f'{quality.min():.4f}'),
        ('Stopped early',     f'{sum(results["stopped_early"])}/{len(cands)}'),
    ]
    y = 0.92
    ax4.text(0.05, y, 'Simulation Summary', color=C, fontsize=11,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for lbl, val in stats:
        ax4.text(0.05, y, lbl+':', color='#888', fontsize=9, transform=ax4.transAxes)
        ax4.text(0.65, y, val, color=C, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.12

    fig.suptitle(
        f'Verifier Health Monitoring — P(t) = E(t)×W(t)  |  θ={threshold}\n'
        f'α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}  |  '
        f'Quality = best found / best available (NOT task accuracy)',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · Stylized simulation · '
                'Validation on Terminal-Bench 2.0 required',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print("Verifier Health Monitor v3")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print("E(t): score / pre-update E[R] (no information leak)")
    print("Quality: best found / best available (NOT task accuracy)\n")

    for th in [0.3, 0.5, 0.7]:
        print_summary(th, run_simulation(threshold=th))

    print("\nGenerating charts (θ=0.5)...")
    plot_results(run_simulation(threshold=0.5), 0.5,
                 save_path='/mnt/user-data/outputs/paper17_simulation_v3.png')
    print("Done.")
