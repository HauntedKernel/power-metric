"""
Adaptive Speculative Decoding via Power Metric
================================================
Paper 10: "Dynamic Draft Length in Speculative Decoding via
           Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies P(t) = E(t) × W(t) to adapt the draft length K in
speculative decoding based on the current token acceptance rate (TAR).

Speculative decoding background (Leviathan et al. 2023):
  A small draft model proposes K tokens. A large target model verifies
  all K in one parallel forward pass. Tokens are accepted greedily
  until the first rejection; the rejected token is corrected.
  Speedup depends critically on TAR — high TAR = more accepted tokens
  per verification step = higher throughput.

The problem:
  Fixed K is suboptimal because TAR varies dramatically across contexts:
  - Easy contexts (code, repetitive text): TAR ~0.85
  - Hard contexts (reasoning, creative): TAR ~0.42
  - Mixed contexts: TAR ~0.65
  A fixed K that is optimal for easy phases wastes draft compute on hard
  phases, and one optimized for hard phases under-speculates on easy ones.

The solution:
  Use P(t) to track TAR trajectory. When P(t) is high (TAR above
  adaptive expected), increase K. When P(t) is low, decrease K.
  K = clip(round(K_min + (K_max - K_min) * P(t)), K_min, K_max)

Signal: token acceptance (1=accepted, 0=rejected)
  TAR is treated as the efficiency signal — E(t) = TAR / E[R](t-1)
  W(t) = EWMA of [E(t) > 1.0], span=8 (wider window for token-level)
  λ=0.3 (longer memory than run-level papers)
  These differ from run-level parameters — token-level dynamics are faster.

Speedup calculation:
  Each verification step: accept n tokens (Bernoulli draws), then 1
  target token. Cost = 1 (verification) + K * c (drafting), c=0.1.
  Speedup = (avg_tokens_per_step) / (1 + avg_K * c)
  Compared to autoregressive = 1 token/step at cost 1.

Calibration:
  Four context phases (500 tokens total), TAR calibrated to published
  speculative decoding results (Leviathan et al. 2023):
  - Easy phases:  TAR ~0.85 (code completion, template filling)
  - Hard phase:   TAR ~0.42 (mathematical reasoning, open-ended)
  - Mixed phase:  TAR ~0.65 (general instruction following)

  This is a stylized simulation. Real TAR depends on model pair,
  task, temperature, and token position. Validation requires running
  actual speculative decoding on a real model pair.

Related papers:
  Paper 5: LIF identity (same exponential decay kernel)
  Paper 14: https://zenodo.org/records/19685841
  Series:   https://github.com/HauntedKernel/power-metric
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict

# ── Parameters ────────────────────────────────────────────────────────────
# Token-level parameters differ from run-level (Papers 1-4, 7, 9)
# because token-level dynamics are faster
ALPHA     = 0.3   # same as series
LAMBDA    = 0.3   # longer memory for token-level (vs 0.5 in run-level)
EWMA_SPAN = 8     # wider window for token-level (vs 3 in run-level)

K_MIN = 2         # minimum draft length
K_MAX = 8         # maximum draft length
DRAFT_COST = 0.1  # draft model cost as fraction of target model cost

# ── Context phases calibrated to published TAR ranges ─────────────────────
PHASES = [
    ('Easy Phase 1',  0.85, 125),  # code/repetitive contexts
    ('Hard Phase',    0.42, 125),  # reasoning/creative
    ('Mixed Phase',   0.65, 125),  # general instruction
    ('Easy Phase 2',  0.85, 125),  # code/repetitive contexts
]
N_TOKENS = sum(n for _, _, n in PHASES)  # 500 total


def compute_pm_update(
    tar_obs: float,
    state: dict,
) -> float:
    """Update P(t) with one TAR observation. Returns new P(t)."""
    er = state['er']
    if er is None:
        eff = 1.0
        state['er'] = max(tar_obs, 1e-6)
    else:
        eff = tar_obs / er
        state['er'] = (1 - ALPHA) * er + ALPHA * tar_obs

    win      = 1.0 if eff > 1.0 else 0.0
    a        = 2.0 / (EWMA_SPAN + 1)
    state['ew'] = a * win + (1 - a) * state['ew']
    inst     = eff * state['ew']
    state['pw'] = (np.exp(-LAMBDA) * state['pw']
                   + (1 - np.exp(-LAMBDA)) * inst)
    return state['pw']


def adaptive_k(power: float) -> int:
    """Compute draft length K from P(t)."""
    return int(np.clip(round(K_MIN + (K_MAX - K_MIN) * power), K_MIN, K_MAX))


def simulate_fixed_k(k: int, phases: list, seed: int = 42) -> dict:
    """Simulate speculative decoding with fixed draft length K."""
    rng = np.random.default_rng(seed)
    steps = 0; n_accepted = 0; n_wasted = 0; tokens_gen = 0

    for phase_name, tar, n_phase in phases:
        for _ in range(n_phase):
            # Sample accepted tokens (stop at first rejection)
            accepted = 0
            for _ in range(k):
                if rng.random() < np.clip(tar + rng.normal(0, 0.04), 0.05, 0.99):
                    accepted += 1
                else:
                    break
            n_accepted += accepted + 1  # +1 target token
            n_wasted   += k - accepted
            steps += 1
            tokens_gen += accepted + 1

    avg_tokens_per_step = n_accepted / steps
    speedup = avg_tokens_per_step / (1 + k * DRAFT_COST)

    return dict(
        k=k, steps=steps, speedup=speedup,
        wasted=n_wasted, avg_tokens_per_step=avg_tokens_per_step,
    )


def simulate_adaptive_k(phases: list, seed: int = 42) -> dict:
    """Simulate speculative decoding with adaptive K via P(t)."""
    rng    = np.random.default_rng(seed)
    state  = {'er': None, 'ew': 0.0, 'pw': 0.0}
    steps  = 0; n_accepted = 0; n_wasted = 0; tokens_gen = 0
    all_k  = []; all_power = []; all_tar = []
    phase_k = {p[0]: [] for p in phases}

    for phase_name, tar, n_phase in phases:
        for _ in range(n_phase):
            tar_obs = np.clip(tar + rng.normal(0, 0.04), 0.05, 0.99)
            power   = compute_pm_update(tar_obs, state)
            k       = adaptive_k(power)

            all_k.append(k)
            all_power.append(power)
            all_tar.append(tar_obs)
            phase_k[phase_name].append(k)

            accepted = 0
            for _ in range(k):
                if rng.random() < tar_obs:
                    accepted += 1
                else:
                    break

            n_accepted += accepted + 1
            n_wasted   += k - accepted
            steps += 1
            tokens_gen += accepted + 1

    avg_k = np.mean(all_k)
    avg_tokens_per_step = n_accepted / steps
    speedup = avg_tokens_per_step / (1 + avg_k * DRAFT_COST)

    return dict(
        steps=steps, speedup=speedup, wasted=n_wasted,
        avg_k=avg_k, avg_tokens_per_step=avg_tokens_per_step,
        all_k=all_k, all_power=all_power, all_tar=all_tar,
        phase_k=phase_k,
    )


def print_summary(fixed_results: List[dict], adaptive: dict):
    print("Speculative Decoding — Adaptive K via P(t)")
    print(f"500 tokens, 4 phases, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print(f"Calibrated to Leviathan et al. (2023) TAR ranges\n")
    print(f"{'Method':<28} {'Steps':>7} {'Speedup':>9} "
          f"{'Wasted':>8} {'Avg K':>7}")
    print("-"*62)
    for r in fixed_results:
        print(f"{'Fixed K=' + str(r['k']):<28} {r['steps']:>7} "
              f"{r['speedup']:>8.2f}x {r['wasted']:>8} {'K='+str(r['k']):>7}")
    print(f"{'Adaptive K via P(t)':<28} {adaptive['steps']:>7} "
          f"{adaptive['speedup']:>8.2f}x {adaptive['wasted']:>8} "
          f"{adaptive['avg_k']:>7.1f}")

    print(f"\nPhase-level K allocation (adaptive):")
    for phase_name, tar, _ in PHASES:
        k_arr = np.array(adaptive['phase_k'][phase_name])
        print(f"  {phase_name:<20} TAR={tar:.2f}  avg K={k_arr.mean():.1f}  "
              f"range=[{k_arr.min()},{k_arr.max()}]")


def plot_results(fixed_results: List[dict], adaptive: dict,
                 save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    x = range(len(adaptive['all_k']))

    # Chart 1: Adaptive K over time
    ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
    ax1.plot(x, adaptive['all_k'], color=C, lw=1.5, label='Adaptive K')
    ax1.axhline(5, color=CB, lw=1, ls='--', label='Fixed K=5')
    # Shade phases
    pos = 0
    phase_colors = [CG, CB, CA, CG]
    for i, (phase_name, tar, n) in enumerate(PHASES):
        ax1.axvspan(pos, pos+n, alpha=0.08, color=phase_colors[i])
        ax1.text(pos + n/2, K_MAX + 0.2, f'TAR={tar}',
                 ha='center', color=phase_colors[i], fontsize=7)
        pos += n
    ax1.set_title('Adaptive K(t) vs Fixed K=5', color=C, fontsize=10)
    ax1.set_xlabel('Token step', color=CGRAY)
    ax1.set_ylabel('Draft length K', color=CGRAY)
    ax1.set_ylim(0, K_MAX + 1)
    ax1.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 2: P(t) and TAR signal
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    ax2.plot(x, adaptive['all_tar'], color=CGRAY, lw=1, alpha=0.5,
             label='TAR (observed)')
    ax2.plot(x, adaptive['all_power'], color=CA, lw=2, label='P(t)')
    pos = 0
    for i, (_, tar, n) in enumerate(PHASES):
        ax2.axvspan(pos, pos+n, alpha=0.05, color=phase_colors[i])
        pos += n
    ax2.set_title('P(t) and Observed TAR', color=C, fontsize=10)
    ax2.set_xlabel('Token step', color=CGRAY)
    ax2.set_ylabel('Value', color=CGRAY)
    ax2.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 3: Speedup vs waste tradeoff
    ax3 = fig.add_subplot(gs[1, 0]); style(ax3)
    for r in fixed_results:
        color = {2: CGRAY, 5: CB, 8: CG}[r['k']]
        ax3.scatter(r['wasted'], r['speedup'], color=color, s=100, zorder=5,
                    label=f'Fixed K={r["k"]}')
    ax3.scatter(adaptive['wasted'], adaptive['speedup'],
                color=C, s=150, marker='*', zorder=6, label='Adaptive K')
    ax3.set_title('Speedup vs Waste Pareto', color=C, fontsize=10)
    ax3.set_xlabel('Wasted draft tokens (lower=better)', color=CGRAY)
    ax3.set_ylabel('Speedup vs autoregressive', color=CGRAY)
    ax3.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 4: Per-phase K distribution
    ax4 = fig.add_subplot(gs[1, 1]); style(ax4)
    phase_labels = [f'{p[0][:8]}\nTAR={p[1]}' for p in PHASES]
    phase_means  = [np.mean(adaptive['phase_k'][p[0]]) for p in PHASES]
    phase_stds   = [np.std(adaptive['phase_k'][p[0]])  for p in PHASES]
    bars = ax4.bar(range(len(PHASES)), phase_means,
                   color=phase_colors, alpha=0.85)
    ax4.errorbar(range(len(PHASES)), phase_means, yerr=phase_stds,
                 fmt='none', color='white', capsize=4)
    ax4.axhline(5, color=CB, lw=1, ls='--', label='Fixed K=5')
    ax4.set_xticks(range(len(PHASES)))
    ax4.set_xticklabels(phase_labels, fontsize=7)
    ax4.set_title('Avg Adaptive K by Phase', color=C, fontsize=10)
    ax4.set_ylabel('K', color=CGRAY)
    ax4.set_ylim(0, K_MAX + 1)
    ax4.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')
    for b, v in zip(bars, phase_means):
        ax4.text(b.get_x()+b.get_width()/2, v+0.1, f'{v:.1f}',
                 ha='center', color=C, fontsize=9, fontweight='bold')

    # Chart 5: Summary
    ax5 = fig.add_subplot(gs[2, :]); style(ax5)
    methods = [f'Fixed K={r["k"]}' for r in fixed_results] + ['Adaptive K']
    speedups = [r['speedup'] for r in fixed_results] + [adaptive['speedup']]
    wastes   = [r['wasted']  for r in fixed_results] + [adaptive['wasted']]
    colors_bar = [CGRAY, CB, CG, C]
    x_pos = np.arange(len(methods))
    ax5_twin = ax5.twinx(); ax5_twin.set_facecolor(PAN)
    bars1 = ax5.bar(x_pos - 0.2, speedups, 0.35, color=colors_bar, alpha=0.85)
    bars2 = ax5_twin.bar(x_pos + 0.2, wastes, 0.35,
                         color=colors_bar, alpha=0.4)
    ax5.set_xticks(x_pos); ax5.set_xticklabels(methods, fontsize=9)
    ax5.set_title('Speedup (solid) and Waste (transparent) by Method',
                  color=C, fontsize=10)
    ax5.set_ylabel('Speedup vs autoregressive', color=CGRAY)
    ax5_twin.set_ylabel('Wasted draft tokens', color=CGRAY)
    ax5_twin.tick_params(colors='#666666')
    for b, v in zip(bars1, speedups):
        ax5.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}x',
                 ha='center', color=C, fontsize=8, fontweight='bold')

    fig.suptitle(
        'Adaptive Speculative Decoding via P(t) = E(t)×W(t)\n'
        'Stylized simulation · TAR calibrated to Leviathan et al. (2023) · '
        'Validation requires real model pair',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                f'λ={LAMBDA}, EWMA span={EWMA_SPAN} · '
                'Cantrell (2026) · Paper 10 · '
                'github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    print("Adaptive Speculative Decoding — Paper 10\n")

    fixed = [simulate_fixed_k(k, PHASES) for k in [2, 5, 8]]
    adaptive = simulate_adaptive_k(PHASES)
    print_summary(fixed, adaptive)

    print("\nGenerating charts...")
    plot_results(fixed, adaptive,
                 save_path='/mnt/user-data/outputs/paper10_simulation.png')
    print("Done.")
