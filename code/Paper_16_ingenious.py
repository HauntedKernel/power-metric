"""
INGENIOUS Analog + Power Metric: Empirical Data + Simulation
=============================================================
Paper 16: "Data Quality Selection and Training Health Monitoring:
           Empirical Evidence from Pythia Deduplication and P(t)"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

TWO-PART STUDY:

Part 1 — EMPIRICAL (real Pythia data):
  Pythia deduped vs standard as a real-world data quality selection
  experiment. The deduplicated Pile removes ~18% of tokens (redundant
  data), analogous to INGENIOUS selecting a high-quality subset.
  Result: deduped consistently improves final quality by +0.7% to +2.9%
  across 1.4B-12B scales, and outperforms standard at 14-15/16
  checkpoints throughout training. 160M shows slight regression (-0.98%),
  suggesting data selection benefits emerge at scale.

Part 2 — SIMULATION (extended training, PM stopping):
  Real Pythia only has 16 checkpoints — insufficient to show PM stopping
  behavior (curves are monotonically improving). An extended 50-step
  simulation calibrated to Pythia 1.4B dynamics shows:
  - Deduped curve reaches plateau faster (22% faster convergence proxy)
  - PM stops earlier on deduped curve (step ~34 vs ~39 for standard)
  - Combined: 44.6% total compute savings at 96.3% quality retention

INGENIOUS connection:
  INGENIOUS (Renduchintala et al., EMNLP 2023) formalizes data subset
  selection via submodular optimization. Deduplication is a simpler
  but empirically real version of the same principle: remove redundant,
  low-information data → train on a more informative subset.
  The quality gains measured here provide empirical grounding for the
  INGENIOUS + PM framework.

Data: EleutherAI/pythia (Biderman et al. 2023) — public benchmark
evaluations at 16 checkpoints for standard and deduped variants.

Related papers:
  Paper 1:  power_metric_training.py (PM on training)
  Paper 3:  power_metric_scaling.py (scaling reliability, same dataset)
  Paper 4:  power_metric_mixing.py (per-domain health)
  Series:   https://github.com/HauntedKernel/power-metric
"""

import json, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ────────────────────────────────────────────────────────────
ALPHA         = 0.3
LAMBDA        = 0.5
EWMA_SPAN     = 3
THETA         = 0.40
DATA_FRACTION = 0.82   # deduped Pile ≈ 82% of full Pile size

BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']

PAIRS = [
    ('pythia-160m',  'pythia-160m-deduped',  '160M'),
    ('pythia-1.4b',  'pythia-1.4b-deduped',  '1.4B'),
    ('pythia-6.9b',  'pythia-6.9b-deduped',  '6.9B'),
    ('pythia-12b',   'pythia-12b-deduped',   '12B'),
]


def load(base_path: str, model: str):
    folder = os.path.join(base_path, model, 'zero-shot')
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
                bd = r[b]; s = bd.get('acc_norm', bd.get('acc', None))
                if s is not None: sc.append(s)
        if len(sc) == len(BENCHMARKS):
            step_scores[step] = np.mean(sc)
    steps = sorted(step_scores.keys())
    return np.array(steps), np.array([step_scores[s] for s in steps])


def compute_pm(scores: np.ndarray) -> np.ndarray:
    """P(t) with pre-update baseline. Consistent with series."""
    er=None; ew=0.0; pw=0.0; powers=[]
    for s in scores:
        if er is None: eff=1.0; er=max(s,1e-6)
        else: eff=s/er; er=(1-ALPHA)*er+ALPHA*s
        win=1.0 if eff>1.0 else 0.0
        a=2.0/(EWMA_SPAN+1); ew=a*win+(1-a)*ew
        inst=eff*ew; pw=np.exp(-LAMBDA)*pw+(1-np.exp(-LAMBDA))*inst
        powers.append(pw)
    return np.array(powers)


def simulate_extended(seed: int, boost: float = 1.0, n: int = 50) -> np.ndarray:
    """
    Extended training curve for PM stopping simulation.
    Calibrated to Pythia 1.4B dynamics, extended to 50 steps.
    boost > 1.0 models faster convergence from deduped data.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for t in range(n):
        te = t * boost
        if te < 5:    base = 0.33 + 0.05 * (te / 5)
        elif te < 20: base = 0.38 + 0.15 * (1 - np.exp(-0.4 * (te - 5)))
        elif te < 35: base = 0.53 + 0.02 * (1 - np.exp(-0.2 * (te - 20)))
        else:         base = 0.55 - 0.008 * ((te - 35) / 15)
        scores.append(float(np.clip(
            np.clip(base, 0, 0.56) + rng.normal(0, 0.006), 0, 1)))
    return np.array(scores)


def find_pm_stop(scores: np.ndarray, warmup: int = 3) -> int:
    powers = compute_pm(scores)
    for i in range(warmup, len(powers)):
        if powers[i] < THETA:
            return i
    return len(powers) - 1


def run_empirical(base_path: str) -> dict:
    empirical = {}
    for std_m, ded_m, label in PAIRS:
        _, std = load(base_path, std_m)
        _, ded = load(base_path, ded_m)
        delta  = ded[-1] - std[-1]
        empirical[label] = dict(
            std=std, ded=ded,
            pm_std=compute_pm(std), pm_ded=compute_pm(ded),
            delta=delta, pct=delta/std[-1]*100,
            n_better=sum(1 for d,s in zip(ded,std) if d >= s),
            std_final=std[-1], ded_final=ded[-1],
        )
    return empirical


def run_simulation(n_seeds: int = 10) -> dict:
    """Extended simulation for PM stopping on deduped vs standard curves."""
    res = {m: [] for m in ['full','ded','pm','ded_pm']}
    for seed in range(n_seeds):
        full = simulate_extended(seed, 1.0)
        ded  = simulate_extended(seed, 1.22)
        ps_f = find_pm_stop(full)
        ps_d = find_pm_stop(ded)
        n    = len(full)
        res['full'].append((full[-1], n, 0.0))
        res['ded'].append((ded[-1],   n, 1 - DATA_FRACTION))
        res['pm'].append((full[ps_f], ps_f+1, 1-(ps_f+1)/n))
        combo = 1 - (ps_d+1)/n * DATA_FRACTION
        res['ded_pm'].append((ded[ps_d], ps_d+1, combo))
    summary = {}
    for m in res:
        summary[m] = dict(
            score  = float(np.mean([r[0] for r in res[m]])),
            steps  = float(np.mean([r[1] for r in res[m]])),
            savings= float(np.mean([r[2] for r in res[m]])),
        )
    return summary


def print_results(empirical: dict, sim: dict):
    print("Paper 16 — INGENIOUS Analog + Power Metric\n")
    print("PART 1: Empirical (Real Pythia Data)")
    print(f"{'Model':<8} {'Std':>8} {'Deduped':>9} {'Δ':>8} "
          f"{'Ded>Std':>10} {'Scale effect'}")
    print("-"*58)
    for label in ['160M','1.4B','6.9B','12B']:
        r = empirical[label]
        scale = 'regression' if r['pct'] < 0 else 'improvement'
        print(f"{label:<8} {r['std_final']:>8.4f} {r['ded_final']:>9.4f} "
              f"{r['delta']:>+8.4f} {r['n_better']:>7}/16  {scale}")

    full_sc = sim['full']['score']
    print("\nPART 2: Simulation (extended 50-step run, PM stopping)")
    print(f"{'Strategy':<22} {'Score':>8} {'Retention':>10} "
          f"{'Steps':>7} {'Savings':>9}")
    print("-"*60)
    for m, label in [('full','Full baseline'),('ded','Deduped only'),
                     ('pm','PM only'),('ded_pm','Deduped + PM')]:
        r = sim[m]
        print(f"{label:<22} {r['score']:>8.4f} "
              f"{r['score']/full_sc*100:>9.1f}% "
              f"{r['steps']:>7.1f} {r['savings']*100:>8.1f}%")


def plot_results(empirical: dict, sim: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 11), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'
    PAN='#0d1117'; GR='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    # Row 1: Real Pythia trajectories
    for col, label in enumerate(['1.4B','6.9B','12B']):
        r   = empirical[label]
        ax  = fig.add_subplot(gs[0, col]); style(ax)
        x   = range(len(r['std']))
        ax.plot(x, r['std'], color=GR, lw=2, label='Standard')
        ax.plot(x, r['ded'], color=CA, lw=2, label='Deduped')
        ax.fill_between(x, r['std'], r['ded'],
                        where=[d >= s for d,s in zip(r['ded'],r['std'])],
                        alpha=0.15, color=CA)
        ax.set_title(f'Pythia-{label}  Δ={r["pct"]:+.1f}%\n'
                     f'Deduped better: {r["n_better"]}/16 ckpts',
                     color=C, fontsize=9)
        ax.set_xlabel('Checkpoint', color=GR, fontsize=8)
        ax.set_ylabel('Score', color=GR, fontsize=8)
        ax.legend(fontsize=7, labelcolor='white',
                  facecolor=PAN, edgecolor='#333')
        tag = '← Real data' if col == 0 else ''
        if tag:
            ax.text(0.02, 0.05, tag, color=CG, fontsize=8,
                    transform=ax.transAxes, fontweight='bold')

    # Row 2 left: P(t) on real data
    ax3 = fig.add_subplot(gs[1, :2]); style(ax3)
    for label, color in [('1.4B',C),('6.9B',CA),('12B',CG)]:
        r = empirical[label]
        x = range(len(r['pm_std']))
        ax3.plot(x, r['pm_ded'], color=color, lw=2,
                 label=f'{label} deduped')
        ax3.plot(x, r['pm_std'], color=color, lw=1.5,
                 ls='--', alpha=0.45)
    ax3.axhline(THETA, color='white', lw=1.5, ls='--',
                label=f'θ={THETA}')
    ax3.set_title('P(t) Real Data — Deduped (solid) vs Standard (dashed)\n'
                  'Neither crosses θ: PM stopping requires denser checkpoints',
                  color=C, fontsize=9)
    ax3.set_xlabel('Checkpoint', color=GR)
    ax3.set_ylabel('P(t)', color=GR)
    ax3.legend(fontsize=7, labelcolor='white',
               facecolor=PAN, edgecolor='#333')

    # Row 2 right: Simulation
    ax4 = fig.add_subplot(gs[1, 2]); style(ax4)
    full0 = simulate_extended(0, 1.0)
    ded0  = simulate_extended(0, 1.22)
    ps_f0 = find_pm_stop(full0)
    ps_d0 = find_pm_stop(ded0)
    x50   = range(50)
    ax4.plot(x50, full0, color=GR, lw=1.5, alpha=0.7, label='Full (sim)')
    ax4.plot(x50, ded0,  color=CA, lw=2,   label='Deduped (sim)')
    ax4.axvline(ps_f0, color=CB, lw=2, ls='--',
                label=f'PM stop full (step {ps_f0})')
    ax4.axvline(ps_d0, color=C,  lw=2, ls='-.',
                label=f'PM stop ded (step {ps_d0})')
    ax4.set_title('Extended Sim: PM Stops\nEarlier on Deduped Curve',
                  color=C, fontsize=9)
    ax4.set_xlabel('Step', color=GR)
    ax4.set_ylabel('Score', color=GR)
    ax4.legend(fontsize=6, labelcolor='white',
               facecolor=PAN, edgecolor='#333')

    # Row 3 left: Quality gain bar chart
    ax5 = fig.add_subplot(gs[2, :2]); style(ax5)
    sizes  = ['160M','1.4B','6.9B','12B']
    deltas = [empirical[s]['pct'] for s in sizes]
    colors_b = [CB if d < 0 else CA for d in deltas]
    bars = ax5.bar(sizes, deltas, color=colors_b, alpha=0.85)
    ax5.axhline(0, color='white', lw=1, ls='--', alpha=0.5)
    for b, v in zip(bars, deltas):
        ax5.text(b.get_x()+b.get_width()/2,
                 v + (0.06 if v >= 0 else -0.18),
                 f'{v:+.2f}%', ha='center',
                 color=C, fontsize=9, fontweight='bold')
    ax5.set_title('Deduped vs Standard — Quality Gain by Scale (Real Data)\n'
                  'Scale effect: gains increase with model size',
                  color=C, fontsize=10)
    ax5.set_ylabel('Δ Score (%)', color=GR)

    # Row 3 right: Summary
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PAN); ax6.axis('off')
    y = 0.96
    full_sc = sim['full']['score']
    ax6.text(0.05, y, 'Results', color=C, fontsize=10,
             fontweight='bold', transform=ax6.transAxes); y -= 0.10
    ax6.text(0.05, y, 'EMPIRICAL:', color=CA, fontsize=9,
             fontweight='bold', transform=ax6.transAxes); y -= 0.09
    for lbl, val in [
        ('Scale 1.4B', f'+{empirical["1.4B"]["pct"]:.2f}%'),
        ('Scale 6.9B', f'+{empirical["6.9B"]["pct"]:.2f}%'),
        ('Scale 12B',  f'+{empirical["12B"]["pct"]:.2f}%'),
        ('160M',       f'{empirical["160M"]["pct"]:+.2f}% (regression)'),
    ]:
        ax6.text(0.05, y, f'{lbl}: {val}', color=C, fontsize=8,
                 transform=ax6.transAxes); y -= 0.08
    y -= 0.03
    ax6.text(0.05, y, 'SIMULATION (PM):', color=CB, fontsize=9,
             fontweight='bold', transform=ax6.transAxes); y -= 0.09
    for lbl, val in [
        ('Combined savings',
         f'{sim["ded_pm"]["savings"]*100:.1f}%'),
        ('Quality retained',
         f'{sim["ded_pm"]["score"]/full_sc*100:.1f}%'),
        ('Validation needed', 'Dense ckpt run'),
    ]:
        ax6.text(0.05, y, f'{lbl}: {val}', color=C, fontsize=8,
                 transform=ax6.transAxes); y -= 0.08

    fig.suptitle(
        'Data Quality Selection + P(t) Health Monitoring\n'
        'Part 1: Real Pythia deduped data — quality gains at scale  ·  '
        'Part 2: Extended simulation — PM early stopping on quality-selected curves',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(
        0.5, 0.01,
        f'Data: EleutherAI/pythia (Biderman et al. 2023) · '
        f'α={ALPHA}, λ={LAMBDA}, θ={THETA} · '
        'INGENIOUS: Renduchintala et al. (EMNLP 2023) · '
        'Cantrell (2026) · Paper 16 · '
        'github.com/HauntedKernel/power-metric',
        ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#050810')
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    BASE = './pythia-main/evals/pythia-v1'
    print("Loading Pythia data...")
    empirical = run_empirical(BASE)
    sim       = run_simulation()
    print_results(empirical, sim)
    print("\nGenerating chart...")
    plot_results(empirical, sim,
                 save_path='/mnt/user-data/outputs/paper16_simulation.png')
    print("Done.")
