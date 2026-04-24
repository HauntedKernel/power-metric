"""
HPO Health Monitoring via Power Metric — Hyperband Slow Starter Recovery
=========================================================================
Paper 7: "Addressing Hyperband's False Positive and Slow Starter Failure
          Modes via Continuous Power Metric Health Scoring"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies P_i(t) = E_i(t) × W_i(t) as a continuous health signal for
each hyperparameter configuration during Hyperband training, identifying
slow starters that Hyperband would prematurely eliminate.

Real data:
  Pythia model suite (Biderman et al. 2023) used as HPO configurations.
  Each Pythia model (70M to 12B) = one hyperparameter configuration.
  Training checkpoints = successive halving rungs.
  Composite benchmark score = validation metric.

Why Pythia models as HPO configs:
  Each Pythia model has different architecture/scale hyperparameters.
  Their benchmark trajectories exhibit exactly the slow-starter behavior
  the paper addresses: large models rank poorly at early rungs but
  achieve superior final performance.

Hyperband simulation:
  eta=3 (standard), rungs at checkpoints 3, 6, 12 (of 16 post-warmup)
  Rung 1 (ckpt 3):  keep top 1/3 by score
  Rung 2 (ckpt 6):  keep top 1/3 of survivors
  Rung 3 (ckpt 12): keep top 1/3 of survivors

Power metric signal:
  P_i(t) monitors each config's improvement trajectory independently.
  A config with low absolute score but high P_i(t) (improving faster
  than its own adaptive expectation) is flagged as a slow starter
  worth retaining.

Related papers:
  Paper 1: power_metric_training.py (same dataset)
  Paper 3: power_metric_scaling.py (same dataset)
  Series:  https://github.com/HauntedKernel/power-metric
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Tuple

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3

# ── Hyperband parameters (Li et al. 2018) ─────────────────────────────────
ETA       = 3      # halving rate: keep top 1/eta at each rung
RUNGS     = [2, 5, 11]  # 0-indexed checkpoint indices (ckpts 3, 6, 12 of 16)

# ── Data ──────────────────────────────────────────────────────────────────
BENCHMARKS = ['lambada_openai','piqa','winogrande','arc_easy','arc_challenge']
SLOW_STARTER_THRESHOLD = 0.05  # improvement > 5% = slow starter


def load_pythia_curves(base_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load benchmark learning curves for all Pythia models.
    Each model = one HPO configuration.
    """
    configs = {}
    target_models = ['pythia-70m','pythia-160m','pythia-410m',
                     'pythia-1.4b','pythia-2.8b','pythia-6.9b','pythia-12b']

    for model in target_models:
        folder = os.path.join(base_path, model, 'zero-shot')
        if not os.path.exists(folder):
            continue
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
        if len(steps) >= 12:
            scores = np.array([step_scores[s] for s in steps])
            configs[model] = (np.array(steps), scores)

    return configs


def compute_power_metric(scores: np.ndarray) -> np.ndarray:
    """Compute P(t) on a learning curve. Pre-update baseline."""
    expected_r = None; ewma_win = 0.0; power = 0.0
    powers = []
    for s in scores:
        if expected_r is None:
            eff = 1.0; expected_r = max(s, 1e-6)
        else:
            eff = s / expected_r
            expected_r = (1 - ALPHA) * expected_r + ALPHA * s
        win = 1.0 if eff > 1.0 else 0.0
        a = 2.0 / (EWMA_SPAN + 1)
        ewma_win = a * win + (1 - a) * ewma_win
        inst = eff * ewma_win
        power = np.exp(-LAMBDA) * power + (1 - np.exp(-LAMBDA)) * inst
        powers.append(power)
    return np.array(powers)


def simulate_hyperband(
    configs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pm_threshold: float = 0.5,
) -> dict:
    """
    Simulate Hyperband + Power Metric on Pythia learning curves.

    Hyperband eliminates bottom 2/3 at each rung by absolute score.
    Power metric flags configs with P_i(t) > threshold for retention
    even if they rank low by absolute score (slow starters).
    """
    results = {}

    for name, (steps, scores) in configs.items():
        powers = compute_power_metric(scores)
        final_score = scores[-1]
        rung1_score = scores[RUNGS[0]]

        # Is this a slow starter?
        is_slow_starter = (final_score - rung1_score) > SLOW_STARTER_THRESHOLD

        # Hyperband decision at Rung 1
        # (Would be eliminated if score at rung 1 is in bottom 2/3)
        # We'll compute this relative to all configs after loading all
        pm_at_rung1 = powers[RUNGS[0]]
        pm_at_rung2 = powers[RUNGS[1]] if len(powers) > RUNGS[1] else powers[-1]

        results[name] = dict(
            steps          = steps,
            scores         = scores,
            powers         = powers,
            rung1_score    = rung1_score,
            final_score    = final_score,
            pm_at_rung1    = pm_at_rung1,
            pm_at_rung2    = pm_at_rung2,
            is_slow_starter = is_slow_starter,
            improvement    = final_score - rung1_score,
        )

    # Hyperband elimination decisions
    rung1_scores = {n: r['rung1_score'] for n, r in results.items()}
    sorted_by_rung1 = sorted(rung1_scores.items(), key=lambda x: x[1])
    n_configs = len(sorted_by_rung1)
    n_eliminate = int(n_configs * (1 - 1/ETA))
    eliminated_hb = set(n for n, _ in sorted_by_rung1[:n_eliminate])

    for name in results:
        r = results[name]
        eliminated = name in eliminated_hb

        # Power metric override: flag if P_i(rung1) > threshold
        # despite low absolute score
        pm_override = (eliminated and r['pm_at_rung1'] > pm_threshold)

        r['eliminated_by_hb'] = eliminated
        r['pm_override'] = pm_override
        r['correct_decision'] = not (eliminated and r['is_slow_starter'])

        # Outcome
        if not eliminated:
            r['outcome'] = 'Kept (HB + PM agree)'
        elif pm_override:
            r['outcome'] = 'Kept by PM override'
        elif r['is_slow_starter']:
            r['outcome'] = 'MISSED (slow starter eliminated)'
        else:
            r['outcome'] = 'Correctly eliminated'

    return results


def print_summary(results: dict, threshold: float):
    print(f"\nHPO Health Monitoring — Real Pythia Data (θ={threshold})")
    print(f"Treating each model as one HPO configuration")
    print(f"Hyperband eta={ETA}, rungs at checkpoints {[r+1 for r in RUNGS]}\n")

    print(f"{'Config':<16} {'Rung1':>8} {'Final':>8} {'Improve':>9} "
          f"{'SlowStart':>10} {'HB Elim':>8} {'PM@R1':>8} {'Override':>9} {'Outcome'}")
    print("-"*100)

    n_slow = 0; n_recovered = 0; n_missed = 0
    for name, r in sorted(results.items(), key=lambda x: x[1]['rung1_score']):
        if r['is_slow_starter']: n_slow += 1
        if r['pm_override']: n_recovered += 1
        if r['eliminated_by_hb'] and r['is_slow_starter'] and not r['pm_override']:
            n_missed += 1
        print(f"{name:<16} {r['rung1_score']:>8.4f} {r['final_score']:>8.4f} "
              f"{r['improvement']:>+9.4f} {str(r['is_slow_starter']):>10} "
              f"{str(r['eliminated_by_hb']):>8} {r['pm_at_rung1']:>8.4f} "
              f"{str(r['pm_override']):>9} {r['outcome']}")

    print(f"\nSummary:")
    print(f"  Total configs:        {len(results)}")
    print(f"  Slow starters:        {n_slow}")
    print(f"  HB would eliminate:   {sum(r['eliminated_by_hb'] for r in results.values())}")
    print(f"  PM recovers:          {n_recovered}")
    print(f"  Still missed:         {n_missed}")
    print(f"  PM threshold:         θ={threshold}")


def plot_results(results: dict, threshold: float, save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'; CY='#f1fa8c'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    # Color map by outcome
    outcome_colors = {
        'Kept (HB + PM agree)':          CG,
        'Kept by PM override':            C,
        'MISSED (slow starter eliminated)': CB,
        'Correctly eliminated':           CGRAY,
    }

    # Row 1-2: Learning curves per config (3 per row)
    config_names = sorted(results.keys(),
                         key=lambda x: results[x]['rung1_score'])

    for idx, name in enumerate(config_names):
        row = idx // 3; col = idx % 3
        if row > 1: break
        r   = results[name]
        ax  = fig.add_subplot(gs[row, col]); style(ax)
        color = outcome_colors[r['outcome']]

        x = range(len(r['scores']))
        ax.plot(x, r['scores'], color=color, lw=2, label='Score')

        # Mark rungs
        for rung_idx in RUNGS:
            if rung_idx < len(r['scores']):
                ax.axvline(rung_idx, color='#444', lw=1, ls=':')

        # Mark Rung 1 elimination point
        ax.scatter(RUNGS[0], r['rung1_score'],
                   color=CB if r['eliminated_by_hb'] else CG,
                   s=80, zorder=5)

        # P(t) on secondary axis
        ax2 = ax.twinx(); ax2.set_facecolor(PAN)
        ax2.plot(x, r['powers'], color=CA, lw=1.2, ls='--', alpha=0.7)
        ax2.axhline(threshold, color=CA, lw=0.8, ls=':', alpha=0.5)
        ax2.set_ylim(0, 1.1); ax2.tick_params(colors='#666666')

        short = name.replace('pythia-','')
        title = f"{short}\n{r['outcome'][:20]}"
        ax.set_title(title, color=color, fontsize=8)
        ax.set_xlabel('Checkpoint', color=CGRAY, fontsize=7)
        ax.set_ylabel('Score', color=CGRAY, fontsize=7)
        ax2.set_ylabel('P(t)', color=CA, fontsize=7)

    # Row 3: Summary charts
    ax3 = fig.add_subplot(gs[2, :2]); style(ax3)
    names = [n.replace('pythia-','') for n in config_names]
    rung1 = [results[n]['rung1_score'] for n in config_names]
    final = [results[n]['final_score']  for n in config_names]
    colors = [outcome_colors[results[n]['outcome']] for n in config_names]
    x_pos  = np.arange(len(names))

    ax3.bar(x_pos - 0.2, rung1, 0.35, color=CGRAY, alpha=0.6, label='Score @ Rung 1')
    ax3.bar(x_pos + 0.2, final, 0.35, color=colors, alpha=0.9, label='Final Score')
    ax3.axhline(np.percentile(rung1, 67), color=CB, lw=1.5, ls='--',
                label='HB elimination cutoff')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, fontsize=9)
    ax3.set_title('Rung 1 vs Final Score — Colored by Outcome', color=C, fontsize=10)
    ax3.set_ylabel('Composite Score', color=CGRAY)
    ax3.legend(fontsize=7, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=v, label=k[:25])
                  for k, v in outcome_colors.items()]
    ax3.legend(handles=legend_els, fontsize=7, labelcolor='white',
               facecolor=PAN, edgecolor='#333', loc='upper left')

    # Summary stats
    ax4 = fig.add_subplot(gs[2, 2]); ax4.set_facecolor(PAN); ax4.axis('off')
    n_slow  = sum(r['is_slow_starter']    for r in results.values())
    n_elim  = sum(r['eliminated_by_hb']   for r in results.values())
    n_rec   = sum(r['pm_override']         for r in results.values())
    n_miss  = sum(r['eliminated_by_hb'] and r['is_slow_starter'] and not r['pm_override']
                  for r in results.values())
    y = 0.96
    ax4.text(0.05, y, 'Results Summary', color=C, fontsize=10,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for lbl, val in [
        ('Configs (Pythia models)', str(len(results))),
        ('Slow starters',          str(n_slow)),
        ('HB eliminates',          str(n_elim)),
        ('PM recovers',            str(n_rec)),
        ('Still missed',           str(n_miss)),
        ('PM threshold θ',         str(threshold)),
        ('Data',                   'Real Pythia benchmarks'),
    ]:
        ax4.text(0.05, y, lbl+':', color=CGRAY, fontsize=8,
                 transform=ax4.transAxes)
        ax4.text(0.65, y, val, color=C, fontsize=8,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.11

    fig.suptitle(
        f'Hyperband + Power Metric HPO — θ={threshold}\n'
        f'Real Pythia learning curves · Each model = one HPO configuration · '
        f'Slow starters = improvement > {SLOW_STARTER_THRESHOLD*100:.0f}% after Rung 1',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · Paper 7 · '
                'Data: EleutherAI/pythia · '
                'github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./pythia-main/evals/pythia-v1')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    print("HPO Health Monitoring — Paper 7")
    print(f"Real Pythia data: {args.data}")
    print(f"Each Pythia model = one HPO configuration\n")

    configs = load_pythia_curves(args.data)
    print(f"Loaded {len(configs)} configurations: {list(configs.keys())}\n")

    for threshold in [0.3, 0.5, 0.7]:
        results = simulate_hyperband(configs, threshold)
        print_summary(results, threshold)

    print("\nGenerating charts...")
    results = simulate_hyperband(configs, args.threshold)
    plot_results(results, args.threshold,
                 save_path='/mnt/user-data/outputs/paper7_simulation.png')
    print("Done.")
