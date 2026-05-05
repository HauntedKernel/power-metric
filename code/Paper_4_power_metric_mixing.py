"""
Data Domain Health Monitoring via Per-Domain Power Metrics
===========================================================
Paper 4: "Per-Domain Health Monitoring for Data Mixing via
          Stochastic Power Metrics"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies per-domain P_i(t) = E_i(t) × W_i(t) to score the health
of each data domain during training, using Pythia benchmark tasks
as domain proxies.

The problem:
  Global training signals (loss, accuracy) mask per-domain health.
  A model may be improving on language tasks while plateauing on
  reasoning — a global signal cannot distinguish these. Aioli (Chen
  et al. 2024) addresses this via mixing law estimation but requires
  paired training runs. The power metric provides a complementary
  single-run signal.

The approach:
  Each benchmark task is treated as a domain proxy:
    lambada_openai → Language/Web
    arc_easy       → Science/Facts
    piqa           → Physical Comm.
    arc_challenge  → Reasoning
    winogrande     → Social Comm.

  P_i(t) is computed independently for each domain using the same
  three-layer framework as all other papers in this series.

  Dynamic mixing proportions are derived as softmax of P_i(t):
    p_i(t) = softmax(P_i(t))

  This is a signal analysis — proportions indicate where the
  health signal would increase allocation, not a controlled
  mixing experiment. Validation requires access to per-domain
  perplexity from actual multi-domain pretraining data.

Important caveat:
  Benchmark accuracy is a proxy for domain health, not a direct
  measurement. Per-domain perplexity from actual training data
  would provide a more direct signal. These results should be
  interpreted as preliminary evidence of per-domain differentiation.

Data: EleutherAI/pythia GitHub (public)
  https://github.com/EleutherAI/pythia

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
from scipy.stats import pearsonr
from typing import Dict, List, Tuple

# ── Parameters (consistent across all papers in series) ───────────────────
ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3

# ── Domain proxies ────────────────────────────────────────────────────────
DOMAINS = {
    'lambada_openai': 'Language/Web',
    'arc_easy':       'Science/Facts',
    'piqa':           'Physical Comm.',
    'arc_challenge':  'Reasoning',
    'winogrande':     'Social Comm.',
}
MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']


def load_per_domain(base_path: str, model: str) -> Tuple[List[int], Dict]:
    """Load per-benchmark scores from real Pythia evaluation files."""
    folder = os.path.join(base_path, model, 'zero-shot')
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Pythia eval folder not found: {folder}\n"
            f"Download from: https://github.com/EleutherAI/pythia"
        )
    step_domain = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        r = data.get('results', {})
        row = {}
        for bench in DOMAINS:
            if bench in r:
                bd = r[bench]
                s  = bd.get('acc_norm', bd.get('acc', None))
                if s is not None: row[bench] = s
        if len(row) == len(DOMAINS):
            step_domain[step] = row
    steps = sorted(step_domain.keys())
    return steps, step_domain


def compute_per_domain_power(
    steps: List[int],
    step_domain: Dict,
) -> Dict[str, List[float]]:
    """
    Compute P_i(t) independently for each domain.

    Each domain's health signal uses the same three-layer computation
    as all other papers — E(t) pre-update, W(t) EWMA, P(t) decay.
    """
    state = {b: {'er': None, 'ew': 0.0, 'pw': 0.0} for b in DOMAINS}
    all_powers = {b: [] for b in DOMAINS}

    for step in steps:
        scores = step_domain[step]
        for bench in DOMAINS:
            s  = scores[bench]
            st = state[bench]
            if st['er'] is None:
                eff = 1.0; st['er'] = max(s, 1e-6)
            else:
                eff = s / st['er']
                st['er'] = (1 - ALPHA) * st['er'] + ALPHA * s
            win      = 1.0 if eff > 1.0 else 0.0
            a        = 2.0 / (EWMA_SPAN + 1)
            st['ew'] = a * win + (1 - a) * st['ew']
            inst     = eff * st['ew']
            st['pw'] = np.exp(-LAMBDA) * st['pw'] + (1 - np.exp(-LAMBDA)) * inst
            all_powers[bench].append(st['pw'])

    return all_powers


def softmax_proportions(p_values: Dict[str, float]) -> Dict[str, float]:
    """Convert P_i values to mixing proportions via softmax."""
    vals   = np.array([p_values[b] for b in DOMAINS])
    exp_v  = np.exp(vals - vals.max())
    props  = exp_v / exp_v.sum()
    return dict(zip(DOMAINS.keys(), props))


def run_analysis(base_path: str) -> dict:
    """Run full per-domain analysis across all three Pythia models."""
    results = {}

    for model in MODELS:
        steps, step_domain = load_per_domain(base_path, model)
        all_powers         = compute_per_domain_power(steps, step_domain)

        # Final state
        final_p = {b: all_powers[b][-1]          for b in DOMAINS}
        final_s = {b: step_domain[steps[-1]][b]   for b in DOMAINS}
        props   = softmax_proportions(final_p)

        # Uniform baseline
        uniform = {b: 1.0 / len(DOMAINS) for b in DOMAINS}

        # Dominant domain
        dominant_bench = max(props, key=props.get)
        dominant       = DOMAINS[dominant_bench]

        # Correlation: P_i vs final benchmark score
        p_list = [final_p[b] for b in DOMAINS]
        s_list = [final_s[b] for b in DOMAINS]
        try:
            r_val, _ = pearsonr(p_list, s_list)
        except Exception:
            r_val = float('nan')

        # Proportion shift from uniform
        prop_shift = {b: props[b] - uniform[b] for b in DOMAINS}
        max_shift  = max(abs(v) for v in prop_shift.values())

        results[model] = dict(
            steps        = steps,
            all_powers   = all_powers,
            final_p      = final_p,
            final_s      = final_s,
            proportions  = props,
            uniform      = uniform,
            prop_shift   = prop_shift,
            max_shift    = max_shift,
            dominant     = dominant,
            correlation  = r_val,
        )

    return results


def print_summary(results: dict):
    print("Per-Domain Health Monitoring — Real Pythia Data")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print("Domain proxies: benchmark task → domain category\n")

    for model, r in results.items():
        print(f"{model}:")
        print(f"  Dominant domain: {r['dominant']}")
        print(f"  P_i vs score correlation: r={r['correlation']:.3f}")
        print(f"  Max proportion shift from uniform: {r['max_shift']:.3f}")
        print(f"  {'Domain':<25} {'P_i(T)':>8} {'Prop':>8} "
              f"{'Δ Uniform':>10} {'Score':>8}")
        print("  " + "-"*62)
        for b, dom in DOMAINS.items():
            print(f"  {dom:<25} {r['final_p'][b]:>8.4f} "
                  f"{r['proportions'][b]:>8.3f} "
                  f"{r['prop_shift'][b]:>+10.3f} "
                  f"{r['final_s'][b]:>8.3f}")
        print()


def plot_results(results: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'; CY='#f1fa8c'
    BG='#050810'; PAN='#0d1117'; CGRAY='#888888'
    MODEL_COLORS = {'pythia-160m': C, 'pythia-1.4b': CA, 'pythia-6.9b': CG}
    DOMAIN_COLORS = [C, CA, CB, CG, CY]

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    for col, (model, r) in enumerate(results.items()):
        color = MODEL_COLORS[model]
        steps = r['steps']
        x     = range(len(steps))

        # Row 1: P_i(t) trajectories per domain
        ax1 = fig.add_subplot(gs[0, col]); style(ax1)
        for i, (bench, dom) in enumerate(DOMAINS.items()):
            ax1.plot(x, r['all_powers'][bench],
                     color=DOMAIN_COLORS[i], lw=1.5,
                     label=dom[:12])
        ax1.axhline(0.5, color='#444', lw=1, ls='--')
        ax1.set_title(f'{model}\nPer-Domain P_i(t)',
                      color=color, fontsize=9)
        ax1.set_xlabel('Checkpoint', color=CGRAY, fontsize=8)
        ax1.set_ylabel('P_i(t)', color=CGRAY, fontsize=8)
        ax1.legend(fontsize=6, labelcolor='white', facecolor=PAN,
                   edgecolor='#333', loc='lower right')

        # Row 2: Final proportions vs uniform
        ax2 = fig.add_subplot(gs[1, col]); style(ax2)
        dom_names  = [DOMAINS[b][:12] for b in DOMAINS]
        prop_vals  = [r['proportions'][b] for b in DOMAINS]
        uniform_v  = [0.2] * len(DOMAINS)
        x_pos      = np.arange(len(DOMAINS))
        ax2.bar(x_pos - 0.2, uniform_v,  0.35, color=CGRAY,
                alpha=0.5, label='Uniform (0.2)')
        ax2.bar(x_pos + 0.2, prop_vals, 0.35,
                color=DOMAIN_COLORS[:len(DOMAINS)], alpha=0.85,
                label='P_i(t) softmax')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(dom_names, fontsize=6, rotation=20)
        ax2.set_title('Mixing Proportions vs Uniform',
                      color=color, fontsize=9)
        ax2.set_ylabel('Proportion', color=CGRAY, fontsize=8)
        ax2.set_ylim(0, 0.35)
        ax2.legend(fontsize=7, labelcolor='white', facecolor=PAN,
                   edgecolor='#333')

    # Row 3: Proportion shift summary across models + correlation
    ax3 = fig.add_subplot(gs[2, :2]); style(ax3)
    x_pos = np.arange(len(DOMAINS))
    width = 0.25
    for i, (model, r) in enumerate(results.items()):
        shifts = [r['prop_shift'][b] for b in DOMAINS]
        color  = list(MODEL_COLORS.values())[i]
        ax3.bar(x_pos + (i-1)*width, shifts, width,
                color=color, alpha=0.8,
                label=model.replace('pythia-',''))
    ax3.axhline(0, color='white', lw=1, ls='--')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([DOMAINS[b][:10] for b in DOMAINS], fontsize=8)
    ax3.set_title('Proportion Shift from Uniform (Δ = P_i softmax − 0.2)',
                  color=C, fontsize=10)
    ax3.set_ylabel('Δ Proportion', color=CGRAY)
    ax3.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Summary table
    ax4 = fig.add_subplot(gs[2, 2]); ax4.set_facecolor(PAN); ax4.axis('off')
    y = 0.96
    ax4.text(0.05, y, 'Results Summary', color=C, fontsize=10,
             fontweight='bold', transform=ax4.transAxes); y -= 0.10
    for model, r in results.items():
        short = model.replace('pythia-', '')
        ax4.text(0.05, y, f'{short}:', color=CA, fontsize=9,
                 fontweight='bold', transform=ax4.transAxes); y -= 0.09
        for lbl, val in [
            ('Dominant', r['dominant'][:12]),
            ('Max Δ prop', f"{r['max_shift']:+.3f}"),
            ('r(P_i, score)', f"{r['correlation']:.3f}"),
        ]:
            ax4.text(0.08, y, f'{lbl}: {val}', color=C, fontsize=8,
                     transform=ax4.transAxes); y -= 0.08
        y -= 0.03

    fig.suptitle(
        'Per-Domain Health Monitoring — P_i(t) = E_i(t)×W_i(t)\n'
        'Real Pythia data · Benchmark tasks as domain proxies · '
        'Signal analysis — not a controlled mixing experiment',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                'Cantrell (2026) · Paper 4 · '
                'Validation requires per-domain perplexity from pretraining data · '
                'github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./pythia-main/evals/pythia-v1')
    args = parser.parse_args()

    print("Per-Domain Health Monitoring — Paper 4")
    print(f"Real Pythia data: {args.data}")
    print(f"α={ALPHA}, λ={LAMBDA}, EWMA span={EWMA_SPAN}")
    print("NOTE: benchmark accuracy is a proxy for domain health.\n"
          "Validation requires per-domain perplexity from pretraining.\n")

    results = run_analysis(args.data)
    print_summary(results)

    print("Generating charts...")
    plot_results(results,
                 save_path='/mnt/user-data/outputs/paper4_simulation.png')
    print("Done.")
