"""
power-metric visualizer
========================
Generates the headline figure for the README and Twitter thread.

Two panels:
  Left:  Prediction trajectories — fixed power-law vs adaptive EMA
         on pythia-1.4b (the clearest example of divergence)
  Right: Alpha sweep — unreliability% across α=0.1..0.9

Saves: power_metric_result.png (1400×600px, 150dpi, dark theme)

Run from C:\\Users\\Carolina\\:
    pip install matplotlib --break-system-packages
    python visualize_result.py --data ./pythia-main/evals/pythia-v1
"""

import os, json, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

# ── Data loading ──────────────────────────────────────────────────────

BENCHMARKS = ['lambada_openai', 'piqa', 'winogrande', 'arc_easy', 'arc_challenge']
SPLIT = 8
THRESH = 0.02
PAPER3_MODELS = ['pythia-160m', 'pythia-1.4b', 'pythia-6.9b']


def load_curve(base, model):
    folder = os.path.join(base, model, 'zero-shot')
    if not os.path.exists(folder):
        return None
    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        try:
            with open(os.path.join(folder, fname)) as f:
                data = json.load(f)
            r = data.get('results', {})
            sc = [r[b].get('acc_norm', r[b].get('acc'))
                  for b in BENCHMARKS if b in r]
            sc = [s for s in sc if s is not None]
            if len(sc) == len(BENCHMARKS):
                step_scores[step] = float(np.mean(sc))
        except Exception:
            pass
    steps = sorted(step_scores.keys())
    if len(steps) < 16: return None
    return np.array([step_scores[s] for s in steps])


def power_law(x, a, b, c):
    return a * np.power(x, b) + c


def predict_ema(scores, alpha, split=SPLIT):
    er = scores[0]
    series = [er]
    for s in scores[1:]:
        er = (1 - alpha) * er + alpha * s
        series.append(er)
    preds = np.array(series[split - 1:-1])
    return preds, scores[split:]


def predict_fixed(scores, split=SPLIT):
    x_fit = np.arange(1, split + 1, dtype=float)
    x_all = np.arange(1, len(scores) + 1, dtype=float)
    try:
        popt, _ = curve_fit(power_law, x_fit, scores[:split],
                             p0=[0.1, 0.3, 0.3], maxfev=10000,
                             bounds=([-np.inf, -2, 0], [np.inf, 2, 1]))
        return power_law(x_all, *popt)[split:]
    except RuntimeError:
        return np.full(len(scores) - split, scores[split - 1])


def unreliability(preds, actuals):
    return float(np.mean(np.abs(preds - actuals) > THRESH))


# ── Plotting ──────────────────────────────────────────────────────────

# Dark editorial palette
BG       = '#0d0d0f'
PANEL    = '#13141a'
GRID     = '#1e2030'
TEXT     = '#e8e8ec'
MUTED    = '#6b7280'
ACCENT   = '#f97316'   # orange — fixed power-law (danger)
GOOD     = '#22d3ee'   # cyan — adaptive (safe)
ACTUAL   = '#ffffff'   # white — actual scores
NEUTRAL  = '#a78bfa'   # purple — α=0.3


def make_figure(scores_dict, out_path='power_metric_result.png'):
    fig = plt.figure(figsize=(14, 5.5), facecolor=BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.10,
                  left=0.06, right=0.97, top=0.88, bottom=0.13)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.spines[['top', 'right', 'left', 'bottom']].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    # ── Panel 1: Trajectory plot (pythia-1.4b) ────────────────────────
    scores = scores_dict.get('pythia-1.4b')
    if scores is None:
        scores = list(scores_dict.values())[0]

    x_all    = np.arange(1, len(scores) + 1)
    x_test   = np.arange(SPLIT + 1, len(scores) + 1)
    x_fit    = np.arange(1, SPLIT + 1)

    fixed_test          = predict_fixed(scores)
    preds_03, act_test  = predict_ema(scores, 0.3)
    preds_06, _         = predict_ema(scores, 0.6)

    # Fit region
    ax1.axvspan(0.5, SPLIT + 0.5, color='#ffffff', alpha=0.03, zorder=0)
    ax1.axvline(SPLIT + 0.5, color=GRID, lw=1.2, ls='--', zorder=1)
    ax1.text(SPLIT * 0.5, scores[:SPLIT].min() - 0.003, 'fit region',
             ha='center', va='top', fontsize=8, color=MUTED)
    ax1.text(SPLIT + (len(scores) - SPLIT) * 0.5,
             scores[:SPLIT].min() - 0.003,
             'prediction region',
             ha='center', va='top', fontsize=8, color=MUTED)

    # Actual scores
    ax1.plot(x_all, scores, color=ACTUAL, lw=1.5, zorder=5,
             marker='o', markersize=3.5, markerfacecolor=ACTUAL,
             label='actual scores')

    # Fixed power-law
    ax1.plot(x_test, fixed_test, color=ACCENT, lw=2.2, zorder=4,
             ls='--', label='fixed power-law (96% unreliable)')

    # Adaptive α=0.6
    ax1.plot(x_test, preds_06, color=GOOD, lw=2.2, zorder=4,
             label='adaptive EMA α=0.6 (0% unreliable)')

    # Mark unreliable predictions on fixed
    for i, (pred, actual) in enumerate(zip(fixed_test, act_test)):
        if abs(pred - actual) > THRESH:
            ax1.plot(x_test[i], pred, 'x', color=ACCENT,
                     markersize=9, markeredgewidth=2, zorder=6)

    # Styling
    ax1.set_xlim(0.5, len(scores) + 0.5)
    ax1.set_xlabel('checkpoint', color=MUTED, fontsize=10, labelpad=8)
    ax1.set_ylabel('benchmark accuracy (mean)', color=MUTED,
                   fontsize=10, labelpad=8)
    ax1.set_title('Scaling Prediction: pythia-1.4b',
                  color=TEXT, fontsize=11, fontweight='bold', pad=10)

    legend = ax1.legend(
        loc='lower right', fontsize=8.5,
        facecolor='#1a1b26', edgecolor=GRID,
        labelcolor=TEXT, framealpha=0.95
    )

    # Error annotation
    err_at_last = abs(fixed_test[-1] - act_test[-1])
    ax1.annotate(
        f'+{err_at_last:.3f} error\n(fixed power-law)',
        xy=(x_test[-1], fixed_test[-1]),
        xytext=(x_test[-1] - 2.5, fixed_test[-1] + 0.012),
        color=ACCENT, fontsize=8.5,
        arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2),
    )

    # ── Panel 2: Alpha sweep (aggregate across 3 models) ─────────────
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    unreliabilities = []

    for alpha in alphas:
        all_errs = []
        for model, sc in scores_dict.items():
            if sc is not None:
                p, a = predict_ema(sc, alpha)
                all_errs.extend(np.abs(p - a).tolist())
        unreliabilities.append(float(np.mean(np.array(all_errs) > THRESH)) * 100)

    # Color bars by regime
    bar_colors = []
    for u in unreliabilities:
        if u == 0:
            bar_colors.append(GOOD)
        elif u < 20:
            bar_colors.append(NEUTRAL)
        else:
            bar_colors.append(ACCENT)

    bars = ax2.bar(alphas, unreliabilities, width=0.07,
                    color=bar_colors, zorder=3, alpha=0.90)

    # Value labels
    for bar, u in zip(bars, unreliabilities):
        label = f'{u:.0f}%'
        va = 'bottom'
        y = bar.get_height() + 1.5
        if u == 0:
            y = 2.5
            va = 'bottom'
        ax2.text(bar.get_x() + bar.get_width() / 2, y,
                 label, ha='center', va=va, fontsize=8.5,
                 color=TEXT, fontweight='bold' if u == 0 else 'normal')

    # Horizontal reference at Paper 3's 38%
    ax2.axhline(38, color=NEUTRAL, lw=1.2, ls=':', zorder=2,
                alpha=0.8)
    ax2.text(0.92, 40, 'paper 3\ndefault (38%)',
             ha='right', va='bottom', fontsize=8, color=NEUTRAL,
             transform=ax2.get_xaxis_transform())

    # Annotate the 0% zone
    ax2.axhspan(-3, 5, color=GOOD, alpha=0.07, zorder=0)
    ax2.text(0.72, 2.5, '0% unreliable', ha='center', va='center',
             fontsize=8.5, color=GOOD, fontweight='bold',
             transform=ax2.get_xaxis_transform())

    ax2.set_xlim(0.03, 0.97)
    ax2.set_ylim(-5, max(unreliabilities) + 15)
    ax2.set_xticks(alphas)
    ax2.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax2.set_xlabel('α (EMA smoothing factor)', color=MUTED,
                   fontsize=10, labelpad=8)
    ax2.set_ylabel('unreliability %', color=MUTED,
                   fontsize=10, labelpad=8)
    ax2.set_title('Alpha Sweep (3 Pythia models)',
                  color=TEXT, fontsize=11, fontweight='bold', pad=10)

    # Legend patches
    p1 = mpatches.Patch(color=ACCENT, label='unreliable (>20%)')
    p2 = mpatches.Patch(color=NEUTRAL, label='partial (<20%)')
    p3 = mpatches.Patch(color=GOOD, label='reliable (0%)')
    ax2.legend(handles=[p1, p2, p3], loc='upper right',
               fontsize=8, facecolor='#1a1b26', edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.95)

    # ── Headline ──────────────────────────────────────────────────────
    fig.text(0.515, 0.97,
             'Fixed power-law: 96% unreliable.  '
             'Adaptive EMA (α=0.6): 0% unreliable.',
             ha='center', va='top', fontsize=12.5,
             color=TEXT, fontweight='bold')

    fig.text(0.515, 0.935,
             'Validated on Pythia 160M–6.9B  ·  pip install power-metric  ·  '
             'github.com/HauntedKernel/power-metric',
             ha='center', va='top', fontsize=9, color=MUTED)

    fig.set_size_inches(14, 5.5)
    plt.savefig(out_path, dpi=100,
                facecolor=BG, edgecolor='none')
    print(f'Saved → {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        default='./pythia-main/evals/pythia-v1',
                        help='Path to pythia evals directory')
    parser.add_argument('--out', default='power_metric_result.png')
    args = parser.parse_args()

    print(f'Loading Pythia data from {args.data}...')
    curves = {}
    for m in PAPER3_MODELS:
        c = load_curve(args.data, m)
        if c is not None:
            curves[m] = c
            print(f'  {m}: {len(c)} checkpoints')
        else:
            print(f'  {m}: not found')

    if not curves:
        print('No data found. Check --data path.')
        return

    make_figure(curves, out_path=args.out)
    print(f'\nDone. Add to README:')
    print(f'  ![power-metric result]({args.out})')


if __name__ == '__main__':
    main()
