"""
Catastrophic Forgetting Detection via Dual Power Metrics
=========================================================
Paper 13: "Real-Time Forgetting Detection During Fine-Tuning via
           Dual Power Metric Health Signals"
Cole Cantrell | cole@paradigmbridge.tech | paradigmbridge.tech

Applies dual P_old(t) / P_new(t) signals to detect catastrophic
forgetting during fine-tuning before it has fully occurred.

The problem:
  Fine-tuning on new tasks causes catastrophic forgetting — previously
  learned capabilities degrade as the model overwrites old knowledge.
  Current detection methods are retrospective: forgetting is discovered
  by running full benchmarks after training. This is expensive and slow.

The solution:
  Run two simultaneous power metric signals during fine-tuning:
  - P_new(t): power metric on new task quality (is fine-tuning working?)
  - P_old(t): power metric on old task quality (is forgetting occurring?)
  When P_old(t) < threshold: forgetting is underway — stop or intervene.

Real data:
  Pythia-1.4B real benchmark scores (Biderman et al. 2023) as pretraining
  baseline (checkpoints 0-7, steps 1K-63K). Simulated fine-tuning phase
  (checkpoints 8-15) with ARC Challenge as target task (+10% gain)
  and the four remaining benchmarks as old tasks (-9% decay).
  Fine-tuning is deliberately aggressive to show signal within 8 checkpoints.

E(t) uses pre-update baseline (consistent with all papers in series).
P_old(t) uses cross-task average: E_old(t) = avg_old_quality(t) / E[R_old](t-1)
P_new(t) uses single task: E_new(t) = new_quality(t) / E[R_new](t-1)

Limitation: only 8 fine-tuning checkpoints — insufficient to fully
calibrate PM stopping vs EWC/LwF baselines. Real validation requires
dense checkpointing over a longer fine-tuning run.

Related papers:
  Paper 1: power_metric_training.py (same dataset, pretraining)
  Series:  https://github.com/HauntedKernel/power-metric
"""

import json, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ────────────────────────────────────────────────────────────
ALPHA     = 0.3
LAMBDA    = 0.5
EWMA_SPAN = 3
THETA_OLD = 0.40   # P_old stopping threshold
PATIENCE  = 1      # steps below threshold before stopping

# ── Data ──────────────────────────────────────────────────────────────────
BENCHMARKS = {
    'lambada_openai': 'Language/Web',
    'piqa':           'Physical Comm.',
    'winogrande':     'Social Comm.',
    'arc_easy':       'Science/Facts',
    'arc_challenge':  'Reasoning (target)',
}
TARGET_TASK = 'arc_challenge'
OLD_TASKS   = [b for b in BENCHMARKS if b != TARGET_TASK]

# ── Fine-tuning simulation parameters ────────────────────────────────────
# Aggressive parameters to show signal within 8 checkpoints
FINETUNING_NEW_GAIN  = 0.10   # +10% on target task over 8 checkpoints
FINETUNING_OLD_DECAY = 0.09   # −9% on old tasks over 8 checkpoints


def load_pythia_1b4(base_path: str):
    """Load real Pythia-1.4B benchmark scores."""
    folder = os.path.join(base_path, 'pythia-1.4b', 'zero-shot')
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Not found: {folder}\nDownload: https://github.com/EleutherAI/pythia")
    step_scores = {}
    for fname in os.listdir(folder):
        m = re.search(r'step(\d+)', fname)
        if not m: continue
        step = int(m.group(1))
        if step < 1000: continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        r = data.get('results', {})
        row = {}
        for b in BENCHMARKS:
            if b in r:
                bd = r[b]; s = bd.get('acc_norm', bd.get('acc', None))
                if s is not None: row[b] = s
        if len(row) == len(BENCHMARKS):
            step_scores[step] = row
    steps = sorted(step_scores.keys())
    return steps, step_scores


def simulate_finetuning(pretrain_end: dict, n_finetune: int = 8):
    """Simulate fine-tuning: target improves, old tasks decay."""
    simulated = {}
    for i in range(n_finetune):
        t = (i + 1) / n_finetune
        row = {}
        for b in BENCHMARKS:
            if b == TARGET_TASK:
                row[b] = pretrain_end[b] * (1 + FINETUNING_NEW_GAIN * t)
            else:
                row[b] = pretrain_end[b] * (1 - FINETUNING_OLD_DECAY * t)
        simulated[i] = row
    return simulated


def compute_pm(signal: list) -> np.ndarray:
    """P(t) with pre-update E[R] baseline."""
    er = None; ew = 0.0; pw = 0.0; powers = []
    for s in signal:
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            eff = s / er
            er  = (1 - ALPHA) * er + ALPHA * s
        win  = 1.0 if eff > 1.0 else 0.0
        a    = 2.0 / (EWMA_SPAN + 1)
        ew   = a * win + (1 - a) * ew
        inst = eff * ew
        pw   = np.exp(-LAMBDA) * pw + (1 - np.exp(-LAMBDA)) * inst
        powers.append(pw)
    return np.array(powers)


def find_stop(p_old: np.ndarray, finetune_start_idx: int,
              theta: float = THETA_OLD) -> int:
    """Find first step in fine-tuning where P_old < theta."""
    for i in range(finetune_start_idx, len(p_old)):
        if p_old[i] < theta:
            return i
    return len(p_old) - 1


def run_analysis(base_path: str):
    steps, scores = load_pythia_1b4(base_path)
    pretrain_steps  = steps[:8]
    finetune_start  = 8
    pretrain_end    = {b: scores[pretrain_steps[-1]][b] for b in BENCHMARKS}

    # Simulated fine-tuning scores (indexed 0-7)
    ft_sim = simulate_finetuning(pretrain_end)

    # Build full trajectory: real pretrain + simulated finetune
    all_steps  = pretrain_steps + [f'ft_{i}' for i in range(8)]
    all_scores = {s: scores[s] for s in pretrain_steps}
    for i in range(8):
        all_scores[f'ft_{i}'] = ft_sim[i]

    # Signals
    old_avg  = [np.mean([all_scores[s][b] for b in OLD_TASKS])
                for s in all_steps]
    new_sig  = [all_scores[s][TARGET_TASK] for s in all_steps]

    p_old = compute_pm(old_avg)
    p_new = compute_pm(new_sig)

    # PM stop
    pm_stop_idx = find_stop(p_old, finetune_start)
    pm_stop_key = all_steps[pm_stop_idx]
    no_stop_key = all_steps[-1]

    results = dict(
        pretrain_end  = pretrain_end,
        all_steps     = all_steps,
        all_scores    = all_scores,
        old_avg       = old_avg,
        new_sig       = new_sig,
        p_old         = p_old,
        p_new         = p_new,
        pm_stop_idx   = pm_stop_idx,
        pm_stop_key   = pm_stop_key,
        finetune_start = finetune_start,
    )

    # Per-task table
    print(f"\nPaper 13 — Catastrophic Forgetting Detection")
    print(f"Pythia-1.4B, 8 pretrain + 8 simulated fine-tuning checkpoints")
    print(f"P_old threshold: θ={THETA_OLD}")
    print(f"PM stop at: index {pm_stop_idx} ({pm_stop_key})\n")

    print(f"{'Task':<25} {'Pretrain End':>13} {'PM Stop':>9} {'No Stop':>9} {'Preserved':>10}")
    print("-"*70)
    for b, name in BENCHMARKS.items():
        pre  = pretrain_end[b]
        pm_q = all_scores[pm_stop_key][b]
        ns_q = all_scores[no_stop_key][b]
        print(f"{name:<25} {pre:>13.4f} {pm_q:>9.4f} {ns_q:>9.4f} "
              f"{pm_q - ns_q:>+10.4f}")

    # Summary stats
    old_pm  = np.mean([all_scores[pm_stop_key][b] for b in OLD_TASKS])
    old_ns  = np.mean([all_scores[no_stop_key][b]  for b in OLD_TASKS])
    old_pre = np.mean([pretrain_end[b]             for b in OLD_TASKS])
    new_pm  = all_scores[pm_stop_key][TARGET_TASK]
    new_ns  = all_scores[no_stop_key][TARGET_TASK]

    print(f"\nSummary:")
    print(f"  Old task avg — pretrain: {old_pre:.4f}, PM stop: {old_pm:.4f}, "
          f"no stop: {old_ns:.4f}, preserved: {old_pm-old_ns:+.4f}")
    print(f"  New task — PM stop: {new_pm:.4f}, no stop: {new_ns:.4f}")
    new_gain_pm = (new_pm - pretrain_end[TARGET_TASK]) / pretrain_end[TARGET_TASK] * 100
    new_gain_ns = (new_ns - pretrain_end[TARGET_TASK]) / pretrain_end[TARGET_TASK] * 100
    print(f"  New task gain — PM: {new_gain_pm:+.1f}%, no stop: {new_gain_ns:+.1f}%")
    print(f"  New capability retained: {new_gain_pm/new_gain_ns*100:.1f}%")

    return results


def plot_results(r: dict, save_path: str = None):
    fig = plt.figure(figsize=(14, 10), facecolor='#050810')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.38)
    C='#00e5ff'; CA='#bd93f9'; CB='#ff6b6b'; CG='#50fa7b'; CY='#f1fa8c'
    BG='#050810'; PAN='#0d1117'; GR='#888888'

    def style(ax):
        ax.set_facecolor(PAN); ax.tick_params(colors='#666666')
        for s in ax.spines.values(): s.set_edgecolor('#333333')

    n      = len(r['all_steps'])
    x      = range(n)
    ft_idx = r['finetune_start']
    stop_i = r['pm_stop_idx']

    # Chart 1: Old task trajectories
    ax1 = fig.add_subplot(gs[0, :]); style(ax1)
    colors_tasks = [C, CA, CG, CY, CB]
    for i, (b, name) in enumerate(BENCHMARKS.items()):
        vals = [r['all_scores'][s][b] for s in r['all_steps']]
        ls   = '-' if b == TARGET_TASK else '--'
        lw   = 2.5 if b == TARGET_TASK else 1.5
        ax1.plot(x, vals, color=colors_tasks[i], lw=lw, ls=ls,
                 label=name[:16])
    ax1.axvline(ft_idx - 0.5, color=GR, lw=1, ls=':', alpha=0.7)
    ax1.axvline(stop_i, color=C, lw=2, ls='--',
                label=f'PM Stop (idx {stop_i})')
    ax1.text(ft_idx - 0.4, 0.75, 'Fine-tuning →', color=GR, fontsize=8)
    ax1.set_title('Task Quality Trajectories — Pretraining + Simulated Fine-tuning',
                  color=C, fontsize=11)
    ax1.set_xlabel('Checkpoint Index', color=GR)
    ax1.set_ylabel('Benchmark Score', color=GR)
    ax1.legend(fontsize=7, labelcolor='white', facecolor=PAN,
               edgecolor='#333', loc='lower right', ncol=2)

    # Chart 2: P_old and P_new
    ax2 = fig.add_subplot(gs[1, 0]); style(ax2)
    ax2.plot(x, r['p_old'], color=CB, lw=2, label='P_old(t)')
    ax2.plot(x, r['p_new'], color=CG, lw=2, label='P_new(t)')
    ax2.axhline(THETA_OLD, color=CB, lw=1.5, ls='--',
                label=f'θ={THETA_OLD}')
    ax2.axvline(ft_idx - 0.5, color=GR, lw=1, ls=':', alpha=0.7)
    ax2.axvline(stop_i, color=C, lw=2, ls='--',
                label=f'P_old < θ (stop)')
    ax2.set_title('P_old(t) and P_new(t)', color=C, fontsize=11)
    ax2.set_xlabel('Checkpoint Index', color=GR)
    ax2.set_ylabel('P(t)', color=GR)
    ax2.legend(fontsize=7, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 3: Per-task quality at PM stop vs no stop
    ax3 = fig.add_subplot(gs[1, 1]); style(ax3)
    task_names  = [BENCHMARKS[b][:12] for b in BENCHMARKS]
    pm_q  = [r['all_scores'][r['pm_stop_key']][b]   for b in BENCHMARKS]
    ns_q  = [r['all_scores'][r['all_steps'][-1]][b] for b in BENCHMARKS]
    xp    = np.arange(len(BENCHMARKS))
    ax3.bar(xp - 0.2, pm_q, 0.35, color=C,  alpha=0.85, label='PM Stop')
    ax3.bar(xp + 0.2, ns_q, 0.35, color=CB, alpha=0.70, label='No Stop')
    ax3.set_xticks(xp)
    ax3.set_xticklabels(task_names, fontsize=7, rotation=15)
    ax3.set_title('Quality at PM Stop vs No Stop', color=C, fontsize=11)
    ax3.set_ylabel('Score', color=GR)
    ax3.legend(fontsize=8, labelcolor='white', facecolor=PAN, edgecolor='#333')

    # Chart 4: Threshold sensitivity
    ax4 = fig.add_subplot(gs[2, :]); style(ax4)
    thetas   = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    stops_t  = []; old_q_t = []; new_q_t = []
    for th in thetas:
        st = find_stop(r['p_old'], ft_idx, th)
        sk = r['all_steps'][st]
        stops_t.append(st)
        old_q_t.append(np.mean([r['all_scores'][sk][b] for b in OLD_TASKS]))
        new_q_t.append(r['all_scores'][sk][TARGET_TASK])

    ax4t = ax4.twinx(); ax4t.set_facecolor(PAN)
    ax4.plot(thetas, old_q_t, color=CB, lw=2, marker='o', ms=6,
             label='Old task avg quality')
    ax4t.plot(thetas, new_q_t, color=CG, lw=2, marker='s', ms=6,
              ls='--', label='New task quality')
    ax4.axvline(THETA_OLD, color=C, lw=1.5, ls='--',
                label=f'Default θ={THETA_OLD}')
    ax4.set_title('Threshold Sensitivity: Old vs New Task Quality Tradeoff',
                  color=C, fontsize=11)
    ax4.set_xlabel('Stopping Threshold θ', color=GR)
    ax4.set_ylabel('Old Task Quality', color=CB)
    ax4t.set_ylabel('New Task Quality', color=CG)
    ax4t.tick_params(colors='#666666')
    ax4.legend(loc='center left', fontsize=8, labelcolor='white',
               facecolor=PAN, edgecolor='#333')
    ax4t.legend(loc='center right', fontsize=8, labelcolor='white',
                facecolor=PAN, edgecolor='#333')

    fig.suptitle(
        'Catastrophic Forgetting Detection — P_old(t) on Old-Task Quality\n'
        'Real Pythia-1.4B pretraining + simulated fine-tuning · '
        'P_old < θ = forgetting underway',
        color=C, fontsize=10, fontweight='bold', y=0.99)
    plt.figtext(0.5, 0.01,
                f'α={ALPHA}, λ={LAMBDA}, θ={THETA_OLD} · '
                'Cantrell (2026) · Paper 13 · '
                'github.com/HauntedKernel/power-metric',
                ha='center', color='#444', fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart: {save_path}")
    return fig


if __name__ == '__main__':
    BASE = './pythia-main/evals/pythia-v1'
    r    = run_analysis(BASE)
    print("\nGenerating charts...")
    plot_results(r, save_path='/mnt/user-data/outputs/paper13_simulation.png')
    print("Done.")
