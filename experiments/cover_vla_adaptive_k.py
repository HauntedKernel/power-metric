"""
CoVer-VLA × P(t) Adaptive K Simulator
======================================
Companion to cover_vla_simulator.py (which tested adaptive M).
Tests whether P(t) helps decide when to STOP TRYING NEW REPHRASES.

Mechanics differ from adaptive M:
  - In CoVer-VLA, K=8 rephrased instructions are precomputed at boot time
  - For each rephrase k, the VLA generates M=5 actions, scored by CoVer
  - The "rephrase-level score" S_k = mean(scores_k) summarizes how well
    that rephrase induces aligned actions
  - The active rephrase is k* = argmax_k S_k

Adaptive K asks: do we need all 8 rephrases? Or after seeing a few good
ones, can we stop generating more? Each new rephrase reveals a new score
distribution, so the signal is the running max of S_k as k advances.

Signal P(t) tracks: max{S_1, S_2, ..., S_k} as k increases. This is
trajectory-like rather than bandit-like — each new rephrase is a
genuinely new region of language space.

Run from C:\\Users\\Carolina\\:
    python cover_vla_adaptive_k.py
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_TASKS_PER_REGIME = 1000

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Score generators (rephrase-level) ────────────────────────────────

def sample_rephrase_score_easy(rng):
    """Easy task: most rephrases work. Mean ~0.78, occasional dud."""
    return rng.beta(8.0, 2.5)

def sample_rephrase_score_medium(rng):
    """Medium: some rephrases work better than others. Mean ~0.5."""
    return rng.beta(4.0, 4.0)

def sample_rephrase_score_hard(rng):
    """Hard: most rephrases fail. Mean ~0.3, hard to find good one."""
    return rng.beta(2.0, 5.0)

REGIMES = {
    'easy':   sample_rephrase_score_easy,
    'medium': sample_rephrase_score_medium,
    'hard':   sample_rephrase_score_hard,
}


# ── Strategies ───────────────────────────────────────────────────────

def strategy_fixed_K(rephrase_scores, K):
    """Try exactly K rephrases, pick best."""
    sampled = rephrase_scores[:K]
    return K, float(sampled.max())


def strategy_pt_adaptive_K(
    rephrase_scores, K_max, theta=0.5, K_min=2,
    alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN,
):
    """
    Try rephrases sequentially. After each, update P(t) on running max.
    Stop when P(t) >= theta and K >= K_min.
    """
    er = None; ew = 0.0; pw = 0.0
    best = -1.0
    for k in range(K_max):
        s = float(rephrase_scores[k])
        best = max(best, s)
        signal = best

        if er is None:
            eff = 1.0
            er = max(signal, 1e-6)
        else:
            eff = signal / er
            er = (1 - alpha) * er + alpha * signal

        win = 1.0 if eff >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst

        if (k + 1) >= K_min and pw >= theta:
            return k + 1, best
    return K_max, best


def strategy_no_improvement_K(rephrase_scores, K_max, patience=2, K_min=2):
    """Stop if best rephrase score hasn't improved for `patience` rephrases."""
    best = -1.0
    no_improve = 0
    for k in range(K_max):
        s = float(rephrase_scores[k])
        if s > best:
            best = s
            no_improve = 0
        else:
            no_improve += 1
        if (k + 1) >= K_min and no_improve >= patience:
            return k + 1, best
    return K_max, best


# ── Experiment ───────────────────────────────────────────────────────

@dataclass
class Result:
    regime: str
    strategy: str
    avg_K_used: float
    avg_best_score: float
    quality_loss: float
    compute_saving_pct: float


def run_per_regime(K_max=12, K_baseline=8, theta_grid=(0.3, 0.4, 0.5, 0.6, 0.7)):
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for regime, gen in REGIMES.items():
        # Pre-generate score pools (each task: K_max possible rephrases)
        pools = []
        for _ in range(N_TASKS_PER_REGIME):
            pool = np.array([gen(rng) for _ in range(K_max)])
            pools.append(pool)

        # Baseline
        bu, bs = [], []
        for pool in pools:
            u, b = strategy_fixed_K(pool, K_baseline)
            bu.append(u); bs.append(b)
        baseline_K = np.mean(bu)
        baseline_score = np.mean(bs)
        results.append(Result(regime, f'fixed_K={K_baseline}',
                              baseline_K, baseline_score, 0.0, 0.0))

        # P(t) adaptive
        for theta in theta_grid:
            uses, scores = [], []
            for pool in pools:
                u, b = strategy_pt_adaptive_K(pool, K_max, theta=theta)
                uses.append(u); scores.append(b)
            avg_K = np.mean(uses)
            avg_score = np.mean(scores)
            losses = [bs[i] - scores[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'pt_theta={theta:.1f}',
                avg_K, avg_score, float(np.mean(losses)),
                (1 - avg_K / baseline_K) * 100,
            ))

        # No-improvement
        for patience in [1, 2, 3]:
            uses, scores = [], []
            for pool in pools:
                u, b = strategy_no_improvement_K(pool, K_max, patience=patience)
                uses.append(u); scores.append(b)
            avg_K = np.mean(uses)
            avg_score = np.mean(scores)
            losses = [bs[i] - scores[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'no_improve_p={patience}',
                avg_K, avg_score, float(np.mean(losses)),
                (1 - avg_K / baseline_K) * 100,
            ))

    return results


def run_mixed(K_max=12, K_baseline=8, theta=0.5, mix=(0.4, 0.4, 0.2)):
    rng = np.random.default_rng(RNG_SEED + 1)
    n_easy = int(N_TASKS_PER_REGIME * mix[0])
    n_med  = int(N_TASKS_PER_REGIME * mix[1])
    n_hard = N_TASKS_PER_REGIME - n_easy - n_med
    pools = {
        'easy':   [np.array([REGIMES['easy'](rng) for _ in range(K_max)])
                   for _ in range(n_easy)],
        'medium': [np.array([REGIMES['medium'](rng) for _ in range(K_max)])
                   for _ in range(n_med)],
        'hard':   [np.array([REGIMES['hard'](rng) for _ in range(K_max)])
                   for _ in range(n_hard)],
    }
    out = {}
    for regime, ps in pools.items():
        bu, bs, au, as_ = [], [], [], []
        for p in ps:
            u, b = strategy_fixed_K(p, K_baseline); bu.append(u); bs.append(b)
            u, b = strategy_pt_adaptive_K(p, K_max, theta=theta)
            au.append(u); as_.append(b)
        out[regime] = dict(
            n=len(ps),
            baseline_K=np.mean(bu), adaptive_K=np.mean(au),
            baseline_score=np.mean(bs), adaptive_score=np.mean(as_),
            quality_loss=np.mean(bs) - np.mean(as_),
        )
    n_total = sum(r['n'] for r in out.values())
    agg = dict(
        baseline_K=sum(r['n']*r['baseline_K'] for r in out.values())/n_total,
        adaptive_K=sum(r['n']*r['adaptive_K'] for r in out.values())/n_total,
        baseline_score=sum(r['n']*r['baseline_score'] for r in out.values())/n_total,
        adaptive_score=sum(r['n']*r['adaptive_score'] for r in out.values())/n_total,
    )
    agg['compute_saving'] = (1 - agg['adaptive_K'] / agg['baseline_K']) * 100
    agg['quality_loss']   = agg['baseline_score'] - agg['adaptive_score']
    return dict(per_regime=out, aggregate=agg)


# ── Reporting ────────────────────────────────────────────────────────

def print_results(results):
    print("\n" + "="*86)
    print("Per-Regime — fixed K=8 baseline vs P(t) adaptive K vs no-improve")
    print("="*86)
    by_regime = {}
    for r in results: by_regime.setdefault(r.regime, []).append(r)
    for regime in ['easy', 'medium', 'hard']:
        print(f"\n  {regime.upper()} regime")
        print(f"  {'Strategy':<22} {'avg K':>8} {'avg best':>10} "
              f"{'quality loss':>14} {'compute saved':>14}")
        print("  " + "-"*82)
        for r in by_regime[regime]:
            print(f"  {r.strategy:<22} {r.avg_K_used:>8.2f} "
                  f"{r.avg_best_score:>10.4f} "
                  f"{r.quality_loss:>+14.4f} "
                  f"{r.compute_saving_pct:>13.1f}%")


def print_mixed(mixed):
    print("\n" + "="*86)
    print("Mixed deployment (40% easy / 40% medium / 20% hard, theta=0.5)")
    print("="*86)
    print(f"\n  {'Regime':<10} {'n':>5} {'base K':>8} {'adapt K':>8} "
          f"{'base best':>10} {'adapt best':>11} {'loss':>9}")
    print("  " + "-"*78)
    for regime in ['easy', 'medium', 'hard']:
        r = mixed['per_regime'][regime]
        print(f"  {regime:<10} {r['n']:>5} {r['baseline_K']:>8.2f} {r['adaptive_K']:>8.2f} "
              f"{r['baseline_score']:>10.4f} {r['adaptive_score']:>11.4f} "
              f"{r['quality_loss']:>+9.4f}")
    a = mixed['aggregate']
    print(f"\n  Aggregate:")
    print(f"    Baseline K:      {a['baseline_K']:.2f}")
    print(f"    Adaptive K:      {a['adaptive_K']:.2f}")
    print(f"    Compute saving:  {a['compute_saving']:.1f}%")
    print(f"    Quality loss:    {a['quality_loss']:+.4f}")


def main():
    print("="*86)
    print("CoVer-VLA × P(t) Adaptive K — Synthetic Simulator")
    print("="*86)
    print()
    print("Tests whether P(t) helps decide HOW MANY REPHRASES to try.")
    print("Each rephrase produces an aggregate score; we sequentially explore")
    print("rephrase space and stop when P(t) on running-best plateaus.")
    print()
    results = run_per_regime(K_max=12, K_baseline=8)
    print_results(results)
    mixed = run_mixed(K_max=12, K_baseline=8, theta=0.5)
    print_mixed(mixed)

    print("\n" + "="*86)
    print("INTERPRETATION")
    print("="*86)
    print("""
  Compare against adaptive M result (cover_vla_simulator.py):
  - Adaptive M aggregate: 44.7% saved, 0.052 quality loss → bandit-shaped, P(t) lost
  - If adaptive K shows similar profile, the rephrase-sampling axis has the
    same i.i.d. structure problem as actions. Pivot to episode monitoring.
  - If adaptive K shows meaningfully better profile (lower quality loss for
    similar saving), the trajectory-vs-bandit framing is correct and we have
    a real fit to bring to Jacky.
""")


if __name__ == '__main__':
    main()
