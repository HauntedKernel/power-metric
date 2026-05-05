"""
CoVer-VLA × P(t) Adaptive Sampling — Synthetic Simulator
==========================================================
Pre-Path-3 prototype: tests whether P(t)-driven adaptive M
(actions per rephrase) plausibly saves compute on top of
CoVer-VLA's verifier scores, using *synthetic* traces calibrated
to their published numbers.

Grounded vs assumed (be honest about both):

GROUNDED (from CoVer-VLA paper, Kwok et al. 2026, arXiv:2602.12281):
  - Standard config:           K=8 rephrases × M=5 actions = 40 candidates
  - SIMPLER ID success @ M=5:  57.0% (π0 + CoVer)
  - SIMPLER OOD success @ M=5: 61.0% (π0 + CoVer)
  - Verifier latency:          ~8 ms per action eval (Table 2)
  - Action-encoder latency dominates: per-action cost is the right axis

ASSUMED (synthetic — flagged because we don't have real score traces):
  - Verifier scores per (rephrase, action) drawn from Beta-mixture:
      * "Success-likely" tasks: most actions score well, top action clearly best
      * "Hard" tasks: most actions score poorly, hard to distinguish
  - "Best action quality" = max score correlates with task success rate
  - Score distribution is i.i.d. across the M actions for one rephrase
    (CoVer-VLA's actual sampling has temperature; this is the right
    direction but real variance may differ)

Why this is still useful:
  If P(t) cannot save compute even on charitable synthetic data
  (where i.i.d. scores make adaptive stopping easier than reality),
  then the idea is dead. If P(t) does save compute on this data, that's
  necessary-but-not-sufficient evidence to invest in real traces.

Usage:
    python cover_vla_simulator.py
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_TASKS_PER_REGIME = 1000
N_REPLICATES = 5

# ── P(t) parameters (V1 EWMA baseline, consistent with paper series) ──
ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Score generators ─────────────────────────────────────────────────

def sample_easy_task_scores(M_max: int, rng: np.random.Generator) -> np.ndarray:
    """
    Easy task: most actions are good, best action clearly stands out.
    Beta(8, 2) gives mean ~0.8, low variance, occasional outliers.
    """
    return rng.beta(8.0, 2.0, M_max)


def sample_hard_task_scores(M_max: int, rng: np.random.Generator) -> np.ndarray:
    """
    Hard task: most actions are poor, occasional good one buried among them.
    Beta(2, 5) gives mean ~0.29, broad variance.
    """
    return rng.beta(2.0, 5.0, M_max)


def sample_medium_task_scores(M_max: int, rng: np.random.Generator) -> np.ndarray:
    """
    Medium task: mixed quality, modest separation between best and rest.
    Beta(4, 4) gives mean 0.5, moderate variance.
    """
    return rng.beta(4.0, 4.0, M_max)


REGIME_GENERATORS = {
    'easy':   sample_easy_task_scores,
    'medium': sample_medium_task_scores,
    'hard':   sample_hard_task_scores,
}


# ── Adaptive M strategies ────────────────────────────────────────────

def strategy_fixed_M(scores: np.ndarray, M: int) -> tuple[int, float]:
    """Baseline: sample exactly M actions, pick best."""
    sampled = scores[:M]
    return M, float(sampled.max())


def strategy_pt_adaptive(
    scores: np.ndarray, M_max: int, theta: float = 0.5,
    M_min: int = 2, alpha: float = ALPHA, lam: float = LAMBDA,
    span: int = EWMA_SPAN,
) -> tuple[int, float]:
    """
    P(t) adaptive: sample one action at a time. After each, update P(t)
    on the running max-score-so-far. Stop when P(t) >= theta AND we have
    at least M_min samples.

    Signal s(t) = current max score (so it's monotone non-decreasing).
    The P(t) framework will detect when max stops improving.

    Args:
        scores: Pre-drawn pool of M_max scores (we sample sequentially).
        M_max:  Max budget. If P(t) never crosses theta, we sample all M_max.
        theta:  Stopping threshold on P(t).
        M_min:  Minimum samples before considering stopping (cold start).

    Returns:
        (samples_used, best_score)
    """
    er = None  # adaptive baseline
    ew = 0.0   # EWMA win rate
    pw = 0.0   # P(t)
    best = -1.0

    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)

        # Use best-so-far as the signal — P(t) detects when best plateaus
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

        # Decision: stop if P(t) high AND we have minimum samples
        if (i + 1) >= M_min and pw >= theta:
            return i + 1, best

    return M_max, best


def strategy_no_improvement(
    scores: np.ndarray, M_max: int, patience: int = 2, M_min: int = 2,
) -> tuple[int, float]:
    """
    Simple baseline: stop if best score hasn't improved for `patience`
    consecutive samples. No P(t).
    """
    best = -1.0
    no_improve = 0
    for i in range(M_max):
        s = float(scores[i])
        if s > best:
            best = s
            no_improve = 0
        else:
            no_improve += 1
        if (i + 1) >= M_min and no_improve >= patience:
            return i + 1, best
    return M_max, best


# ── Quality criterion ─────────────────────────────────────────────────

def quality_loss(adaptive_best: float, baseline_best: float) -> float:
    """
    Loss from using adaptive vs baseline.
    Negative = adaptive picked a better action than baseline (lucky).
    Positive = adaptive picked worse (paid quality cost for compute saving).
    """
    return baseline_best - adaptive_best


# ── Experiment ───────────────────────────────────────────────────────

@dataclass
class Result:
    regime: str
    strategy: str
    avg_M_used: float
    avg_best_score: float
    quality_loss_vs_baseline: float
    compute_saving_pct: float


def run_experiment(M_max: int = 10, M_baseline: int = 5,
                   theta_grid=(0.3, 0.4, 0.5, 0.6, 0.7)) -> list[Result]:
    """
    For each regime (easy/medium/hard), compare:
      - Fixed M=5 baseline (CoVer-VLA's published config)
      - P(t) adaptive at multiple theta values
      - No-improvement-patience baseline

    Score generator pool size = M_max so all strategies see the same draws
    (paired comparison, lower variance).
    """
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for regime_name, gen in REGIME_GENERATORS.items():
        # Pre-generate score pools for this regime
        pools = [gen(M_max, rng) for _ in range(N_TASKS_PER_REGIME)]

        # Baseline: fixed M = M_baseline
        baseline_uses = []
        baseline_scores = []
        for pool in pools:
            u, b = strategy_fixed_M(pool, M_baseline)
            baseline_uses.append(u)
            baseline_scores.append(b)
        avg_baseline_M = np.mean(baseline_uses)
        avg_baseline_score = np.mean(baseline_scores)

        results.append(Result(
            regime=regime_name,
            strategy=f'fixed_M={M_baseline}',
            avg_M_used=avg_baseline_M,
            avg_best_score=avg_baseline_score,
            quality_loss_vs_baseline=0.0,
            compute_saving_pct=0.0,
        ))

        # P(t) adaptive across thresholds
        for theta in theta_grid:
            uses = []; scores_out = []
            for pool in pools:
                u, b = strategy_pt_adaptive(pool, M_max, theta=theta)
                uses.append(u); scores_out.append(b)
            avg_M = np.mean(uses)
            avg_score = np.mean(scores_out)
            # Per-task quality loss
            losses = [baseline_scores[i] - scores_out[i] for i in range(len(pools))]
            avg_loss = float(np.mean(losses))
            saving = (1 - avg_M / avg_baseline_M) * 100

            results.append(Result(
                regime=regime_name,
                strategy=f'pt_theta={theta:.1f}',
                avg_M_used=avg_M,
                avg_best_score=avg_score,
                quality_loss_vs_baseline=avg_loss,
                compute_saving_pct=saving,
            ))

        # No-improvement baseline
        for patience in [1, 2, 3]:
            uses = []; scores_out = []
            for pool in pools:
                u, b = strategy_no_improvement(pool, M_max, patience=patience)
                uses.append(u); scores_out.append(b)
            avg_M = np.mean(uses)
            avg_score = np.mean(scores_out)
            losses = [baseline_scores[i] - scores_out[i] for i in range(len(pools))]
            avg_loss = float(np.mean(losses))
            saving = (1 - avg_M / avg_baseline_M) * 100

            results.append(Result(
                regime=regime_name,
                strategy=f'no_improve_p={patience}',
                avg_M_used=avg_M,
                avg_best_score=avg_score,
                quality_loss_vs_baseline=avg_loss,
                compute_saving_pct=saving,
            ))

    return results


# ── Mixed-regime experiment ──────────────────────────────────────────

def run_mixed_regime(M_max: int = 10, M_baseline: int = 5,
                     theta: float = 0.5,
                     mix=(0.4, 0.4, 0.2)) -> dict:
    """
    Realistic deployment: tasks come from a mix of difficulties.
    mix = (easy, medium, hard) fractions.

    Tests whether P(t) self-allocates: spending less on easy tasks,
    more on hard ones — which is the actual point of adaptive M.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    n_easy = int(N_TASKS_PER_REGIME * mix[0])
    n_med  = int(N_TASKS_PER_REGIME * mix[1])
    n_hard = N_TASKS_PER_REGIME - n_easy - n_med

    pools_by_regime = {
        'easy':   [sample_easy_task_scores(M_max, rng) for _ in range(n_easy)],
        'medium': [sample_medium_task_scores(M_max, rng) for _ in range(n_med)],
        'hard':   [sample_hard_task_scores(M_max, rng) for _ in range(n_hard)],
    }

    results_by_regime = {}
    for regime, pools in pools_by_regime.items():
        baseline_uses = []; baseline_scores = []
        adaptive_uses = []; adaptive_scores = []
        for pool in pools:
            bu, bs = strategy_fixed_M(pool, M_baseline)
            au, as_ = strategy_pt_adaptive(pool, M_max, theta=theta)
            baseline_uses.append(bu); baseline_scores.append(bs)
            adaptive_uses.append(au); adaptive_scores.append(as_)
        results_by_regime[regime] = dict(
            n=len(pools),
            baseline_M=np.mean(baseline_uses),
            adaptive_M=np.mean(adaptive_uses),
            baseline_score=np.mean(baseline_scores),
            adaptive_score=np.mean(adaptive_scores),
            quality_loss=np.mean(baseline_scores) - np.mean(adaptive_scores),
        )

    # Aggregate
    total_n = sum(r['n'] for r in results_by_regime.values())
    weighted_baseline_M = sum(r['n'] * r['baseline_M'] for r in results_by_regime.values()) / total_n
    weighted_adaptive_M = sum(r['n'] * r['adaptive_M'] for r in results_by_regime.values()) / total_n
    weighted_baseline_score = sum(r['n'] * r['baseline_score'] for r in results_by_regime.values()) / total_n
    weighted_adaptive_score = sum(r['n'] * r['adaptive_score'] for r in results_by_regime.values()) / total_n

    return dict(
        per_regime=results_by_regime,
        aggregate=dict(
            baseline_M=weighted_baseline_M,
            adaptive_M=weighted_adaptive_M,
            baseline_score=weighted_baseline_score,
            adaptive_score=weighted_adaptive_score,
            compute_saving=(1 - weighted_adaptive_M / weighted_baseline_M) * 100,
            quality_loss=weighted_baseline_score - weighted_adaptive_score,
        )
    )


# ── Reporting ────────────────────────────────────────────────────────

def print_results(results: list[Result]):
    print("\n" + "="*86)
    print("Per-Regime Results — fixed M=5 baseline vs P(t) adaptive vs no-improve baseline")
    print("="*86)

    by_regime = {}
    for r in results:
        by_regime.setdefault(r.regime, []).append(r)

    for regime in ['easy', 'medium', 'hard']:
        print(f"\n  {regime.upper()} regime ({N_TASKS_PER_REGIME} tasks)")
        print(f"  {'Strategy':<22} {'avg M':>8} {'avg best':>10} "
              f"{'quality loss':>14} {'compute saved':>14}")
        print(f"  {'-'*86}")
        for r in by_regime[regime]:
            print(f"  {r.strategy:<22} {r.avg_M_used:>8.2f} "
                  f"{r.avg_best_score:>10.4f} "
                  f"{r.quality_loss_vs_baseline:>+14.4f} "
                  f"{r.compute_saving_pct:>13.1f}%")


def print_mixed(mixed: dict):
    print("\n" + "="*86)
    print("Mixed-regime deployment (40% easy / 40% medium / 20% hard, theta=0.5)")
    print("="*86)
    print(f"\n  {'Regime':<10} {'n':>5} {'baseline M':>11} {'adaptive M':>11} "
          f"{'baseline best':>14} {'adaptive best':>14} {'quality loss':>14}")
    print(f"  {'-'*86}")
    for regime in ['easy', 'medium', 'hard']:
        r = mixed['per_regime'][regime]
        print(f"  {regime:<10} {r['n']:>5} {r['baseline_M']:>11.2f} {r['adaptive_M']:>11.2f} "
              f"{r['baseline_score']:>14.4f} {r['adaptive_score']:>14.4f} "
              f"{r['quality_loss']:>+14.4f}")
    a = mixed['aggregate']
    print(f"\n  Aggregate:")
    print(f"    Baseline avg M:     {a['baseline_M']:.2f}")
    print(f"    Adaptive avg M:     {a['adaptive_M']:.2f}")
    print(f"    Compute saving:     {a['compute_saving']:.1f}%")
    print(f"    Quality loss:       {a['quality_loss']:+.4f}")
    print(f"    Baseline avg best:  {a['baseline_score']:.4f}")
    print(f"    Adaptive avg best:  {a['adaptive_score']:.4f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("="*86)
    print("CoVer-VLA × P(t) Adaptive Sampling — Synthetic Simulator")
    print("="*86)
    print()
    print("CALIBRATION FROM PAPER:")
    print("  - CoVer-VLA published config:    K=8 rephrases × M=5 actions per rephrase")
    print("  - SIMPLER ID success rate:        57.0% (π0 + CoVer)")
    print("  - SIMPLER OOD success rate:       61.0% (π0 + CoVer)")
    print("  - This sim varies M (actions per rephrase). K is held constant.")
    print()
    print("SYNTHETIC ASSUMPTIONS (flagged honestly):")
    print("  - Score distributions: Beta(8,2) easy / Beta(4,4) medium / Beta(2,5) hard")
    print("  - i.i.d. scores within a rephrase (real CoVer-VLA has temperature; closely related)")
    print()

    # Per-regime experiment
    results = run_experiment(M_max=10, M_baseline=5)
    print_results(results)

    # Mixed-regime
    mixed = run_mixed_regime(M_max=10, M_baseline=5, theta=0.5,
                             mix=(0.4, 0.4, 0.2))
    print_mixed(mixed)

    # Verdict heuristic
    print("\n" + "="*86)
    print("VERDICT GUIDE")
    print("="*86)
    print("""
  Look at the mixed-regime aggregate row.

  STRONG positive signal: compute_saving > 30% with quality_loss < 0.02
    → adaptive M plausibly works on top of CoVer-VLA. Worth requesting
       real traces from Jacky.

  WEAK positive signal: compute_saving 10-30% with quality_loss < 0.05
    → suggestive but not strong. Request data only if conversation
       continues naturally; don't push.

  NO signal: compute_saving < 10%, OR quality_loss > 0.05
    → idea doesn't survive even on charitable synthetic data. Drop the
       adaptive-M angle for the blog post; pivot to a different fit
       (e.g., adaptive K rephrases instead of adaptive M actions).

  Also compare P(t) adaptive vs no-improvement baseline. If
  no-improvement-patience saves comparable compute with similar quality,
  then P(t) isn't earning its complexity in this regime — and we should
  flag that honestly when talking to Jacky.
""")


if __name__ == '__main__':
    main()
