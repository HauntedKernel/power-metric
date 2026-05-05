"""
P(t) Offensive Variant — Prior-Anchored Allocation
====================================================
Tests the conceptual fork identified in the GPT exchange:

  DEFENSIVE P(t) (current ML formulation):
    E(t) = s(t) / R(t-1) where R adapts to recent observations.
    Asks: "Is the signal consistent with recent expectations?"
    Use case: plateau detection, stopping, anomaly detection.

  OFFENSIVE P(t) (original trading intuition):
    E(t) = s(t) / B_prior  where B_prior is a FIXED rational expectation.
    Asks: "Is the signal exceeding what we expected before observing?"
    Use case: resource expansion, allocation, "press the edge".

The masochism diagnosis: defensive P(t) cannot distinguish "performing as
expected" from "exceeding expectations" because the baseline absorbs
outperformance into itself. If you expect 3R and get 4R, the baseline
becomes 4R, and now further 4R outcomes are merely "average."

The hypothesis: offensive P(t) with a fixed prior baseline should
correctly distinguish "doing fine" from "exceeding expectations" and
should be the right tool for adaptive scaling decisions.

This simulator tests offensive variant on the SAME bandit-shaped
problem (CoVer-VLA-style adaptive M action sampling) where defensive
P(t) failed (44.7% saved, 0.052 quality loss).

If offensive P(t) succeeds where defensive failed, this validates the
fork and unlocks the entire offensive lane of applications.

Usage:
    python pt_offensive.py
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_TASKS_PER_REGIME = 1000

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Score generators (same as cover_vla_simulator) ───────────────────

def sample_easy(rng, M):    return rng.beta(8.0, 2.0, M)   # mean ~0.80
def sample_medium(rng, M):  return rng.beta(4.0, 4.0, M)   # mean ~0.50
def sample_hard(rng, M):    return rng.beta(2.0, 5.0, M)   # mean ~0.29

REGIMES = {'easy': sample_easy, 'medium': sample_medium, 'hard': sample_hard}

# What we'd "rationally expect" before sampling — the prior anchor.
# In CoVer-VLA's case, the prior is the mean expected score given the task.
# This is the analog of "expected R = 3" in the trading framing.
REGIME_PRIORS = {'easy': 0.8, 'medium': 0.5, 'hard': 0.3}


# ── Defensive P(t) (current ML version, V1 EWMA baseline) ────────────

def pt_defensive(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """
    E(t) = s(t) / R(t-1)    where R adapts to s.
    The baseline IS the expectation. Outperformance gets absorbed.
    """
    er = None; ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            eff = s / er
            er = (1 - alpha) * er + alpha * s
        win = 1.0 if eff >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


# ── Offensive P(t) (prior-anchored) ──────────────────────────────────

def pt_offensive(scores, prior_baseline, lam=LAMBDA, span=EWMA_SPAN):
    """
    E(t) = s(t) / B_prior    where B_prior is FIXED.
    Original trading intuition: "actual / what we expected before observing".
    Outperformance accumulates instead of being absorbed.
    """
    ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        eff = s / max(prior_baseline, 1e-6)
        win = 1.0 if eff >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


def pt_offensive_smoothed(scores, prior_baseline,
                            alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """
    Hybrid: E(t) anchored to prior, but the efficiency itself is smoothed.
    GPT's suggested cleanest form:
        efficiency_smoothed = EMA(s(t) / B_prior)
    Avoids per-step noise without redefining the benchmark.
    """
    eff_smoothed = None
    ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        eff_raw = s / max(prior_baseline, 1e-6)
        if eff_smoothed is None:
            eff_smoothed = eff_raw
        else:
            eff_smoothed = (1 - alpha) * eff_smoothed + alpha * eff_raw
        win = 1.0 if eff_smoothed >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff_smoothed * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


# ── Adaptive M strategies under each variant ─────────────────────────

def strategy_fixed_M(scores, M):
    return M, float(scores[:M].max())


def strategy_offensive_continue_while_hot(
    scores, M_max, prior_baseline, theta_hot=1.2, M_min=2, M_max_safety=10,
):
    """
    Offensive interpretation: keep sampling AS LONG AS we're exceeding
    expectations. P(t) here represents "are we still in a hot streak?"
    
    Stop when:
      - We've sampled M_min minimum AND
      - P(t) drops below theta_hot (no longer outperforming prior)
      - OR we hit M_max safety cap
    
    This is the "scale into opportunity" variant: don't stop early on
    a hot streak, do stop early when the prior was overoptimistic.
    """
    powers = pt_offensive(scores[:M_max_safety], prior_baseline)
    best = -1.0
    for i in range(M_max_safety):
        s = float(scores[i])
        best = max(best, s)
        if (i + 1) >= M_min:
            # Stop if no longer exceeding prior — task isn't yielding the
            # outperformance we hoped for, no point continuing.
            if powers[i] < theta_hot:
                return i + 1, best
    return M_max_safety, best


def strategy_offensive_stop_when_satisfied(
    scores, M_max, prior_baseline, theta_satisfied=1.5, M_min=2,
):
    """
    Different offensive interpretation: stop EARLY when we have a clear
    win. If P(t) crosses theta_satisfied, we have strong evidence this
    task is yielding much better than prior expected — we have a
    confidence-bounded best, no need to keep sampling.
    
    Stop when:
      - M >= M_min AND P(t) >= theta_satisfied (clearly winning)
      - OR M = M_max (cap)
    """
    powers = pt_offensive(scores[:M_max], prior_baseline)
    best = -1.0
    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)
        if (i + 1) >= M_min and powers[i] >= theta_satisfied:
            return i + 1, best
    return M_max, best


def strategy_defensive_baseline(scores, M_max, theta=0.5, M_min=2):
    """For comparison — the defensive variant we already tested."""
    powers = pt_defensive(scores[:M_max])
    best = -1.0
    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)
        if (i + 1) >= M_min and powers[i] >= theta:
            return i + 1, best
    return M_max, best


# ── Per-regime experiment ────────────────────────────────────────────

@dataclass
class Result:
    regime: str
    strategy: str
    avg_M: float
    avg_best: float
    quality_loss: float
    compute_saved_pct: float


def run(M_max=10, M_baseline=5):
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for regime, gen in REGIMES.items():
        prior = REGIME_PRIORS[regime]
        pools = [gen(rng, M_max) for _ in range(N_TASKS_PER_REGIME)]

        # Fixed M=5 baseline
        b_uses, b_scores = [], []
        for p in pools:
            u, b = strategy_fixed_M(p, M_baseline)
            b_uses.append(u); b_scores.append(b)
        results.append(Result(
            regime, f'fixed_M={M_baseline}',
            np.mean(b_uses), np.mean(b_scores), 0.0, 0.0,
        ))
        baseline_M = np.mean(b_uses)
        baseline_score = np.mean(b_scores)

        # Defensive P(t) (the previously tested approach)
        for theta in [0.4, 0.5, 0.6]:
            uses, scs = [], []
            for p in pools:
                u, b = strategy_defensive_baseline(p, M_max, theta=theta)
                uses.append(u); scs.append(b)
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'defensive_θ={theta}',
                np.mean(uses), np.mean(scs),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

        # Offensive: continue while hot
        for theta_hot in [0.8, 1.0, 1.2]:
            uses, scs = [], []
            for p in pools:
                u, b = strategy_offensive_continue_while_hot(
                    p, M_max, prior, theta_hot=theta_hot)
                uses.append(u); scs.append(b)
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'off_continue_θhot={theta_hot}',
                np.mean(uses), np.mean(scs),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

        # Offensive: stop when satisfied
        for theta_sat in [1.2, 1.5, 1.8]:
            uses, scs = [], []
            for p in pools:
                u, b = strategy_offensive_stop_when_satisfied(
                    p, M_max, prior, theta_satisfied=theta_sat)
                uses.append(u); scs.append(b)
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'off_stop_θsat={theta_sat}',
                np.mean(uses), np.mean(scs),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

    return results


# ── Mixed regime ─────────────────────────────────────────────────────

def run_mixed(M_max=10, M_baseline=5, mix=(0.4, 0.4, 0.2)):
    rng = np.random.default_rng(RNG_SEED + 1)
    n_easy = int(N_TASKS_PER_REGIME * mix[0])
    n_med  = int(N_TASKS_PER_REGIME * mix[1])
    n_hard = N_TASKS_PER_REGIME - n_easy - n_med
    counts = {'easy': n_easy, 'medium': n_med, 'hard': n_hard}

    pools_with_priors = []
    for regime, n in counts.items():
        gen = REGIMES[regime]
        prior = REGIME_PRIORS[regime]
        for _ in range(n):
            pools_with_priors.append((gen(rng, M_max), prior, regime))

    strategies = {
        'fixed_M=5': lambda pool, prior: strategy_fixed_M(pool, M_baseline),
        'defensive_θ=0.5': lambda pool, prior:
            strategy_defensive_baseline(pool, M_max, theta=0.5),
        'off_continue_θhot=1.0': lambda pool, prior:
            strategy_offensive_continue_while_hot(pool, M_max, prior, theta_hot=1.0),
        'off_stop_θsat=1.5': lambda pool, prior:
            strategy_offensive_stop_when_satisfied(pool, M_max, prior, theta_satisfied=1.5),
    }

    out = {}
    for sname, sfn in strategies.items():
        all_uses, all_scs = [], []
        per_reg = {'easy': [], 'medium': [], 'hard': []}
        for pool, prior, regime in pools_with_priors:
            u, b = sfn(pool, prior)
            all_uses.append(u); all_scs.append(b)
            per_reg[regime].append((u, b))

        baseline_uses = []
        baseline_scs = []
        for pool, _, _ in pools_with_priors:
            u, b = strategy_fixed_M(pool, M_baseline)
            baseline_uses.append(u); baseline_scs.append(b)

        out[sname] = dict(
            avg_M=float(np.mean(all_uses)),
            avg_best=float(np.mean(all_scs)),
            quality_loss=float(np.mean(baseline_scs) - np.mean(all_scs)),
            saving=(1 - np.mean(all_uses) / np.mean(baseline_uses)) * 100,
            per_regime={
                r: dict(avg_M=np.mean([x[0] for x in v]),
                        avg_best=np.mean([x[1] for x in v]))
                for r, v in per_reg.items()
            },
        )
    return out


# ── Reporting ────────────────────────────────────────────────────────

def print_per_regime(results):
    print("\n" + "="*86)
    print("Per-Regime — Defensive vs Offensive P(t)")
    print("="*86)
    by_regime = {}
    for r in results: by_regime.setdefault(r.regime, []).append(r)
    for regime in ['easy', 'medium', 'hard']:
        print(f"\n  {regime.upper()} regime  (prior expectation = {REGIME_PRIORS[regime]})")
        print(f"  {'Strategy':<28} {'avg M':>7} {'avg best':>10} "
              f"{'quality loss':>14} {'saved':>8}")
        print("  " + "-"*82)
        for r in by_regime[regime]:
            print(f"  {r.strategy:<28} {r.avg_M:>7.2f} "
                  f"{r.avg_best:>10.4f} "
                  f"{r.quality_loss:>+14.4f} "
                  f"{r.compute_saved_pct:>7.1f}%")


def print_mixed(out):
    print("\n" + "="*86)
    print("Mixed deployment (40% easy / 40% medium / 20% hard)")
    print("="*86)
    print(f"\n  {'Strategy':<28} {'avg M':>7} {'avg best':>10} "
          f"{'quality loss':>14} {'saved':>8}")
    print("  " + "-"*82)
    for sname, m in out.items():
        print(f"  {sname:<28} {m['avg_M']:>7.2f} {m['avg_best']:>10.4f} "
              f"{m['quality_loss']:>+14.4f} {m['saving']:>7.1f}%")

    print(f"\n  Per-regime breakdown:")
    print(f"  {'Strategy':<28} {'easy M':>8} {'med M':>8} {'hard M':>8} "
          f"{'easy best':>10} {'med best':>10} {'hard best':>10}")
    print("  " + "-"*92)
    for sname, m in out.items():
        pr = m['per_regime']
        print(f"  {sname:<28} "
              f"{pr['easy']['avg_M']:>8.2f} "
              f"{pr['medium']['avg_M']:>8.2f} "
              f"{pr['hard']['avg_M']:>8.2f} "
              f"{pr['easy']['avg_best']:>10.4f} "
              f"{pr['medium']['avg_best']:>10.4f} "
              f"{pr['hard']['avg_best']:>10.4f}")


def main():
    print("="*86)
    print("P(t) Offensive Variant — Testing Prior-Anchored Allocation")
    print("="*86)
    print()
    print("CONCEPTUAL FORK (from GPT discussion):")
    print("  Defensive P(t):  E(t) = s(t) / R(t-1) — baseline adapts")
    print("                   → 'is process plateauing?' (current ML version)")
    print("  Offensive P(t):  E(t) = s(t) / B_prior — baseline FIXED")
    print("                   → 'are we exceeding prior expectation?' (original)")
    print()
    print("The masochism: defensive baseline absorbs outperformance into itself.")
    print("Cannot distinguish 'doing fine' from 'exceeding expectations'.")
    print()
    print("Test: Adaptive M on bandit problems where defensive P(t) failed")
    print("(44.7% saved, 0.052 quality loss in the original CoVer-VLA test).")
    print()
    print("Two offensive strategies tested:")
    print("  - off_continue: keep sampling while P(t) > θ_hot (press the edge)")
    print("  - off_stop:     stop early when P(t) > θ_sat (confident win)")

    results = run()
    print_per_regime(results)
    mixed = run_mixed()
    print_mixed(mixed)

    print(f"\n{'='*86}")
    print("INTERPRETATION")
    print("="*86)
    print("""
  The decisive comparison is offensive vs defensive in mixed regime.

  STRONG positive signal: offensive variant achieves higher quality
    (lower loss) at comparable or better compute saving than defensive.
    → validates the offensive/defensive fork. Unlocks new lane of
       applications. Worth a major catalog reorganization.

  REGIME-SPECIFIC: offensive wins in some regimes (e.g. easy where
    outperformance is real) and loses in others (e.g. hard where
    prior is overoptimistic). → both variants have their place.
    Class story strengthens.

  NO improvement: offensive doesn't help even with prior anchoring.
    → the masochism diagnosis is incomplete. Bandit problems may
       be fundamentally outside P(t)'s natural shape, regardless of
       baseline philosophy.

  Key question: does offensive_continue allocate MORE compute to
  easy tasks (where we're exceeding expectations) and LESS to hard
  tasks (where we're not)? That would be the original trading
  intuition vindicated: 'press the edge while it exists, fold when
  expectations were wrong.'

  Look at the per-regime breakdown of avg_M for off_continue:
  - If easy_M > medium_M > hard_M, offensive variant correctly
    self-allocates by exceeding-expectations signal. Win.
  - If pattern is flat or reversed, the framework still struggles.
""")


if __name__ == '__main__':
    main()
