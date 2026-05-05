"""
P(t) Meta-Controller — CUSUM-Gated Offensive/Defensive Switching
=================================================================

The synthesis hypothesis:

  P(t) is not a single signal. It's a family of signals, each appropriate
  for a different regime. CUSUM detects the active regime and selects
  the right family member.

  - DEFENSIVE P(t): adaptive baseline. Asks "is process plateauing?"
    Right tool for: stable regimes where we want to stop wasting compute.
  - OFFENSIVE P(t): prior-anchored baseline. Asks "are we exceeding
    expectations?"  Right tool for: hot-streak regimes where we want to
    allocate more compute.
  - CUSUM gate: detects regime change. Selects which P(t) is active.

Architecture:

      ┌─────────────────────┐
      │  defensive P(t)     │
      │  (adaptive baseline)│
      └──────────┬──────────┘
                 │
                 ▼ when stable/plateau regime
      ┌──────────────────────┐
      │   CUSUM regime gate  │ ◄── input s(t)
      └──────────┬───────────┘
                 ▲ when exceeding-prior regime
                 │
      ┌──────────┴──────────┐
      │  offensive P(t)     │
      │  (prior-anchored)   │
      └─────────────────────┘

Also fixes the threshold issue from pt_offensive.py — uses regime-relative
thresholds instead of absolute ones.

Test problem: same adaptive M bandit setup that defensive P(t) failed on.
The key question: does the meta-controller correctly self-allocate compute
across easy/medium/hard tasks, where neither standalone variant did?

Usage:
    python pt_meta_cusum.py
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_TASKS_PER_REGIME = 1000

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Score generators ─────────────────────────────────────────────────

def sample_easy(rng, M):    return rng.beta(8.0, 2.0, M)
def sample_medium(rng, M):  return rng.beta(4.0, 4.0, M)
def sample_hard(rng, M):    return rng.beta(2.0, 5.0, M)

REGIMES = {'easy': sample_easy, 'medium': sample_medium, 'hard': sample_hard}
REGIME_PRIORS = {'easy': 0.8, 'medium': 0.5, 'hard': 0.3}


# ── Defensive P(t): adaptive baseline ────────────────────────────────

def update_defensive(state, s, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """One step of defensive P(t) update."""
    er, ew, pw = state['er'], state['ew'], state['pw']
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
    state['er'], state['ew'], state['pw'] = er, ew, pw
    return pw, eff


# ── Offensive P(t): prior-anchored baseline ──────────────────────────

def update_offensive(state, s, prior, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """
    One step of offensive P(t) update.
    Uses smoothed efficiency against a fixed prior.
    """
    eff_smoothed = state['eff_smoothed']
    ew, pw = state['ew'], state['pw']
    eff_raw = s / max(prior, 1e-6)
    if eff_smoothed is None:
        eff_smoothed = eff_raw
    else:
        eff_smoothed = (1 - alpha) * eff_smoothed + alpha * eff_raw
    win = 1.0 if eff_smoothed >= 1.0 else 0.0
    a = 2.0 / (span + 1)
    ew = a * win + (1 - a) * ew
    inst = eff_smoothed * ew
    pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
    state['eff_smoothed'], state['ew'], state['pw'] = eff_smoothed, ew, pw
    return pw, eff_smoothed


# ── CUSUM regime detector ────────────────────────────────────────────

def update_cusum(state, s, prior, k=0.05, h=0.3):
    """
    Bidirectional CUSUM tracking deviation from prior.

    Args:
        state: dict with 'cusum_up', 'cusum_down', 'regime'
        s:     current signal value
        prior: anchored prior expectation (the "rational" expectation)
        k:     reference value (slack/dead zone) — typically σ/2 of signal
        h:     decision threshold — when to declare regime change

    Returns:
        regime: 'offensive' (s > prior consistently) or 'defensive' (default)
    """
    deviation = s - prior

    # CUSUM up: detects sustained outperformance
    state['cusum_up'] = max(0.0, state['cusum_up'] + deviation - k)
    # CUSUM down: detects sustained underperformance / stability
    state['cusum_down'] = max(0.0, state['cusum_down'] - deviation - k)

    # Regime decision with hysteresis (prevent flapping)
    if state['regime'] == 'defensive':
        if state['cusum_up'] > h:
            state['regime'] = 'offensive'
            state['cusum_up'] = 0.0  # reset on switch
    else:  # currently offensive
        if state['cusum_down'] > h:
            state['regime'] = 'defensive'
            state['cusum_down'] = 0.0

    return state['regime']


# ── Meta-controller: combines defensive + offensive via CUSUM ────────

def meta_controller_pt(scores, prior, k_cusum=0.05, h_cusum=0.3,
                        warmup=2):
    """
    Run scores through the meta-controller.
    Returns:
        active_pt:   array of active P(t) value (whichever variant fired)
        regime_log:  array of strings ('defensive' or 'offensive') per step
        defensive_pt: defensive P(t) trajectory (for inspection)
        offensive_pt: offensive P(t) trajectory (for inspection)
    """
    def_state = dict(er=None, ew=0.0, pw=0.0)
    off_state = dict(eff_smoothed=None, ew=0.0, pw=0.0)
    cusum_state = dict(cusum_up=0.0, cusum_down=0.0, regime='defensive')

    n = len(scores)
    active_pt = np.zeros(n)
    regime_log = []
    def_pt = np.zeros(n)
    off_pt = np.zeros(n)

    for t, s in enumerate(scores):
        s = float(s)

        # Update both variants in parallel
        d_pt, _ = update_defensive(def_state, s)
        o_pt, _ = update_offensive(off_state, s, prior)
        def_pt[t] = d_pt
        off_pt[t] = o_pt

        # Update CUSUM regime detector
        if t >= warmup:
            regime = update_cusum(cusum_state, s, prior, k=k_cusum, h=h_cusum)
        else:
            regime = 'defensive'  # default during warmup

        regime_log.append(regime)

        # Active P(t) is the variant matching current regime
        active_pt[t] = o_pt if regime == 'offensive' else d_pt

    return active_pt, regime_log, def_pt, off_pt


# ── Strategies ───────────────────────────────────────────────────────

def strategy_fixed_M(scores, M):
    return M, float(scores[:M].max())


def strategy_meta(scores, M_max, prior, theta_def=0.5, theta_off_factor=0.85,
                   k_cusum=0.05, h_cusum=0.3, M_min=2):
    """
    Meta-controller strategy:
      - When in defensive regime: stop if defensive P(t) >= theta_def
        (we've plateaued, no more gains)
      - When in offensive regime: continue while offensive P(t) >= theta_off
        (still in hot streak), stop when it drops
      
      theta_off is regime-relative: theta_off = theta_off_factor * prior
      This addresses the threshold issue from pt_offensive.py.
    """
    active_pt, regime_log, def_pt, off_pt = meta_controller_pt(
        scores[:M_max], prior, k_cusum=k_cusum, h_cusum=h_cusum)
    
    theta_off = theta_off_factor * prior
    best = -1.0
    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)
        if (i + 1) < M_min:
            continue

        regime = regime_log[i]
        pt_val = active_pt[i]

        if regime == 'defensive':
            # Stop if defensive plateau detected
            if pt_val >= theta_def:
                return i + 1, best
        else:  # offensive
            # Stop if offensive signal weakens
            if pt_val < theta_off:
                return i + 1, best

    return M_max, best


def strategy_defensive_only(scores, M_max, theta=0.5, M_min=2):
    """For comparison: pure defensive P(t)."""
    state = dict(er=None, ew=0.0, pw=0.0)
    best = -1.0
    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)
        pt, _ = update_defensive(state, s)
        if (i + 1) >= M_min and pt >= theta:
            return i + 1, best
    return M_max, best


def strategy_offensive_only(scores, M_max, prior, theta_factor=0.85, M_min=2):
    """
    For comparison: pure offensive P(t) with FIXED threshold issue.
    Uses regime-relative threshold = theta_factor * prior.
    Stops when offensive signal weakens (no longer outperforming).
    """
    state = dict(eff_smoothed=None, ew=0.0, pw=0.0)
    theta = theta_factor * prior
    best = -1.0
    for i in range(M_max):
        s = float(scores[i])
        best = max(best, s)
        pt, _ = update_offensive(state, s, prior)
        if (i + 1) >= M_min and pt < theta:
            return i + 1, best
    return M_max, best


# ── Experiment ───────────────────────────────────────────────────────

@dataclass
class Result:
    regime: str
    strategy: str
    avg_M: float
    avg_best: float
    quality_loss: float
    saved_pct: float


def run(M_max=10, M_baseline=5):
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for regime, gen in REGIMES.items():
        prior = REGIME_PRIORS[regime]
        pools = [gen(rng, M_max) for _ in range(N_TASKS_PER_REGIME)]

        # Baseline
        b_uses, b_scores = zip(*[strategy_fixed_M(p, M_baseline) for p in pools])
        baseline_M = np.mean(b_uses); baseline_score = np.mean(b_scores)
        results.append(Result(regime, f'fixed_M={M_baseline}',
                              baseline_M, baseline_score, 0.0, 0.0))

        # Defensive only (with corrected presentation)
        for theta in [0.4, 0.5, 0.6]:
            uses, scs = zip(*[strategy_defensive_only(p, M_max, theta=theta)
                              for p in pools])
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'def_θ={theta}',
                float(np.mean(uses)), float(np.mean(scs)),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

        # Offensive only — with FIXED regime-relative threshold
        for tf in [0.7, 0.85, 1.0]:
            uses, scs = zip(*[strategy_offensive_only(p, M_max, prior, theta_factor=tf)
                              for p in pools])
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'off_factor={tf}',
                float(np.mean(uses)), float(np.mean(scs)),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

        # Meta-controller (defensive + offensive + CUSUM gate)
        for h_c in [0.15, 0.30, 0.50]:
            uses, scs = zip(*[strategy_meta(p, M_max, prior, h_cusum=h_c)
                              for p in pools])
            losses = [b_scores[i] - scs[i] for i in range(len(pools))]
            results.append(Result(
                regime, f'meta_h={h_c}',
                float(np.mean(uses)), float(np.mean(scs)),
                float(np.mean(losses)),
                (1 - np.mean(uses) / baseline_M) * 100,
            ))

    return results


def run_mixed(M_max=10, M_baseline=5, mix=(0.4, 0.4, 0.2)):
    rng = np.random.default_rng(RNG_SEED + 1)
    counts = {
        'easy':   int(N_TASKS_PER_REGIME * mix[0]),
        'medium': int(N_TASKS_PER_REGIME * mix[1]),
        'hard':   N_TASKS_PER_REGIME - int(N_TASKS_PER_REGIME * mix[0])
                  - int(N_TASKS_PER_REGIME * mix[1]),
    }

    pools_with_meta = []
    for regime, n in counts.items():
        gen = REGIMES[regime]
        prior = REGIME_PRIORS[regime]
        for _ in range(n):
            pools_with_meta.append((gen(rng, M_max), prior, regime))

    strategies = {
        'fixed_M=5': lambda pool, prior: strategy_fixed_M(pool, M_baseline),
        'def_θ=0.5': lambda pool, prior: strategy_defensive_only(pool, M_max, theta=0.5),
        'off_factor=0.85': lambda pool, prior:
            strategy_offensive_only(pool, M_max, prior, theta_factor=0.85),
        'meta_h=0.30': lambda pool, prior: strategy_meta(pool, M_max, prior, h_cusum=0.30),
    }

    out = {}
    for sname, sfn in strategies.items():
        all_data = []
        per_regime = {'easy': [], 'medium': [], 'hard': []}
        baseline_data = []
        for pool, prior, regime in pools_with_meta:
            u, b = sfn(pool, prior)
            all_data.append((u, b))
            per_regime[regime].append((u, b))
            ub, bb = strategy_fixed_M(pool, M_baseline)
            baseline_data.append((ub, bb))

        avg_M = np.mean([x[0] for x in all_data])
        avg_best = np.mean([x[1] for x in all_data])
        avg_baseline_M = np.mean([x[0] for x in baseline_data])
        avg_baseline_best = np.mean([x[1] for x in baseline_data])

        out[sname] = dict(
            avg_M=float(avg_M),
            avg_best=float(avg_best),
            quality_loss=float(avg_baseline_best - avg_best),
            saved=float((1 - avg_M / avg_baseline_M) * 100),
            per_regime={
                r: dict(
                    avg_M=float(np.mean([x[0] for x in v])) if v else 0.0,
                    avg_best=float(np.mean([x[1] for x in v])) if v else 0.0,
                )
                for r, v in per_regime.items()
            },
        )
    return out


# ── Diagnostic: regime detection trace on synthetic episodes ─────────

def diagnostic_trace():
    """Show regime decisions on individual sample episodes."""
    print("\n" + "="*86)
    print("DIAGNOSTIC: Regime detection on individual episodes")
    print("="*86)
    rng = np.random.default_rng(RNG_SEED + 2)
    for regime, gen in REGIMES.items():
        prior = REGIME_PRIORS[regime]
        scores = gen(rng, 10)
        active_pt, regime_log, def_pt, off_pt = meta_controller_pt(
            scores, prior, h_cusum=0.30)
        print(f"\n  {regime.upper()} regime (prior={prior})")
        print(f"  {'t':>3} {'s(t)':>7} {'def_pt':>8} {'off_pt':>8} "
              f"{'regime':>11} {'active':>8}")
        print("  " + "-"*60)
        for t in range(10):
            mark = '←' if regime_log[t] == 'offensive' else ' '
            print(f"  {t:>3} {scores[t]:>7.4f} {def_pt[t]:>8.4f} {off_pt[t]:>8.4f} "
                  f"{regime_log[t]:>11} {active_pt[t]:>8.4f} {mark}")


# ── Reporting ────────────────────────────────────────────────────────

def print_per_regime(results):
    print("\n" + "="*86)
    print("Per-Regime — Defensive vs Offensive (fixed) vs Meta-controller")
    print("="*86)
    by_regime = {}
    for r in results: by_regime.setdefault(r.regime, []).append(r)
    for regime in ['easy', 'medium', 'hard']:
        prior = REGIME_PRIORS[regime]
        print(f"\n  {regime.upper()} regime  (prior={prior})")
        print(f"  {'Strategy':<22} {'avg M':>7} {'avg best':>10} "
              f"{'quality loss':>14} {'saved':>8}")
        print("  " + "-"*76)
        for r in by_regime[regime]:
            print(f"  {r.strategy:<22} {r.avg_M:>7.2f} "
                  f"{r.avg_best:>10.4f} "
                  f"{r.quality_loss:>+14.4f} "
                  f"{r.saved_pct:>7.1f}%")


def print_mixed(out):
    print("\n" + "="*86)
    print("Mixed deployment (40% easy / 40% medium / 20% hard)")
    print("="*86)
    print(f"\n  {'Strategy':<22} {'avg M':>7} {'avg best':>10} "
          f"{'quality loss':>14} {'saved':>8}")
    print("  " + "-"*76)
    for sname, m in out.items():
        print(f"  {sname:<22} {m['avg_M']:>7.2f} {m['avg_best']:>10.4f} "
              f"{m['quality_loss']:>+14.4f} {m['saved']:>7.1f}%")

    print(f"\n  Per-regime breakdown — KEY DIAGNOSTIC:")
    print(f"  {'Strategy':<22} {'easy M':>8} {'med M':>8} {'hard M':>8} "
          f"{'easy best':>11} {'med best':>11} {'hard best':>11}")
    print("  " + "-"*92)
    for sname, m in out.items():
        pr = m['per_regime']
        print(f"  {sname:<22} "
              f"{pr['easy']['avg_M']:>8.2f} "
              f"{pr['medium']['avg_M']:>8.2f} "
              f"{pr['hard']['avg_M']:>8.2f} "
              f"{pr['easy']['avg_best']:>11.4f} "
              f"{pr['medium']['avg_best']:>11.4f} "
              f"{pr['hard']['avg_best']:>11.4f}")


def main():
    print("="*86)
    print("P(t) Meta-Controller — CUSUM-Gated Offensive/Defensive Switching")
    print("="*86)
    print()
    print("ARCHITECTURE:")
    print("  defensive P(t)  —— asks 'is process plateauing?' (current ML version)")
    print("  offensive P(t)  —— asks 'are we exceeding prior?' (original trading)")
    print("  CUSUM gate      —— selects which question is active per timestep")
    print()
    print("FIXES from previous offensive simulator:")
    print("  - Regime-relative thresholds (factor × prior, not absolute)")
    print("  - Smoothed offensive efficiency to reduce per-step volatility")
    print("  - Hysteresis on regime switching to prevent flapping")
    print()
    print("KEY QUESTION: does meta-controller correctly self-allocate?")
    print("  - More compute on easy (where outperformance is real)")
    print("  - Less compute on hard (where prior is overoptimistic)")
    print("  - Better quality at comparable or better compute than either alone?")

    results = run()
    print_per_regime(results)
    mixed = run_mixed()
    print_mixed(mixed)
    diagnostic_trace()

    print(f"\n{'='*86}")
    print("INTERPRETATION")
    print("="*86)
    print("""
  In the per-regime breakdown of the meta_h=0.30 row:

  STRONG positive signal: meta_easy_M > meta_med_M > meta_hard_M
    → meta-controller correctly allocates more compute where outperformance
      is real, less where prior is overoptimistic. Original trading
      intuition vindicated. Major catalog reorganization warranted.

  COMPARABLE TO STANDALONE: meta achieves quality similar to defensive
    or offensive alone, no improvement from gating
    → CUSUM isn't earning its complexity. Class story holds (defensive
      and offensive are valid variants) but no synthesis claim.

  WORSE THAN STANDALONE: meta produces worse quality due to switching
    instability or threshold mismatch → architecture has hidden
    pathology. Document and don't pursue meta-controller direction.

  Look at the diagnostic trace: does CUSUM correctly switch regimes
  for hard tasks (should stay defensive) vs easy tasks (might switch
  to offensive when consistent outperformance is observed)?
""")


if __name__ == '__main__':
    main()
