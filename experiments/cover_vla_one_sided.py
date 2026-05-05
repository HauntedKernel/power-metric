"""
P(t) Episode Monitor — One-Sided Efficiency Variant
====================================================
Tests whether a one-sided efficiency formulation rescues the episode
monitoring result that failed with standard P(t).

Diagnosis from previous run:
  Standard P(t) formulation has symmetric efficiency: E(t) = s(t)/R(t-1).
  On a noisy-but-healthy episode, s oscillates around the baseline R.
  When s dips slightly below R, E < 1, win=0, W decays, P decays.
  This is the same dynamic whether the dip is "healthy noise" or
  "real degradation." Result: 81.6% false positive rate on noisy success.

Hypothesis:
  The asymmetry is the fix. We don't actually care about beating the
  baseline (saturating wins). We care about anomalous shortfall. So:

  E(t) = min(1, s(t) / R(t-1))     [one-sided: cap at 1]
  win  = 1[E(t) >= 1 - tol]         [tolerance band]

  This way:
  - When s > R, E saturates at 1, win=1 (normal success)
  - When s ≈ R (noise), E ≈ 1, win=1 (still healthy)
  - When s << R, E drops below 1, win=0, W decays, P decays
    (this is the actual failure signal)

We test 3 variants here against standard P(t) baseline:
  V1_standard       — published baseline (the one that failed)
  V_one_sided       — capped efficiency, normal win threshold
  V_floor_baseline  — R cannot fall below R_min once established
                      (hard-floor variant; protects baseline from chasing
                       the failure trajectory down)
  V_combined        — both fixes together

Same episode types and detectors as cover_vla_episode_monitor.py.
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_EPISODES_PER_TYPE = 500
EPISODE_LENGTH = 30

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Episode generators (same as before) ──────────────────────────────

def gen_success(rng, T=EPISODE_LENGTH):
    base = 0.75 + 0.10 * np.linspace(0, 1, T)
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_gradual_failure(rng, T=EPISODE_LENGTH, fail_start=0.4):
    fail_idx = int(T * fail_start)
    base = np.zeros(T)
    base[:fail_idx] = 0.78 + np.linspace(0, 0.05, fail_idx)
    decline = np.linspace(0, 0.5, T - fail_idx)
    base[fail_idx:] = 0.78 - decline
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_cliff_failure(rng, T=EPISODE_LENGTH, cliff_at=0.7):
    cliff_idx = int(T * cliff_at)
    base = np.zeros(T)
    base[:cliff_idx] = 0.80
    base[cliff_idx:] = 0.30
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_misdirected(rng, T=EPISODE_LENGTH):
    base = 0.70 - 0.30 * np.linspace(0, 1, T)
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_noisy_success(rng, T=EPISODE_LENGTH):
    base = 0.72 * np.ones(T)
    noise = rng.normal(0, 0.12, T)
    return np.clip(base + noise, 0.0, 1.0)

EPISODE_TYPES = {
    'success':           (gen_success, False),
    'gradual_failure':   (gen_gradual_failure, True),
    'cliff_failure':     (gen_cliff_failure, True),
    'misdirected':       (gen_misdirected, True),
    'noisy_success':     (gen_noisy_success, False),
}


# ── P(t) variants ────────────────────────────────────────────────────

def pt_standard(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """V1: Published. E = s/R, symmetric, win = 1[E >= 1]."""
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


def pt_one_sided(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN, tol=0.02):
    """
    One-sided: E = min(1, s/R), win = 1[E >= 1 - tol].
    The cap means outperforming doesn't help (already at max).
    The tolerance means small dips don't immediately count as losses.
    """
    er = None; ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            raw_eff = s / er
            eff = min(1.0, raw_eff)  # one-sided cap
            er = (1 - alpha) * er + alpha * s
        win = 1.0 if eff >= (1.0 - tol) else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


def pt_floor_baseline(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN,
                       warmup=5):
    """
    Hard-floor R: once warmup observations seen, R cannot fall below
    the warmup mean. Prevents baseline from chasing a failure trajectory
    down (which would mask the failure).
    """
    er = None; ew = 0.0; pw = 0.0
    r_floor = None
    warmup_obs = []
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            eff = s / er
            er = (1 - alpha) * er + alpha * s
            if r_floor is not None:
                er = max(er, r_floor)  # apply floor
        warmup_obs.append(s)
        if t == warmup - 1:
            r_floor = float(np.mean(warmup_obs))
        win = 1.0 if eff >= 1.0 else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


def pt_combined(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN,
                tol=0.02, warmup=5):
    """One-sided efficiency + floor baseline."""
    er = None; ew = 0.0; pw = 0.0
    r_floor = None
    warmup_obs = []
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0; er = max(s, 1e-6)
        else:
            raw_eff = s / er
            eff = min(1.0, raw_eff)
            er = (1 - alpha) * er + alpha * s
            if r_floor is not None:
                er = max(er, r_floor)
        warmup_obs.append(s)
        if t == warmup - 1:
            r_floor = float(np.mean(warmup_obs))
        win = 1.0 if eff >= (1.0 - tol) else 0.0
        a = 2.0 / (span + 1)
        ew = a * win + (1 - a) * ew
        inst = eff * ew
        pw = np.exp(-lam) * pw + (1 - np.exp(-lam)) * inst
        powers[t] = pw
    return powers


VARIANTS = {
    'V1_standard':      pt_standard,
    'V_one_sided':      pt_one_sided,
    'V_floor':          pt_floor_baseline,
    'V_combined':       pt_combined,
}


# ── Detection ────────────────────────────────────────────────────────

def detect(powers, theta, patience=2, t_min=5):
    consecutive_low = 0
    for t in range(t_min, len(powers)):
        if powers[t] < theta:
            consecutive_low += 1
            if consecutive_low >= patience:
                return True, t
        else:
            consecutive_low = 0
    return False, -1


# ── Experiment ───────────────────────────────────────────────────────

def run_variant(variant_fn, theta, n=N_EPISODES_PER_TYPE):
    rng = np.random.default_rng(RNG_SEED)
    episodes = {ep: [gen(rng) for _ in range(n)]
                for ep, (gen, _) in EPISODE_TYPES.items()}
    out = {}
    for ep_type, eps in episodes.items():
        is_fail = EPISODE_TYPES[ep_type][1]
        detected = []
        steps = []
        for scores in eps:
            powers = variant_fn(scores)
            d, t = detect(powers, theta)
            detected.append(d)
            if d: steps.append(t)
        out[ep_type] = dict(
            is_failure=is_fail,
            detect_rate=np.mean(detected),
            avg_step=np.mean(steps) if steps else float('nan'),
        )
    return out


def find_best_theta(variant_fn, theta_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """
    Find theta that maximizes (avg TP - avg FP).
    Returns best theta and its metrics.
    """
    best = None
    for theta in theta_grid:
        out = run_variant(variant_fn, theta)
        tp = np.mean([v['detect_rate'] for k, v in out.items() if v['is_failure']])
        fp = np.mean([v['detect_rate'] for k, v in out.items() if not v['is_failure']])
        margin = tp - fp
        if best is None or margin > best[0]:
            best = (margin, theta, tp, fp, out)
    return best


def main():
    print("="*92)
    print("P(t) Variants — Episode Monitoring with One-Sided Efficiency")
    print("="*92)
    print()
    print("Diagnosis: V1 standard had 81.6% FP on noisy_success because symmetric")
    print("efficiency punishes downward noise as much as upward. Testing variants")
    print("that should be more noise-tolerant.")
    print()

    # Per-variant best operating point
    print("="*92)
    print("Best operating point per variant (maximizes TP - FP)")
    print("="*92)
    print(f"\n  {'Variant':<18} {'best θ':>8} {'avg TP':>9} {'avg FP':>9} {'margin':>9}")
    print("  " + "-"*78)
    for vname, vfn in VARIANTS.items():
        margin, theta, tp, fp, out = find_best_theta(vfn)
        print(f"  {vname:<18} {theta:>8.2f} {tp*100:>8.1f}% {fp*100:>8.1f}% "
              f"{margin*100:>+8.1f}%")

    # Detailed breakdown per variant at best theta
    for vname, vfn in VARIANTS.items():
        margin, theta, tp, fp, out = find_best_theta(vfn)
        print(f"\n{'─'*92}")
        print(f"{vname}  (best θ={theta})")
        print(f"{'─'*92}")
        print(f"  {'Episode type':<22} {'fail?':>6} {'detect rate':>13} {'avg step':>10}")
        print("  " + "-"*60)
        for ep_type in ['success', 'noisy_success',
                       'gradual_failure', 'cliff_failure', 'misdirected']:
            v = out[ep_type]
            mark = '✓' if v['is_failure'] else 'X'
            print(f"  {ep_type:<22} {mark:>6} "
                  f"{v['detect_rate']*100:>12.1f}% "
                  f"{v['avg_step']:>10.1f}")

    # Threshold sensitivity for one_sided (the hypothesis)
    print(f"\n{'='*92}")
    print("Threshold sensitivity — V_one_sided")
    print("="*92)
    print(f"\n  {'θ':<6} {'TP grad':>9} {'TP cliff':>10} {'TP misdir':>11} "
          f"{'FP succ':>9} {'FP noisy':>10}")
    print("  " + "-"*70)
    for theta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        out = run_variant(pt_one_sided, theta)
        print(f"  {theta:<6.2f} "
              f"{out['gradual_failure']['detect_rate']*100:>8.1f}% "
              f"{out['cliff_failure']['detect_rate']*100:>9.1f}% "
              f"{out['misdirected']['detect_rate']*100:>10.1f}% "
              f"{out['success']['detect_rate']*100:>8.1f}% "
              f"{out['noisy_success']['detect_rate']*100:>9.1f}%")

    print(f"\n{'='*92}")
    print("INTERPRETATION")
    print("="*92)
    print("""
  Goal: V_one_sided or V_combined achieves >80% TP on all three failure
        types AND <15% FP on both success types at some theta.

  If yes → fundamental P(t) class supports one-sided variants for
           failure detection regimes. Real lead worth bringing to Jacky.

  If no  → static-signal failure detection is genuinely outside P(t)'s
           comfortable regime. Document and move on. Not every problem
           is a P(t) problem.

  Compare to RunningAvg from previous test (100/100/99.8 TP, 0/0 FP).
  P(t) variants need to AT LEAST match this to be worth using.
  Beating it would require some property RunningAvg lacks (e.g.,
  earlier detection lead time at comparable accuracy).
""")


if __name__ == '__main__':
    main()
