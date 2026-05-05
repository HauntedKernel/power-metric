"""
CoVer-VLA × P(t) Episode-Level Monitoring Simulator
=====================================================
The deeper fit: instead of using P(t) to decide how many candidates
to sample within a single timestep (the bandit problem that P(t)
doesn't fit), use P(t) to monitor episode health ACROSS timesteps.

The setup:
  CoVer-VLA runs at 2.2 Hz on a robot. At each timestep t the verifier
  produces a top score s(t) = max over the K*M candidates of the
  alignment score for the action it chose to execute. Across an episode
  of T timesteps, we have a trajectory s(1), s(2), ..., s(T).

  Their published method does NOT track this trajectory. They pick the
  best action at each timestep and execute. There's no "is the episode
  going off track" detector.

P(t) on the verifier-top-score sequence is exactly the kind of
trajectory-monitoring problem the framework was built for — same
template as Paper 13's P_old(t) on forgetting, Paper 1's training
health, etc.

The proposed claim:
  P(t) on the verifier top score detects failing episodes early enough
  to trigger replanning, with low false-positive rate on healthy
  episodes.

Synthetic episode types (all assumed; flagged):
  - SUCCESS: scores stable or rising over T timesteps
  - GRADUAL FAILURE: scores decline smoothly mid-episode
  - CLIFF FAILURE: scores collapse sharply near end
  - MISDIRECTED: scores high early then drift down (drift)
  - NOISY SUCCESS: scores oscillate but trend stable

For each episode we measure:
  - Did P(t) cross below threshold?
  - At what timestep did it cross? (compared to actual failure point)
  - On success episodes, did P(t) falsely fire?

This is the simulation we couldn't run on real data without going to
cloud GPUs and running their full pipeline. If the synthetic result
shows P(t) cleanly separates failure from success trajectories, that's
the strongest possible case for asking Jacky for real rollout traces.

Run from C:\\Users\\Carolina\\:
    python cover_vla_episode_monitor.py
"""

import numpy as np
from dataclasses import dataclass

RNG_SEED = 42
N_EPISODES_PER_TYPE = 500
EPISODE_LENGTH = 30  # ~13 second rollout at 2.2 Hz

ALPHA = 0.3
LAMBDA = 0.5
EWMA_SPAN = 3


# ── Episode generators ───────────────────────────────────────────────

def gen_success(rng, T=EPISODE_LENGTH):
    """Healthy episode: scores stable or rising. Mean ~0.7-0.85, low noise."""
    base = 0.75 + 0.10 * np.linspace(0, 1, T)  # gentle improvement
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_gradual_failure(rng, T=EPISODE_LENGTH, fail_start=0.4):
    """
    Scores high early, gradually decline starting at fail_start fraction.
    Models robot drifting away from task semantics over time.
    """
    fail_idx = int(T * fail_start)
    base = np.zeros(T)
    base[:fail_idx] = 0.78 + np.linspace(0, 0.05, fail_idx)
    decline = np.linspace(0, 0.5, T - fail_idx)
    base[fail_idx:] = 0.78 - decline
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_cliff_failure(rng, T=EPISODE_LENGTH, cliff_at=0.7):
    """
    Scores high until late-episode collapse. Models catastrophic
    misidentification (robot suddenly grabs wrong object, etc.).
    """
    cliff_idx = int(T * cliff_at)
    base = np.zeros(T)
    base[:cliff_idx] = 0.80
    base[cliff_idx:] = 0.30  # sharp drop
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_misdirected(rng, T=EPISODE_LENGTH):
    """
    Scores moderate-high early, drift down throughout. Models the robot
    pursuing a plausible-but-wrong interpretation from start.
    """
    base = 0.70 - 0.30 * np.linspace(0, 1, T)
    noise = rng.normal(0, 0.05, T)
    return np.clip(base + noise, 0.0, 1.0)

def gen_noisy_success(rng, T=EPISODE_LENGTH):
    """High variance but trend stable. Tests false-positive rate."""
    base = 0.72 * np.ones(T)
    noise = rng.normal(0, 0.12, T)  # higher noise
    return np.clip(base + noise, 0.0, 1.0)

EPISODE_TYPES = {
    'success':           (gen_success, False),
    'gradual_failure':   (gen_gradual_failure, True),
    'cliff_failure':     (gen_cliff_failure, True),
    'misdirected':       (gen_misdirected, True),
    'noisy_success':     (gen_noisy_success, False),
}


# ── P(t) tracker ─────────────────────────────────────────────────────

def compute_pt_trajectory(scores, alpha=ALPHA, lam=LAMBDA, span=EWMA_SPAN):
    """Run P(t) over an episode score trajectory."""
    er = None; ew = 0.0; pw = 0.0
    powers = np.zeros(len(scores))
    for t, s in enumerate(scores):
        s = float(s)
        if er is None:
            eff = 1.0
            er = max(s, 1e-6)
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


def detect_failure(powers, theta=0.4, patience=2, t_min=5):
    """
    Trigger detection if P(t) < theta for `patience` consecutive timesteps,
    after a warmup of t_min steps.
    Returns: (detected: bool, detection_timestep: int or -1)
    """
    consecutive_low = 0
    for t in range(t_min, len(powers)):
        if powers[t] < theta:
            consecutive_low += 1
            if consecutive_low >= patience:
                return True, t
        else:
            consecutive_low = 0
    return False, -1


# ── Baselines ────────────────────────────────────────────────────────

def detect_failure_running_avg(scores, theta=0.5, patience=3, t_min=5,
                                window=5):
    """Baseline: trigger if running mean over `window` < theta."""
    consecutive_low = 0
    for t in range(t_min, len(scores)):
        win = scores[max(0, t-window+1):t+1]
        if win.mean() < theta:
            consecutive_low += 1
            if consecutive_low >= patience:
                return True, t
        else:
            consecutive_low = 0
    return False, -1


def detect_failure_drop_threshold(scores, drop=0.2, patience=2, t_min=5,
                                    baseline_window=5):
    """Baseline: trigger if score dropped by `drop` from running early avg."""
    consecutive_low = 0
    for t in range(t_min, len(scores)):
        baseline = scores[:baseline_window].mean()
        if scores[t] < baseline - drop:
            consecutive_low += 1
            if consecutive_low >= patience:
                return True, t
        else:
            consecutive_low = 0
    return False, -1


# ── Experiment ───────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    detector_name: str
    episode_type: str
    is_failure: bool
    detection_rate: float       # fraction detected
    false_positive_rate: float  # only meaningful for non-failure types
    avg_detect_step: float      # only meaningful when detected
    avg_lead_time: float        # how much earlier than episode end (failure cases)


def run_experiment(theta_pt=0.4, theta_running=0.5, drop_threshold=0.2):
    rng = np.random.default_rng(RNG_SEED)

    # Generate episodes
    episodes = {}
    for ep_type, (gen, is_fail) in EPISODE_TYPES.items():
        episodes[ep_type] = [gen(rng) for _ in range(N_EPISODES_PER_TYPE)]

    detectors = {
        'P(t)':            lambda scores: detect_failure(
                                compute_pt_trajectory(scores), theta=theta_pt),
        'RunningAvg':      lambda scores: detect_failure_running_avg(
                                scores, theta=theta_running),
        'DropThreshold':   lambda scores: detect_failure_drop_threshold(
                                scores, drop=drop_threshold),
    }

    results = []
    for det_name, det_fn in detectors.items():
        for ep_type, eps in episodes.items():
            is_fail = EPISODE_TYPES[ep_type][1]
            detected = []
            detect_steps = []
            for ep in eps:
                d, step = det_fn(ep)
                detected.append(d)
                if d:
                    detect_steps.append(step)
            detect_rate = np.mean(detected)
            avg_step = np.mean(detect_steps) if detect_steps else float('nan')
            lead = (EPISODE_LENGTH - avg_step) if detect_steps else float('nan')
            results.append(DetectorResult(
                detector_name=det_name,
                episode_type=ep_type,
                is_failure=is_fail,
                detection_rate=detect_rate,
                false_positive_rate=detect_rate if not is_fail else float('nan'),
                avg_detect_step=avg_step,
                avg_lead_time=lead,
            ))
    return results


def print_results(results):
    print("\n" + "="*92)
    print("Episode-Level Failure Detection — P(t) vs RunningAvg vs DropThreshold")
    print("="*92)

    by_det = {}
    for r in results: by_det.setdefault(r.detector_name, []).append(r)

    for det_name, rows in by_det.items():
        print(f"\n  Detector: {det_name}")
        print(f"  {'Episode type':<22} {'fail?':>6} {'detect rate':>13} "
              f"{'avg step':>10} {'lead time':>11}")
        print("  " + "-"*88)
        for r in rows:
            mark = '✓' if r.is_failure else 'X'
            print(f"  {r.episode_type:<22} {mark:>6} "
                  f"{r.detection_rate*100:>12.1f}% "
                  f"{r.avg_detect_step:>10.1f} {r.avg_lead_time:>10.1f}")

    # Summary table: TP rate (failures detected) vs FP rate (success false-fires)
    print("\n" + "="*92)
    print("Summary: True Positive Rate (failures detected) vs False Positive Rate")
    print("="*92)
    print(f"\n  {'Detector':<18} {'TP gradual':>10} {'TP cliff':>10} {'TP misdir':>10} "
          f"{'FP success':>11} {'FP noisy':>9}")
    print("  " + "-"*82)
    for det_name, rows in by_det.items():
        d = {r.episode_type: r.detection_rate for r in rows}
        print(f"  {det_name:<18} "
              f"{d['gradual_failure']*100:>9.1f}% "
              f"{d['cliff_failure']*100:>9.1f}% "
              f"{d['misdirected']*100:>9.1f}% "
              f"{d['success']*100:>10.1f}% "
              f"{d['noisy_success']*100:>8.1f}%")


def print_threshold_sensitivity():
    print("\n" + "="*92)
    print("P(t) Threshold Sensitivity — TPR vs FPR")
    print("="*92)
    print()
    print(f"  {'theta':<8} {'TP gradual':>11} {'TP cliff':>10} {'TP misdir':>11} "
          f"{'FP success':>12} {'FP noisy':>10}")
    print("  " + "-"*78)
    for theta in [0.2, 0.3, 0.4, 0.5, 0.6]:
        results = run_experiment(theta_pt=theta)
        d = {(r.detector_name, r.episode_type): r.detection_rate for r in results}
        print(f"  {theta:<8.2f} "
              f"{d[('P(t)','gradual_failure')]*100:>10.1f}% "
              f"{d[('P(t)','cliff_failure')]*100:>9.1f}% "
              f"{d[('P(t)','misdirected')]*100:>10.1f}% "
              f"{d[('P(t)','success')]*100:>11.1f}% "
              f"{d[('P(t)','noisy_success')]*100:>9.1f}%")


def main():
    print("="*92)
    print("CoVer-VLA × P(t) Episode-Level Monitoring — Synthetic Simulator")
    print("="*92)
    print()
    print("CALIBRATION:")
    print(f"  Episode length:  {EPISODE_LENGTH} timesteps (~13 sec at 2.2 Hz)")
    print(f"  Episodes per type: {N_EPISODES_PER_TYPE}")
    print(f"  Five episode types: success, noisy_success (healthy)")
    print(f"                       gradual_failure, cliff_failure, misdirected (failing)")
    print()
    print("WHAT THIS TESTS:")
    print("  Can P(t) on per-timestep verifier scores reliably distinguish")
    print("  healthy episodes from failing ones, with high TP and low FP rate?")
    print("  This is the application closest to P(t)'s natural shape:")
    print("  trajectory monitoring, not bandit sampling.")

    results = run_experiment()
    print_results(results)
    print_threshold_sensitivity()

    print("\n" + "="*92)
    print("INTERPRETATION")
    print("="*92)
    print("""
  STRONG result: P(t) achieves > 80% TP on all three failure types AND
                 < 15% FP on both success types at some theta.
                 → publishable claim. Bring real-trace request to Jacky.

  MIXED result:  P(t) detects some failure types well but not others, OR
                 has elevated FP on noisy_success.
                 → frame as "P(t) detects gradual/drift failures cleanly;
                   cliff failures need different mechanism." Still useful.

  WEAK result:   P(t) FP rate near or above TP rate, OR baselines match
                 P(t) performance.
                 → episode monitoring is also bandit-shaped or trivially
                   handled by simpler heuristics. Drop CoVer-VLA blog post
                   angle, pivot to a different application or close out.

  Compare P(t) to RunningAvg and DropThreshold. If P(t) decisively
  outperforms, that's the case for the framework. If they tie, P(t)'s
  complexity isn't earning its keep.

  Threshold sensitivity matters: a result that only works at one specific
  theta is fragile. We want a range of theta values (e.g., 0.3-0.5)
  that all give reasonable TP/FP tradeoff.
""")


if __name__ == '__main__':
    main()
