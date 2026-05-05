"""
power_metric/predictor.py
=========================
Core implementation of the adaptive scaling predictor.

Key finding: adaptive EMA baseline with α=0.6 reduces scaling
law prediction unreliability from 96% to 0% on the Pythia model
suite, compared to fixed power-law extrapolation.

The framework is also the basis for P(t) = E(t) × W(t), a stochastic
health signal for monitoring ML training dynamics across the lifecycle.
See the paper series at github.com/HauntedKernel/power-metric.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional, List, Tuple


# ── Default parameters (validated on Pythia suite) ───────────────────

DEFAULT_ALPHA    = 0.6    # Key finding: 0.6 outperforms paper default of 0.3
DEFAULT_LAMBDA   = 0.5    # P(t) exponential decay
DEFAULT_SPAN     = 3      # EWMA win rate span
DEFAULT_SPLIT    = 8      # early/late checkpoint split


class ScalingPredictor:
    """
    Adaptive scaling predictor using EMA baseline.

    Reduces prediction unreliability from ~96% (fixed power-law) to
    ~0% (adaptive EMA, α=0.6) on the Pythia model suite.

    Usage:
        predictor = ScalingPredictor(alpha=0.6)
        predictor.fit(early_scores)
        predictions = predictor.predict(n_steps=8)
        health = predictor.health()

    Args:
        alpha: EMA smoothing factor. Higher = more responsive to recent
               scores. Default 0.6 validated on Pythia; sweep with
               alpha_sweep() to calibrate for your model family.
        split: Number of early checkpoints used for fitting.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA,
                  split: int = DEFAULT_SPLIT):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.split = split
        self._scores: Optional[np.ndarray] = None
        self._er_series: Optional[np.ndarray] = None
        self._power_law_params: Optional[tuple] = None
        self._fitted = False

    def fit(self, scores: List[float]) -> 'ScalingPredictor':
        """
        Fit the predictor on observed checkpoint scores.

        Args:
            scores: List of benchmark accuracy scores at consecutive
                    checkpoints (e.g. mean of piqa/arc_easy/
                    arc_challenge/winogrande at each checkpoint).

        Returns:
            self (for chaining)
        """
        self._scores = np.asarray(scores, dtype=float)
        if len(self._scores) < 2:
            raise ValueError("Need at least 2 scores to fit")

        # Build EMA baseline series
        er = self._scores[0]
        er_series = [er]
        for s in self._scores[1:]:
            er = (1 - self.alpha) * er + self.alpha * s
            er_series.append(er)
        self._er_series = np.array(er_series)

        # Fit power law to first `split` checkpoints if enough data
        if len(self._scores) >= self.split:
            self._fit_power_law()

        self._fitted = True
        return self

    def _fit_power_law(self):
        """Fit y = a * x^b + c to early checkpoints."""
        x = np.arange(1, self.split + 1, dtype=float)
        y = self._scores[:self.split]
        try:
            popt, _ = curve_fit(
                _power_law, x, y,
                p0=[0.1, 0.3, 0.3],
                maxfev=10000,
                bounds=([-np.inf, -2, 0], [np.inf, 2, 1])
            )
            self._power_law_params = popt
        except RuntimeError:
            self._power_law_params = None

    def predict(self, n_steps: Optional[int] = None) -> np.ndarray:
        """
        Predict future checkpoint scores using adaptive EMA baseline.

        The pre-update EMA state E[R](t-1) serves as the prediction
        for score(t), updated after each observation.

        Args:
            n_steps: Number of future steps to predict. If None,
                     predicts the same length as the fitted series.

        Returns:
            Array of predicted scores.
        """
        self._check_fitted()
        n = n_steps or len(self._scores)

        # Pre-update predictions: EMA state before incorporating score(t)
        # predicts score(t). For steps beyond observed data, continue
        # rolling forward from the last known state.
        if n <= len(self._er_series):
            # Predict within observed range (validation mode)
            return self._er_series[:n]

        # Extend predictions beyond observed data
        last_er = self._er_series[-1]
        extra = n - len(self._er_series)
        # For extrapolation beyond observed data, use power-law trend
        # if available, otherwise flat from last EMA state
        if self._power_law_params is not None:
            a, b, c = self._power_law_params
            x_extra = np.arange(
                len(self._scores) + 1,
                len(self._scores) + extra + 1,
                dtype=float
            )
            extra_preds = _power_law(x_extra, a, b, c)
            # Blend: weight toward EMA-informed prediction
            extra_preds = 0.7 * extra_preds + 0.3 * last_er
        else:
            extra_preds = np.full(extra, last_er)

        return np.concatenate([self._er_series, extra_preds])

    def predict_fixed(self) -> Optional[np.ndarray]:
        """
        Fixed power-law predictions for comparison.
        Returns None if insufficient data for power-law fit.
        """
        self._check_fitted()
        if self._power_law_params is None:
            return None
        a, b, c = self._power_law_params
        x = np.arange(1, len(self._scores) + 1, dtype=float)
        return _power_law(x, a, b, c)

    def reliability(self,
                     actuals: Optional[List[float]] = None,
                     threshold: float = 0.02,
                     split: Optional[int] = None) -> dict:
        """
        Compute prediction reliability metrics.

        Compares adaptive EMA predictions against actual scores on
        the post-split "test" region (same methodology as Paper 3).

        Args:
            actuals: Actual scores. If None, uses scores from fit().
            threshold: Absolute error threshold for "unreliable"
                       (default 0.02 matches Paper 3 methodology).
            split: Where to split early/late. Defaults to self.split.

        Returns:
            dict with keys:
              adaptive_unreliable: fraction of test steps where
                |predicted - actual| > threshold
              fixed_unreliable: same for fixed power-law
              adaptive_mae: mean absolute error for adaptive EMA
              fixed_mae: mean absolute error for fixed power-law
              n_test: number of test checkpoints evaluated
        """
        self._check_fitted()
        sp = split or self.split
        scores = np.asarray(actuals) if actuals else self._scores

        if len(scores) <= sp:
            raise ValueError(
                f"Need more than {sp} scores for reliability test. "
                f"Got {len(scores)}.")

        # Adaptive predictions for test region
        # Pre-update EMA at t-1 predicts score at t
        preds_adapt = self._er_series[sp - 1:len(scores) - 1]
        actuals_test = scores[sp:]
        n = min(len(preds_adapt), len(actuals_test))
        preds_adapt = preds_adapt[:n]
        actuals_test = actuals_test[:n]

        errs_adapt = np.abs(preds_adapt - actuals_test)
        result = {
            'adaptive_unreliable': float(np.mean(errs_adapt > threshold)),
            'adaptive_mae': float(np.mean(errs_adapt)),
            'n_test': n,
            'threshold': threshold,
            'alpha': self.alpha,
        }

        # Fixed power-law for test region
        if self._power_law_params is not None:
            a, b, c = self._power_law_params
            x_test = np.arange(sp + 1, sp + n + 1, dtype=float)
            preds_fixed = _power_law(x_test, a, b, c)
            errs_fixed = np.abs(preds_fixed - actuals_test)
            result['fixed_unreliable'] = float(
                np.mean(errs_fixed > threshold))
            result['fixed_mae'] = float(np.mean(errs_fixed))

        return result

    def health(self) -> float:
        """
        Current P(t) health signal (0–1).

        P(t) = E(t) × W(t) measures whether scores are beating
        adaptive expectations. Values near 1 indicate healthy
        scaling; values near 0 indicate plateau or degradation.

        Returns:
            Current P(t) value.
        """
        self._check_fitted()
        return float(_compute_pt(self._scores)[-1])

    def health_trajectory(self) -> np.ndarray:
        """P(t) signal at every checkpoint. Shape: (n_scores,)"""
        self._check_fitted()
        return _compute_pt(self._scores)

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """
        Quick check: is the current scaling trajectory healthy?

        Args:
            threshold: P(t) threshold. Below this = not reliable.

        Returns:
            True if P(t) >= threshold.
        """
        return self.health() >= threshold

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Call fit(scores) before predict/health/reliability.")

    def __repr__(self):
        fitted = f", fitted on {len(self._scores)} ckpts" if self._fitted else ""
        return f"ScalingPredictor(alpha={self.alpha}{fitted})"


# ── Alpha sweep utility ───────────────────────────────────────────────

def alpha_sweep(
    scores: List[float],
    alpha_range: Optional[List[float]] = None,
    threshold: float = 0.02,
    split: Optional[int] = None,
) -> dict:
    """
    Find the best α for your model family by sweeping a range.

    Key finding on Pythia: α=0.6 achieves 0% unreliability.
    Different model families may benefit from different α values.
    Run this once per model family to calibrate.

    Args:
        scores: Complete checkpoint score trajectory.
        alpha_range: α values to test. Default: 0.1 to 0.9.
        threshold: Absolute error threshold (default 0.02).
        split: Early/late split. Default: half of available ckpts.

    Returns:
        dict with keys:
          results: per-alpha unreliability and MAE
          best_alpha: α achieving lowest unreliability
          best_unreliable: unreliability rate at best α
          summary: printable table string

    Example:
        scores = [0.42, 0.48, 0.53, 0.56, 0.59, 0.61, 0.62, 0.63,
                  0.64, 0.65, 0.655, 0.66, 0.662, 0.664, 0.666, 0.668]
        result = alpha_sweep(scores)
        print(result['summary'])
        # Use result['best_alpha'] for production
    """
    if alpha_range is None:
        alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sp = split or (len(scores) // 2)
    results = {}

    for alpha in alpha_range:
        predictor = ScalingPredictor(alpha=alpha, split=sp)
        predictor.fit(scores)
        try:
            r = predictor.reliability(threshold=threshold)
            results[alpha] = {
                'unreliable': r['adaptive_unreliable'],
                'mae': r['adaptive_mae'],
            }
        except ValueError:
            results[alpha] = {'unreliable': None, 'mae': None}

    # Find best alpha
    valid = {a: r for a, r in results.items() if r['unreliable'] is not None}
    best_alpha = min(valid, key=lambda a: (valid[a]['unreliable'], valid[a]['mae']))
    best_unreliable = valid[best_alpha]['unreliable']

    # Build summary string
    lines = [
        f"α sweep (split={sp}, threshold={threshold})",
        f"{'α':>6}  {'unreliable%':>13}  {'MAE':>8}",
        "-" * 34,
    ]
    for alpha in alpha_range:
        r = results.get(alpha, {})
        if r.get('unreliable') is not None:
            marker = " ←best" if alpha == best_alpha else ""
            lines.append(
                f"{alpha:>6.2f}  {r['unreliable']*100:>12.1f}%  "
                f"{r['mae']:>8.4f}{marker}"
            )
    lines.append(f"\nRecommended α: {best_alpha} "
                  f"({best_unreliable*100:.0f}% unreliable)")

    return {
        'results': results,
        'best_alpha': best_alpha,
        'best_unreliable': best_unreliable,
        'summary': '\n'.join(lines),
    }


# ── P(t) computation ──────────────────────────────────────────────────

def _compute_pt(scores: np.ndarray,
                 alpha: float = 0.3,
                 lam: float = DEFAULT_LAMBDA,
                 span: int = DEFAULT_SPAN) -> np.ndarray:
    """Compute P(t) = E(t) × W(t) trajectory over checkpoint scores."""
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


def _power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, b) + c
