"""
power-metric
============
Adaptive scaling predictor for language model training.

Key result: adaptive EMA baseline (α=0.6) reduces scaling law
prediction unreliability from ~96% to ~0% on the Pythia suite,
compared to standard fixed power-law extrapolation.

Quickstart:
    from power_metric import ScalingPredictor, alpha_sweep

    # Fit on early checkpoints
    predictor = ScalingPredictor(alpha=0.6)
    predictor.fit(early_scores)

    # Predict future checkpoints
    predictions = predictor.predict(n_steps=8)

    # Check training health
    health = predictor.health()          # P(t) signal, 0–1
    reliable = predictor.is_reliable()  # True/False

    # Calibrate α for your model family
    result = alpha_sweep(all_scores)
    print(result['summary'])
    best_alpha = result['best_alpha']

Papers:
    github.com/HauntedKernel/power-metric
"""

from .predictor import ScalingPredictor, alpha_sweep

__version__ = "0.1.0"
__author__  = "Cole Cantrell / Paradigm Bridge"
__all__     = ["ScalingPredictor", "alpha_sweep"]
