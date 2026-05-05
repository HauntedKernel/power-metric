# P(t) Module Catalog
**Cole Cantrell — Paradigm Bridge — Living Document**

This file catalogs structural variants of the P(t) framework, when each
applies, what we've tested, and what we haven't. It is the working
specification of P(t) as a *class* of methods rather than a single formula.

Updated: May 4, 2026. Add to this file whenever a new variant is tested
or a new regime is identified.

---

## Common scaffold

All variants share:

```
At each observation t:
  E(t) = some_efficiency_function(s(t), R-state)
  W(t) = β · 1[E(t) ≥ 1] + (1-β) · W(t-1)        EWMA win rate
  P(t) = e^(-λ) · P(t-1) + (1-e^(-λ)) · E(t)·W(t)  exponential decay integral
```

The variants differ in how E(t) is computed and how the baseline state R is
maintained. The W and P layers stay constant.

Defaults across all variants: β = 2/(span+1) with span=3, λ = 0.5.

---

## Variant catalog

### V1. EWMA baseline (published)

```
E(t) = s(t) / R(t-1)
R(t) = (1-α) R(t-1) + α s(t)
```

Single-timescale baseline. R fast-tracks recent observations.

**Defaults:** α = 0.3.
**Best for:** medium-noise signals with monotone or slowly-shifting trajectories.
**Tested on:** Papers 1, 3, 4, 7, 13, 16, 18.
**Known regime:** matches simple exponential smoothing structure but with
ratio-based efficiency. Lourie's smoothing-literature critique applies most
directly to this variant.

---

### V2. Holt-style two-timescale

```
E(t) = s(t) / R̃(t-1)
R(t)  = (1-α) R(t-1) + α s(t)         fast tracker
R̃(t) = (1-γ) R̃(t-1) + γ R(t)         slow tracker on top of fast
```

Two timescales: fast R follows current scores, slow R̃ provides the
benchmark E(t) measures against.

**Defaults to test:** α = 0.3, γ ∈ {0.05, 0.10, 0.20}.
**Best for:** trajectories with stable underlying trends and noisy
near-term observations.
**Tested on:** Paper 7 (HPO). Strengthens 1.4B recovery margin from
P=0.544 to P=0.570 at canonical settings without changing recovery count
across broader grid.
**Known regime:** principled response to Lourie's two-timescale critique.
Maps onto Holt's linear trend method but uses ratio efficiency.
**Open questions:** does it strengthen Paper 18 chain selection? Probably
not (chains are short, two timescales need observation horizon). Worth
testing on Paper 13 forgetting if we run that on real data.

---

### V3. Pre-smoothed input

```
s̃(t) = (1-μ) s̃(t-1) + μ s(t)         smooth raw input first
E(t) = s̃(t) / R(t-1)
R(t) = (1-α) R(t-1) + α s̃(t)
```

Denoise input before pipeline. Adds lag in exchange for noise tolerance.

**Defaults to test:** μ ∈ {0.2, 0.4, 0.6}.
**Best for:** very noisy per-observation signals where trajectory matters
more than instantaneous value.
**Tested on:** Paper 7. At low μ (heavy smoothing) breaks slow-starter
recovery — the lag prevents detection. At high μ (μ=0.6, light smoothing)
recovers similarly to V1. So this variant is structurally inferior for
HPO-style problems.
**Known regime:** likely useful for inference sampling (Paper 2) where
per-sample binary correctness is highly variable. Untested there.
**Avoid for:** any problem requiring fast detection of regime change.

---

### V4. Additive deviation

```
E(t) = 1 + (s(t) - R(t-1))           additive surprise centered at 1
R(t) = (1-α) R(t-1) + α s(t)
```

Replaces ratio with signed deviation from baseline.

**Defaults to test:** α ∈ {0.3, 0.5}.
**Best for:** signals where ratios are unstable (R near zero) or where
absolute differences are more interpretable than relative ones.
**Tested on:** Paper 7. Underperforms V1 — fails canonical recovery
(P=0.46 < θ=0.5). Additive surprise on already-ratio-like data
double-dampens the signal.
**Known regime:** plausibly useful for binary-signal applications where
ratio formulation gets unstable initialization (PRM800K chain selection
required ε-floor on E[R] to avoid division blowup). Untested there.
**Avoid for:** any problem where relative improvement (rather than
absolute change) is the natural comparison.

---

### V5. Full SDE formulation (patent, untested in published papers)

```
dE_t = μE_t dt + σE_t dW_t           Geometric Brownian Motion
dW_t = θ(W̄ - W_t)dt + σ_w dB_t      Ornstein-Uhlenbeck mean reversion
P_t  = E_t × W_t                      composite
```

Plus MLE parameter estimation and CUSUM regime change detection.

**Best for:** rich time series with many observations per signal,
where small-sample noise of MLE doesn't dominate.
**Tested on:** Paper 20 (safety monitoring). False positive rate 50-100%
vs 0-15% for V1. The full formulation needs more data to fit cleanly than
the safety regime provided. NOT viable for low-observation regimes.
**Known regime:** dense-checkpoint training runs, long inference
trajectories, financial backtests with hundreds of returns. Untested in
ML applications because we don't have dense enough data.
**Open: CUSUM as an add-on to V1.** Could provide regime detection
without needing the full GBM/OU machinery. Worth testing on Paper 3
scaling reliability where regime changes are the central phenomenon.

---

### V6 (untested). Aggregation-aware multi-signal P(t)

For applications with multiple parallel signals (per-domain training,
multi-layer monitoring), the aggregation rule matters and *is not part
of the EWMA scaffold*.

```
P_aggregate(t) = aggregate({P_i(t) for each signal i})

aggregations tested:
  geometric_mean   — Paper 19 (ACE stack), default for health monitoring
  min              — required for safety (Paper 20 finding)
  arithmetic_mean  — untested
  weighted         — by inverse variance, untested
```

**Known regime:** geometric mean appropriate when signals share failure
modes (correlated degradation). MIN appropriate when any single signal
failing is catastrophic (safety, multi-layer integrity). Different
regime → different aggregation, no universal rule.

---

## Regime → variant mapping (working hypotheses)

| Regime | Best variant | Reason |
|---|---|---|
| HPO with monotone trajectories, ranking noise | V2 (Holt) | Two-timescale separates ranking noise from real velocity |
| Scaling law extrapolation | V1 with high α, or V5 with CUSUM | Detects regime change rather than smooths through it |
| Reasoning chain selection (binary) | V1 with ε-floor, possibly V4 | Handles initialization at zero correctness |
| Inference sampling | Untested. V3 plausible if per-sample noise is the dominant problem | |
| Catastrophic forgetting | Untested on real data. V2 plausible (slow degradation, monotone) | |
| Multi-signal safety monitoring | MIN aggregation regardless of underlying variant | Failure on any signal must propagate |
| Multi-signal health monitoring | Geometric mean if failures correlate | Empirically validated on ACE stack (Paper 19) |

---

## Things we do NOT yet have evidence for

- That V2 (Holt) generalizes beyond HPO. Tested on one application.
- That V5 (full SDE) ever beats V1 in any published-paper regime. So far
  it has been worse where tested.
- That CUSUM regime detection adds value over adaptive baseline alone.
- That any variant rescues Paper 4 (per-domain) or Paper 5 (per-domain
  empirical claim) — the data shows different scales prioritize different
  domains, and no smoothing fixes data fact.
- That P(t) is a *strict* superset of exponential smoothing methods. We
  have evidence (Section 4 verification) that V1 outperforms Holt-Winters
  and SES on Pythia scaling, but on a small dataset.

---

## Things we should not claim until tested

- "P(t) is robust to formulation choice." (True at canonical settings on
  Paper 7. Possibly false elsewhere.)
- "Two-timescale formulations strengthen detection in monotone-trajectory
  problems." (Tested on Paper 7 only. Need second confirmation.)
- "P(t) generalizes beyond exponential smoothing." (Defensible from Paper
  3 verification but small N.)

---

## Format for adding to this catalog

When testing a new variant on a new application, log:

1. **Variant ID and parameters used.**
2. **Regime characteristics:** signal type (continuous/binary), noise
   level (estimated), observation density, target detection sensitivity.
3. **Result vs V1 baseline:** strictly better / strictly worse /
   regime-specific tradeoff.
4. **Mechanistic explanation:** why we expected this variant to behave
   this way, and whether the result matches the prediction.
5. **Open questions raised by the test.**

Keep the catalog living. Don't delete failed tests — they're as
informative as successes for establishing the class boundaries.
