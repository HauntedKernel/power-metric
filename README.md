# Power Metric — Adaptive Compute Allocation via Stochastic Health Signals

**P(t) = E(t) × W(t)**

A stochastic framework for real-time compute allocation across the full ML stack — training, inference, routing, and optimization.

Developed by [Cole Cantrell](https://paradigmbridge.tech) | cole@paradigmbridge.tech | [@HauntedKernel](https://twitter.com/HauntedKernel)

---

## What This Is

The power metric is a simple, implementable signal that answers one question at every layer of the ML stack:

> Is this process still worth continuing?

- **E(t)** — is current performance outperforming its own adaptive expected trajectory?
- **W(t)** — has it been doing so consistently?
- **P(t)** — the integrated health signal, robust to single-step noise

When P(t) drops below a threshold, stop. Reallocate compute elsewhere.

The same equation applies across training runs, inference sampling, hyperparameter search, expert routing, and test-time scaling verification.

---

## Quick Start

```python
from power_metric_verifier import VerifierHealthMonitor

# Paper 17: Verifier health monitoring during test-time scaling
monitor = VerifierHealthMonitor(threshold=0.5)

for candidate_score in your_verifier_scores:
    result = monitor.update(candidate_score)
    if result.should_stop:
        best = result.best_candidate_idx
        break
```

Run the simulation:

```bash
git clone https://github.com/HauntedKernel/power-metric
cd power-metric
pip install numpy matplotlib
python power_metric_verifier.py
```

---

## Papers

| # | Topic | Link |
|---|-------|------|
| 1 | Training compute allocation — Pythia empirical evidence | coming soon |
| 2 | Inference compute — adaptive repeated sampling allocation | coming soon |
| 3 | Scaling law reliability — real-time pre-commitment signal | coming soon |
| 4 | Data domain health monitoring — per-source signals | coming soon |
| 5 | LIF theory — Wiener-Hopf identity proof | coming soon |
| 6 | FlashAttention stack — hardware + allocation compounding | coming soon |
| 7 | HPO / Hyperband — complementary continuous health signal | coming soon |
| 8 | Early exit — depth efficiency via layer-level P(t) | coming soon |
| 9 | Checkpoint selection — post-training deployment signal | coming soon |
| 10 | Speculative decoding — adaptive draft length K | coming soon |
| 11 | MoE routing — cross-expert health comparison | coming soon |
| 12 | RLHF safety — reward hacking detection | coming soon |
| 13 | Continual learning — dual-signal forgetting detection | coming soon |
| 14 | You Are Wasting 96% of Your Inference Compute | [zenodo.org/records/19685841](https://zenodo.org/records/19685841) |
| 15 | Synthesis — The Resource Commitment Principle | coming soon |
| 16 | INGENIOUS + Power Metric — data selection + stopping | coming soon |
| 17 | Verifier health monitoring — LLM-as-a-Verifier extension | [zenodo.org/records/19688476](https://zenodo.org/records/19688476) |

Papers uploading to Zenodo progressively. Contact cole@paradigmbridge.tech for any paper directly.

---

## Code

| File | Paper | Description |
|------|-------|-------------|
| `power_metric_verifier.py` | Paper 17 | Verifier health monitor + simulation |

More simulation scripts coming as papers are validated.

---

## Simulation Results — Paper 17

Verifier health monitoring calibrated to Terminal-Bench 2.0 (Mirhoseini et al., 2026).
500 problems, three regimes: active discrimination / convergence plateau / degradation.

**Quality metric:** best score found / best score available — measures search efficiency within the score stream. This is NOT downstream task accuracy. Real validation requires Terminal-Bench 2.0 infrastructure.

| Threshold θ | Compute Reduction | Quality Score |
|-------------|------------------|---------------|
| 0.3 (conservative) | 89% | 0.962 |
| 0.5 (balanced) | 95% | 0.943 |
| 0.7 (aggressive) | 96% | 0.937 |

---

## The Math

**Core parameters** (consistent across all 17 papers):
- α = 0.3 — adaptive expected R mean reversion speed
- λ = 0.5 — exponential decay constant
- EWMA span = 3 — win rate smoothing window

**The three-layer computation:**

```
Layer 1 — Adaptive Expected R (pre-update baseline):
  E[R](t) = (1-α) · E[R](t-1) + α · score(t)
  [updated AFTER E(t) is computed — no information leak]

Layer 2 — Power Metric:
  E(t) = score(t) / E[R](t-1)       # efficiency: above or below expectation?
  W(t) = EWMA of [E(t) > 1.0]       # win rate: consistency of outperformance

Layer 3 — Integrated Health Signal:
  P(t) = exp(-λ)·P(t-1) + (1-exp(-λ))·[E(t)×W(t)]
```

**Theoretical grounding:**
- Consistent with classical optimal stopping and sequential decision theory
- P(t) is mathematically equivalent to the Leaky Integrate-and-Fire neuron model (Lapicque 1907) — proven via Wiener-Hopf optimal filtering theorem, error < 10⁻¹⁶
- The Resource Commitment Principle: any system allocating resources based on noisy evidence streams under squared-error loss converges on this solution

---

## Applications Across the Stack

| Layer | Decision | Signal |
|-------|----------|--------|
| Pretraining | When to stop a training run | Benchmark improvement trajectory |
| Data mixing | Which domains are still learning | Per-domain loss improvement |
| HPO | Which configs to keep running | Validation metric trajectory |
| Inference sampling | When to stop generating candidates | Per-sample solve rate |
| Early exit | Which layer to exit at | Layer-level confidence |
| Speculative decoding | How many tokens to draft | Token acceptance rate |
| MoE routing | Which experts to favor | Cross-expert score comparison |
| RLHF | When reward hacking begins | Held-out capability signal |
| Checkpoint selection | Which checkpoint to deploy | Training health trajectory |
| Test-time verification | When verifier loses discrimination | Candidate score vs expectation |

---

## Contact & Collaboration

Cole Cantrell | Independent Researcher  
cole@paradigmbridge.tech  
paradigmbridge.tech  
[@HauntedKernel](https://twitter.com/HauntedKernel)

Actively seeking collaborators with access to:
- Pretraining infrastructure (Papers 1–13, 15–16)
- Test-time scaling infrastructure / Terminal-Bench 2.0 (Paper 17)
- ArXiv endorsement for cs.AI also welcomed

Patent pending: *Dynamic Risk Assessment and Decision Optimization System Using Stochastic Performance Metrics* (2025)
