# Edge-of-Chaos Experiment Summary

## 1. Baseline Experiment (edge_of_chaos_experiment.py)
- Tested logistic-map chaos parameter `r` across ordered → chaotic regimes.
- Dataset: full MNIST, architecture `784-128-64-10` with chaos layers in hidden units.
- Best accuracy: **97.82% at r = 3.2** (transition regime).
- Theoretical edge (`r = 3.57`) reached **97.54%**, confirming the sweet spot lies slightly before the edge.
- Fully chaotic (`r ≥ 3.9`) degraded accuracy (down to **94.31%** at `r = 4.0`).

## 2. Improvement Experiments (edge_of_chaos_improved.py)

### 2.1 Fine-tuned `r` sweep (3.0 → 3.5)
- Eleven evenly spaced values; `r = 3.10` peaked at **98.11%**.
- Reinforces that the optimal region is in the transition zone, not the theoretical edge.

### 2.2 Chaos strength sweep (strength ∈ [0.0 … 0.5] at r = 3.20)
- Best range: **0.15–0.20**, with **98.10%** at strength `0.20`.
- No chaos (strength 0.0) dropped to **97.33%**.

### 2.3 Layer-specific injection
- Configurations: all layers, first only, second only, both explicitly.
- Second layer only (and both layers) tied for best at **97.89%**.
- Suggests later hidden layers benefit more from chaos than earlier ones.

### 2.4 Grid search (`r ∈ {3.10…3.40}`, strength ∈ {0.05, 0.10, 0.15, 0.20})
- Best combo: **r = 3.40, strength = 0.15 → 98.01%**.
- Several nearby configs (e.g. `r = 3.40, strength = 0.10`) remain in the **97.9%** range.

### 2.5 Visualization
- Plot saved as `edge_of_chaos_improvements.png` summarises all four experiments.

## 3. Training Efficiency Study (test_training_efficiency.py)

### 3.1 Quick subset sanity check (5k train / 1k test, 3 epochs)
- Baseline: **90.20%** in 24.5s.
- Chaos (r=3.40, strength=0.15): **90.10%** in 32.0s.
- Takeaway: chaos introduces ~30% overhead and needs enough epochs/data to show benefits.

### 3.2 Full MNIST, 10 epochs (CPU)
- Baseline: **97.61%**, 192s, reached 97.5% in **5 epochs**.
- Chaos configuration: **97.65%**, 249s, reached 97.5% in **4 epochs**.
- Net effect: **+0.04%** accuracy, **+25% faster convergence**, **~29% training-time overhead** at 10 epochs.
- Conclusion: chaos layers trade extra per-epoch cost for slightly faster convergence and marginal accuracy gain on MNIST.

## 4. Recommended Next Steps
- Validate the optimal configuration on additional datasets: Fashion-MNIST (drop-in) or CIFAR-10 (requires CNN adaptation).
- Profile GPU training to quantify overhead in realistic setups.
- Explore adaptive chaos schedules (start strong, decay) to balance early exploration with late stability.
- Document methodology and results in a short technical report before considering publication; include comparisons to non-chaotic baselines and related literature.

## 5. Extension Study: Fashion-MNIST (edge_of_chaos_improved.py --dataset fashion-mnist)
- Settings: batch size 256, 5 epochs per configuration (reduced for time).
- Fine-tuned r sweep: best accuracy **87.89% at r = 3.30**; performance flattens across 3.1–3.4 range.
- Strength sweep (r = 3.20): chaos provides marginal gains; **0.10** delivers **87.87%**, while no chaos already hits **87.82%**. Higher strengths hurt.
- Layer-specific chaos: differences within ±1.1%; second layer only or both layers ≈ **87.5%**.
- Grid search: best combo **r = 3.25, strength = 0.05 → 87.72%**. Chaos benefit over baseline is within noise (<0.1%).
- Takeaway: Fashion-MNIST supports the “transition-before-edge” optimum but shows minimal chaos advantage, implying dataset sensitivity.


