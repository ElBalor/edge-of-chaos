# Edge of Chaos Neural Networks

## Experiment 11: Edge of Chaos Networks

**Question:** Do neural networks work BEST at the boundary between order and chaos?

### Hypothesis

The brain operates at the "edge of chaos" (critical state—proven in neuroscience):
- **Too ordered** = rigid (can't adapt)
- **Too chaotic** = random (can't learn)
- **Edge of chaos** = optimal (flexible + stable)

### What This Does

1. Builds neural networks with tunable chaos parameters
2. Trains multiple versions at different chaos levels (r = 2.0, 3.5, 3.9 in logistic map)
3. Measures performance vs. chaos level
4. Plots accuracy vs. chaos parameter to find the optimal r

### Running the Experiment

```bash
pip install -r requirements.txt
python edge_of_chaos_experiment.py
```

### Expected Output

- Training logs for each chaos level
- Performance comparison plot (accuracy vs. chaos parameter)
- Model checkpoints saved for each configuration

### Chaos Levels

- **r < 3.0**: Ordered (stable fixed points)
- **r ≈ 3.57**: Edge of chaos (period-doubling cascade)
- **r > 3.57**: Chaotic (sensitive to initial conditions)

### Tools

- PyTorch for neural networks
- Logistic map and Lorenz attractor for chaos dynamics

---

*From the Grimoire of Elbàlor — The Digital Necromancer 💀🔥*

