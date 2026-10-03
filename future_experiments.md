# Future Experiments to Explore

## High Priority Experiments

### Experiment 3: Quantum Neural Networks (Gate-Based)
**Question:** Can I train ACTUAL quantum circuits (not just quantum-inspired)?

**What I'd build:**
- Variational Quantum Circuits (VQC)—quantum gates trainable via gradients
- Hybrid classical-quantum model (classical NN + quantum layer)
- Train on quantum simulator (or real quantum computer via IBM/Google)

**Why it's cool:**
- Quantum computers exist NOW (IBM Quantum, Google Sycamore—53 qubits)
- You can run REAL experiments (not just theory)
- Quantum advantage for certain problems (chemistry, optimization)

**Experiment:**
- Quantum classifier (train VQC to classify MNIST)
- Compare to classical NN (does quantum help? when?)

**Tools:** Qiskit (IBM), Cirq (Google), PennyLane (quantum ML library)

---

### Experiment 10: Predictive Coding Networks
**Question:** Can networks learn by PREDICTING inputs (not classifying)?

**What you'd build:**
- Network that predicts NEXT input (not label)
- Error = difference between prediction and reality
- Learning = minimize prediction error (Bayesian brain hypothesis)

**Why it's cool:**
- Brain works like this (constantly predicting sensory input)
- Unsupervised (no labels needed—learns from raw data)
- Explains perception (we "see" predictions, not raw input)

**Experiment:**
- Train predictive coding network on video (predict next frame)
- Test: Does it learn object permanence? Physics? Cause-effect?

**Tools:** PyTorch, implement Rao & Ballard's predictive coding equations

---

## Moderate Feasibility Experiments

### Experiment 5: Liquid State Machines (LSM)
**Question:** Can you build a "liquid" that COMPUTES?

**What you'd build:**
- Reservoir of spiking neurons (randomly connected)
- Input = perturbs the reservoir (like dropping stone in water—ripples)
- Output = read-out layer (trained to interpret ripples)

**Why it's cool:**
- Brain works like this (sensory input → cortical ripples → interpretation)
- No training the reservoir (only the read-out)—FAST
- Good for temporal data (speech, video, time-series)

**Experiment:**
- Build LSM for speech recognition (spoken digits)
- Compare to RNN/LSTM (accuracy, speed, energy)

**Tools:** Brian2 (SNN simulator), BindsNET (PyTorch SNNs)

---

### Experiment 1: Thermodynamic Neural Networks
**Question:** Can neural networks learn by MINIMIZING FREE ENERGY (like physical systems)?

**What you'd build:**
- Replace backprop with Free Energy Principle (Friston's active inference)
- Neurons update to minimize "surprise" (prediction error + entropy)
- Network learns by finding EQUILIBRIUM states (like cooling metal)

**Why it's cool:**
- Brain doesn't use backprop (biologically implausible)
- Thermodynamic learning = more brain-like (local updates, no global gradient)
- Could be MORE efficient than backprop (less compute)

**Experiment:**
- Train MNIST classifier using free energy minimization (no backprop)
- Compare to standard backprop (accuracy, speed, energy use)

**Tools:** PyTorch, implement Friston's FEP equations

---

### Experiment 2: Fluid Dynamics Neural Networks
**Question:** Can neural networks simulate FLUID FLOW (Navier-Stokes equations)?

**What you'd build:**
- Neural network that predicts how fluids move (air, water, blood)
- Train on physics simulations (CFD data)
- Use it to design aerodynamic shapes (cars, planes, drones)

**Why it's cool:**
- Traditional CFD = SLOW (hours/days for one simulation)
- Neural CFD = FAST (milliseconds—1000x speedup)
- Applications: aerospace, medicine (blood flow), climate modeling

**Experiment:**
- Train on 2D fluid simulations (smoke, water)
- Test: Can it predict turbulence? Vortices? Eddies?

**Tools:** PyTorch, Physics-Informed Neural Networks (PINNs)

---

## Notes
- Start with Quantum NNs (#3) or Predictive Coding (#10) for high feasibility
- LSM (#5) is proven but has limited applications
- Thermodynamic (#1) and Fluid Dynamics (#2) need more research but are promising

