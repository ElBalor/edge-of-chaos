# Edge of Chaos Improvements

## What We're Testing

Your initial experiment found that **r = 3.2** (transition regime) performs best (97.82%), not exactly at the theoretical edge of chaos (r = 3.57). Here are 4 improvements to optimize this further:

---

## Improvement 1: Fine-tune r Parameter ⚙️

**Problem:** We only tested 12 r values, missing the precise optimum.

**Solution:** Test 11 values between r = 3.0 and 3.5 (the promising region).

**Expected:** Find the exact optimal r value (maybe 3.15? 3.25?).

---

## Improvement 2: Vary Chaos Injection Strength 💪

**Problem:** We fixed chaos strength at 0.1. Maybe different strengths work better!

**Solution:** Test strengths: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

**Expected:** Find optimal balance between chaos and stability.

---

## Improvement 3: Layer-Specific Chaos Injection 🎯

**Problem:** We inject chaos in all layers. Maybe some layers need more/less chaos?

**Solution:** Test:
- All layers (current)
- First layer only
- Second layer only
- Both layers (explicit)

**Expected:** Early layers might need more chaos, later layers less (or vice versa).

---

## Improvement 4: Optimal Combination (Grid Search) 🔍

**Problem:** r and strength might interact! Need to find best combination.

**Solution:** Grid search over:
- r values: 3.1 to 3.4 (7 values)
- Strengths: 0.05, 0.1, 0.15, 0.2 (4 values)
- Total: 28 configurations

**Expected:** Find the **absolute best** r + strength combination.

---

## Running the Improvements

```bash
python edge_of_chaos_improved.py
```

This will:
1. Run all 4 improvement experiments
2. Generate a 2×2 plot showing all results
3. Save the best configuration found
4. Output JSON with all results

**Time:** ~30-60 minutes (depending on GPU)

---

## Expected Outcomes

1. **Precise optimal r:** Maybe 3.18 or 3.22 instead of 3.2
2. **Optimal strength:** Maybe 0.12 or 0.08 instead of 0.1
3. **Layer strategy:** Maybe first layer only is best
4. **Best combo:** Could achieve 97.9%+ accuracy

---

## Next Steps After This

If we find improvements:
- Test on harder dataset (CIFAR-10)
- Implement adaptive chaos (starts high, decreases during training)
- Test on different network architectures
- Compare to baseline (no chaos) more rigorously

