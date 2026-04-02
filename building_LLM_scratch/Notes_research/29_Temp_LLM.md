# Decoding Strategies: Temperature Scaling

## 1. Lecture Context: Improving Generation Quality

**Previous achievement**: Trained GPT-2 (162M params) on *"The Verdict"* → **overfitting**, repetitive outputs:
```

Input: "every effort moves you"
Output: "you was one of the xmc laid down across the SE..." (memorized, incoherent)

```

**Today's goal**: **Control randomness** in autoregressive generation:
```

Greedy (argmax):      Deterministic → repetitive
Temperature scaling:  Probabilistic → **balanced creativity**

```

**Next**: Top-k sampling (combine with temperature).

---

## 2. Problem: Greedy Decoding (Current Method)

### 2.1. Current `generate_text_simple()`
```

Input: "every effort moves you" → GPT → logits[^1]
↓ softmax
probs: [0.001, 0.06, 0.35, 0.57, 0.002, ...] ("closer", "toward", "forward", ...)
↑ argmax → token ID 3 → "forward"

```

**Issues**:
- **Always** picks highest prob → **repetitive** ("forward, forward, forward...").
- No **creativity/diversity**.

### 2.2. Visual Workflow
```

"every effort moves" → GPT → logits → softmax → probs → **argmax** → "forward"
↓
**Sample** → ?

```

---

## 3. Solution: Probabilistic Sampling (Multinomial)

**Replace `argmax` → `torch.multinomial(probs, 1)`**:
```

probs: [0.001, 0.06, 0.35, 0.57, 0.002, ...] ("closer", "toward", "forward", "pizza", ...)
Sample → 57% "forward", 35% "toward", 6% "closer", <1% "pizza"

```

**Demo** (vocab=9 tokens, 1000 trials):
```

forward: 582 (58%)  ← Most likely ✓
toward:  343 (34%)
closer:   73 (7%)
inches:    2 (0.2%)
pizza:     0 (0%)

```

**Result**: **Diverse** outputs:
```

"every effort moves you forward"
"every effort moves you toward"
"every effort moves you closer"

```

---

## 4. Temperature Scaling: Control Probability Sharpness

### 4.1. Formula
```

scaled_logits = logits / temperature    \# T > 0
probs = softmax(scaled_logits)
next_token = multinomial(probs, 1)

```

### 4.2. Temperature Effects

| Temperature | Effect | Probability Distribution | Output Behavior |
|-------------|--------|-------------------------|-----------------|
| **T < 1** (0.1) | **Sharpens** | `forward: 99.1%` | **Greedy-like** (repetitive) |
| **T = 1** | **Original** | `forward: 57%`<br>`toward: 35%` | **Balanced** |
| **T > 1** (5.0) | **Flattens** | `forward: 28%`<br>`toward: 25%`<br>`pizza: 4%` | **Creative** (nonsensical) |

### 4.3. Visual Effect
```

T=0.1:  ████████████████████████████████████ forward (99%)
T=1.0:  ██████████████ forward (57%)    ████████ toward (35%)
T=5.0:  █████ forward (28%) ████ toward (25%) █ pizza (4%)

```

**Why "temperature"?**
- **Low T**: Low entropy → **deterministic** (frozen).
- **High T**: High entropy → **chaotic** (random).

---

## 5. Step-by-Step Code Implementation

### Step 1: Base Sampling (No Temperature)
```python
def generate_with_sampling(model, idx, max_new_tokens, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.context_length:]  # Context window
        
        # Forward pass
        logits = model(idx_cond)[:, -1, :]  # Last position [B, vocab_size]
        
        # Sample (replace argmax)
        probs = F.softmax(logits / temperature, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
        
        # Append
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```


### Step 2: Temperature Demo (Small Vocab)

```python
# Tiny vocab for visualization
vocab = ["closer", "effort", "forward", "inches", "moves", "pizza", "toward", "you"]
inv_vocab = {i: token for i, token in enumerate(vocab)}

logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 0.1, 2.5, 1.8])  # Raw scores

# T=1.0 (baseline)
probs1 = F.softmax(logits, dim=0)
print(probs1)  # tensor([0.06, 0.10, 0.57, 0.03, 0.16, 0.01, 0.35, 0.12])

# T=0.1 (sharp)
probs_low = F.softmax(logits/0.1, dim=0)
print(probs_low)  # tensor([0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00])

# T=5.0 (flat)
probs_high = F.softmax(logits/5.0, dim=0)
print(probs_high)  # tensor([0.12, 0.13, 0.15, 0.11, 0.13, 0.11, 0.14, 0.13])
```


### Step 3: 1000 Trials Statistics

```python
samples = torch.multinomial(probs1, 1000, replacement=True)
freq = torch.bincount(samples, minlength=len(vocab)) / 1000
for i, token in enumerate(vocab):
    print(f"{token}: {freq[i]*100:.1f}%")
```

```
forward: 57.2%
toward:  34.3%
closer:   7.3%
...
```


### Step 4: Full Integration

```python
output_ids = generate_with_sampling(model, enc.encode("every effort moves you"), 
                                   max_new_tokens=25, temperature=0.8)
print(enc.decode(output_ids))
```

**Varied outputs**:

```
"every effort moves you toward the horizon..."
"every effort moves you forward with grace..."
```


---

## 6. Practical Guidelines

```
temperature=0.7  → Production sweet spot (ChatGPT default)
temperature=0.1  → Deterministic (boring)
temperature=1.0  → Original (baseline)
temperature=2.0+ → Creative (risky)
```

**Usage**:

```python
# Balanced creativity
generate(..., temperature=0.8)

# Safe/repetitive  
generate(..., temperature=0.3)

# Experimental
generate(..., temperature=1.2)
```


---

## 7. Workflow Summary

```
1. Input → GPT → logits [B, vocab_size]
2. Scale:    logits / temperature
3. Softmax:  probs = softmax(scaled_logits)
4. Sample:   next_token = multinomial(probs, 1)
5. Append → Repeat
```

**Before**: Greedy → **always** "forward".
**After**: Probabilistic → "forward", "toward", "closer" (proportional).

---

## 8. Key Takeaways

1. **Greedy = boring**: `argmax` → repetitive.
2. **Sampling = creative**: `multinomial(probs)` → diverse.
3. **Temperature controls sharpness**:
    - Low → **deterministic**.
    - High → **random**.
4. **T=0.7-0.9**: **Production gold standard**.

**Next lecture**: **Top-k sampling** → filter nonsense while preserving creativity.

**Victory**: From **comma spam** → **grammatical diversity**! 🎉

```


<div align="center">⁂</div>

[^1]: 28-And-29-Spam-Ham-Projects-Using-Word2vec-AvgWord2vec.ipynb```

