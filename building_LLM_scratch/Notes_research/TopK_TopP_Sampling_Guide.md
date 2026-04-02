# Top‑K and Top‑P Sampling in LLM / SLM Design  
---

# Table of Contents

1. Why Sampling Matters  
2. Temperature Scaling  
3. Top‑K Sampling  
4. Top‑P (Nucleus) Sampling  
5. Top‑K vs Top‑P Comparison  
6. Designing a Novel SLM or LLM (Step‑by‑Step)  
7. Production Engineering Stack  
8. Advanced Techniques  
9. Evaluation Strategy  
10. Common Mistakes  
11. When NOT to Use Sampling  
12. PyTorch Implementation  
13. CUDA‑Optimized Pseudocode  
14. Entropy & Theoretical Insight  
15. Modern Production Defaults (2026)

---

# 1. Why Sampling Matters

LLMs generate text autoregressively:

\[
P(w_t \mid w_{<t})
\]

At each step, the model outputs logits over the vocabulary.

If we choose:

\[
\arg\max P(w_t)
\]

we get **greedy decoding**, which is:

- Deterministic  
- Repetitive  
- Low diversity  
- Often dull  

Modern systems use **stochastic decoding**, primarily:

- Temperature scaling  
- Top‑K sampling  
- Top‑P (Nucleus) sampling  
- Repetition penalties  

---

# 2. Temperature Scaling

Before applying Top‑K or Top‑P, logits are scaled.

Given logits \( z_i \):

\[
z_i' = \frac{z_i}{T}
\]

Where:

- \(T < 1\): Sharper distribution (more deterministic)  
- \(T = 1\): No change  
- \(T > 1\): Flatter distribution (more creative)  

Softmax:

\[
P_i = \frac{e^{z_i'}}{\sum_j e^{z_j'}}
\]

Temperature is applied **before filtering**.

---

# 3. Top‑K Sampling

## Definition

Keep the top \(K\) highest‑probability tokens and sample from them.

## Algorithm

1. Compute logits  
2. Apply temperature  
3. Compute softmax  
4. Sort tokens  
5. Keep top \(K\)  
6. Zero others  
7. Renormalize  
8. Sample  

## Mathematical Form

Let \(V_K\) be the top‑K set:

\[
V_K = \text{TopK}(P, K)
\]

Then:

\[
P'_i =
\begin{cases}
\frac{P_i}{\sum_{j \in V_K} P_j} & i \in V_K \\
0 & \text{otherwise}
\end{cases}
\]

## Typical Values

| K | Behavior |
|----|----------|
| 1 | Greedy |
| 5 | Conservative |
| 40–50 | Balanced |
| 100+ | Creative / noisy |

## Limitations

- Fixed size  
- Not distribution‑adaptive  
- Can truncate too aggressively  

---

# 4. Top‑P (Nucleus Sampling)

## Definition

Select the smallest set of tokens such that cumulative probability ≥ \(p\).

\[
\sum_{i \in V_p} P_i \ge p
\]

## Algorithm

1. Compute logits  
2. Apply temperature  
3. Softmax  
4. Sort descending  
5. Compute cumulative sum  
6. Select minimal set satisfying cumulative ≥ \(p\)  
7. Renormalize  
8. Sample  

## Mathematical Form

\[
P'_i =
\begin{cases}
\frac{P_i}{\sum_{j \in V_p} P_j} & i \in V_p \\
0 & \text{otherwise}
\end{cases}
\]

## Typical Values

| p | Behavior |
|----|----------|
| 0.8 | Conservative |
| 0.9 | Balanced |
| 0.95 | Creative |
| 0.98 | Very diverse |

## Why It’s Better

- Adaptive to distribution shape  
- More stable across contexts  
- Standard in modern chat LLMs  

---

# 5. Top‑K vs Top‑P Comparison

| Feature | Top‑K | Top‑P |
|----------|--------|--------|
| Fixed size | ✅ | ❌ |
| Adaptive | ❌ | ✅ |
| Production use | Moderate | Very High |
| Stability | Medium | High |
| Creative control | Medium | High |

---

# 6. Designing a Novel SLM or LLM

## Step 1 — Model Size Consideration

| Model Size | Recommendation |
|-------------|----------------|
| ≤3B (SLM) | Lower temperature, lower p |
| 7–13B | Moderate temperature + p=0.9 |
| 30B+ | Flexible tuning |

Small models hallucinate more → decode conservatively.

---

## Step 2 — Baseline Default

Recommended safe baseline:

```
temperature = 0.7
top_p = 0.9
top_k = None
```

---

## Step 3 — Task-Specific Tuning

### Chatbot
```
temperature = 0.7–0.9
top_p = 0.9–0.95
```

### Code Generation
```
temperature = 0.2–0.5
top_p = 0.8–0.9
```

### Creative Writing
```
temperature = 0.9–1.2
top_p = 0.95–0.98
```

---

## Step 4 — Add Penalties

Modern decoding stack:

\[
\text{Logits}
\rightarrow \text{Temperature}
\rightarrow \text{Repetition Penalty}
\rightarrow \text{Top‑P}
\rightarrow \text{Sampling}
\]

Common additions:

- Repetition penalty (1.05–1.2)  
- Frequency penalty  
- Presence penalty  
- Logit bias  

---

# 7. Production Engineering Stack

Modern (2026) conversational systems use:

```
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.1
frequency_penalty = 0.2
top_k = optional safety cap (e.g., 100)
```

---

# 8. Advanced Techniques

## Dynamic Top‑P

Adjust \(p\) based on entropy:

- High entropy → lower p  
- Low entropy → higher p  

---

## Entropy-Aware Sampling

Entropy:

\[
H(P) = -\sum_i P_i \log P_i
\]

Use entropy to modulate temperature dynamically.

---

## Contrastive Decoding

Balances:

\[
\text{Score} = \log P_{large} - \alpha \log P_{small}
\]

Reduces hallucinations.

---

# 9. Evaluation Strategy

Measure:

- Perplexity  
- Self‑BLEU (diversity)  
- Repetition rate  
- Human evaluation  
- Toxicity rate  
- Hallucination rate  

Sampling affects all metrics.

---

# 10. Common Mistakes

❌ High temperature + high p  
❌ Fixed K without validation  
❌ No renormalization  
❌ No repetition penalty  
❌ Ignoring entropy diagnostics  

---

# 11. When NOT to Use Sampling

Use greedy or beam search for:

- Deterministic benchmarks  
- Mathematical proofs  
- Safety‑critical generation  
- Evaluation pipelines  

---

# 12. PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, temperature=0.7, top_p=0.9):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(-1, next_token)
```

---

# 13. CUDA‑Optimized Pseudocode

```cpp
// Warp-level parallel Top-P sketch

load logits into shared memory
apply temperature scaling
compute softmax (block reduction)
sort via parallel radix sort
compute prefix sum (warp scan)
truncate when cumulative >= p
renormalize
sample using curand
```

---

# 14. Entropy & Theoretical Insight

Sampling modifies the effective support of the distribution.

Top‑K:

\[
|V_K| = K
\]

Top‑P:

\[
|V_p| = f(H(P))
\]

Where entropy \(H(P)\) determines nucleus size.

Higher entropy → larger nucleus.

---

# 15. Final Design Philosophy

When building a novel LLM/SLM:

1. Train for calibrated probabilities  
2. Apply temperature  
3. Prefer Top‑P over Top‑K  
4. Add repetition controls  
5. Tune per domain  
6. Evaluate with human + automatic metrics  
7. Monitor entropy during decoding  

---

# Key Takeaways

- Top‑K = fixed size  
- Top‑P = adaptive mass  
- Temperature shapes distribution  
- Modern LLMs prefer Top‑P  
- Decoding design is as important as model architecture  

---

*End of File*