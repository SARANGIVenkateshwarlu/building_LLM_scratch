
# LLM Training: Loss Functions & Text Evaluation

## 1. Lecture Context: Stage 2 – Training Begins

**Stage 1 complete**: Data prep → Attention → GPT architecture (124M params, text generation).  
**Stage 2 focus**: **Training** pipeline (7 steps):  
1. **Text generation** (recap)  
2. **Text evaluation** (loss functions) ← **Today**  
3. Dataset processing  
4. Training/validation loss  
5. LLM training loop (backprop)  
6. Pretrained weights  

**Problem**: Untrained GPT produces gibberish ("hello I am" → random tokens).  
**Goal**: Define **quantitative loss** to measure "badness" → enable gradient descent.

**Training intuition**: Convert qualitative "this output sucks" → number (loss) → minimize via backprop.

---

## 2. GPT Config Recap (Training Setup)

```

vocab_size:     50,257  (tiktoken BPE tokenizer)
context_length: 256     (reduced for laptop training; GPT-2 uses 1024)
embedding_dim:  768
n_heads:        12
n_layers:       12      (Transformer blocks)
dropout:        0.1
qkv_bias:       False

```

**Model**: `GPT(config)` → input token IDs → **logits tensor**.

---

## 3. Inputs vs Targets: Autoregressive Setup

### 3.1. Example (batch_size=2, context_len=3)

**Inputs** `[2, 3]`:
```

Batch 1:   → "every effort moves"
Batch 2: [   40, 1107, 588]  → "I really like"

```

**Targets** `[2, 3]` (inputs **shifted by 1**):
```

Batch 1:   → "effort moves you"
Batch 2:  → "really like chocolate"

```

### 3.2. Why Shifted Targets? (3 Predictions per Sequence)

For 3 tokens → **3 prediction tasks**:
```

Batch 1:

- Pos 0: "every"     → predict "effort"    (target)
- Pos 1: "every effort" → predict "moves"    (target)[^1]
- Pos 2: "every effort moves" → predict "you" (target)[^2]

Batch 2: Similar...

```

**Key**: Same tensor for inputs/targets, just **offset by 1**. Enables self-supervision.

---

## 4. Forward Pass: Inputs → Predicted Outputs

### 4.1. Model Forward
```

inputs  → GPT → logits[^3][^2]

```

### 4.2. Logits → Probabilities → Token IDs

**Step 1**: Softmax over vocab dimension:
```

logits[batch=0, pos=0, :] = [-0.2, 1.3, -2.1, 3.7, ...]  (raw scores)
probs[0, 0, :] = [0.01, 0.12, 0.02, 0.45, ...]  (∑=1 per position)

```

**Step 2**: Argmax → predicted token IDs:
```

pred_ids = argmax(probs) = 16657  ("armed"?)
pred_ids = argmax(probs) = 339    ("H"?)[^1]
pred_ids = argmax(probs) = 42826  ("Netflix"?)[^2]

```

**Untrained output**: Gibberish (expected).

**Simple vocab example (vocab_size=7)**:
```

probs = [0.1, 0.6, 0.2, 0.05, 0, 0.02, 0.01]  ("a", "effort", "every", "forward", "moves", "U", "zoo")
argmax → index=1 → "effort" ✓ (if trained)

```

---

## 5. Loss Function: Cross Entropy (Negative Log Likelihood)

### 5.1. Extract Target Probabilities

From `probs [2, 3, 50257]` and `targets [2, 3]`:
```

Batch 0 targets:
→ p_target_0 = probs, probs, probs[^1][^2]
→ [0.001, 0.023, 0.007]  (low → untrained!)

Batch 1: Similar → 6 total p_target values

```

**Gather targets**:
```

p_targets = [p11, p12, p13, p21, p22, p23]

```

### 5.2. Negative Log Likelihood (Cross Entropy)

```

loss = -mean(log(p11), log(p12), ..., log(p23))
= - (1/6) * Σ log(p_target_i)

```

**Why?**
- Want each `p_target → 1` → `log(1)=0` → `loss → 0`.
- Low p_target (e.g., 0.001) → large negative log → high loss → strong training signal.

**Example**:
```

p_targets = [0.001, 0.023, 0.007, ...]
logs = [-6.9, -3.8, -4.96, ...]
mean_log = -5.0 → loss = 5.0  (high → bad model)

```

### 5.3. PyTorch One-Liner
```python
logits_flat = logits.view(-1, vocab_size)  #[^4]
targets_flat = targets.view(-1)            #[^4]

loss = F.cross_entropy(logits_flat, targets_flat)
```

- **Internally**: Softmax → gather target probs → -mean(log(probs)).

**Computed loss**: ~10.79 (very high).

---

## 6. Perplexity: Interpretable Loss Metric

```
perplexity = exp(loss) = e^10.79 ≈ 48,725
```

**Intuition**:

- Model as uncertain as **randomly choosing** from 48,725 tokens.
- Vocab=50,257 → nearly uniform (bad!).
- **Good model**: perplexity ≈ 2-10 (chooses from few likely tokens).

**Why useful?** Directly relates to vocab size → human-interpretable.

---

## 7. Code Walkthrough

### 7.1. Setup

```python
config = GPTConfig(...)  # vocab=50257, context=256, etc.
model = GPT(config)
inputs = torch.tensor([, ])  #[^3][^2]
targets = torch.tensor([, ]) #[^2][^3]
```


### 7.2. Forward + Loss

```python
logits = model(inputs)  #[^3][^2]

# Manual (educational)
probs = F.softmax(logits, dim=-1)
p_targets = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
log_p = torch.log(p_targets)
manual_loss = -log_p.mean()  # ~10.79

# PyTorch (production)
logits_flat = logits.view(-1, config.vocab_size)  #[^4]
targets_flat = targets.view(-1)                    #[^4]
ce_loss = F.cross_entropy(logits_flat, targets_flat)  # ~10.79 ✓

perplexity = torch.exp(ce_loss)  # ~48,725
```


### 7.3. Decode for Intuition

```
Decode inputs: "every effort moves"
Decode targets: "effort moves you"
Decode preds: "armed H Netflix"  (gibberish)
```


---

## 8. Training Pipeline Preview (Next Lectures)

1. **Dataset**: Tokenize book (e.g., "The Verdict") → input/target pairs.
2. **Batch loop**: Compute loss over entire dataset → train/val splits.
3. **Backprop**: `loss.backward()` → optimizer.step().
4. **Monitor**: Loss ↓, perplexity ↓ → coherent generation.
5. **Pretrained weights**: Load OpenAI GPT-2 → fine-tune.

**Hands-on next**: Scale to full book dataset → compute dataset loss.

[file:21]

```
<span style="display:none">[^5]</span>

<div align="center">⁂</div>

[^1]: 28-And-29-Spam-Ham-Projects-Using-Word2vec-AvgWord2vec.ipynb
[^2]: https://www.dictionary.com/browse/contemplating
[^3]: https://www.merriam-webster.com/dictionary/contemplate
[^4]: https://www.youtube.com/watch?v=QRacdebxVHg
[^5]: paste.txt```

