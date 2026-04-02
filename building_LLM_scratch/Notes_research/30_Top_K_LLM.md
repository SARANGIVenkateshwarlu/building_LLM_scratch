# LLM Decoding Strategy: Top‑k Sampling (with Temperature)

## 1. Why We Need Decoding Strategies

- Basic decoding: take the token with **maximum probability** (greedy / argmax).
- Pipeline so far:
  1. Input text → GPT model → logits vector for next token.
  2. Apply softmax → probability distribution over vocabulary.
  3. Choose argmax index → next token.

- Problems:
  - Outputs can be random‑looking and incoherent for small models.
  - Deterministic argmax encourages **overfitting** and repetition.
  - With temperature alone, even **nonsensical tokens** (e.g., “pizza”) can get sampled.

**Goal**: Keep creativity, but **constrain** which tokens are allowed as candidates.

---

## 2. Recap: Temperature Scaling + Multinomial Sampling

### 2.1. From greedy to probabilistic sampling

Instead of:
- `next_id = argmax(probs)`

Use:
- `next_id = multinomial(probs)`  (sample according to probabilities)

Effect:
- Most probable token still chosen **most often**, but
- Other plausible tokens sometimes chosen → **diversity** and **creativity**.

### 2.2. Temperature scaling effect

Steps:
1. Start with logits \(z\) (un-normalized scores).
2. Divide by temperature \(T > 0\):  
   \(\tilde{z} = z / T\).
3. Apply softmax: \(p = \text{softmax}(\tilde{z})\).
4. Sample with multinomial from \(p\).

Behavior:
- **Low T (< 1)**: sharp distribution, almost greedy, low randomness.
- **T = 1**: original model confidence.
- **High T (> 1)**: flat distribution, high randomness (may produce nonsense, like “pizza”).

Issue: with only temperature, **every** token still has some chance to be sampled.

---

## 3. Idea of Top‑k Sampling

**Problem with temperature alone**:
- Even after scaling, rare/unrelated tokens still have **non‑zero** probability.
- Example: “every effort moves you pizza” may appear with T=5 because “pizza” has ~4% probability.

**Top‑k intuition**:
- Before sampling, **restrict** to only the **k most likely tokens**.
- Completely **remove** all other tokens from consideration.

High‑level procedure:
1. Take next‑token logits.
2. Find **top k** logits and their indices.
3. Set all other logits to **−∞**.
4. Apply temperature scaling (optional).
5. Softmax → probabilities now non‑zero only on top‑k tokens.
6. Sample from this reduced distribution with multinomial.

Result:
- **No chance** for very low‑probability, nonsensical tokens to be generated.
- Randomness is **contained** within a small, plausible set.

---

## 4. Step‑by‑Step: Top‑k on a Single Logit Vector

Assume:
- Next‑token logits: \([z_0, z_1, ..., z_{V-1}]\).
- We choose \(k = 3\).

### Step 1: Compute top‑k

- Example logits (simplified):
  - \([4.51, -0.3, 1.2, 6.75, 0.0, -2.1, 6.28, 0.8, ...]\)
- `topk` returns:
  - `top_values = [6.75, 6.28, 4.51]`
  - `top_indices = [3, 6, 0]`

### Step 2: Mask all non top‑k logits

- Create new logits:
  - index 3 → 6.75  
  - index 6 → 6.28  
  - index 0 → 4.51  
  - all others → \(-\infty\)

So the new logit vector is:
- \([4.51, -∞, -∞, 6.75, -∞, -∞, 6.28, -∞, ...]\)

### Step 3: (Optional) apply temperature

- If using temperature \(T\), divide only these (non‑∞) logits by \(T\):
  - \(\tilde{z}_i = z_i / T\) for indices 0,3,6.

### Step 4: Softmax

- Softmax over the masked & scaled logits:
  - Non‑top‑k positions (−∞) → probability 0.
  - Top‑k positions → renormalized to sum to 1.

Example probabilities (k=3):
- token at idx 3 (largest logit): ~0.57
- token at idx 6: ~0.35
- token at idx 0: ~0.08
- all others: 0

### Step 5: Multinomial sampling

- Now **sample** from this 3‑element distribution:
  - Most often: token idx 3.
  - Sometimes: token idx 6.
  - Rarely: token idx 0.
  - Never: any non‑top‑k token (probability exactly 0).

---

## 5. Full Decoding Workflow: Top‑k + Temperature + Multinomial

### 5.1. Logic sequence for one step

For each decoding step:

1. **Run model**:
   - Input context IDs → GPT → logits for last position: `logits_last`.

2. **Top‑k restriction**:
   - Choose k (e.g., 25).
   - Get top‑k logits and indices.
   - Replace all other logits with \(-∞\).

3. **Temperature scaling**:
   - Divide remaining logits by `temperature` (e.g., 1.0–1.4).

4. **Softmax**:
   - Convert to probabilities over top‑k tokens (others 0).

5. **Sample**:
   - Use multinomial sampling to pick one next token ID.

6. **Append & repeat**:
   - Append the new token ID to the context.
   - Repeat until max_new_tokens or an EOS token is reached.

### 5.2. Pseudocode

```python
def generate_with_topk(
    model, 
    input_ids, 
    max_new_tokens, 
    context_size, 
    temperature=1.0, 
    top_k=25,
    eos_id=None
):
    for _ in range(max_new_tokens):
        # 1) Limit context
        idx_cond = input_ids[:, -context_size:]
        
        # 2) Forward pass
        logits = model(idx_cond)         # [B, T, V]
        logits = logits[:, -1, :]        # last position [B, V]
        
        # 3) Top-k masking
        if top_k is not None:
            # Get top-k logits and indices
            top_vals, top_idx = logits.topk(top_k, dim=-1)
            # Create mask: all logits < smallest top-k value → -inf
            min_top = top_vals[..., -1, None]
            logits = torch.where(
                logits < min_top,
                torch.full_like(logits, float("-inf")),
                logits
            )
        
        # 4) Temperature scaling + softmax
        if temperature > 0:
            logits = logits / temperature
            probs = logits.softmax(dim=-1)   # [B, V]
            # 5) Multinomial sampling
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
        else:
            # Fallback: greedy
            next_id = logits.argmax(dim=-1, keepdim=True)
        
        # Optional: EOS early stop
        if eos_id is not None and (next_id == eos_id).all():
            break
        
        # 6) Append
        input_ids = torch.cat([input_ids, next_id], dim=1)
    
    return input_ids
```


---

## 6. Example: Before vs After Top‑k

### 6.1. Old decoding (argmax only)

Input:

```text
"every effort moves you"
```

Output (example from training lecture):

```text
"you was one of the xmc laid down across the seers and silver of an exquisitely appointed ..."
```

Issues:

- Overfitting and direct copying from training text.
- Sometimes nonsensical phrases.


### 6.2. With temperature + top‑k

Call:

- `top_k = 25`
- `temperature = 1.4`
- `max_new_tokens = 15`

Output (example from lecture):

```text
"every effort moves you stand to work on surprise one of us had gone with random it is ..."
```

Observations:

- Tokens **weren’t** present as a contiguous phrase in the original book.
- Model produces **novel combinations** (true generative behavior).
- Overfitting/memorization is **reduced** because:
    - Sampling introduces stochasticity.
    - Top‑k removes irrelevant extreme‑tail tokens.

---

## 7. Practical Usage Guidelines

### 7.1. Choosing `top_k` and `temperature`

- `top_k`:
    - 5–20: Very conservative, constrained, low diversity.
    - 20–50: Good trade‑off for small models.
    - 50+: More flexible but can reintroduce strange tokens.
- `temperature`:
    - 0.7–1.0: Safe default.
    - <0.7: More deterministic, safer, less creative.
    - >1.0: More diverse, but risk of nonsense.

Typical config:

```python
top_k = 25
temperature = 0.8–1.0
```


### 7.2. When to use what

- For **factual or code**:
    - Lower `temperature` (0.2–0.7), moderate `top_k` (~10–20).
- For **creative writing**:
    - Higher `temperature` (0.9–1.2), larger `top_k` (~40–50).
- For **debugging the model**:
    - Turn off sampling (temperature→0 or just argmax) to see “pure” behavior.

---

## 8. Conceptual Summary (Mental Model)

Full next‑token pipeline:

1. **Model**:
    - Context → Transformer → logits for each vocab token.
2. **Top‑k mask**:
    - Keep only **K most likely** tokens, drop all others (→ −∞).
3. **Temperature**:
    - Adjust the **sharpness** of remaining scores.
    - Low T → **confident**, high T → **exploratory**.
4. **Softmax**:
    - Convert adjusted logits → probabilities.
5. **Multinomial sampling**:
    - Draw one token according to these probabilities.
6. **Append \& repeat**:
    - Add token to sequence, continue until stopping condition.

This combination:

- Prevents the model from **always** choosing the same word (anti‑repetition).
- Prevents it from choosing **completely random** words (anti‑nonsense).
- Gives you a **controlled knob** over creativity vs reliability.

```
```

