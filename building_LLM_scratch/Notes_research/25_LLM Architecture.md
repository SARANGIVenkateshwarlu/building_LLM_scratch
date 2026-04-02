
# GPT Text Generation – Final Lecture: From Logits to Text

## 1. Autoregressive Text Generation Process

**Core mechanism**: LLMs generate **one token at a time** in an iterative loop:

```

Iteration 1:  ["hello", "I", "am"]  → predict "a"
Iteration 2:  ["hello", "I", "am", "a"] → predict "model"
Iteration 3:  ["hello", "I", "am", "a", "model"] → predict "ready"
...
Iteration 6:  ["hello", "I", "am", "a", "model", "ready", "to", "help"] → STOP

```

**Key concepts**:
- **Context window**: Max tokens model considers (GPT-2: 1024).
- **Max new tokens**: How many tokens to generate (e.g., 6).
- **Append & repeat**: Each predicted token becomes input for next iteration.

**Example output**: `hello I am a model ready to help.` (6 new tokens generated)

---

## 2. Complete GPT Pipeline Recap

```

Text → Token IDs → Token Embeds(768D) + Positional Embeds → Dropout
→ Transformer Blocks(12×) → Final LayerNorm → Output Head → LOGITS[B, T, 50257]

```

**Output tensor shape**: `[batch_size, num_tokens, vocab_size]`
- **Example**: Input "every effort moves you" (4 tokens) → `[1, 4, 50257]`

**What each row represents** (autoregressive predictions):
```

Row 0: After "every" → predict "effort"
Row 1: After "every effort" → predict "moves"
Row 2: After "every effort moves" → predict "you"
Row 3: After "every effort moves you" → predict "forward"

```

**Today's goal**: Extract **last row** → generate **next token** → **repeat**.

---

## 3. 5-Step Token Generation Process (Whiteboard Breakdown)

### Step 1: Extract Last Time Step
```

Logits:  → logits[:, -1, :] →[^11][^12]

```
- **Why last row?** Only final position predicts **next** token.
- Shape: `[batch_size, vocab_size]`

### Step 2: Logits → Probabilities (Softmax)
```

logits[-1] = [-0.2, 1.3, -2.1, 3.7, ...]  (raw scores)
softmax(logits[-1]) = [0.01, 0.12, 0.02, 0.45, ...]  (probabilities)

```
- **Softmax**: Converts scores to valid probabilities (∑=1).
- Each value = probability of that vocabulary token being next.

### Step 3: Greedy Selection (Argmax)
```

probs = [0.01, 0.12, 0.02, 0.45, 0.03, ...]
next_token_id = argmax(probs) = 57  (highest probability index)

```
**Note**: Softmax optional for argmax (monotonic), but shows probabilities.

### Step 4: Decode Token ID → Text
```

token_id=57 → tokenizer.decode(57) → "forward"

```

### Step 5: Append & Loop
```

Original: ["every", "effort", "moves", "you"]
New:      ["every", "effort", "moves", "you", "forward"]

```
**Repeat** until `max_new_tokens` reached.

---

## 4. Code Implementation: `generate_text_simple()`

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 1. Respect context window
        idx_cond = idx[:, -context_size:]
        
        # 2. Forward pass → logits
        logits = model(idx_cond)
        
        # 3. Extract last time step
        logits = logits[:, -1, :]  # [B, vocab_size]
        
        # 4. Softmax → probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 5. Argmax → next token
        idx_next = torch.argmax(probs, dim=-1)  # [B]
        
        # 6. Append to sequence
        idx = torch.cat((idx, idx_next.unsqueeze(0)), dim=1)
    
    return idx
```

**Example execution**:

```
Input:  ["hello", "I", "am"] → token_ids=[^1]
max_new_tokens=6, context_size=1024

After 6 iterations:
Output:[^1]
Decode: "hello I am <random gibberish>"
```

**Why random?** Model untrained (random 124M parameters).

---

## 5. Dimension Flow Through Generation Loop

```
Iteration 1:
Input:    (B=1, T=3)[^13][^11]
→ logits:[^11][^13]
→ last:[^11]
→ probs:  (∑=1)[^11]
→ next_id:[^11]
→ new_input:[^12][^11]

Iteration 2:
Input:    (appended)[^12][^11]
→ Repeat...
```

**Context window enforcement**:

```
if len(tokens) > context_size:
    input = tokens[-context_size:]  # Keep recent tokens only
```


---

## 6. Why Softmax? (Even When Not Needed)

**Current**: `argmax(softmax(logits)) = argmax(logits)` ✓ Identical results.

**Future modules** (foreshadowed):

- **Temperature sampling**: `probs ** (1/temperature)` → more/less random.
- **Top-k/top-p sampling**: Sample from top N tokens or cumulative probability.
- **Beam search**: Keep multiple candidate sequences.

Softmax **required** for probabilistic sampling → **creative/diverse** outputs.

---

## 7. Full End-to-End Demo

```
# Setup
model = GPT(config)  # 124M params, random weights
model.eval()  # Disable dropout/norm during inference
input_text = "hello I am"
input_ids = encoder.encode(input_text)  #[^12][^11]

# Generate
output_ids = generate_text_simple(
    model, input_ids, max_new_tokens=6, context_size=1024
)

# Decode
generated_text = decoder.decode(output_ids)
print(generated_text)  # "hello I am <random tokens>"
```

**Result**: Works! Architecture complete. Training next.

---

## 8. GPT Architecture Module Complete ✅

**6-7 lectures covered**:

1. Dummy GPT → LayerNorm → GeLU → FFN → Residuals
2. TransformerBlock (attention core)
3. Full GPT model (124M params)
4. **Text generation** (today)

**Milestones achieved**:

- [x] Full GPT-2 small architecture (local laptop)
- [x] Forward pass: text → logits
- [x] Autoregressive generation: logits → tokens → text
- [x] Dimension mastery through every layer

**Next module**: **Training 124M parameters** → coherent outputs.

**Validation**: Compare untrained vs ChatGPT:

```
Untrained: "hello I am ï»¿î±î±¦î±¦î±¦î±¦"
ChatGPT:  "hello I am here to help you today"
```

**You did it!** Built GPT-2 from scratch. 95% of users never see this level.

---
<span style="display:none">[^10][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://pm.dartus.fr/posts/2025/how-llm-generate-text/

[^2]: https://www.solutelabs.com/blog/step-by-step-llm-process

[^3]: https://towardsai.net/p/l/how-does-an-llm-generate-text

[^4]: https://www.linkedin.com/pulse/how-exactly-llm-generates-text-ivan-reznikov

[^5]: https://milvus.io/ai-quick-reference/how-do-llms-generate-text

[^6]: https://www.youtube.com/watch?v=NKnZYvZA7w4

[^7]: https://zactax.com/blog/2026/02/how-do-llms-generate-text

[^8]: https://www.reddit.com/r/deeplearning/comments/1npk1fn/how_llms_generate_text_a_clear_and_comprehensive/

[^9]: https://www.youtube.com/watch?v=t1iONhCmzg4

[^10]: https://dev.to/mahakfaheem/decoding-demystified-how-llms-generate-text-iii-3a0d

[^11]: 28-And-29-Spam-Ham-Projects-Using-Word2vec-AvgWord2vec.ipynb

[^12]: https://www.oxfordlearnersdictionaries.com/definition/english/contemplate

[^13]: https://www.merriam-webster.com/dictionary/contemplate

