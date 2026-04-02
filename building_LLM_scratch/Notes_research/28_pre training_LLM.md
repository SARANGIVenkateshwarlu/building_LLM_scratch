
# LLM Pre-Training: Full Training Loop with Backpropagation

## 1. Lecture Context: Stage 2 Complete – **Train the Model!**

**Milestones achieved**:
```

Stage 1: ✓ GPT-2 architecture (124M→162M params)
Stage 2:

1. ✓ Text generation (autoregressive)
2. ✓ Loss (cross-entropy)
3. ✓ Dataset → batches (Verdict book, 5K tokens)
4. **LLM training loop** ← Today (**backprop**!)
```

**Hands-on**: **10 epochs** on laptop → **loss ↓9.8→3.9**, ppl ↓51K→50. **Overfitting** observed (expected on tiny data).

**Dataset**: *"The Verdict"* (~20K chars → 5K tokens, 90/10 train/val).

---

## 2. Training Loop Schematic

```

for epoch in range(num_epochs):           \# 10 epochs
for batch_idx, (xb, yb) in enumerate(train_loader):  \# 9 batches
logits = model(xb)                 \# Forward: [B,256,50K] → loss
loss = F.cross_entropy(...)        \# Target probs → NLL
model.zero_grad()                  \# Reset grads
loss.backward()                    \# **BACKPROP**: ∂loss/∂params (162M!)
optimizer.step()                   \# Update: params -= lr * grads
optimizer.zero_grad()              \# (AdamW)

        if batch_idx % 5 == 0:             # Eval freq
            print(train_loss, val_loss)
    
    generate_sample()  # Visualize after epoch
    ```

**Key insight**: `loss.backward()` computes **all 162M gradients** in **one line**!

---

## 3. Parameter Breakdown (Why 162M?)

```

Token embeddings:    vocab(50K) × embed(768) =  38M
Positional embeds:   ctx(256)   × embed(768) =   0.2M
───────────────────────────────────────────────
Embeddings total:                         38M

**Per Transformer block** (×12 blocks):
Multi-head attn (QKV): 768×768×3 =  1.8M
FFN:          768×3072×2 =  4.7M
Output proj:  768×768     =  0.6M
─────────────────────────────────────
Block total:                              7.1M ×12 = 85M

LM head:             embed(768) × vocab(50K)  =  38M
───────────────────────────────────────────────
**GRAND TOTAL: 161M** (GPT-2 small: 124M w/ tied weights)

```

**Update rule**: `param_new = param_old - lr × ∂loss/∂param` (×162M!).

---

## 4. Step-by-Step Code Instructions

### Step 1: Setup (from previous lectures)
```python
# Config (laptop-friendly)
config = GPTConfig(
    vocab_size=50257, context_length=256, embed_dim=768,
    n_heads=12, n_layers=12, dropout=0.1
)
model = GPT(config).to(device='cpu')  # Or 'cuda'

# DataLoaders (Verdict book)
train_loader = create_dataloader(train_tokens, batch_size=2, ...)
val_loader   = create_dataloader(val_tokens, batch_size=2, ...)

# Hyperparams
num_epochs = 10
eval_freq = 5
learning_rate = 1e-3
```


### Step 2: Optimizer (AdamW)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),  # All 162M params!
    lr=learning_rate, weight_decay=1e-4
)
```


### Step 3: Training Loop (`pretrain_llm`)

```python
def pretrain_llm(model, train_loader, val_loader, num_epochs=10):
    global_step = 0
    for epoch in range(num_epochs):
        # Training phase
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward
            logits = model(xb)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                yb.view(-1)
            )
            
            # Backward + Update
            optimizer.zero_grad()  # Reset grads
            loss.backward()        # Compute ∂loss/∂params
            optimizer.step()       # params -= lr * grads
            
            global_step += xb.numel()  # Tokens seen (~512/batch)
            
            # Eval every 5 batches
            if batch_idx % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model)
                val_loss   = calc_loss_loader(val_loader, model)
                print(f"Step {global_step}: Train {train_loss:.3f}, Val {val_loss:.3f}")
        
        # Sample generation (visualize progress)
        generate_print_sample(model, "every effort moves you")
    
    print("Training complete!")
```


### Step 4: Eval Helpers (from prev lecture)

```python
def calc_loss_loader(loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def generate_print_sample(model, start="every effort moves you", max_new=50):
    # Tokenize → autoregressive generate → decode → print
    print(enc.decode(generate_text_simple(model, enc.encode(start), max_new)))
```


### Step 5: Run!

```python
start_time = time.time()
pretrain_llm(model, train_loader, val_loader, num_epochs=10)
print(f"Completed in {(time.time()-start_time)/60:.1f} minutes")
```

**Laptop timing**: **6.6 minutes** (MacBook Air).

---

## 5. Results Analysis

### 5.1. Loss Curves

```
Step    Train   Val     PPL (exp(loss))
  0     9.78 → 3.91   51K → 50
Init    9.93          54K
```

- **Train**: Drops dramatically (overfits tiny data).
- **Val**: Drops then **plateaus ~6.3** (stagnates).

**Plot**:

```
Train loss: \\\\\\\\\ (steep ↓)
Val loss:   \\\\____ (↓ then flat → overfitting)
Tokens:     Linear ↑ (~50K total)
```


### 5.2. Generation Progress (Input: "every effort moves you")

```
Epoch 1:  ",,,,,,,,,,,,,,,"
Epoch 3:  " and I had been"
Epoch 7:  "you? Yes, quite insensible to the irony..."
Epoch 10: "you was one of the xmc laid down across the SE..."
```

- **Early**: Repetitive commas.
- **Mid**: Fragments from data.
- **Late**: **Grammatical** (but memorized: "insensible to the irony" verbatim from book).

**Overfitting signs**:

- Train << Val loss.
- **Memorization** (quotes training text).

**Tiny data (5K tokens)** → expected. Production: **billions/trillions**.

---

## 6. Debugging \& Sanity Checks

```
# Monitor gradients (shouldn't explode)
print(f"Max grad: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0):.2f}")

# Tokens seen
assert final_tokens_seen ≈ 5K tokens × 10 epochs ≈ 50K

# No NaNs
assert not torch.isnan(loss).any()
```

**Hyperparam tuning**:

- `lr`: 1e-4 → 1e-2 (too high → diverge).
- `epochs`: 5-50.
- `batch_size`: 4-32 (GPU).
- `weight_decay`: 1e-4.

---

## 7. Key Insights \& Extensions

**Why it works**:

- **Differentiable pipeline**: Embed → Transformer → logits → softmax → cross-entropy.
- **PyTorch magic**: `loss.backward()` handles chain rule ×162M params.
- **Self-supervised**: Targets from input (shifted).

**Scale up**:

```
# Larger context (slower)
config.context_length = 1024

# GPU + bigger batches
model.to('cuda'), batch_size=16

# Huge data: 1T+ tokens, 100s GPUs, weeks
```

**Next**: **Decoding strategies** (temperature, top-k) → reduce overfitting → **creative** generation.

**Victory**: **Full GPT-2 trained from scratch** on laptop → coherent text!

[file:23]

```
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: paste.txt```

