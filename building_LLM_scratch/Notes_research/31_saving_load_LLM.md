# PyTorch Model Weights: Saving & Loading (Checkpoints)

## 1. Lecture Context: Preparing for Pre-trained Weights

**Progress recap**:
```

Stage 1: ✓ GPT-2 architecture (162M params)
Stage 2:
1-4. ✓ Loss → Dataset → Training → Decoding (temp + top-k)
5. **Checkpoints** ← Today (save/load weights)
6. **Next**: Load OpenAI GPT-2 pretrained weights!

```

**Why checkpoints?**
- **162M parameters** × **Adam optimizer state** = **gigabytes**.
- Training: **hours/days** → **Don't lose progress!**
- **Collaborate**: Share trained model with team.
- **Resume**: Close laptop → continue tomorrow.

---

## 2. Core PyTorch Commands

### 2.1. Save Model Weights
```python
torch.save(model.state_dict(), "model.pth")
```

**What is `model.state_dict()`?**

- **Dictionary**: `{layer_name: parameter_tensor}`
- Contains **all learnable parameters** (weights/biases).
- Example:

```python
{
  'token_embedding.weight': torch.Size(),
  'transformer.0.attn.W_Q.weight': torch.Size(),
  'lm_head.weight': torch.Size(),
  # ... 162M total params
}
```

**`.pth`**: PyTorch convention (binary format).

### 2.2. Load Model Weights

```python
model = GPT(config)  # Fresh model (random weights)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Ready for inference
```

**Flow**:

```
Random GPT → load_state_dict("model.pth") → Trained GPT ✓
```


---

## 3. Problem: Optimizer State Loss

**Issue**: `model.state_dict()` saves **only model weights**.

- **AdamW** maintains **per-parameter history**:
    - `exp_avg` (gradient history).
    - `exp_avg_sq` (squared gradient history).
    - Learning rate, weight decay, etc.

**Without optimizer state**:

```
Day 1: Train 10 epochs → good convergence
Day 2: Load model → new AdamW → resets history → unstable training!
```


### 3.1. Save Model + Optimizer Checkpoint

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "model+optimizer.pth")
```


### 3.2. Load Full Checkpoint

```python
checkpoint = torch.load("model+optimizer.pth")

# Fresh model + optimizer
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Load states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.train()  # Resume training seamlessly
```


---

## 4. Step-by-Step Implementation

### Step 4.1: During Training (Save Every N Epochs)

```python
def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"✅ Checkpoint saved: {filepath}")

# In training loop:
for epoch in range(num_epochs):
    # ... training code ...
    if epoch % 5 == 0:
        save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch_{epoch}.pth")
```


### Step 4.2: Resume Training

```python
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"✅ Loaded checkpoint from epoch {start_epoch-1}")
    return start_epoch

# Resume:
start_epoch = load_checkpoint(model, optimizer, "checkpoint_epoch_5.pth")
for epoch in range(start_epoch, num_epochs):
    # Continue training...
```


### Step 4.3: Inference Only (Model Weights)

```python
# Load for generation only
model = GPT(config)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Generate!
output = generate_with_topk(model, input_ids, max_new=50, temperature=0.8, top_k=25)
```


---

## 5. Complete Example: Save/Load Demo

```python
# 1. Setup (your trained model)
config = GPTConfig(vocab_size=50257, context_length=256, ...)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train a bit...
for epoch in range(3):
    # ... training loop ...
    pass

# 2. Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.__dict__,  # Save config too!
    'epoch': epoch,
}, "my_gpt_checkpoint.pth")

# 3. Later: Load & resume
new_model = GPT(config)
new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
checkpoint = torch.load("my_gpt_checkpoint.pth")

new_model.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"Resumed from epoch {checkpoint['epoch']}")

# 4. Continue training or generate
new_model.train()  # or .eval()
```


---

## 6. File Size Reality Check

```
Model params: 162M × 4 bytes/float32 = ~650MB
Optimizer state: ~2-3× model size     = ~1.5-2GB
──────────────────────────────────────
Total checkpoint: **~2.5GB**

GPT-2 small (124M):   ~1.8GB total
GPT-2 medium (350M):  ~5GB total
GPT-2 large (774M):  ~12GB total
```

**Pro tip**: Use `torch.save(..., pickle_protocol=4)` for large files.

---

## 7. Best Practices

### 7.1. Save Frequency

```
# Every epoch (small datasets)
if epoch % 1 == 0: save()

# Every 5-10 epochs (large datasets)
if epoch % 5 == 0: save()

# Every N steps (production)
if global_step % 10000 == 0: save()
```


### 7.2. Multiple Checkpoints

```
# Keep top-3 best validation loss
checkpoint_dir = "checkpoints/"
save_checkpoint(model, optimizer, epoch, f"{checkpoint_dir}epoch_{epoch}.pth")
```


### 7.3. Config Preservation

```python
torch.save({
    'config': config.__dict__,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, "checkpoint.pth")
```


---

## 8. Why This Matters for Next Lecture

**Next**: Load **OpenAI GPT-2 weights** into **your custom GPT**:

```
1. Download OpenAI GPT-2 weights (~500MB)
2. Map layer names: 'h.0.attn.c_attn.weight' → your naming
3. torch.load("gpt2-small.pth") → your_model.load_state_dict()
4. Generate Shakespeare/Harry Potter instantly!

No retraining needed → **millions saved**!
```

**Today's toolbox** → **Tomorrow's pretrained magic** 🎉

```
Commands learned:
✅ torch.save(model.state_dict(), "model.pth")
✅ model.load_state_dict(torch.load("model.pth"))
✅ Save/Load optimizer state_dict()
✅ Full checkpoint workflow
```

```
```

