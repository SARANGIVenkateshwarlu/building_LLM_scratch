<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# generate .ipynb file

I can generate the `.ipynb` file, but I can’t attach/create it in this turn because file-generation access is unavailable right now.

Use this exact notebook content locally to create `llm_production_optuna.ipynb`:

1. Open Jupyter Lab or VS Code.
2. Create a new notebook named `llm_production_optuna.ipynb`.
3. Add the cells below in order.
4. Copy your existing data-preparation/tokenizer/dataloader cells from your original notebook before the “Run Optuna” section.
5. Run top to bottom on your Lightning.IAI GPU.

## Notebook cells

### Cell 1 — Markdown

```markdown
# Production LLM Fine-Tuning with Optuna

This notebook is a production-oriented training pipeline for a GPT-style model with:
- Consistent config
- Selective unfreezing of last transformer blocks
- Final norm tuning
- Dropout / LR / weight decay / epoch optimization
- Optuna tuning
- Save only improved validation checkpoints
- Final best model export with torch.save
```


### Cell 2 — Code

```python
import os
import json
import math
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    !pip -q install optuna
    import optuna

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("DEVICE:", DEVICE)
```


### Cell 3 — Code

```python
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TUNING_CONFIG = {
    "n_trials": 20,
    "study_name": "gpt_production_optuna",
    "direction": "maximize",
    "max_grad_norm": 1.0,
    "checkpoint_metric": "val_accuracy"
}
```


### Cell 4 — Code

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```


### Cell 5 — Code

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def set_trainable_layers(model, train_last_n_blocks=1, train_final_norm=True, train_embeddings=False, train_out_head=True):
    freeze_all(model)

    if train_embeddings:
        for p in model.tok_emb.parameters():
            p.requires_grad = True
        for p in model.pos_emb.parameters():
            p.requires_grad = True

    if train_last_n_blocks == -1 or train_last_n_blocks >= len(model.trf_blocks):
        blocks_to_train = model.trf_blocks
    else:
        blocks_to_train = model.trf_blocks[-train_last_n_blocks:]

    for block in blocks_to_train:
        for p in block.parameters():
            p.requires_grad = True

    if train_final_norm:
        for p in model.final_norm.parameters():
            p.requires_grad = True

    if train_out_head:
        for p in model.out_head.parameters():
            p.requires_grad = True
```


### Cell 6 — Code

```python
def compute_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_train_batches=20, eval_val_batches=None):
    model.eval()

    def _loss_on_loader(loader, max_batches=None):
        losses = []
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            loss = compute_loss_batch(x, y, model, device)
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    def _accuracy_on_loader(loader, max_batches=None):
        correct = 0
        total = 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / total if total > 0 else 0.0

    train_loss = _loss_on_loader(train_loader, eval_train_batches)
    val_loss = _loss_on_loader(val_loader, eval_val_batches)
    train_acc = _accuracy_on_loader(train_loader, eval_train_batches)
    val_acc = _accuracy_on_loader(val_loader, eval_val_batches)

    model.train()
    return train_loss, val_loss, train_acc, val_acc
```


### Cell 7 — Code

```python
def save_if_best(model, optimizer, epoch, metrics, best_state, save_prefix):
    improved = False

    if metrics["val_accuracy"] > best_state["best_val_accuracy"]:
        improved = True
    elif metrics["val_accuracy"] == best_state["best_val_accuracy"] and metrics["val_loss"] < best_state["best_val_loss"]:
        improved = True

    if improved:
        best_state["best_val_accuracy"] = metrics["val_accuracy"]
        best_state["best_val_loss"] = metrics["val_loss"]
        best_state["best_epoch"] = epoch
        best_state["best_metrics"] = metrics.copy()

        torch.save(model.state_dict(), OUTPUT_DIR / f"{save_prefix}_best_state_dict.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }, OUTPUT_DIR / f"{save_prefix}_best_checkpoint.pt")

    return best_state, improved
```


### Cell 8 — Code

```python
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    save_prefix="run",
    eval_every_epoch=True
):
    model.to(device)
    best_state = {
        "best_val_accuracy": -1.0,
        "best_val_loss": float("inf"),
        "best_epoch": -1,
        "best_metrics": {}
    }

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TUNING_CONFIG["max_grad_norm"])
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_step_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")

        if eval_every_epoch:
            train_loss, val_loss, train_acc, val_acc = evaluate_model(
                model, train_loader, val_loader, device
            )

            metrics = {
                "epoch": epoch,
                "avg_train_step_loss": avg_train_step_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            }

            best_state, improved = save_if_best(
                model, optimizer, epoch, metrics, best_state, save_prefix
            )

            metrics["improved"] = improved
            history.append(metrics)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"improved={improved}"
            )

    return history, best_state
```


### Cell 9 — Markdown

```markdown
## Important

Before running the Optuna section below, make sure these are already defined from your original notebook:
- `train_loader`
- `val_loader`

If your original notebook uses a custom tokenizer/dataset pipeline, keep those cells above this point.
```


### Cell 10 — Code

```python
# Sanity check
assert "train_loader" in globals(), "train_loader not found. Run your data loader cells first."
assert "val_loader" in globals(), "val_loader not found. Run your validation loader cells first."

sample_cfg = copy.deepcopy(BASE_CONFIG)
model = GPTModel(sample_cfg)
total_params, trainable_params = count_parameters(model)
print("Total params:", total_params)
print("Trainable params before freezing:", trainable_params)
```


### Cell 11 — Code

```python
def objective(trial):
    cfg = copy.deepcopy(BASE_CONFIG)

    cfg["drop_rate"] = trial.suggest_float("drop_rate", 0.0, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int("epochs", 2, 8)

    train_last_n_blocks = trial.suggest_categorical(
        "train_last_n_blocks", [1, 2, 3, 4, 6, -1]
    )
    train_final_norm = trial.suggest_categorical("train_final_norm", [True, False])
    train_embeddings = trial.suggest_categorical("train_embeddings", [False, True])

    model = GPTModel(cfg).to(DEVICE)
    set_trainable_layers(
        model,
        train_last_n_blocks=train_last_n_blocks,
        train_final_norm=train_final_norm,
        train_embeddings=train_embeddings,
        train_out_head=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss_batch(input_batch, target_batch, model, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TUNING_CONFIG["max_grad_norm"])
            optimizer.step()

        train_loss, val_loss, train_acc, val_acc = evaluate_model(
            model, train_loader, val_loader, DEVICE
        )

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "optuna_best_trial_state_dict.pt"
            )

            with open(OUTPUT_DIR / "optuna_best_trial_metrics.json", "w") as f:
                json.dump({
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "train_loss": train_loss,
                    "params": trial.params
                }, f, indent=2)

        trial.report(best_val_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc
```


### Cell 12 — Code

```python
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)

study = optuna.create_study(
    study_name=TUNING_CONFIG["study_name"],
    direction=TUNING_CONFIG["direction"],
    sampler=sampler,
    pruner=pruner
)

study.optimize(objective, n_trials=TUNING_CONFIG["n_trials"])

print("Best value:", study.best_value)
print("Best params:", study.best_params)

with open(OUTPUT_DIR / "best_optuna_summary.json", "w") as f:
    json.dump({
        "best_value": study.best_value,
        "best_params": study.best_params
    }, f, indent=2)
```


### Cell 13 — Code

```python
best_params = study.best_params
final_cfg = copy.deepcopy(BASE_CONFIG)
final_cfg["drop_rate"] = best_params["drop_rate"]

final_model = GPTModel(final_cfg).to(DEVICE)

set_trainable_layers(
    final_model,
    train_last_n_blocks=best_params["train_last_n_blocks"],
    train_final_norm=best_params["train_final_norm"],
    train_embeddings=best_params["train_embeddings"],
    train_out_head=True
)

final_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, final_model.parameters()),
    lr=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"]
)

total_params, trainable_params = count_parameters(final_model)
print("Final model total params:", total_params)
print("Final model trainable params:", trainable_params)
```


### Cell 14 — Code

```python
final_history, final_best_state = train_model(
    model=final_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=final_optimizer,
    device=DEVICE,
    num_epochs=best_params["epochs"],
    save_prefix="final_model"
)

with open(OUTPUT_DIR / "final_training_history.json", "w") as f:
    json.dump(final_history, f, indent=2)
```


### Cell 15 — Code

```python
torch.save(final_model.state_dict(), OUTPUT_DIR / "final_best_model_state_dict.pt")

torch.save({
    "model_state_dict": final_model.state_dict(),
    "config": final_cfg,
    "best_params": best_params,
    "best_validation": final_best_state
}, OUTPUT_DIR / "final_model_bundle.pt")

print("Saved files:")
for p in OUTPUT_DIR.iterdir():
    print("-", p.name)
```


### Cell 16 — Markdown

```markdown
## Notes

- Use your original dataloader/tokenizer cells above this pipeline.
- For 40 GB VRAM, increase:
  - `BASE_CONFIG["context_length"]`
  - batch size in your dataloaders
  - `TUNING_CONFIG["n_trials"]`
- Recommended next tuning range:
  - epochs: 3 to 12
  - learning rate: 1e-5 to 3e-4
  - weight decay: 1e-6 to 1e-2
  - dropout: 0.0 to 0.35
  - train_last_n_blocks: [1, 2, 4, 6, -1]
```


## Quick fix

If you want, in the next message I can give you the same notebook as:

- raw `.ipynb` JSON format, or
- a Python script that auto-generates the `.ipynb` file from this content.

