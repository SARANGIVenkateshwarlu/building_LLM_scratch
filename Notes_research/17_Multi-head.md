# Multi‑Head Attention (Build LLMs From Scratch Series)

This module implements **multi‑head causal self‑attention** on top of a previously built `CausalAttention` block, as part of a series on building large language models from first principles.

## Background

Before this component, the series covers:

- Simplified self‑attention
- Full self‑attention with trainable weights
- Causal self‑attention (masked, autoregressive)

These form the foundation for understanding **multi‑head attention**, which is the standard attention mechanism in modern LLMs (e.g., GPT‑style models).

## Recap: Causal Self‑Attention

Given token embeddings \(X \in \mathbb{R}^{B \times T \times d_{in}}\):

1. Project to **queries**, **keys**, and **values**:
   - \(Q = X W_Q\)
   - \(K = X W_K\)
   - \(V = X W_V\)

2. Compute **attention scores**:
   - \(S = Q K^\top\) per sequence, shape \((T, T)\).

3. Apply **causal mask**:
   - All positions **above the main diagonal** are masked so each token only attends to **itself and previous tokens**.

4. Softmax over rows → **attention weights**.

5. Multiply by values:
   - \(C = \text{softmax}(S_{\text{masked}})\, V\)
   - Context \(C\) has shape \((B, T, d_{out})\).

The context vectors are enriched representations that encode how each token relates to others in the sequence.

## What Is Multi‑Head Attention?

A **single head** uses one set of \((W_Q, W_K, W_V)\). Multi‑head attention:

- Creates multiple independent heads, each with its **own** \(W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\).
- Each head performs causal self‑attention separately.
- Their outputs are **concatenated along the feature dimension**.

If:

- Input embeddings have dimension `d_in`
- Each head outputs `d_out`
- There are `H` heads

then the final context has dimension:

- `d_out_total = d_out * H`

and the output tensor shape is:

- `(batch_size, num_tokens, d_out * H)`

### Example Shapes

- Input batch: 2 sequences, 6 tokens each, embedding size 3  
  - Shape: `(2, 6, 3)`
- Single causal head with `d_out = 2`  
  - Output: `(2, 6, 2)`
- Multi‑head wrapper with `num_heads = 2`, `d_out = 2`  
  - Each head: `(2, 6, 2)`  
  - Concatenated: `(2, 6, 4)` (because `2 heads * 2 dims = 4`)

## Implementation Overview

We implement `MultiHeadAttention` as a **wrapper** around the existing `CausalAttention` module.

### CausalAttention (Prerequisite)

`CausalAttention`:

- Input: `(batch_size, num_tokens, d_in)`
- Parameters:
  - `d_in`: input embedding dimension
  - `d_out`: output (per‑head) dimension
  - `context_length`: maximum number of tokens
  - `dropout`: attention dropout rate
- Output: `(batch_size, num_tokens, d_out)`

### MultiHeadAttention Wrapper

Key ideas:

- Instantiate `num_heads` copies of `CausalAttention`, each with its own weights.
- For each forward pass, run all heads on the same input and **concatenate** the outputs.

Efficiency Considerations
The wrapper above is conceptually simple but inefficient:

Each head runs sequentially in Python.

Modern LLMs compute all heads in parallel using a weight‑split formulation:

Single large projection for Q/K/V

Reshape/split across heads

Parallel batched matmuls

The next step (next lecture) is to implement multi‑head attention with weight splits, which matches how large models like GPT‑3/4 are actually implemented.

Real‑World Scale
Large models use dozens or hundreds of attention heads.

For example, GPT‑3’s largest variant uses 96 heads; GPT‑4 likely uses even more.

Concatenating so many heads greatly increases representational power but also requires significant compute and memory.

Files / Components (Suggested)
causal_attention.py – single causal self‑attention head.

multihead_attention.py – wrapper implementing multi‑head attention via multiple CausalAttention instances.

notebooks/ – demos showing:

Shape checks

Visualizations of attention matrices

Comparison of single vs multi‑head outputs