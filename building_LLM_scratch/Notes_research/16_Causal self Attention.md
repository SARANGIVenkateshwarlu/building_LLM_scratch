# Causal Attention Lecture Summary

## Introduction and Recap
- Part of "Build LLMs from Scratch" series; deep dive into attention (prior: simplified self-attention, self-attention w/ trainable weights, QKV).
- Today: Causal (masked) attention; next: Multi-head attention (used in GPT).
- Sequential learning essential: Basics → Causal → Multi-head.

**Example**: Sentence "your journey starts with one step" → 6 tokens → 3D embeddings (for demo; GPT: 100k+ dims).

## Self-Attention Recap
- Input embeddings (semantic meaning, no context) → Context vectors (meaning + relations/attention to other tokens).
- Steps:
  - Embeddings × WQ/WK/WV → Queries (Q), Keys (K), Values (V).
  - Q × KT → Attention scores (row: query's attention to all keys).
  - Scores / √dk → Softmax → Attention weights (rows sum=1).
  - Weights × V → Context matrix.

**Example**: "journey" row shows attention weights to "your" (0.15), "starts" (0.22), etc.; richer than raw embedding.

## Causal Attention Motivation
- Self-attention: Full bidirectional access (fine-tune/research, not generation).
- Causal: Restrict to past/current tokens (autoregressive: predict next without future leak).
- GPT training/generation: Mask future tokens (upper triangle of attention matrix → 0).

**Example**: For "with": Attend to "your/journey/starts/with" only; mask "one/step".

## Implementing Causal Mask (Method 1)
- Compute attention weights as usual.
- Create lower triangular mask (torch.tril(ones)) → Upper triangle=0.
- Weights × mask → Zero upper triangle.
- Renormalize rows (divide by row sums) → Rows sum=1.

**Example**: Post-mask, "journey" row: Only first 2 non-zero; renormalize so they sum=1.

## Data Leakage Issue & Fix
- Problem: Softmax before mask → Future influences denominators (leakage).
- Solution: Mask attention *scores* first:
  - Scores → Upper triangular mask (torch.triu(ones)) → Fill upper w/ -∞ (mask_fill).
  - /√dk → Softmax → Weights (auto-zero upper, rows sum=1; no leakage).

**Example**: "journey" scores: Upper → -∞; softmax: exp(-∞)=0, priors normalize perfectly.

## Causal Attention Class (w/ Batches)
- Handles batches: Input [B, T, C] (batch, tokens, embed dim).
- Q/K/V: Linear projections.
- Scores: (Q @ K.transpose(-2,-1)) / √dk.
- Mask: triu(ones(T,T)) → scores.mask_fill(mask, -inf).
- Softmax(Dim=-1) → Dropout → @ V → Context [B, T, C].
- register_buffer for mask (auto GPU move).

**Example**: Batch of 2 sentences (6 tokens each): Output [2,6,C]; processes independently.

## Dropout in Causal Attention
- Prevents overfitting/lazy neurons: Randomly zero ~p fraction of attention weights (scale survivors by 1/(1-p)).
- Applied post-softmax (p=0.1-0.2 training; 0 inference).
- Forces all neurons to contribute.

**Example**: p=0.5 demo: Row [0.2,0.3,0.5] → Some zeroed (e.g., [0.4,0,1.0]); survivors ×2.

## Key Advantages & Edge Cases
- No future leakage; efficient autoregression.
- Batches: Flexible token lengths (mask :ntokens).
- PyTorch nn.Dropout/Linear; forward handles dims dynamically.

**Example**: Context=4096; short seq pads implicitly via mask.

## Next Steps
- Multi-head: Parallel causal heads → Concat → Linear (GPT-style).
- Theory + code mix; master dims/matrices for LLMs.

**Example**: Stack heads for parallel relations (syntax, semantics).


```
Text
 ↓
Tokenization
 ↓
Tokens
 ↓
Token IDs (integers)
 ↓
Embedding Layer
 ↓
Token Embeddings (vectors)(X)
        ↓
Linear Projections (Wq, Wk, Wv)
        ↓
Q = X · Wq
K = X · Wk
V = X · Wv
        ↓
Attention Scores = Q · Kᵀ
        ↓
Apply Mask (set future positions to -∞)
        ↓
Scale: Attention Scores / √d_k
       ↓
Dropout
        ↓
Softmax
        ↓
Attention Weights (rows sum to 1)   
        ↓
Output (Context vector matrix) = Attention Weights · V
```