<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# LLM from Scratch — 1-Page Cheat Sheet

```
BUILDING LLMs: NEXT TOKEN PREDICTION
=====================================

INPUT TEXT → TOKENS → EMBEDDINGS → TRANSFORMER → LOGITS → NEXT TOKEN

1. DATA PIPELINE
├── Text → Tokens → Token IDs
├── Token IDs → Token Embeddings (d=768 GPT-2)
└── + Positional Embeddings → Input Embeddings

2. ATTENTION MECHANISM (THE HEART)
   Q, K, V = Input × Wq, Wk, Wv
   Scores = QKᵀ/√dk → Mask → Softmax → Weights × V = CONTEXT
   
   MULTI-HEAD: Multiple perspectives on same data
   
3. TRANSFORMER BLOCK (x12 in GPT-2)
   LayerNorm → MultiHeadAttn → Residual
   ↓
   LayerNorm → FeedForward(MLP) → Residual
   
4. OUTPUT
   Transformer → Linear → Logits → Argmax = Next Token
   Loss = CrossEntropy(true_next, predicted_next)

5. TRAINING LOOP
   for batch in data:
     logits = model(batch)
     loss = cross_entropy(logits, targets)
     loss.backward()
     optimizer.step()

6. FINE-TUNING
   ├── Classification: Spam/Not Spam
   └── Instruction: "Convert to passive" → response

7. EVALUATION
   ├── MMLU (57 tasks)
   ├── Human judgment
   └── LLM-as-judge (LLaMA3 8B)

KEY INSIGHTS
• Attention captures context relationships
• Residuals + LayerNorm = stable deep training  
• Scale = Intelligence (data + params)
• Causal mask = autoregressive prediction

INTERVIEW QUICK HITS
"Why √dk?" → Prevents softmax saturation
"Why multi-head?" → Multiple relationship types
"Embedding vs Positional?" → What + Where
"Residuals?" → Gradient flow
"Pretrain vs Finetune?" → General vs Task-specific

MATH ESSENTIALS
Attention: softmax(QKᵀ/√dk)V
Loss: -∑y_true * log(y_pred)

STACK: Embed → [Transformer Blocks] → Linear
```

**Print this. Memorize this. Ace the interview.** 🚀

