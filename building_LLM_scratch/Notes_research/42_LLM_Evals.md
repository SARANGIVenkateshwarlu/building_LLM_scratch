<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Lecture Overview

This transcript covers the completion of an instruction fine-tuning project for a GPT-2 LLM using a dataset of 1,100 instruction-input-output pairs. The instructor reviews prior stages (data prep and fine-tuning), demonstrates results, and focuses on evaluation: extracting responses, qualitative analysis, and quantitative scoring via another LLM (Llama 3 via Ollama).[](file:1)

Key goals include assessing how well the fine-tuned model follows instructions compared to ground truth, addressing challenges like partial correctness (e.g., "prepared" vs. "cooked").

## Prior Stages Recap

- **Stage 1: Data Preparation** – Downloaded dataset, batched for equal token lengths, created train (85%), test (10%), validation (5%) loaders.[](file:1)
- **Stage 2: Fine-Tuning** – Loaded GPT-2 medium (355M params), trained 1 epoch on CPU (loss curves shown; recommends 2+ epochs on GPU/M3 Mac). Pre-fine-tune: random outputs; post: improved but imperfect (e.g., "The meal is prepared by the chef every day" vs. true "The meal is cooked by the chef every day").[](file:1)


## Stage 3: LLM Evaluation (New Focus)

Evaluation splits into three sub-stages, emphasizing metrics beyond binary accuracy due to generative nuance.

### 1. Extract Responses

- Loop over test set inputs, generate model outputs with fine-tuned LLM, strip prompts, save to `instruction_data_with_response.json` (adds "model_response" field alongside instruction/input/output).[](file:1)
- Example for first 3 test samples:


| Instruction | True Output | Model Response | Qualitative Note |
| :-- | :-- | :-- | :-- |
| Rewrite "The car is very fast" using simile | The car is as fast as lightning | The car is as fast as a bullet | Close/similar quality |
| Cloud type for thunderstorms? | Cumulonimbus | Thunderstorms form in high-pressure... (rambling, inaccurate) | Poor/directly off-topic |
| Author of Pride and Prejudice? | Jane Austen | George Bernard Shaw | Factually wrong |

- Save model: `torch.save(model.state_dict(), 'gpt2-medium-355m-sft.pth')`; reload: `model.load_state_dict(torch.load(...))`.


### 2. Qualitative Evaluation

- Human-like review: Spot patterns (e.g., good similes, factual errors from 1-epoch limit).
- Three common LLM eval methods discussed:

1. MMLU (57 tasks across domains like algebra, ethics; benchmark general knowledge).[](file:1)
2. Human preference (side-by-side ranking).
3. LLM-as-judge (used here: Llama 3-8B instruct via Ollama compares output vs. model_response).[](file:1)


### 3. Quantitative Scoring (Automated)

- **Setup Ollama**:

1. Download from ollama.com (Mac: direct; Win: `ollama serve` then `ollama run llama3`).
2. Terminal: `ollama run llama3` (downloads ~4.7GB; keep running).
3. Python API: Define `query_model(prompt, model='llama3')` for inference.[](file:1)
- **Prompt Llama3**: "Given [instruction+input], correct output: [output]. Score model response: [model_response] on 0-100 (100=perfect)." Refine to "Respond with integer only" for clean scores.[](file:1)
- Example scores (from Llama3):


| Sample | Score | Rationale Snippet |
| :-- | :-- | :-- |
| Simile (bullet) | 85 | Correct structure; vivid but less dramatic than lightning |
| Cloud | 20 | Off-topic, inaccurate (e.g., high-pressure claim wrong) |
| Author | 0 | Factually incorrect |

- Full test avg: ~50-55/100 (run `evaluate_all(test_data)` if GPU/M3 available; Ollama non-deterministic).[](file:1)


## Improvements Suggested

- Hyperparams: Learning rate, batch size, epochs (2+).
- Data: Scale to Alpaca (52k pairs) for better generalization.
- Model: GPT-2 large/XL (774M/1.5B params).
- Prompts: Experiment formats.
- PEFT: LoRA for efficiency.[](file:1)


## Full Pipeline Steps

1. Data prep/batching/loaders.
2. Load GPT-2, fine-tune.
3. Extract/save test responses.
4. Qualitative check.
5. Quantitative score via Ollama/LLM-judge.
6. Save model.
7. Iterate improvements.[](file:1)

This provides a complete, runnable framework for instruction-tuned LLM eval—ideal for your ML/quant work; try on larger datasets like Alpaca.[^1]

<div align="center">⁂</div>

[^1]: paste.txt

