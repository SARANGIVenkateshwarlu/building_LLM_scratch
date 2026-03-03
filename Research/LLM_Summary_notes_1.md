# Building a Data Pipeline for Large Language Models (LLMs)

Evry one can ‑friendly way, how text is turned into numbers that a Large Language Model can understand and learn from.

---

## 1. Big Picture: From Storybook to Numbers

Before an LLM can “read” a book, the book must be converted into numbers.[file:1] The whole process is called **data preprocessing** or **data processing pipeline**.[file:1]

### 1.1 Four main steps

1. **Tokenization** – split text into small pieces called *tokens* and give each token an ID (a number).[file:1]  
2. **Token embeddings** – turn each token ID into a vector (a list of numbers) that tries to capture meaning.[file:1]  
3. **Positional embeddings** – add information about where each token is in the sentence.[file:1]  
4. **Input embeddings** – add token and positional vectors together; this final vector is sent into the LLM.[file:1]

### 1.2 Simple example

Text:  
> “The cat sat on the mat.”[file:1]

- Tokenization → tokens: `["The", "cat", "sat", "on", "the", "mat", "."]`, IDs: `[10, 523, 87, 34, 15, 620, 4]` (just an example).[file:1]  
- Token embeddings → each ID becomes a vector, for example a 3‑D or 256‑D vector like \([1.2, -0.5, 0.3]\) for `"cat"`.[file:1]  
- Positional embeddings → position 0, 1, 2, … each has its own vector, like \([0.1, 0.0, 0.2]\) for position 0, \([0.2, 0.1, 0.0]\) for position 1, etc.[file:1]  
- Input embedding at each position = token vector + position vector.[file:1]

### 1.3 3‑D “meaning space” illustration

To make vectors easier for school students, imagine **3‑dimensional vectors** instead of 256‑dimensional ones.[file:1]

We choose three simple “features” (axes):  

1. **Is it an animal?** (0 to 1)  
2. **Can we eat it?** (0 to 1)  
3. **Is it usually a pet?** (0 to 1)[file:1]

Example 3‑D vectors:  

- `dog`  → \([1.0, 0.1, 0.9]\)  
- `cat`  → \([1.0, 0.1, 0.9]\)  
- `apple` → \([0.0, 1.0, 0.0]\)  
- `banana` → \([0.0, 1.0, 0.0]\)[file:1]

Here, dog and cat are close together in 3‑D space, and apple and banana are close together, which matches our intuition about meaning.[file:1]

---

## 2. Tokenization: From Text to Token IDs

LLMs cannot read raw paragraphs directly; they need text broken into small, consistent pieces.[file:1]

### 2.1 Why tokenization matters

Tokenization affects:

- How big the **vocabulary** is (how many unique tokens).[file:1]  
- How fast the model can train.[file:1]  
- How well the model handles new or rare words.[file:1]

The lecture explains three main types:[file:1]

- **Word-based** tokenization  
- **Character-based** tokenization  
- **Subword-based** tokenization (Byte Pair Encoding, BPE)[file:1]

---

### 2.2 Word-based tokenization

**Goal:** split text into words and punctuation tokens.[file:1]

Start with a sentence:  
> `"Hello, world. This is a test"`[file:1]

1. Split only on spaces → `"Hello,"`, `"world."`, `"This"`, `"is"`, `"a"`, `"test"` (punctuation stuck to words).[file:1]  
2. Improve using regular expressions: split on spaces **and** punctuation like `, . : ; ? ! " ( ) / -`.[file:1]  
3. Remove pure whitespace tokens.[file:1]

After improvement, tokens become:  
`["Hello", ",", "world", ".", "This", "is", "a", "test"]`.[file:1]

**Problems with simple word tokenization**:[file:1]

- Punctuation stuck to words hides useful structure.  
- Whitespace as tokens is usually not needed for normal text.[file:1]

So we split punctuation into separate tokens and drop whitespace tokens.[file:1]

#### Vocabulary and token IDs

Steps after tokenizing an entire dataset:[file:1]

1. Put all tokens from the dataset into a list.  
2. Remove duplicates using a `set`.[file:1]  
3. Sort tokens alphabetically.[file:1]  
4. Assign token IDs: first token → 0, second → 1, etc.[file:1]

This mapping is called the **vocabulary** (token → token ID).[file:1]

**Toy example**:

- Tokens: `["brown", "dog", "fox", "jumps", "lazy", "over", "quick", "the"]`.[file:1]  
- Sorted: `["brown", "dog", "fox", "jumps", "lazy", "over", "quick", "the"]`.[file:1]  
- IDs:  
  - `brown: 0`  
  - `dog: 1`  
  - `fox: 2`  
  - `jumps: 3`  
  - `lazy: 4`  
  - `over: 5`  
  - `quick: 6`  
  - `the: 7`.[file:1]

A simple **Tokenizer class** can then provide:

- `encode(text)` → text → list of IDs.[file:1]  
- `decode(ids)` → list of IDs → text (with a fix to avoid extra spaces before punctuation).[file:1]

#### Out-of-vocabulary (OOV) problem

If a word never appears in the training book, there is no ID for it.[file:1] For example, if `"hello"` does not appear in “The Verdict”, the tokenizer cannot encode it.[file:1]

To fix this, we add special tokens:[file:1]

- `<UNK>` – for unknown words.  
- `<EOT>` – end-of-text marker between documents.[file:1]

These are added to the vocabulary and get their own IDs.[file:1] The improved tokenizer (`v2`) replaces unknown words with `<UNK>` and can insert `<EOT>` between unrelated texts.[file:1]

**Example**:

Input: `"Hello do you like tea"`[file:1]

- `"Hello"` not in vocabulary → becomes `<UNK> do you like tea`.[file:1]  
- IDs: `[UNK_ID, id("do"), id("you"), id("like"), id("tea")]`.[file:1]

---

### 2.3 Character-based tokenization

Character-based tokenization treats **each character** as a token.[file:1]

Example:  

Text: `"my hobby"` → tokens: `["m", "y", " ", "h", "o", "b", "b", "y"]`.[file:1]

**Advantages**:[file:1]

- Very small vocabulary (around 256 characters in ASCII).  
- No OOV problem: any word is just a sequence of characters.[file:1]

**Disadvantages**:[file:1]

- Sequences get much longer: `"hobby"` is 5 tokens instead of 1.  
- Word structure and meaning are broken apart at the token level.  
- Harder for the model to learn long‑range meaning efficiently.[file:1]

---

### 2.4 Subword-based tokenization & Byte Pair Encoding (BPE)

Subword tokenization aims to combine the good parts of word‑ and character‑level tokenization.[file:1]

Rules:[file:1]

- Very frequent words → keep as whole tokens.  
- Rare words → split into smaller “meaningful pieces” (like roots and suffixes).[file:1]

**Idea example**:[file:1]

- `"boy"` appears often → keep `"boy"` as one token.  
- `"boys"` is less common → split into `"boy"` + `"s"`.[file:1]  
Now both words share the `"boy"` subword and keep the meaning connection.[file:1]

This helps the model realize that `"tokens"` and `"tokenizing"` share the root `"token"`.[file:1]

#### BPE as a compression algorithm

Original BPE (1994) was used for data compression:[file:1]

1. Start with a sequence of symbols.  
2. Find the most frequent adjacent pair (a “byte pair”).  
3. Merge that pair into a new symbol.  
4. Repeat until a stopping rule is met.[file:1]

Example:[file:1]

- Data: `a a b d a a`  
- Most common pair: `a a` → replace with `Z` → `Z b d Z a a`  
- Repeat with next most frequent pair.[file:1]

#### Using BPE to build a subword vocabulary

Steps for LLMs:[file:1]

1. Start with words and their frequencies, e.g., `old`, `older`, `finest`, `lowest` with counts.[file:1]  
2. Add an end-of-word marker: `old</w>`, `older</w>`, etc.[file:1]  
3. Split each word into characters:  
   - `old</w>` → `o l d </w>`  
   - `finest</w>` → `f i n e s t </w>`.[file:1]  
4. Count adjacent pairs and merge the most frequent pair into a new symbol (e.g., `e`+`s` → `es`).[file:1]  
5. Repeat merges: `e s t` → `est`, `o l` → `ol`, `o l d` → `old`, etc.[file:1]

After several merges, you end up with a vocabulary containing:[file:1]

- Whole words like `old`, `est`.  
- Subwords like `low`, `er`.  
- Some leftover characters.[file:1]

In the toy example, about 11 tokens are enough to build all original words.[file:1]

**Important properties**:[file:1]

- Frequent patterns (roots, suffixes) get their own tokens (e.g., `est`, `old`).  
- Rare words are split into known subwords and characters.  
- Vocabulary stays a reasonable size (GPT‑2 uses ~50k tokens).[file:1]

#### BPE in real GPT tokenizers

OpenAI’s GPT models use BPE tokenization via the `tiktoken` library.[file:1]

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(text, allowed_special={"<|endoftext|>"})
decoded = enc.decode(ids)
Behaviors:[file:1]

No <UNK> needed: BPE can always split a new word into existing pieces.

Can handle weird strings like "someunknownplace" by splitting into known parts.

GPT‑2’s BPE vocabulary has around 50,000 tokens.[file:1]

3. Building Inputs and Targets: Context, Stride, and DataLoader
After tokenization, the entire dataset is one long list of token IDs.[file:1] To train the LLM, we need to create input–target pairs so the model can learn to predict the next token.[file:1]

3.1 Context length and next-token prediction
LLMs are trained to predict the next token given a context of previous tokens.[file:1]

Context size (max sequence length) = how many tokens the model can see at one time.[file:1]

Example in the lecture: context size = 4.[file:1]

Text: "one word at a time"
Tokens: ["one", "word", "at", "a", "time"].[file:1]

Build input–target pairs:[file:1]

Input: ["one"] → Target: "word"

Input: ["one", "word"] → Target: "at"

Input: ["one", "word", "at"] → Target: "a"

Input: ["one", "word", "at", "a"] → Target: "time".[file:1]

If we fix context size = 4, then a single input row ["one", "word", "at", "a"] really contains 4 prediction tasks inside (predict each next token at each position).[file:1] In IDs, the target row is the input row shifted by one token.[file:1]

3.2 Sliding window and stride
To cover the whole text, we slide a window over the token ID sequence.[file:1]

Context size = window length (e.g., 4).

Stride = how many positions we move each time.[file:1]

Example: token IDs [t0, t1, t2, t3, t4, t5, …], context size = 4.[file:1]

Stride = 1 (overlapping):

[t0, t1, t2, t3]

[t1, t2, t3, t4]

[t2, t3, t4, t5].[file:1]

Stride = 4 (non‑overlapping):

[t0, t1, t2, t3]

[t4, t5, t6, t7].[file:1]

Stride = 1 → more training examples but more repetition; larger stride → fewer, more spaced examples.[file:1]

3.3 Dataset and DataLoader (PyTorch)
Implementation idea:[file:1]

Encode the full text with BPE into encoded_text.[file:1]

Loop with context size and stride to build:

input_ids (X) – each row length = context_size.

target_ids (y) – each row is input_ids shifted by one.[file:1]

Wrap (input_ids, target_ids) into a custom Dataset and then a DataLoader.[file:1]

DataLoader parameters:[file:1]

batch_size – how many sequences in one batch (e.g., 8).

shuffle – whether to randomize order during training.

num_workers – CPU workers for faster loading.[file:1]

Example shapes for context size = 4, batch size = 8:[file:1]

inputs: [8, 4] token IDs.

targets: [8, 4] token IDs.[file:1]

4. Token Embeddings: From IDs to Meaningful Vectors
4.1 Why we need embeddings
Token IDs (like 34 for "cat" and 91 for "kitten") are just labels; they do not say that cat and kitten are similar.[file:1] One‑hot vectors also treat every word as equally far apart.[file:1]

We want dense vectors (embeddings) where similar words end up close together.[file:1]

The lecture gives a toy 5‑D example with features like has_tail, is_eatable, has_four_legs, makes_sound, is_pet and shows that dog and cat get similar vectors while apple and banana get similar ones.[file:1]

4.2 Embedding matrix (lookup table)
In real LLMs, embeddings are learned, not hand‑designed.[file:1]

We create a big embedding matrix:[file:1]

Rows = vocabulary size (e.g., 50,257 for GPT‑2).

Columns = embedding dimension (e.g., 256 or 768).[file:1]

Shape example: [50257, 256].[file:1]

Each row is the embedding vector for one token ID.[file:1]

In PyTorch, this is:

python
token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
To get embeddings for one batch:[file:1]

Input: input_ids has shape [batch_size, context_size], e.g., [8, 4].

Output: token_embeddings = token_embedding(input_ids) has shape [8, 4, 256].[file:1]

Every number in this 3‑D tensor will be adjusted during training so that the model becomes good at predicting next tokens.[file:1]

4.3 3‑D geometry view for students
For school students, instead of thinking about 256‑D space, imagine a 3‑D cube:[file:1]

x‑axis = “animal‑ness”

y‑axis = “food‑ness”

z‑axis = “pet‑ness”[file:1]

Then words like dog, cat, hamster sit near one corner (high animal, maybe high pet), while apple, banana, bread sit near a different corner (high food, not animal).[file:1] Training an LLM is like moving all these points around in 3‑D until sentences can be predicted well.[file:1]

5. Positional Embeddings and Final Input Embeddings
5.1 Why positional information is needed
Token embeddings tell us what the word is, but not where it appears.[file:1]

Example sentences:[file:1]

"the cat sat on the mat"

"on the mat the cat sat"

The word "cat" appears at different positions, but its token embedding is the same.[file:1] Transformers are naturally order‑agnostic, so we must add position information.[file:1]

5.2 Absolute vs relative positional encoding
Two styles:[file:1]

Absolute positional encoding

Each position index (0, 1, 2, …, context_size−1) gets its own position embedding vector.[file:1]

Final input embedding = token embedding + that position’s embedding.[file:1]

GPT models use absolute positional embeddings.[file:1]

Relative positional encoding

Focus on distances between tokens like “2 tokens apart”.[file:1]

Helpful for very long sequences.[file:1]

The lecture focuses on absolute positional encoding.[file:1]

5.3 Building positional embeddings
Given:[file:1]

Context size = 4

Embedding dimension = 256

We create a positional embedding layer of shape [4, 256]:[file:1]

Row 0 → position 0

Row 1 → position 1

Row 2 → position 2

Row 3 → position 3.[file:1]

This is also an nn.Embedding layer, but indexed by position instead of token ID.[file:1]

5.4 Adding token and positional embeddings (broadcasting)
We have:[file:1]

token_embeddings: shape [batch_size, context_size, embed_dim] (e.g., [8, 4, 256]).

positional_embeddings: shape [context_size, embed_dim] (e.g., [4, 256]).[file:1]

We “stretch” the positional embeddings across the batch dimension (broadcasting) and add:[file:1]

text
input_embeddings = token_embeddings + positional_embeddings
Result: input_embeddings has shape [8, 4, 256].[file:1]

Now each token is represented by a vector that includes both its meaning and its position, which is exactly what the LLM needs to start its work.[file:1]

6. End-to-End Architecture Summary (For Students)
Putting all the pieces together:[file:1]

Read text

Load a book like “The Verdict” into raw_text.[file:1]

Tokenize

Use BPE (e.g., tiktoken with GPT‑2 encoding) to convert raw_text into encoded_text (a list of token IDs).[file:1]

Build training samples

Choose context size (e.g., 4) and stride (e.g., 1 or 4).[file:1]

Use a sliding window to produce many [context_size]‑length sequences and their shifted targets.[file:1]

Create DataLoader

Group sequences into batches (e.g., 8 sequences per batch) using PyTorch’s DataLoader.[file:1]

Token embedding layer

Convert token IDs [batch_size, context_size] into token embeddings [batch_size, context_size, embed_dim].[file:1]

Positional embedding layer

Create a [context_size, embed_dim] matrix for positions 0…context_size−1 and add it to the token embeddings.[file:1]

Input embeddings

The sum of token + positional embeddings is the final 3‑D tensor that enters the transformer blocks for training.[file:1]

For a student, you can imagine the whole architecture as:

Line of words → broken into tokens → turned into 3‑D or 256‑D points that also remember where they are in the sentence → these points are then processed by the “thinking machine” (the transformer) to predict the next word.[file:1]