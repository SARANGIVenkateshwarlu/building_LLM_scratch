1. Big Picture: Data Pipeline for LLMspaste.txt​
Before a large language model (LLM) can learn from text, the raw data (books, web pages, code, etc.) must be transformed into a numerical format the model can process. This lecture focuses on that data preprocessing pipeline.paste.txt​
The full pipeline has four main steps:paste.txt​
Tokenization → break text into discrete units called tokens and map them to token IDs.paste.txt​
Token embeddings → convert token IDs into dense vectors that capture meaning.paste.txt​
Positional embeddings → encode where each token appears in the sequence.paste.txt​
Input embeddings → add token and positional embeddings to get the final input to the model.paste.txt​
Example:
Text: “The cat sat on the mat.”
Tokenization might give tokens: ["The", "cat", "sat", "on", "the", "mat", "."] and IDs [10, 523, 87, 34, 15, 620, 4].paste.txt​
Token embeddings map each ID to a 256‑ or 768‑dimensional vector.paste.txt​
Positional embeddings add information like “this is position 0, this is position 1, …”.paste.txt​
Input embedding at each position = token vector + position vector.paste.txt​
2. Tokenization: From Text to Token IDspaste.txt​
2.1 Why tokenization matters
LLMs cannot take a whole PDF or raw text directly; they need a standardized sequence of small units. Tokenization defines those units and heavily influences vocabulary size, memory usage, and how well the model generalizes to new words.paste.txt​
The lecture discusses three main types:paste.txt​
Word-based tokenization
Character-based tokenization
Subword-based tokenization (e.g., Byte Pair Encoding, BPE)
2.2 Word-based tokenizationpaste.txt​
Goal: split text into words and punctuation tokens.paste.txt​
Basic approach using regex:paste.txt​
Start with a sentence: "Hello, world. This is a test".
First split only on whitespace → gives tokens like "Hello,", "world.", "This", … (punctuation still attached).paste.txt​
Improve split: use a regex that splits on spaces and punctuation characters such as , . : ; ? ! " ( ) / -.paste.txt​
Filter out pure whitespace tokens.paste.txt​
After improvement, tokens become: ["Hello", ",", "world", ".", "This", "is", "a", "test"].paste.txt​
Key issues with naive word tokenization:paste.txt​
Punctuation stuck to words (e.g., "Hello,") loses structure between word and punctuation.paste.txt​
Whitespace as separate tokens is usually not helpful for plain text examples.paste.txt​
The lecture shows how to fix both: split punctuation into their own tokens and drop whitespace tokens.paste.txt​
Vocabulary and token IDs
Once you have a token list for the whole dataset:paste.txt​
Remove duplicates (use set).
Sort tokens alphabetically.paste.txt​
Assign integer IDs consecutively: first token → 0, second → 1, etc.paste.txt​
This mapping is called the vocabulary: a dictionary from token → token ID.paste.txt​
Example (toy):
Tokens: ["brown", "dog", "fox", "jumps", "lazy", "over", "quick", "the"]
Sorted: ["brown", "dog", "fox", "jumps", "lazy", "over", "quick", "the"]
IDs: {"brown": 0, "dog": 1, "fox": 2, "jumps": 3, "lazy": 4, "over": 5, "quick": 6, "the": 7}.paste.txt​
The lecture then builds a Python tokenizer class with:paste.txt​
encode(text) → text → list of token IDs
decode(ids) → list of token IDs → reconstructed text (with a small fix to avoid extra spaces before punctuation).paste.txt​
Out-of-vocabulary (OOV) problem
Word-based tokenizers have a big issue: if the input contains a word not seen in the training corpus (e.g., "hello" never appeared in the book “The Verdict”), the tokenizer cannot map it to an ID and fails.paste.txt​
To handle OOV in this simple setup, the lecture adds special tokens:paste.txt​
<UNK> (unknown) → used when a word is not in the vocabulary.paste.txt​
<EOT> (end-of-text) → used to mark boundaries between documents.paste.txt​
These are appended to the vocabulary and given IDs (e.g., last two IDs). The tokenizer v2 then replaces any unknown word with <UNK> and can also insert <EOT> between unrelated text segments.paste.txt​
Example:
Input: "Hello do you like tea"
"Hello" not in vocabulary → tokenized as <UNK> do you like tea.paste.txt​
IDs: [UNK_ID, id("do"), id("you"), id("like"), id("tea")].paste.txt​
2.3 Character-based tokenizationpaste.txt​
Character-based tokenization treats each character as a token.paste.txt​
Example:
Text: "my hobby"
Tokens: ["m", "y", " ", "h", "o", "b", "b", "y"].paste.txt​
Advantages:paste.txt​
Very small vocabulary (roughly number of characters, e.g., 256 ASCII chars).paste.txt​
No OOV problem: any word can be represented as character sequence.paste.txt​
Disadvantages:paste.txt​
Sequences become much longer (e.g., "hobby" becomes 5 tokens instead of 1).paste.txt​
Completely destroys word-level structure—semantic meaning of words is lost at the tokenization level.paste.txt​
Less efficient for long texts and harder to learn long-range semantics.paste.txt​
2.4 Subword-based tokenization & Byte Pair Encoding (BPE)paste.txt​
Subword tokenization tries to get the best of both worlds:paste.txt​
Do not split very frequent words; keep them as single tokens.paste.txt​
Split rare words into meaningful pieces (subwords) such as roots and suffixes.paste.txt​
Example idea:paste.txt​
"boy" appears frequently → keep "boy" as a single token.
"boys" appears less frequently → split into "boy" + "s".
Thus both boy and boys share the root "boy", and vocabulary size remains smaller.paste.txt​
This helps encode that "tokens" and "tokenizing" share the root "token" and are semantically related.paste.txt​
BPE algorithm (compression origin)paste.txt​
Original BPE (1994) was a data compression algorithm:paste.txt​
Start with a sequence of symbols.
Find the most frequent pair of adjacent symbols (a “byte pair”).
Merge that pair into a new symbol.
Repeat until some stopping criterion (max merges or target vocab size).paste.txt​
Toy compression example:paste.txt​
Data: a a b d a a
Most common pair: a a → replace with Z: Z b d Z a a
Next common pair: a b or b d etc; continue merging pairs.paste.txt​
Using BPE for subword vocabulariespaste.txt​
To create a subword vocabulary for LLMs:paste.txt​
Start from a list of words and their frequencies.paste.txt​
Example words: old, older, finest, lowest with counts.paste.txt​
Add a special end-of-word marker (e.g., </w>) to each word: old</w>, older</w>, etc.paste.txt​
Initially, split each word into characters:paste.txt​
old</w> → o l d </w>
finest</w> → f i n e s t </w> etc.
Count frequencies of all adjacent symbol pairs, find most frequent pair, merge it into a new symbol.paste.txt​
Example: e s appears most (in finest, lowest) → merge to es.paste.txt​
Repeat the merge steps: maybe e s t → est, o l → ol, o l d → old, etc.paste.txt​
After a number of merges, you get a vocabulary that contains:paste.txt​
Some full words: old, est
Some subwords: er, low
Some characters still present.paste.txt​
In the toy example, this process ends with a vocabulary of about 11 tokens that can build all original words using combinations.paste.txt​
Important properties:paste.txt​
Frequent patterns (roots/suffixes) become tokens (e.g., est, old).paste.txt​
Rare words are decomposed into known subwords and characters.paste.txt​
Vocabulary size stays moderate; GPT‑2 uses about 50,000 BPE tokens.paste.txt​
BPE in practice (GPT / tiktoken)paste.txt​
OpenAI’s GPT models use a BPE tokenizer implemented in the tiktoken library. The lecture demonstrates:paste.txt​
python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(text, allowed_special={"<|endoftext|>"})
decoded = enc.decode(ids)

Key behaviors:paste.txt​
No explicit <UNK> token is needed: BPE breaks any unknown word into subword/character pieces.paste.txt​
Handles weird or random strings like "someunknownplace" by splitting into known subpieces.paste.txt​
GPT‑2 BPE vocabulary size ≈ 50,000 tokens (later models slightly larger).paste.txt​
This gives three big advantages:paste.txt​
Vocabulary reasonably small.
Encodes root/shared subwords, helping semantics.
Robust to unseen words or misspellings.
3. Building Inputs and Targets: Context, Stride, and DataLoaderpaste.txt​
After tokenization, we have a long sequence of token IDs for the whole dataset. For training, we need to turn this into input–target pairs that reflect “predict next token” behavior.paste.txt​
3.1 Context length and next-token predictionpaste.txt​
LLMs are trained to predict the next token given a context of previous tokens. We choose:paste.txt​
Context size (a.k.a. max sequence length) = how many tokens the model sees at a time.paste.txt​
For the lecture’s toy example, context size = 4.paste.txt​
Text: "one word at a time"
Tokens: ["one", "word", "at", "a", "time"].paste.txt​
We construct input–target pairs:paste.txt​
Input: ["one"] → Target: "word"
Input: ["one", "word"] → Target: "at"
Input: ["one", "word", "at"] → Target: "a"
Input: ["one", "word", "at", "a"] → Target: "time".paste.txt​
When we represent contexts of fixed length 4, an input row like ["one", "word", "at", "a"] has four prediction tasks internally:paste.txt​
At position 0 (seeing "one") → model predicts "word".
At position 1 (seeing "one word") → model predicts "at".
At position 2 (seeing "one word at") → predict "a".
At position 3 (seeing "one word at a") → predict "time".paste.txt​
In ID form, the target row is just the input row shifted by one token.paste.txt​
3.2 Sliding window and stridepaste.txt​
To cover the whole dataset, we slide a window of size context_size over the token ID sequence.paste.txt​
Two important parameters:paste.txt​
Context size = window length (e.g., 4).
Stride = how many positions we move the window each time.paste.txt​
Example token IDs: [t0, t1, t2, t3, t4, t5, ...] and context size = 4.paste.txt​
Stride = 1 (overlapping windows):
Window 1: [t0, t1, t2, t3]
Window 2: [t1, t2, t3, t4]
Window 3: [t2, t3, t4, t5] etc.paste.txt​
Stride = 4 (non-overlapping windows):
Window 1: [t0, t1, t2, t3]
Window 2: [t4, t5, t6, t7] etc.paste.txt​
The lecture emphasizes stride as a “sliding window” concept: stride=1 gives maximal overlap (more training examples but more redundancy); larger stride gives fewer, more spaced batches.paste.txt​
3.3 Dataset and DataLoader (PyTorch)paste.txt​
Implementation idea:paste.txt​
Encode the full text with BPE into encoded_text (list of token IDs).paste.txt​
Loop over encoded_text with context size and stride, build two tensors:
input_ids (X) → each row length = context_size
target_ids (y) → each row is input_ids shifted by one.paste.txt​
Wrap (input_ids, target_ids) in a custom Dataset and then a PyTorch DataLoader.paste.txt​
DataLoader parameters:paste.txt​
batch_size → how many input–target pairs to process together before updating model parameters.
shuffle → whether to randomize order (during training).
num_workers → number of CPU workers for parallel data loading.paste.txt​
Example:
Context size = 4, batch size = 8 → each batch from DataLoader has shape:
inputs: [8, 4] token IDs
targets: [8, 4] token IDs.paste.txt​
Each row is one sequence of 4 tokens; each row’s targets are the next tokens for each position in that sequence.paste.txt​
4. Token Embeddings: From IDs to Semantic Vectorspaste.txt​
4.1 Why we need embeddings
Token IDs (e.g., 34 for "cat", 91 for "kitten") are arbitrary integers; they do not encode semantic similarity.paste.txt​
Problems:paste.txt​
Random IDs do not tell the model that "cat" and "kitten" are related.
One‑hot vectors (huge sparse vectors with a single 1) also treat all words as equally distant.paste.txt​
We want dense vectors where semantically similar words have similar representations.paste.txt​
Toy example with 5‑dimensional vectors:paste.txt​
Features: [has_tail, is_eatable, has_four_legs, makes_sound, is_pet].
dog → [0.9, 0.1, 0.9, 0.9, 0.9]
cat → [0.9, 0.1, 0.9, 0.9, 0.9] (close to dog)
apple → [0.0, 0.9, 0.0, 0.0, 0.0]
banana → [0.0, 0.9, 0.0, 0.0, 0.0].paste.txt​
In this space, dog and cat cluster; apple and banana cluster; dog and apple are far apart. This is the intuition behind embeddings.paste.txt​
4.2 Embedding matrix (lookup table)paste.txt​
In practice, embeddings are learned, not hand-crafted features.paste.txt​
We define an embedding matrix (token embedding layer):paste.txt​
Rows = vocabulary size (e.g., 50,257 rows for GPT‑2).paste.txt​
Columns = embedding dimension (e.g., 256 or 768).paste.txt​
Each row is the vector for one token ID.paste.txt​
If vocab size is 50,257 and embedding dim is 256, the matrix has shape [50257, 256].paste.txt​
In PyTorch, this is nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim), which initializes the matrix with random values.paste.txt​
To get embeddings for a batch of token IDs:paste.txt​
Input: input_ids of shape [batch_size, context_size] (e.g., [8, 4]).
Output: token_embeddings = embedding_layer(input_ids) → shape [8, 4, 256].paste.txt​
Interpretation:paste.txt​
For each of the 8 × 4 positions, we now have a 256‑dimensional embedding vector.
These vectors will be learned/optimized during training together with the rest of the LLM parameters.paste.txt​
5. Positional Embeddings and Final Input Embeddingspaste.txt​
5.1 Why positional information is needed
Token embeddings capture “what” the token is, but not “where” it is in the sequence.paste.txt​
Example:paste.txt​
Sentence A: "the cat sat on the mat"
Sentence B: "on the mat the cat sat"
The word "cat" appears in both but at different positions. Its token embedding is identical in both cases if we only use token embeddings. Without position, the model cannot distinguish these different structures.paste.txt​
Transformers (used by GPT) are permutation-invariant by nature, so positional information must be injected explicitly.paste.txt​
5.2 Absolute vs relative positional encodingpaste.txt​
Two main styles:paste.txt​
Absolute positional encoding
Each position (0, 1, 2, …, context_size−1) has its own position embedding vector.paste.txt​
Final input embedding at each position = token embedding + its position embedding.paste.txt​
GPT models use absolute positional embeddings.paste.txt​
Relative positional encoding
Focuses on distances between tokens (e.g., “token A is 2 positions before token B”).paste.txt​
Useful for very long sequences and patterns that can appear in many places.paste.txt​
The lecture concentrates on absolute positional embeddings because that is what GPT‑style models use.paste.txt​
5.3 Building positional embeddingspaste.txt​
Given:paste.txt​
Context size = 4.
Embedding dimension = 256.
We create a positional embedding layer:
Shape: [context_size, embed_dim] = [4, 256].paste.txt​
Row 0: position 0 embedding (for first token in sequence).
Row 1: position 1 embedding, etc.paste.txt​
Again, this is an nn.Embedding layer but indexed by position rather than token ID. Values are randomly initialized and learned during training.paste.txt​
5.4 Adding token and positional embeddings (broadcasting)paste.txt​
We have:paste.txt​
Token embeddings: shape [batch_size, context_size, embed_dim] → e.g., [8, 4, 256].
Positional embeddings: shape [context_size, embed_dim] → [4, 256].paste.txt​
To add them:paste.txt​
Use broadcasting: treat positional embeddings as [1, 4, 256] and broadcast over batch dimension to match [8, 4, 256].paste.txt​
Then: input_embeddings = token_embeddings + positional_embeddings.paste.txt​
Result: input_embeddings of shape [8, 4, 256].paste.txt​
Interpretation:paste.txt​
Each token’s final representation encodes both what the token is (semantics) and where it is (position).
These input embeddings are the actual inputs fed into the transformer blocks during pretraining.paste.txt​
Both token and positional embedding matrices are optimized end‑to‑end during training, along with attention and feedforward layers.paste.txt​
6. Putting It All Together: End-to-End Examplepaste.txt​
Here is a compact walk‑through from raw text to LLM input, using the concepts from the lecture.paste.txt​
Raw dataset
Example: public domain book “The Verdict” (1908).paste.txt​
Loaded from disk via Python, read into raw_text.paste.txt​
Tokenization with BPE (tiktoken)
Use GPT‑style BPE encoder: enc = tiktoken.get_encoding("gpt2").paste.txt​
encoded_text = enc.encode(raw_text) → list of token IDs.paste.txt​
Vocabulary size ≈ 50,000 tokens.paste.txt​
Build input/target tensors
Choose context_size = 4 and stride (e.g., 1 or 4).paste.txt​
Slide window over encoded_text to build sequences of length 4.paste.txt​
For each sequence, create a target sequence shifted by one ID.paste.txt​
Store all sequences in input_ids and target_ids tensors.paste.txt​
DataLoader
Wrap (input_ids, target_ids) in a Dataset and PyTorch DataLoader with batch_size = 8.paste.txt​
Each batch: inputs [8, 4], targets [8, 4].paste.txt​
Token embedding layer
Define token_embedding = nn.Embedding(vocab_size=50257, embed_dim=256).paste.txt​
Compute token_embeds = token_embedding(inputs) → [8, 4, 256].paste.txt​
Positional embedding layer
Define pos_embedding = nn.Embedding(num_positions=4, embed_dim=256).paste.txt​
Create a position index [0, 1, 2, 3], expand to batch, and call pos_embedding.paste.txt​
Get pos_embeds → [4, 256] broadcast to [8, 4, 256] when added.paste.txt​
Final input embeddings
input_embeddings = token_embeds + pos_embeds.paste.txt​
Shape [8, 4, 256] → fed into the transformer layers which will learn to predict targets.paste.txt​
With this pipeline, the model sees text that is numerically encoded, semantically rich, and position‑aware, which is essential for effective LLM training