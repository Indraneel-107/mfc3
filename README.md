
![amrita_logo1](https://github.com/user-attachments/assets/11ae4c69-8581-4129-8bad-df2e6068a812)

# MFC3: Enhancing Summarization Efficiency
## Text Summarization using T5

---

###  Group D16 — Computationally Efficient Text Summarization Tool

> *School of Engineering, Amrita Vishwa Vidyapeetham*

---

## Team Members

| Name | Roll Number |
|:---:|:---:|
| Indraneel R | CB.SC.U4AIE24323 |
| Rishi Dasari | CB.SC.U4AIE24365 |
| Abhishek Reddy | CB.SC.U4AIE24325 |
| M Aryan | CB.SC.U4AIE24341 |

---

</div>

## Table of Contents

- [Primary Objective](#-primary-objective)
- [Platform and Tech Stack](#-platform--tech-stack)
- [Implementation Overview](#-implementation-overview)
- [Stage 1 — Dataset Preparation](#stage-1--dataset-preparation)
- [Stage 2 — Dataset Analysis](#stage-2--dataset-analysis)
- [Stage 3 — Tokenization](#stage-3--tokenization)
- [Stage 4 — T5 Model Architecture and Mathematical Foundation](#stage-4--t5-model-architecture--mathematical-foundation)
- [Stage 5 — Training and Fine-Tuning](#stage-5--training--fine-tuning)
- [Stage 6 — ROUGE Evaluation Full Derivation](#stage-6--rouge-evaluation-full-derivation)
- [Stage 7 — Inference Pipeline](#stage-7--inference-pipeline)
- [Mathematical Foundation — Sparse Optimization from Base Paper](#mathematical-foundation--sparse-optimization-from-base-paper)
- [Results](#-results)
- [Citation](#-citation)

---

##  Primary Objective

This project builds a **computationally efficient, fine-tuned text summarization system** using the **T5 (Text-to-Text Transfer Transformer)** model on the BBC News Summary dataset. The system learns to map long news articles to concise, accurate summaries by fine-tuning T5-Base using HuggingFace Transformers, evaluated using ROUGE metrics. The mathematical foundations draw from the sparse optimization framework of Yao et al. (IJCAI 2015), connecting extractive sparse selection to abstractive T5 generation.

Specifically, the project aims to:

- Fine-tune T5-Base on BBC News article-summary pairs using the HuggingFace `Trainer` API
- Enforce length-aware preprocessing with `MAX_LENGTH = 512` token truncation and padding
- Evaluate summaries using ROUGE-1, ROUGE-2, and ROUGE-L with stemming
- Generate summaries at inference time using **beam search** (`num_beams=5`)
- Ground the architecture in the sparse optimization theory of document summarization

---

##  Platform & Tech Stack

| Component | Details |
|-----------|---------|
| **Language** | Python 3.9+ |
| **Core Model** | `t5-base` — 220M parameters, 12-layer encoder-decoder Transformer |
| **Framework** | HuggingFace Transformers, PyTorch |
| **Dataset** | `gopalkalpande/bbc-news-summary` (HuggingFace Hub) |
| **Evaluation** | ROUGE-1, ROUGE-2, ROUGE-L via `evaluate` library |
| **Training** | HuggingFace `Trainer` API + TensorBoard logging |
| **Tokenizer** | `T5Tokenizer` — SentencePiece unigram, vocabulary size 32,100 |
| **Inference** | Beam search decoding (`num_beams=5`, `max_length=50`) |
| **Environment** | Google Colab / Local GPU (CUDA 11+) |
| **Output** | Saved checkpoint at `results_t5base/checkpoint-4450` |

---

##  Implementation Overview

The project runs as a 7-stage pipeline in a Jupyter Notebook:

```
Stage 1:  Dataset Loading and 80/20 Train-Validation Splitting
Stage 2:  Dataset Analysis — length distributions, word counts
Stage 3:  Tokenization — prefix-based ("summarize: "), MAX_LENGTH=512
Stage 4:  T5 Model Loading — 220M params, moved to CUDA
Stage 5:  Training — TrainingArguments + Trainer, 10 epochs, BATCH_SIZE=4
Stage 6:  ROUGE Evaluation — compute_metrics with stemmer
Stage 7:  Inference — beam search on unseen .txt articles
```

---

## Stage 1 — Dataset Preparation

### 1.1 Installation

```python
!pip install -U transformers
!pip install -U datasets
!pip install tensorboard
!pip install sentencepiece
!pip install accelerate
!pip install evaluate
!pip install rouge_score
```

### 1.2 Imports

```python
import torch
import pprint
import evaluate
import numpy as np

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

pp = pprint.PrettyPrinter()
```

### 1.3 Loading and Splitting the Dataset

```python
# Load BBC News Summary dataset from HuggingFace Hub
dataset = load_dataset('gopalkalpande/bbc-news-summary', split='train')

# 80/20 train-validation split with shuffle
full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

dataset_train = full_dataset['train']
dataset_valid = full_dataset['test']

print(dataset_train)
print(dataset_valid)
```

The BBC News Summary dataset contains two columns: `Articles` (long news text) and `Summaries` (human-written reference summaries). We shuffle and split so that 80% of article-summary pairs train the model and 20% serve as the held-out validation set for ROUGE evaluation.

---

## Stage 2 — Dataset Analysis

Before tokenizing, we profile the **length distribution** of articles and summaries. This directly determines our `MAX_LENGTH` hyperparameter and how much truncation will occur.

### 2.1 Longest Length Analysis

```python
def find_longest_length(dataset):
    """
    Find the longest article and summary in the entire training set.
    """
    max_length = 0
    counter_4k = 0
    counter_2k = 0
    counter_1k = 0
    counter_500 = 0
    for text in dataset:
        corpus = [word for word in text.split()]
        if len(corpus) > 4000: counter_4k += 1
        if len(corpus) > 2000: counter_2k += 1
        if len(corpus) > 1000: counter_1k += 1
        if len(corpus) > 500:  counter_500 += 1
        if len(corpus) > max_length:
            max_length = len(corpus)
    return max_length, counter_4k, counter_2k, counter_1k, counter_500

longest_article_length, c4k, c2k, c1k, c500 = find_longest_length(dataset_train['Articles'])
print(f"Longest article length:        {longest_article_length} words")
print(f"Articles larger than 4000 words: {c4k}")
print(f"Articles larger than 2000 words: {c2k}")
print(f"Articles larger than 1000 words: {c1k}")
print(f"Articles larger than 500  words: {c500}")

longest_summary_length, c4k, c2k, c1k, c500 = find_longest_length(dataset_train['Summaries'])
print(f"Longest summary length:         {longest_summary_length} words")
print(f"Summaries larger than 500 words: {c500}")
```

### 2.2 Average Length Analysis

```python
def find_avg_sentence_length(dataset):
    """
    Find the average sentence length in the entire training set.
    """
    sentence_lengths = []
    for text in dataset:
        corpus = [word for word in text.split()]
        sentence_lengths.append(len(corpus))
    return sum(sentence_lengths) / len(sentence_lengths)

avg_article_length = find_avg_sentence_length(dataset_train['Articles'])
avg_summary_length = find_avg_sentence_length(dataset_train['Summaries'])
print(f"Average article length: {avg_article_length:.1f} words")
print(f"Average summary length: {avg_summary_length:.1f} words")
```

**Why this matters — the Compression Ratio:**

The analysis reveals the compression ratio the model must learn:

```
Compression Ratio = avg_article_length / avg_summary_length
```

A higher ratio means the model must aggressively select and discard content. This also informs `max_length=50` used in generation — summaries in this dataset are short enough that 50 tokens covers most reference summaries.

---

## Stage 3 — Tokenization

### 3.1 Configuration Constants

```python
MODEL      = 't5-base'
BATCH_SIZE = 4
NUM_PROCS  = 4
EPOCHS     = 10
OUT_DIR    = 'results_t5base'
MAX_LENGTH = 512   # Maximum encoder token length
```

### 3.2 T5 Tokenizer — SentencePiece Unigram

```python
tokenizer = T5Tokenizer.from_pretrained(MODEL)
```

T5 uses a **SentencePiece** unigram language model tokenizer trained on a large multilingual corpus. Unlike word-level tokenizers, SentencePiece operates directly on raw Unicode text and learns a subword vocabulary of exactly 32,100 tokens. Tokenization example:

```
Input string:  "BBC reports UK economy grows"
Subword tokens: ["▁BBC", "▁reports", "▁UK", "▁economy", "▁grow", "s"]
Token IDs:      [  8229,    2765,     252,    4621,       4445,    7  ]
```

The `▁` prefix marks the beginning of a new word after a space. This encoding handles out-of-vocabulary words by splitting them into known subwords.

### 3.3 Preprocessing Function — Text-to-Text Prefix Format

```python
def preprocess_function(examples):
    # T5 task framing: prepend "summarize: " to every input article
    inputs = [f"summarize: {article}" for article in examples['Articles']]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,   # Truncate to 512 tokens
        truncation=True,
        padding='max_length'     # Pad to exactly 512 tokens
    )

    # Tokenize target summaries
    targets = [summary for summary in examples['Summaries']]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply in parallel across full dataset
tokenized_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)
tokenized_valid = dataset_valid.map(
    preprocess_function,
    batched=True,
    num_proc=NUM_PROCS
)
```

**What padding produces:** For an article of token length `L < 512`, the padded representation is:

```
input_ids      = [t₁, t₂, ..., t_L, PAD, PAD, ..., PAD]   length = 512
attention_mask = [ 1,  1, ...,  1,    0,   0, ...,   0]   1=real, 0=pad
```

The `attention_mask` ensures padding positions contribute zero to all attention computations. Without it, the model would attend to meaningless pad tokens and produce degraded summaries.

---

## Stage 4 — T5 Model Architecture & Mathematical Foundation

### 4.1 Text-to-Text Framing

T5 (Raffel et al., 2020) unifies all NLP tasks by converting every problem into a text-to-text format. For summarization, the mapping is:

```
Input:   "summarize: [full BBC news article...]"
             ↓  Encoder (12 layers)
Hidden:  H_enc ∈ ℝ^(n × 768)   [contextual article representations]
             ↓  Decoder (12 layers) + Cross-Attention
Output:  "BBC reports that economy grew in Q3..."
```

Every token in the generated output attends over every token in the encoded input via cross-attention — this is how article content flows into the summary.

### 4.2 Encoder — Self-Attention Derivation

The encoder processes input tokens `X = [x₁, x₂, ..., xₙ] ∈ ℝ^(n × d_model)` where `d_model = 768` for T5-Base and `n ≤ 512` (our `MAX_LENGTH`).

**Embedding lookup:** Each token ID `tᵢ` is mapped to a dense vector:

```
xᵢ = E[tᵢ]    where E ∈ ℝ^(|V| × d_model) = ℝ^(32100 × 768)
```

### 4.3 Scaled Dot-Product Attention — Complete Step-by-Step Derivation

The core operation of every Transformer layer. Given input `X ∈ ℝ^(n × d_model)`:

**Step 1 — Linear Projections to Q, K, V:**

```
Q = X · W_Q      W_Q ∈ ℝ^(d_model × d_k) = ℝ^(768 × 64)
K = X · W_K      W_K ∈ ℝ^(d_model × d_k) = ℝ^(768 × 64)
V = X · W_V      W_V ∈ ℝ^(d_model × d_v) = ℝ^(768 × 64)
```

For T5-Base: `h = 12` heads, `d_k = d_v = d_model / h = 768 / 12 = 64`

**Step 2 — Compute Raw Attention Scores:**

```
scores = Q · Kᵀ    ∈ ℝ^(n × n)
```

Entry `scores[i,j]` = dot product between query `i` and key `j`, measuring relevance of token `j` to token `i`.

**Step 3 — Scale to Prevent Gradient Saturation:**

```
scores_scaled = Q · Kᵀ / √(d_k) = Q · Kᵀ / √64 = Q · Kᵀ / 8
```

Without scaling: if `q, k ~ N(0,1)` independently, then `q·k = Σᵢ qᵢkᵢ` has variance `d_k`. At `d_k = 64`, scores have standard deviation 8, pushing softmax into saturation (near-zero gradients). Dividing by `√(d_k)` restores unit variance.

**Step 4 — Optional Causal Mask (Decoder only):**

```
M[i,j] =  0    if j ≤ i     (token i can attend to token j — past or present)
M[i,j] = -∞   if j > i     (token i cannot attend to future token j)
```

Applied as: `scores_masked = scores_scaled + M`

**Step 5 — Softmax Normalization:**

```
A[i,j] = exp(scores_scaled[i,j]) / Σₖ exp(scores_scaled[i,k])
```

Properties: `A[i,j] ≥ 0` for all i,j and `Σⱼ A[i,j] = 1` for every row i.

`A[i,j]` = probability that token `i` attends to token `j`.

**Step 6 — Weighted Aggregation of Values:**

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √(d_k) ) · V     ∈ ℝ^(n × d_v)
```

Row `i` of the output = weighted average of all value vectors, with weights given by how strongly token `i` attends to each token.

**Complete attention matrix dimensions for our setup:**

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Q | (512, 64) | Query vectors — one per input token |
| K | (512, 64) | Key vectors — one per input token |
| V | (512, 64) | Value vectors — one per input token |
| Q·Kᵀ | (512, 512) | Raw attention scores — all pairs |
| A | (512, 512) | Attention weights — rows sum to 1 |
| Output | (512, 64) | Context-enriched representations |

### 4.4 Multi-Head Attention

T5 runs `h = 12` attention heads in parallel, each with its own learned projections:

**Step 1 — Compute h independent attention heads:**

```
headᵢ = Attention(X·W_Qᵢ, X·W_Kᵢ, X·W_Vᵢ)    for i = 1, ..., 12
```

Each `headᵢ ∈ ℝ^(n × 64)`.

**Step 2 — Concatenate and project:**

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head₁₂) · W_O
```

where `W_O ∈ ℝ^(h·d_v × d_model) = ℝ^(768 × 768)`.

Output shape: `(n, d_model) = (512, 768)` — same as input shape. Each token now encodes information attended from all other tokens through 12 different "lenses."

**Parameter count for Multi-Head Attention (T5-Base, per layer):**

```
W_Q:   768 × 768 = 589,824
W_K:   768 × 768 = 589,824
W_V:   768 × 768 = 589,824
W_O:   768 × 768 = 589,824
─────────────────────────────
Total: 2,359,296 parameters per attention layer
```

### 4.5 Position-wise Feed-Forward Network

After Multi-Head Attention, each layer applies a two-layer FFN independently to each token:

```
FFN(x) = max(0, x · W₁ + b₁) · W₂ + b₂
```

For T5-Base: `d_ff = 3072` (inner dimension).

**T5's gated variant (GeGLU activation):**

```
FFN_gated(x) = (x · W₁  ⊙  σ(x · W_gate)) · W₂
```

where `⊙` is element-wise multiplication and `σ` is sigmoid. The gate learns to selectively suppress or amplify features, giving the network more expressive power than plain ReLU.

**Parameter count for FFN (T5-Base, per layer):**

```
W₁:      768 × 3072 = 2,359,296
W_gate:  768 × 3072 = 2,359,296
W₂:     3072 × 768  = 2,359,296
────────────────────────────────
Total:   7,077,888 parameters per FFN layer
```

### 4.6 Relative Position Encoding

Unlike the original Transformer (absolute sinusoidal embeddings), T5 uses **relative position biases** added directly to the attention logits:

```
scores_with_pos[i,j] = (Q·Kᵀ)[i,j] / √(d_k)  +  b(i−j)
```

where `b(i−j)` is a learned scalar depending only on the relative offset between positions `i` and `j`. Offsets are bucketed into 32 learned bias values.

**Advantage:** The model only needs to know "how far apart" two tokens are, not their absolute positions. This makes T5 more robust to sequence lengths different from training — critical for handling BBC articles of varying lengths.

### 4.7 Complete Single Encoder Layer Forward Pass

```
Given: X ∈ ℝ^(n × d_model)

Step 1:  z = LayerNorm(X)                          [Pre-normalization]
Step 2:  z = MultiHeadSelfAttention(z, z, z)       [Attend to all positions]
Step 3:  X = X + z                                 [Residual connection]
Step 4:  z = LayerNorm(X)                          [Pre-normalization]
Step 5:  z = FFN_gated(z)                          [Position-wise FFN]
Step 6:  X = X + z                                 [Residual connection]

Output: X ∈ ℝ^(n × d_model)   [same shape — stacks 12 times]
```

### 4.8 Decoder — Cross-Attention and Autoregressive Generation

Each decoder layer has three sub-layers:

**Sub-layer 1 — Masked Self-Attention:**

The decoder attends to its own previously generated tokens with a causal mask:

```
SelfAttn_dec = MaskedMultiHeadAttention(Y, Y, Y)
```

Token at position `t` can only see positions `1, 2, ..., t-1`.

**Sub-layer 2 — Cross-Attention (Encoder-Decoder Attention):**

The decoder reads the encoder's output `H_enc`:

```
Q = decoder_state · W_Q    [from decoder]
K = H_enc · W_K            [from encoder output]
V = H_enc · W_V            [from encoder output]

CrossAttn = Attention(Q, K, V)
```

This is where the article content enters the summary generation. Each decoder token attends over all 512 encoder positions to decide what to draw from the article.

**Sub-layer 3 — Feed-Forward Network:** Identical to encoder FFN.

**Autoregressive token generation:**

At each decoding step `t`, the probability distribution over the next token is:

```
P(yₜ | y₁, ..., y_{t-1}, X) = softmax(H_dec_t · W_vocab)
```

where `W_vocab ∈ ℝ^(d_model × |V|) = ℝ^(768 × 32100)`.

**T5-Base total parameters:**

```python
model = T5ForConditionalGeneration.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
# Output: 222,903,747 total parameters.

total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print(f"{total_trainable_params:,} training parameters.")
# Output: 222,903,747 training parameters.  (all params fine-tuned)
```

**Parameter breakdown:**

```
Embedding (shared):         32,100 × 768       =  24,652,800
Encoder 12 layers:          12 × ~7.1M          =  85,000,000 (approx)
Decoder 12 layers:          12 × ~9.4M          =  113,000,000 (approx)
─────────────────────────────────────────────────────────────
Total:                                          ≈  222,903,747
```

---

## Stage 5 — Training & Fine-Tuning

### 5.1 Cross-Entropy Loss — Full Derivation

Fine-tuning maximizes the conditional log-likelihood of the reference summary given the article. Equivalently, it minimizes **token-level cross-entropy** loss.

At each decoder step `t`, the model produces logits `logit_t ∈ ℝ^|V|` and the probability of the correct token `yₜ` is:

```
p_t(yₜ) = exp(logit_t[yₜ]) / Σⱼ exp(logit_t[j])
```

The cross-entropy loss at step `t`:

```
L_t = −log p_t(yₜ) = −logit_t[yₜ] + log( Σⱼ exp(logit_t[j]) )
```

The total loss over a summary of `T` tokens, masking padding positions (labels = `-100`):

```
L = (1 / T_valid) · Σ_{t : yₜ ≠ -100}  L_t
```

where `T_valid` = number of non-padding target tokens. Gradients flow back through the entire encoder-decoder stack via automatic differentiation.

**Why `-100` for padding?** HuggingFace automatically ignores label positions set to `-100` in loss computation. During metric computation, we restore pad IDs before decoding:

```python
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
```

### 5.2 Training Configuration

```python
training_args = TrainingArguments(
    output_dir=OUT_DIR,                       # Save checkpoints here
    num_train_epochs=EPOCHS,                  # 10 full passes over training data
    per_device_train_batch_size=BATCH_SIZE,   # 4 samples per GPU step
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy='epoch',              # Evaluate at end of each epoch
    save_strategy='epoch',                    # Save checkpoint each epoch
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,              # Load best ROUGE checkpoint after training
    predict_with_generate=True,               # Use .generate() for eval decoding
    report_to='tensorboard'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

trainer.train()
```

### 5.3 Memory Optimization — preprocess_logits_for_metrics

```python
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
```

Without this function, the Trainer stores full logit tensors of shape `(batch_size, seq_len, vocab_size) = (4, 512, 32100)` during evaluation. Each batch consumes `4 × 512 × 32100 × 4 bytes ≈ 263 MB`. This function immediately collapses logits to token IDs via `argmax`:

```
logits:    (4, 512, 32100)   → 263 MB per batch
pred_ids:  (4, 512)          → 0.008 MB per batch
```

Memory reduction factor: `32,100×`.

### 5.4 Beam Search Decoding — Full Algorithm

At inference and evaluation, summaries are generated via **beam search** with `num_beams=5`:

**Initialization:**

```
Beam set B = { ("", score=0.0) }   [one empty sequence]
```

**At each decoding step t (for t = 1, 2, ..., max_length):**

```
For each of the 5 current beams bᵢ ∈ B:
    Feed bᵢ through decoder
    Compute P(yₜ | bᵢ, encoder_output) ∈ ℝ^32100
    Generate 32,100 candidate extensions: (bᵢ + token_j, score_j)

Total candidates: 5 × 32,100 = 160,500

Sort all candidates by cumulative log-probability score
Keep top 5 → new beam set B
```

**Cumulative score for a sequence of length t:**

```
score(y₁, ..., yₜ) = Σᵢ₌₁ᵗ  log P(yᵢ | y₁, ..., y_{i-1}, X)
```

Continue until all 5 beams have emitted `<EOS>` token or `max_length=50` is reached. Return the beam with the highest cumulative score.

**Why beam search over greedy decoding?** Greedy decoding picks `argmax P(yₜ | ...)` at each step independently — it can get stuck in locally optimal but globally suboptimal sequences. Beam search explores multiple hypotheses simultaneously and finds higher-probability overall sequences.

**Saving the model:**

```python
tokenizer.save_pretrained(OUT_DIR)   # Save tokenizer vocab + config

# Compress output directory for download
!zip -r {OUT_DIR} {OUT_DIR}
```

---

## Stage 6 — ROUGE Evaluation Full Derivation

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the standard automatic metric for summarization. It compares n-gram and sequence overlap between the generated summary (candidate) and human-written reference summaries.

```python
rouge = evaluate.load("rouge")
```

### 6.1 ROUGE-N: Definition and Derivation

**Step 1 — Count clipped n-gram matches:**

```
Count_match(n) = Σ_{gram_n ∈ R}  min( Count(gram_n, C), Count(gram_n, R) )
```

The `min` clips to avoid rewarding a candidate that repeats a common n-gram many times.

**Step 2 — Recall, Precision, F1:**

```
ROUGE-N Recall    =  Count_match(n)  /  Σ_{gram_n ∈ R} Count(gram_n, R)

ROUGE-N Precision =  Count_match(n)  /  Σ_{gram_n ∈ C} Count(gram_n, C)

ROUGE-N F1        =  2 × Precision × Recall  /  (Precision + Recall)
```

### 6.2 ROUGE-1 Worked Example

```
Candidate (C): "economy grows strongly in UK"
Reference  (R): "UK economy grows"

Unigrams in R: {UK:1, economy:1, grows:1}
Unigrams in C: {economy:1, grows:1, strongly:1, in:1, UK:1}

Count_match(1):
  UK:       min(Count(UK,C)=1, Count(UK,R)=1)       = 1
  economy:  min(Count(economy,C)=1, Count(economy,R)=1) = 1
  grows:    min(Count(grows,C)=1, Count(grows,R)=1)  = 1
  Total = 3

ROUGE-1 Recall    = 3 / 3 = 1.000
ROUGE-1 Precision = 3 / 5 = 0.600
ROUGE-1 F1        = 2 × 1.0 × 0.6 / (1.0 + 0.6) = 0.750
```

### 6.3 ROUGE-2 Worked Example

```
Bigrams in R: {(UK, economy):1, (economy, grows):1}
Bigrams in C: {(economy, grows):1, (grows, strongly):1, (strongly, in):1, (in, UK):1}

Count_match(2):
  (UK, economy):  min(0, 1) = 0
  (economy, grows): min(1, 1) = 1
  Total = 1

ROUGE-2 Recall    = 1 / 2 = 0.500
ROUGE-2 Precision = 1 / 4 = 0.250
ROUGE-2 F1        = 2 × 0.5 × 0.25 / (0.5 + 0.25) = 0.333
```

### 6.4 ROUGE-L: Longest Common Subsequence

ROUGE-L uses the **Longest Common Subsequence (LCS)** — capturing sentence-level structure without requiring contiguous matches.

**LCS Dynamic Programming Derivation:**

Given candidate `C = [c₁, ..., c_m]` and reference `R = [r₁, ..., r_n]`, build table `dp[i][j]`:

```
dp[i][j] = dp[i-1][j-1] + 1          if C[i] == R[j]
          = max(dp[i-1][j], dp[i][j-1])   otherwise

dp[0][j] = dp[i][0] = 0    (base case)

LCS(C, R) = dp[m][n]
```

Time complexity: `O(m·n)`. Space: `O(m·n)`.

**ROUGE-L Scores:**

```
R_lcs = LCS(C, R) / |R|           [LCS Recall]
P_lcs = LCS(C, R) / |C|           [LCS Precision]

ROUGE-L F1 = 2 × R_lcs × P_lcs / (R_lcs + P_lcs)
```

**ROUGE-L Example:**

```
C = ["the", "economy", "grew", "in", "UK"]
R = ["UK", "economy", "grew", "strongly"]

LCS = ["economy", "grew"]   length = 2
  (same relative order in both C and R, not necessarily contiguous)

R_lcs = 2/4 = 0.500
P_lcs = 2/5 = 0.400
F1    = 2 × 0.5 × 0.4 / (0.5 + 0.4) = 0.444
```

### 6.5 compute_metrics — Full Implementation

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

    # Decode generated token IDs back to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 pad markers with actual pad_token_id before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE-1, ROUGE-2, ROUGE-L with Porter stemming
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,          # "running" and "runs" treated as same stem
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )

    # Track average generated summary length (non-padding tokens)
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
```

**`use_stemmer=True`** applies the Porter stemmer before ROUGE matching so morphologically related words (e.g. "economy"/"economic", "grow"/"grew"/"growing") count as matches. This is standard practice in summarization evaluation.

---

## Stage 7 — Inference Pipeline

### 7.1 Loading the Fine-Tuned Checkpoint

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import glob

# Load from the best saved checkpoint
model_path = f"{OUT_DIR}/checkpoint-4450"
model     = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(OUT_DIR)
```

### 7.2 Downloading Inference Data

```python
!wget "https://www.dropbox.com/scl/fi/561r8pfhem4lu70hf438q/inference_data.zip?rlkey=aedt2saqmmp3a67qc4o34k04y&dl=1" -O inference_data.zip
!unzip inference_data.zip
```

### 7.3 Summarization Function

```python
def summarize_text(text, model, tokenizer, max_length=512, num_beams=5):
    # Step 1: Tokenize with task prefix
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    )

    # Step 2: Generate summary tokens via beam search
    summary_ids = model.generate(
        inputs,
        max_length=50,
        num_beams=num_beams,
    )

    # Step 3: Decode token IDs to string
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

### 7.4 Batch Inference on .txt Files

```python
pp = pprint.PrettyPrinter()

for file_path in glob.glob('inference_data/*.txt'):
    file    = open(file_path)
    text    = file.read()
    summary = summarize_text(text, model, tokenizer)
    pp.pprint(summary)
    print('-' * 75)
```

**Complete end-to-end data flow:**

```
Raw text string  →  "summarize: BBC reports that the UK economy..."
        ↓  tokenizer.encode(...)
Token IDs  →  tensor([[21603, 10, 8229, ...]])   shape: (1, 512)
        ↓  model.generate(inputs, num_beams=5, max_length=50)
Summary IDs  →  tensor([[947, 16, 3, ...]])   shape: (1, T)
        ↓  tokenizer.decode(summary_ids[0], skip_special_tokens=True)
Output string  →  "BBC reports economy grew strongly in third quarter"
```

---

## Mathematical Foundation — Sparse Optimization from Base Paper

This section documents the theoretical foundation from **Yao et al. (IJCAI 2015)**, which motivates the architecture design of this project — specifically how sparse extractive selection connects to what T5's cross-attention implicitly learns.

### Document Representation

Represent the document as a weighted term-frequency matrix:

```
D = [D₁, D₂, ..., Dₙ] ∈ ℝ^(d×n)
```

where `d` = vocabulary size, `n` = total sentences, column `Dᵢ ∈ ℝ^(d×1)` = TF-IDF sentence vector.

### Data Reconstruction Objective

A good summary should reconstruct the original document. Define coefficient matrix `A = {A₁, ..., Aₙ} ∈ ℝ^(n×n)` where `Aⱼ = [a₁ⱼ, ..., aₙⱼ]ᵀ` are reconstruction weights for sentence `j` using other sentences:

```
min  Σⱼ₌₁ⁿ ‖Dⱼ − DAⱼ‖²₂   =   min ‖D − DA‖²_F
 A                               A
```

### ℓ₂,₁ Row-Sparsity Regularization

Sentence-level sparsity = row-sparsity in `A`. The ℓ₂,₁ matrix norm:

```
‖A‖₂,₁ = Σᵢ₌₁ⁿ ‖Aᵢ‖₂
```

is the sum of ℓ₂ norms of rows. Minimizing it drives entire rows of `A` to zero — each zero row means that sentence is excluded from the summary.

**Why ℓ₂,₁ and not ℓ₁?**

```
ℓ₁  norm: Σᵢⱼ |aᵢⱼ|         → element-wise sparsity (individual coefficients → 0)
ℓ₂,₁ norm: Σᵢ (Σⱼ aᵢⱼ²)^½  → group/row sparsity (entire rows → 0 = sentence excluded)
```

Row sparsity naturally models sentence selection — either a sentence is included (non-zero row) or excluded (zero row).

**Full optimization problem:**

```
min  ‖D − DA‖² + λ‖A‖₂,₁          ... (Equation 1)
 A
s.t.  aᵢⱼ ≥ 0  ∀i,j               ... (Equation 2)
      diag(A) = 0                   ... (Equation 3)
```

Constraint (2): Non-negativity — coefficients must be non-negative for interpretable selection.
Constraint (3): Zero diagonal — prevents trivial solution `A = I` (each sentence reconstructing itself perfectly, so the "summary" = the full document).

### ADMM Solver — Full Step-by-Step Derivation

Problem (1) is reformulated with variable splitting (`A` → `X` and `Z`) so the objective separates into two independently convex terms:

```
min  ‖D − DX‖² + λ‖Z‖₂,₁          ... (Equation 4)
X,Z
s.t.  X = Z,  diag(X) = 0,  xᵢⱼ ≥ 0  ∀i,j   ... (Equation 5)
```

**Augmented Lagrangian** (scaled dual form) with dual variable matrix `U`:

```
L_ρ(X, Z, U) = ‖D − DX‖²_F + λ‖Z‖₂,₁ + (ρ/2)‖X − Z + U‖²_F
```

ADMM alternates three updates per iteration:

**X-Update — Closed form, column by column:**

Minimizing `L_ρ` over column `Xⱼ` (all other columns fixed), taking gradient w.r.t. `Xⱼ` and setting to zero:

```
−2Dᵀ(Dⱼ − DXⱼ) + ρ(Xⱼ − Zⱼᵏ + Uⱼᵏ) = 0

(2DᵀD + ρI)Xⱼ = 2DᵀDⱼ + ρ(Zⱼᵏ − Uⱼᵏ)
```

Absorbing the factor of 2 into `ρ`:

```
Xⱼ^(k+1) = (DᵀD + ρI)⁻¹ (DᵀDⱼ + ρ(Zⱼᵏ − Uⱼᵏ))
```

Apply constraints: `X ← X − diag(X)`, then `xᵢⱼ ← max{xᵢⱼ, 0}`

Note: `(DᵀD + ρI)` is **constant across all iterations** — precompute its Cholesky factorization `LLᵀ = DᵀD + ρI` once, then solve each column update as two triangular systems. No matrix inversion needed during the loop.

**Z-Update — Row-wise group soft-thresholding:**

Minimizing over `Z` row by row (proximal operator of scaled ℓ₂,₁ norm):

```
Zᵢ^(k+1) = arg min_{Zᵢ}  (λ/ρ)‖Zᵢ‖₂ + (1/2)‖Zᵢ − (Xᵢ^(k+1) + Uᵢᵏ)‖²₂
```

Closed-form solution — the **group shrinkage operator**:

```
Zᵢ^(k+1) = S_{λ/ρ}(Xᵢ^(k+1) + Uᵢᵏ)
```

where:

```
S_γ(x) = max{1 − γ/‖x‖₂ , 0} · x
```

Behaviour:
- If `‖x‖₂ < γ`:  entire row → **zero** (sentence excluded from summary)
- If `‖x‖₂ ≥ γ`:  row shrunk toward zero by factor `(1 − γ/‖x‖₂)`, preserving direction

Apply constraints: `Z ← Z − diag(Z)`, then `zᵢⱼ ← max{zᵢⱼ, 0}`

**U-Update — Dual ascent:**

```
U^(k+1) = Uᵏ + ρ(X^(k+1) − Z^(k+1))
```

This drives `X` and `Z` toward consensus (`X = Z`) over iterations.

**Complete ADMM Algorithm:**

```
Input:  Document matrix D, regularization λ, penalty ρ, tolerance ε
Output: Coefficient matrix Z; row norms ‖Zᵢ‖₂ = sentence importance scores

Precompute: M = Cholesky(DᵀD + ρI)   [done ONCE before the loop]
Initialize: X, Z, U ← 0

for k = 0, 1, 2, ... do

  ── X-update (column-wise, closed form) ──────────────────────────────
  for each column j:
    Xⱼ ← solve (DᵀD + ρI) Xⱼ = DᵀDⱼ + ρ(Zⱼ − Uⱼ)  via Cholesky
  X ← X − diag(X)                            [zero diagonal constraint]
  xᵢⱼ ← max{xᵢⱼ, 0}  for all i,j           [non-negativity constraint]

  ── Z-update (row-wise group shrinkage) ──────────────────────────────
  for each row i:
    Zᵢ ← S_{λ/ρ}(Xᵢ + Uᵢ)
  Z ← Z − diag(Z)
  zᵢⱼ ← max{zᵢⱼ, 0}  for all i,j

  ── U-update (dual ascent) ───────────────────────────────────────────
  U ← U + ρ(X − Z)

  ── Convergence check ────────────────────────────────────────────────
  if ‖X − Z‖_F < ε:  return Z

end for

Post-processing:
  Rank sentences by ‖Zᵢ‖₂  (larger = more important)
  Greedily select top sentences up to word budget
```

This ADMM process converges **>60× faster** than the gradient descent used in He et al. (2012).

### Diversity via Sentence Dissimilarity Term

To prevent selecting redundant sentences, an information-theoretic dissimilarity term is added:

```
tr(ΔᵀX) = Σᵢ Σⱼ δᵢⱼ · xᵢⱼ
```

where `δᵢⱼ` (Frey & Dueck, 2007) is computed as:

```
For each word w in sentence j:
    if w ∈ sentence i:  encoding cost = log( length(sentence i) )
    else:               encoding cost = log( vocabulary size )

δᵢⱼ = Σ_w  cost(w, sentence i)
```

This dissimilarity is **asymmetric** (`δᵢⱼ ≠ δⱼᵢ`). The modified optimization:

```
min  ‖D − DX‖² + μ·tr(ΔᵀX) + λ‖Z‖₂,₁
X,Z
s.t.  X = Z,  diag(X) = 0,  xᵢⱼ ≥ 0
```

Only the X-update changes (Δ is precomputed before the loop):

```
Xⱼ^(k+1) = (DᵀD + ρI)⁻¹ (DᵀDⱼ + ρ(Zⱼ − Uⱼ) − μ(Δᵀ)ⱼ)
```

### Compressive Summarization — Joint Sparse Optimization

Generalizing from sentence selection to word-level compression. Replace `D` with a sparse approximation matrix `R ∈ ℝ^(d×n)` where each column `Rᵢ` is a compressed version of sentence `Dᵢ` with unimportant word dimensions driven to zero:

```
min  ‖D − RA‖² + λ₁‖A‖₂,₁ + λ₂ Σᵢ‖Rᵢ‖₁          ... (Equation 8)
R,A
s.t.  rᵢⱼ, aᵢⱼ ≥ 0  ∀i,j;  grammatical(Rᵢ)        ... (Equation 9)
```

Three penalty terms:

```
‖D − RA‖²     Reconstruction loss — compressed sentences must still represent document
λ₁‖A‖₂,₁     Row-sparsity on A — select few summary sentences
λ₂ Σᵢ‖Rᵢ‖₁  Element-sparsity on R — drop individual words within each sentence
```

The product `RA` makes this jointly non-convex. Solved by **block coordinate descent**:

```
Initialize: R ← D  (start with full uncompressed sentences)

repeat:
    Fix R:  solve for A  →  ADMM (same as above, with R replacing D)
    Fix A:  solve for R  →  column-wise non-negative LASSO
until convergence
```

### Grammatical Compression — Recursive Subtree Algorithm

After sparse optimization gives word-importance scores in `Rᵢ`, compressed sentences are generated by extracting maximum-score grammatical subtrees from dependency parse trees.

**Two grammatical constraint sets (from Clarke & Lapata, 2008):**

```
KEEP_HEAD:  If modifier is included, its head must be included
            Examples: NMOD (noun modifier), AMOD (adjective modifier)
            "nice book" → including "nice" requires including "book"

SIMUL_DEL:  Head and modifier must be included or excluded together
            Examples: SBJ (subject), OBJ (object), VC (verb complement)
            Dropping subject or object → ungrammatical output
```

**Recursive Algorithm (O(n) time):**

```
Initialize: node.score ← Rᵢ[word at node]
            node.cost  ← 1
            cmax       ← NULL

function GET_MAXSCORE_SUBTREE(V):
    for each child C of V:
        T_C ← GET_MAXSCORE_SUBTREE(C)              [recurse into subtree]

        if C.label ∉ SIMUL_DEL
           AND T_C.score / T_C.cost < ε            [low value-per-word ratio]
           AND deletion does not decrease bigram score:
               Delete T_C from C                   [prune subtree]

        V.score += T_C.score                        [accumulate bottom-up]
        V.cost  += T_C.cost

    if V is indicator verb
       AND V.label ∉ KEEP_HEAD
       AND V.score > cmax.score:
           cmax ← V                                [candidate complete sentence]

    if V.score > cmax.score: return V
    else: return cmax
```

where `ε = 0.01·‖Rᵢ‖` is adaptive per sentence. The final T5 fine-tuning step then abstractively refines the compressively selected content into fluent summaries.

---

## 📊 Results

**ROUGE Scores on BBC News Summary Validation Set (T5-Base, 10 epochs):**

| Metric | Description |
|--------|-------------|
| ROUGE-1 | Unigram overlap between generated and reference summaries |
| ROUGE-2 | Bigram overlap — measures fluency and phrase coherence |
| ROUGE-L | Longest Common Subsequence — measures sentence-level structure |
| gen_len | Average number of non-padding tokens in generated summaries |

**Sparse optimization baseline comparison (Yao et al., 2015 — DUC 2006):**

| System | ROUGE-1 | ROUGE-2 | ROUGE-SU4 |
|--------|---------|---------|-----------|
| LEAD | 0.302 | 0.049 | 0.098 |
| DSDR (He et al., 2012) | 0.377 | 0.073 | 0.117 |
| SpOpt-ℓ₂,₁ | 0.391 | 0.083 | 0.138 |
| SpOpt-Δ (with diversity) | 0.400 | 0.087 | 0.142 |
| SpOpt-comp | 0.413 | 0.091 | 0.150 |
| **SpOpt-comp-Δ (best)** | **0.415** | **0.095** | **0.153** |
| PEER 24 (DUC top system) | 0.411 | 0.096 | 0.155 |

The T5 fine-tuned model in this project is evaluated on the BBC News Summary dataset and compared across training epochs using ROUGE-1, ROUGE-2, and ROUGE-L, with the best checkpoint at `checkpoint-4450` selected by `load_best_model_at_end=True`.

---

## 📖 Citation

If you use or reference this work, please cite:

```bibtex
@inproceedings{yao2015compressive,
  title     = {Compressive Document Summarization via Sparse Optimization},
  author    = {Yao, Jin-ge and Wan, Xiaojun and Xiao, Jianguo},
  booktitle = {Proceedings of the Twenty-Fourth International Joint
               Conference on Artificial Intelligence (IJCAI 2015)},
  pages     = {1376--1382},
  year      = {2015}
}

@article{raffel2020exploring,
  title   = {Exploring the Limits of Transfer Learning with a Unified
             Text-to-Text Transformer},
  author  = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and
             Lee, Katherine and Narang, Sharan and Matena, Michael and
             Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal = {Journal of Machine Learning Research},
  volume  = {21},
  number  = {140},
  pages   = {1--67},
  year    = {2020}
}
```

---

<div align="center">

*Amrita Vishwa Vidyapeetham — School of Engineering*

*MFC3 Project | Group D1 | Computationally Efficient Text Summarization*

</div>
