# Transformers everywhere

The Grumpy old Coders do some research and example code about Transformers in the context of LLM-s.


## Requirements

Requires Python 3.10 or newer. Install dependencies in a virtual environment
from the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Activate the environment in each new terminal before running the example.
If you are already inside `transformer`, use `source ../.venv/bin/activate`.


## Step 1: Word tokenization


`WordTokenizer` splits text into a list of strings using whitespace:

```python
from transformer.tokenizer import WordTokenizer

tokenizer = WordTokenizer()
tokens = tokenizer.tokenize("The cat sat on the mat.")
print(tokens)
# ['The', 'cat', 'sat', 'on', 'the', 'mat.']
```

Run the example:

```sh
python3 -m transformer.main
```

Pass a custom sentence with `--sentence`, or omit it to use `DEFAULT_SENTENCE`:

```sh
python3 -m transformer.main --sentence "Transformers process tokens."
```

You can also run directly from inside the `transformer` folder:

```sh
python3 main.py
python3 main.py --sentence "Transformers process tokens."
```

Repeated spaces, tabs, and newlines separate words without creating empty
tokens. Empty or whitespace-only input returns `[]`. Case is preserved, and
punctuation stays attached to words (for example, `mat.`).

## Step 2: Word embeddings

`main.py` also prints a three-dimensional embedding for each word:

```python
from transformer.embedder import TokenEmbedder

embedder = TokenEmbedder(dimensions=3)
vectors = embedder.embed(["cat", "dog", "cat"])
```

The embedder uses the pretrained 300-dimensional GloVe word table from
[Hugging Face](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d).
The first run downloads approximately 480 MB of weights plus the vocabulary
into the project's `.cache/huggingface` directory. Later runs reuse those files.
Once downloaded, set `HF_HUB_OFFLINE=1` to skip network checks.

> **GloVe** stands for **Global Vectors for Word Representation**. It is a method for learning word embeddings: numerical vectors that capture patterns in how words are used. It learns from **word co-occurrence**—how often words appear near one another across a large text collection. Words used in similar contexts tend to develop similar vectors

PCA is fitted once per instance on the first 10,000 non-padding vocabulary
entries, independently of the input sentence. Reuse the instance for multiple
sentences. No neural network training is required. The output dimension is
configurable from 1 to 300, defaulting to 3.

> **PCA** stands for **Principal Component Analysis**. It reduces the number of dimensions in data while preserving as much variation as possible.

Lookup lowercases tokens and strips surrounding ASCII punctuation. Unknown
words get a zero vector; empty input produces an array of shape `(0, dimensions)`.
Repeated words receive identical vectors. Three dimensions lose substantial
semantic information and the axes have no predefined meanings.

## Step 3: Positional encoding

`PositionalEncoder` generates a fixed sinusoidal matrix `P`. Each row represents
a token position, starting at zero, and each column is an embedding dimension.
It requires no training or download.


> In a sinusoidal matrix, the sine and cosine are used to express a point on a circle. On a circle with radius 1, at angle θ:

| Angle | Cosine (x) | Sine (y) | Location |
|---|---:|---:|---|
| 0° | 1 | 0 | Right |
| 90° | 0 | 1 | Top |
| 180° | −1 | 0 | Left |
| 270° | 0 | −1 | Bottom |

For position `p` and embedding dimension `d`, the standard formula is:

```text
P[p, 2i]     = sin(p / 10000^(2i/d))
P[p, 2i + 1] = cos(p / 10000^(2i/d))
```


Only columns below `d` are included. With our three dimensions this becomes
`[sin(p), cos(p), sin(p / 10000^(2/3))]`. 

For our default sentence, **“The cat sat on the mat.”**:

| Position | Token | sin(p) | cos(p) | sin(p / 464.159) |
|---|---|---:|---:|---:|
| 0 | The | 0\.0000 | 1\.0000 | 0\.0000 |
| 1 | cat | 0\.8415 | 0\.5403 | 0\.0022 |
| 2 | sat | 0\.9093 | −0.4161 | 0\.0043 |
| 3 | on | 0\.1411 | −0.9900 | 0\.0065 |
| 4 | the | −0.7568 | −0.6536 | 0\.0086 |
| 5 | mat. | −0.9589 | 0\.2837 | 0\.0108 |


The third coordinate changes slowly for short sentences. These values describe
position, not word meaning. The encoder supports any positive integer dimension;
`encode(0)` returns an empty matrix of shape `(0, dimensions)`.

## Step 4: Combining words and positions

Add the matrices element by element to form `X`, the input for the future
attention layer:

```python
from transformer.positional_encoder import PositionalEncoder

E = embedder.embed(tokens)
encoder = PositionalEncoder(dimensions=embedder.dimensions)
P = encoder.encode(len(tokens))
X = E + P
```

`E`, `P`, and `X` all have shape `(number_of_tokens, 3)` in our example.
The vectors are added, not concatenated. The same word at different positions
has the same word embedding but a different combined representation.

Run `python3 -m transformer.main --sentence "The cat sat on the mat."` from
the project root with the virtual environment activated to print the embeddings,
`P`, and `X`. Empty input prints an empty token list and skips subsequent steps.

## Step 5: Queries, keys, and values

`QKVProjector` transforms `X` with three separate weight matrices:

```python
from transformer.qkv_projector import QKVProjector

projector = QKVProjector(dimensions=3, seed=42)
Q, K, V = projector.project(X)
# Q = X @ projector.W_Q
# K = X @ projector.W_K
# V = X @ projector.W_V
```

Each weight matrix has shape `(3, 3)`. `Q`, `K`, and `V` each have shape
`(number_of_tokens, 3)`. The same weights apply independently to every token;
this step does not yet mix information between tokens. Biases are omitted.
An empty input matrix of shape `(0, 3)` produces three empty output matrices.

Weights are randomly initialized once per instance using a fixed seed, with
standard deviation `1 / sqrt(dimensions)`. Reusing the instance reuses the
weights. These are **untrained demonstration weights**, separate from the
pretrained GloVe embeddings; they do not guarantee meaningful attention.
`main.py` prints the weight matrices and the resulting `Q`, `K`, and `V`.

Queries and keys determine attention weights; values supply the information
combined using those weights. The optional training example below implements
one causal attention head.

## Step 6: Calculating the attention matrix

```python
from transformer.self_attention import SelfAttention

A = SelfAttention().calculate(Q, K)
```

This computes `A = row_softmax(Q @ K.T / sqrt(d_k) + M)`, where `d_k` is
the number of query/key dimensions (3 here). The causal mask `M` is zero
on and below the diagonal and negative infinity above it, so future tokens
receive exactly zero attention. Softmax is applied separately to each row.

For N tokens, `A` has shape `(N, N)`. Entry `A[i, j]` tells us how strongly
token i attends to token j. Each row sums to 1 before display rounding.
Tokens can attend to themselves; the first row is always `[1, 0, ...]`.
Empty Q and K matrices produce an empty `(0, 0)` matrix.

The example prints A as a table with `position:token` labels on both axes,
so repeated words remain distinguishable. It works with random weights by
default and learned weights when using `--train`. The training code shares
the same attention calculation. V does not enter this calculation; it is
used afterward to form the context vectors `A @ V`.

## Step 7: Following value vectors back to tokens

The demo labels every row of `V = X @ W_V` with its source token and zero-based
position. For example, `1:cat` identifies the value vector offered by the second
token. Its three coordinates are projected features, not probabilities or
human-readable concepts. Because X includes position, repeated words can have
different value vectors at different positions.

The next step calculates `Y = A @ V`. For one receiving token i, it prints
each source token's contribution `A[i, j] * V[j]`, followed by their sum `Y[i]`.
It also prints Y for all tokens. By default the walkthrough focuses on the last
token; choose another position with `--explain-token`:

```sh
python3 -m transformer.main --sentence "the cat likes milk" --explain-token 2
```

Here, the receiving token is `2:likes`. The walkthrough shows contributions
from `0:the`, `1:cat`, and `2:likes`; `3:milk` has a zero contribution because
it is in the future. Add `--train` to inspect the learned demo weights.
The option also works when running `python3 main.py` inside `transformer`.

**A says how much to take; V says what is available to take.** V is computed
independently of A and is shared across receiving tokens. Each receiving token
uses a different row of A to mix those same source vectors. A small attention
weight scales a vector down, but final influence also depends on that vector's
magnitude and direction. Neither the weights nor the vectors are permanent
word-importance scores. Y is a context vector, not a predicted word.

Calculations use full precision; printed values are rounded to four decimals,
so displayed products and sums may differ slightly from hand calculations.

## Step 8: Predicting the next token with an output layer

Run a partial sentence with training enabled:

```sh
python3 -m transformer.main --train --sentence "the cat likes"
```

`OutputLayer` uses the same `W_out` and `bias` learned by `AttentionTrainer`:

```python
output = trainer.output_layer
logits = output.logits(Y[-1:])
probabilities = output.predict(Y[-1:])
next_id = int(probabilities[0].argmax())
next_token = vocabulary[next_id]
```

It calculates `logits = Y @ W_out + bias`, then applies row-wise softmax.
For vocabulary size C, `W_out` has shape `(3, C)`, the bias has shape `(C,)`,
and the output probabilities have shape `(number_of_tokens, C)`. Vocabulary
IDs follow the sorted training vocabulary, in exactly the same order used for
training targets and output weight columns.

To continue the prompt, we use only the **last row of Y**, regardless of which
token was selected for the value walkthrough with `--explain-token`. Step 8
prints all vocabulary tokens with their IDs, logits, and probabilities, sorted
by descending probability. It selects the highest-probability token (`argmax`)
and displays a one-token continuation. This is not a multi-token generation loop.

Without `--train`, this step prints a reminder to enable training. No new random
output weights are introduced after training. Predictions can only select words
in the training vocabulary; the tiny model can still produce incorrect results.
There is no end-of-sequence token, so even a complete sentence gets a next-word
prediction. Empty input skips the output step along with the other model steps.

## Optional: Learning the projection weights

Activate the virtual environment and run from the project root:

```sh
python3 -m transformer.main --train
python3 -m transformer.main --train --epochs 2000 --sentence "the cat likes"
```

Inside `transformer`, the equivalent command is `python3 main.py --train`.
Without `--train`, the example continues to use its original random weights.
Training uses six built-in sentences, such as `the cat likes milk`. To supply
your own UTF-8 file containing one sentence per line:

```sh
python3 -m transformer.main --train --training-file sentences.txt --epochs 2000
```

Blank lines are ignored. Every other line must contain at least two tokens,
and the corpus must contain at least two distinct tokens. Training text is
lowercased and split on whitespace; punctuation remains attached to target
words. Prefer consistent spelling and punctuation in this small corpus.

`AttentionTrainer` uses explicit NumPy backpropagation, with no new dependencies.
For `the cat likes milk`, input tokens are `the cat likes` and targets are
`cat likes milk`. Every input position predicts the following token. Positions
restart at zero for each sentence; sentences are never joined together.

The forward pass is:

```text
X = fixed GloVe embeddings + fixed positional encoding
Q = X @ W_Q; K = X @ W_K; V = X @ W_V
A = row_softmax(Q @ K.T / sqrt(dimensions) + causal_mask)
context = A @ V
logits = context @ W_out + bias
loss = next-token cross-entropy
```

The causal mask prevents each row from seeing later input tokens. Backpropagation
calculates gradients of the loss for `W_Q`, `W_K`, `W_V`, `W_out`, and `bias`.
Full-batch gradient descent updates these weights with a default learning rate
of 0.05 and global gradient norm clipping at 1. The GloVe vectors, PCA projection,
and positional encoding remain fixed. Loss is averaged across target tokens.

`main.py` reports loss before and after training, prints a final-word prediction
for each training sentence, and uses the updated projector to calculate Q/K/V
for your `--sentence`. That sentence is for inspection and is not automatically
added to the training corpus. Weights live in memory only: each run starts from
the same seed and trains again. Empty `--sentence` skips embedding and training.

In a local run, 500 updates reduced loss from approximately 2.8220 to 1.6917,
but several training predictions remained wrong. These are training-set results,
not evidence of generalization. Three dimensions and six sentences are highly
restrictive, and some prefixes have multiple valid continuations. This is a
single attention head and output layer, not a full Transformer or useful LLM.

Run the offline numerical checks with:

```sh
python3 -m unittest discover -s tests -v
```

They compare every analytic parameter gradient with finite differences, verify
causal masking, and check that training reduces loss and updates all three
projection matrices. They do not require a model download.
