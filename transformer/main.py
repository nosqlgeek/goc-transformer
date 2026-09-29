import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .attention_trainer import AttentionTrainer, DEFAULT_SENTENCES
    from .embedder import TokenEmbedder
    from .positional_encoder import PositionalEncoder
    from .qkv_projector import QKVProjector
    from .self_attention import SelfAttention
    from .tokenizer import WordTokenizer
else:
    from attention_trainer import AttentionTrainer, DEFAULT_SENTENCES
    from embedder import TokenEmbedder
    from positional_encoder import PositionalEncoder
    from qkv_projector import QKVProjector
    from self_attention import SelfAttention
    from tokenizer import WordTokenizer


DEFAULT_SENTENCE = "The cat sat on the mat."


def format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in vector) + "]"


def explain_values(tokens: list[str], A: np.ndarray, V: np.ndarray, index: int) -> None:
    """Show how source value vectors form one receiving token's output."""
    labels = [f"{i}:{token}" for i, token in enumerate(tokens)]
    print("Step 7 - Mixing token information: Y = A @ V ...")
    print(f"Receiving token: {labels[index]}")
    print("Each source contributes its attention weight multiplied by its value vector.")
    contributions = A[index, :, np.newaxis] * V
    for j, (label, value, contribution) in enumerate(zip(labels, V, contributions)):
        masked = " (future token: masked)" if j > index else ""
        print(
            f"  {label}: {A[index, j]:.4f} * {format_vector(value)}"
            f" = {format_vector(contribution)}{masked}"
        )
    Y = A @ V
    print(f"Sum of contributions = Y[{index}] = {format_vector(Y[index])}")
    print("Y for all receiving tokens:")
    for label, row in zip(labels, Y):
        print(f"  {label}: {format_vector(row)}")
    print("V supplies content; A supplies mixing weights. Neither is a fixed word-importance score.")


def main(
    sentence: str = DEFAULT_SENTENCE,
    train: bool = False,
    epochs: int = 500,
    training_file: str = None,
    explain_token: int = None,
) -> None:
    training_tokens = None
    if train:
        sentences = (
            Path(training_file).read_text(encoding="utf-8").splitlines()
            if training_file else DEFAULT_SENTENCES
        )
        training_tokens = [WordTokenizer().tokenize(s.lower()) for s in sentences if s.strip()]
        if not training_tokens or any(len(t) < 2 for t in training_tokens):
            raise ValueError("Each training sentence must contain at least two tokens")
        if epochs < 1:
            raise ValueError("epochs must be positive")
    print("Step 1 - Tokenizing words ...")
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(sentence)
    print(tokens)

    if explain_token is not None and not 0 <= explain_token < len(tokens):
        raise ValueError("explain_token must be a valid zero-based token position")

    if tokens:
        print("Step 2 - Embedding words (first run downloads the GloVe model) ...")
        embedder = TokenEmbedder(dimensions=3)
        embeddings = embedder.embed(tokens)
        for token, vector in zip(tokens, embeddings):
            print(f"{token}: {vector.round(4).tolist()}")

        print("Step 3 - Positional encoding P ...")
        encoder = PositionalEncoder(dimensions=embedder.dimensions)
        positions = encoder.encode(len(tokens))
        print(positions.round(4))

        print("Step 4 - Combining embeddings and positions: X = E + P ...")
        X = embeddings + positions
        print(X.round(4))

        projector = QKVProjector(dimensions=embedder.dimensions)
        if train:
            vocabulary = sorted({token for words in training_tokens for token in words})
            word_ids = {word: i for i, word in enumerate(vocabulary)}
            examples = []
            for words in training_tokens:
                # Shift the targets by one word; do not include the last word in X.
                inputs = embedder.embed(words[:-1]) + encoder.encode(len(words) - 1)
                targets = np.array([word_ids[word] for word in words[1:]])
                examples.append((inputs, targets))
            print(f"Training one attention head on {len(examples)} sentences ...")
            trainer = AttentionTrainer(projector, len(vocabulary))
            history = trainer.fit(examples, epochs=epochs)
            print(f"Training loss: {history[0]:.4f} -> {history[-1]:.4f}")
            for words, (inputs, _) in zip(training_tokens, examples):
                prediction = vocabulary[int(trainer.predict(inputs)[-1].argmax())]
                print(f"{' '.join(words[:-1])} -> {prediction} (target: {words[-1]})")
        label = "learned demo weights" if train else "untrained demonstration weights"
        print(f"Step 5 - Projecting Q, K, V ({label}) ...")
        Q, K, V = projector.project(X)
        for name, matrix in (
            ("W_Q", projector.W_Q), ("W_K", projector.W_K),
            ("W_V", projector.W_V), ("Q", Q), ("K", K),
        ):
            print(f"{name}:\n{matrix.round(4)}")

        print("V = X @ W_V: one value vector per source token")
        print("Coordinates are projected features, not named meanings or attention weights.")
        for i, (token, row) in enumerate(zip(tokens, V)):
            print(f"  {i}:{token}: {format_vector(row)}")

        print("Step 6 - Calculating causal attention matrix A ...")
        A = SelfAttention().calculate(Q, K)
        print("Rows attend to columns; labels are position:token (positions start at 0).")
        labels = [f"{i}:{token}" for i, token in enumerate(tokens)]
        width = max(10, max(len(label) for label in labels) + 2)
        print(" " * width + "".join(f"{label:>{width}}" for label in labels))
        for label, row in zip(labels, A):
            print(f"{label:>{width}}" + "".join(f"{value:>{width}.4f}" for value in row))

        explain_values(tokens, A, V, len(tokens) - 1 if explain_token is None else explain_token)

        print("Step 8 - Output layer: next-token probabilities ...")
        if train:
            Y = A @ V
            output = trainer.output_layer
            logits = output.logits(Y[-1:])[0]
            probabilities = output.predict(Y[-1:])[0]
            print(f"Using Y[{len(tokens) - 1}] for the last token {tokens[-1]!r}.")
            print(f"logits = Y[-1] @ W_out + bias; W_out shape: {output.weights.shape}")
            print("probabilities = softmax(logits)")
            print(f"{'ID':>5}  {'Token':<20} {'Logit':>10} {'Probability':>12}")
            for token_id in np.argsort(-probabilities, kind="stable"):
                print(
                    f"{token_id:>5}  {vocabulary[token_id]:<20}"
                    f" {logits[token_id]:>10.4f} {probabilities[token_id]:>11.2%}"
                )
            next_id = int(probabilities.argmax())
            print(f"Selected next token (argmax): {vocabulary[next_id]!r}, ID {next_id}")
            print(f"One-token continuation: {sentence.rstrip()} {vocabulary[next_id]}")
            print("Predictions are limited to the training vocabulary and may be incorrect.")
        else:
            print("Run with --train to learn an output layer and predict the next token.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect attention and, with --train, predict the next token."
    )
    parser.add_argument(
        "--sentence",
        default=DEFAULT_SENTENCE,
        help="Sentence to tokenize (default: %(default)s)",
    )
    parser.add_argument("--train", action="store_true", help="Learn Q/K/V weights on simple sentences")
    parser.add_argument("--epochs", type=int, default=500, help="Training updates (default: 500)")
    parser.add_argument("--training-file", help="Optional UTF-8 corpus, one sentence per line")
    parser.add_argument(
        "--explain-token", type=int,
        help="Zero-based receiving token position for the value walkthrough (default: last token)",
    )
    args = parser.parse_args()
    if args.training_file and not args.train:
        parser.error("--training-file requires --train")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.explain_token is not None and not 0 <= args.explain_token < len(WordTokenizer().tokenize(args.sentence)):
        parser.error("--explain-token must be a valid zero-based token position")
    main(
        args.sentence, train=args.train, epochs=args.epochs,
        training_file=args.training_file, explain_token=args.explain_token,
    )
