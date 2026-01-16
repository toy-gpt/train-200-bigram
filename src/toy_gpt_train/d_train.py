"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using a bigram context (previous token, current token).

Responsibilities:
- Create ((previous_token, current_token) -> next_token) training pairs from the corpus
- Run a basic gradient-descent training loop
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss
  - think descending to find the bottom of a valley in a landscape
  - where the valley floor corresponds to lower prediction error

Notes:
- This is intentionally simple: no deep learning framework, no Transformer.
- The model is a softmax regression classifier over bigram contexts.
- Training updates the weight rows corresponding to the observed bigram context.
- token_embeddings.csv is a visualization-friendly projection derived from weights;
  embeddings are not yet a learned standalone table.
"""

import logging
import math
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.io_artifacts import (
    RowLabeler,
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)

LOG: logging.Logger = get_logger("P01", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"


type BigramContext = tuple[int, int]
type BigramPair = tuple[BigramContext, int]


def token_row_index_bigram(token_id: int, vocab_size: int) -> int:
    """Return the row index for a bigram context where previous_id == current_id == token_id.

    Used for bootstrapping the first bigram step.
    """
    return token_id * vocab_size + token_id


def row_labeler_bigram(vocab: VocabularyLike, vocab_size: int) -> RowLabeler:
    """Map a bigram row index to a label like 'prev|curr'."""

    def label(row_idx: int) -> str:
        previous_id: int = row_idx // vocab_size
        current_id: int = row_idx % vocab_size

        prev_tok: str = vocab.get_id_token(previous_id) or f"id_{previous_id}"
        curr_tok: str = vocab.get_id_token(current_id) or f"id_{current_id}"

        return f"{prev_tok}|{curr_tok}"

    return label


def make_training_pairs(token_ids: list[int]) -> list[BigramPair]:
    """Convert token IDs into ((prev, curr), next) pairs."""
    pairs: list[BigramPair] = []
    for i in range(len(token_ids) - 2):
        pairs.append(((token_ids[i], token_ids[i + 1]), token_ids[i + 2]))
    return pairs


def argmax(values: list[float]) -> int:
    """Return index of maximum value."""
    best_idx = 0
    best_val = values[0]
    for i in range(1, len(values)):
        if values[i] > best_val:
            best_val = values[i]
            best_idx = i
    return best_idx


def cross_entropy_loss(probs: list[float], target_id: int) -> float:
    """Compute -log(p[target]) with numerical safety."""
    p = probs[target_id]
    # Guard against log(0)
    p = max(p, 1e-12)
    return -math.log(p)


def train_model(
    model: SimpleNextTokenModel,
    pairs: list[BigramPair],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the bigram model using gradient descent on softmax cross-entropy."""
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for (previous_id, current_id), target_id in pairs:
            probs: list[float] = model.forward(previous_id, current_id)
            total_loss += cross_entropy_loss(probs, target_id)

            if argmax(probs) == target_id:
                correct += 1

            row_idx: int = previous_id * model.vocab_size + current_id
            row: list[float] = model.weights[row_idx]
            for j in range(model.vocab_size):
                y: float = 1.0 if j == target_id else 0.0
                grad: float = probs[j] - y
                row[j] -= learning_rate * grad

        avg_loss: float = total_loss / len(pairs) if pairs else float("nan")
        accuracy: float = correct / len(pairs) if pairs else 0.0
        history.append(
            {"epoch": float(epoch), "avg_loss": avg_loss, "accuracy": accuracy}
        )

        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run a simple training demo end-to-end."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Softmax Regression")

    # Step 0: Identify the corpus file (single file rule).
    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.error("No tokens found. Check corpus file.")
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error(f"Token not found in vocabulary: {tok!r}")
            return
        token_ids.append(tok_id)

    # Step 4: Create training pairs (input -> target).
    pairs: list[BigramPair] = make_training_pairs(token_ids)
    LOG.info(f"Created {len(pairs)} training pairs.")

    # Step 5: Initialize model with random weights.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(TRAIN_LOG_PATH, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="bigram",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_bigram(vocab, vocab.vocab_size()),
    )

    # Step 8: Qualitative check - what does the model predict after first token?
    previous_token = tokens[0]
    current_token = tokens[1]
    previous_id = vocab.get_token_id(previous_token)
    current_id = vocab.get_token_id(current_token)
    if previous_id is not None and current_id is not None:
        probs = model.forward(previous_id, current_id)
        best_next_id = argmax(probs)
        best_next_tok = vocab.get_id_token(best_next_id)
        LOG.info(
            f"After training, most likely next token after {previous_token!r}|{current_token!r} is {best_next_tok!r} (ID: {best_next_id})."
        )


if __name__ == "__main__":
    main()
