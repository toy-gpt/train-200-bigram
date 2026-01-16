"""c_model.py - Simple model module.

Defines a minimal next-token prediction model for a bigram context
(uses two tokens in sequence: previous token, current token).

Responsibilities:
- Represent a simple parameterized model that maps a
  2-tuple of token IDs (previous token, current token)
  to a score for each token in the vocabulary.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight table (conceptually a 3D tensor: prev x curr x next,
  flattened for storage as a 2D matrix)
- one forward computation
- no learning here

Training is handled in a different module.
"""

import logging
import math
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

LOG: logging.Logger = get_logger("MODEL", level="INFO")


class SimpleNextTokenModel:
    """A minimal next-token prediction model (bigram context)."""

    def __init__(self, vocab_size: int) -> None:
        """Initialize the model with a given vocabulary size."""
        self.vocab_size: Final[int] = vocab_size

        # Weight table (flattened):
        # - Conceptually: vocab_size x vocab_size x vocab_size
        #   where:
        #     (previous_id, current_id) selects a row (context),
        #     and each column is a possible next token.
        #
        # - Stored as: (vocab_size * vocab_size) rows x vocab_size columns.
        #   row_index = previous_id * vocab_size + current_id
        self.weights: list[list[float]] = [
            [0.0 for _ in range(vocab_size)] for _ in range(vocab_size * vocab_size)
        ]

        LOG.info(f"Model initialized with vocabulary size {vocab_size} (bigram).")

    def _row_index(self, previous_id: int, current_id: int) -> int:
        """Map a (previous_id, current_id) pair to a row index in the flattened weight table."""
        return previous_id * self.vocab_size + current_id

    def forward(self, previous_id: int, current_id: int) -> list[float]:
        """Perform a forward pass.

        Args:
            previous_id: Integer ID of the previous token.
            current_id: Integer ID of the current token.

        Returns:
            Probability distribution over next tokens.
        """
        row_idx = self._row_index(previous_id, current_id)
        scores = self.weights[row_idx]
        return self._softmax(scores)

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        """Convert raw scores into probabilities.

        Args:
            scores: Raw score values.

        Returns:
            Normalized probability distribution.
        """
        max_score: float = max(scores)
        exp_scores: list[float] = [math.exp(s - max_score) for s in scores]
        total: float = sum(exp_scores)
        return [s / total for s in exp_scores]


def main() -> None:
    """Demonstrate a forward pass of the simple bigram model."""
    # Local imports keep modules decoupled.
    from toy_gpt_train.a_tokenizer import SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Simple Next-Token Model Demo (Bigram Context)")

    # Step 1: Tokenize input text.
    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < 2:
        LOG.info("Need at least two tokens for bigram demonstration.")
        return

    # Step 2: Build vocabulary.
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Initialize model.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 4: Select previous and current tokens.
    previous_token: str = tokens[0]
    current_token: str = tokens[1]

    previous_id: int | None = vocab.get_token_id(previous_token)
    current_id: int | None = vocab.get_token_id(current_token)

    if previous_id is None or current_id is None:
        LOG.info("One of the sample tokens was not found in vocabulary.")
        return

    # Step 5: Forward pass (bigram context).
    probs: list[float] = model.forward(previous_id, current_id)

    # Step 6: Inspect results.
    LOG.info(
        f"Input tokens: {previous_token!r} (ID {previous_id}), {current_token!r} (ID {current_id})"
    )
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
