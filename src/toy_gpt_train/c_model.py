"""c_model.py - Simple model module.

Defines a minimal next-token prediction model using bigram context.
  A bigram models P(next | current).

Responsibilities:
- Represent a simple parameterized model that maps a
  token ID (current token) to a score for each token in the vocabulary.
- Convert scores into probabilities using softmax.
- Provide a forward pass (no training in this module).

This model is intentionally simple:
- one weight table (2D matrix: current x next)
- one forward computation
- no learning here

Training is handled in a different module.
"""

import logging
import math
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

__all__ = ["SimpleNextTokenModel"]

LOG: logging.Logger = get_logger("MODEL", level="INFO")


class SimpleNextTokenModel:
    """Bigram next-token prediction model.

    Maps (current_token_id) -> distribution over next tokens.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the model with a given vocabulary size."""
        self.vocab_size: Final[int] = vocab_size

        # Weight table:
        # - Shape: vocab_size x vocab_size
        #   where:
        #     row = current_id (context),
        #     column = possible next token.
        self.weights: list[list[float]] = [
            [0.0 for _ in range(vocab_size)] for _ in range(vocab_size)
        ]

        LOG.info(f"Model initialized with vocabulary size {vocab_size} (bigram).")

    def forward(self, current_id: int) -> list[float]:
        """Perform a forward pass.

        Args:
            current_id: Integer ID of the current token.

        Returns:
            Probability distribution over next tokens.
        """
        if not (0 <= current_id < self.vocab_size):
            raise ValueError(f"current_id out of range: {current_id}")
        scores = self.weights[current_id]
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

    log_header(LOG, "Simple Next-Token Model Demo (Bigram / Context-1)")

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

    # Step 4: Select current token.
    current_token: str = tokens[0]
    current_id: int | None = vocab.get_token_id(current_token)

    if current_id is None:
        LOG.info("Sample token was not found in vocabulary.")
        return

    # Step 5: Forward pass (bigram context).
    probs: list[float] = model.forward(current_id)

    # Step 6: Inspect results.
    LOG.info(f"Input token: {current_token!r} (ID {current_id})")
    LOG.info("Output probabilities for next token:")
    for idx, prob in enumerate(probs):
        tok: str | None = vocab.get_id_token(idx)
        LOG.info(f"  {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
