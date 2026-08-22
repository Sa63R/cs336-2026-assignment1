from __future__ import annotations

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if logits.ndim != 1:
        raise ValueError(
            "logits must have shape (vocab_size,)"
        )

    if temperature < 0:
        raise ValueError(
            "temperature must be non-negative"
        )

    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            "top_p must be in the interval (0, 1]"
        )

    # temperature=0 作为贪心解码的特殊情况。
    if temperature == 0:
        return int(
            torch.argmax(logits).item()
        )

    scaled_logits = logits / temperature

    probabilities = softmax(
        scaled_logits,
        dim=-1,
    )

    # top_p=1 时不需要过滤。
    if top_p == 1.0:
        sampled_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        return int(sampled_token.item())

    sorted_probabilities, sorted_indices = (
        torch.sort(
            probabilities,
            descending=True,
        )
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # 某个 token 之前的累计概率。
    previous_cumulative = (
        cumulative_probabilities
        - sorted_probabilities
    )

    # 保留最小的概率前缀，使累计概率达到 top_p。
    keep_mask = previous_cumulative < top_p

    filtered_probabilities = (
        sorted_probabilities * keep_mask
    )

    # 截断之后必须重新归一化。
    filtered_probabilities = (
        filtered_probabilities
        / filtered_probabilities.sum()
    )

    sampled_sorted_position = torch.multinomial(
        filtered_probabilities,
        num_samples=1,
    )

    sampled_token = sorted_indices[
        sampled_sorted_position
    ]

    return int(sampled_token.item())