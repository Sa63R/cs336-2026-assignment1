# inputs targets就在这来了

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length

    starting_indices = np.random.randint(
        low=0,
        high=max_start,
        size=batch_size,
    )

    offsets = np.arange(context_length)

    positions = (
        starting_indices[:, None]
        + offsets[None, :]
    )

    inputs = torch.as_tensor(
        dataset[positions],
        dtype=torch.long,
        device=device,
    )

    targets = torch.as_tensor(
        dataset[positions + 1],
        dtype=torch.long,
        device=device,
    )

    return inputs, targets