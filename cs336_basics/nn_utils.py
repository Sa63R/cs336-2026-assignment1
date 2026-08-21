import torch
from collections.abc import Iterable


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = x.max(dim=dim, keepdim=True).values
    shifted_x = x - max_value
    exp_x = torch.exp(shifted_x)
    denominator = exp_x.sum(dim=dim, keepdim=True)

    # 类似于前面的RMSnorm，不难理解
    return exp_x / denominator

def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    max_values = inputs.max(
        dim=-1,
        keepdim=True,
    ).values

    shifted_inputs = inputs - max_values

    log_partition = torch.log(
        torch.exp(shifted_inputs).sum(dim=-1)
    )

    target_logits = shifted_inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_partition - target_logits

    return losses.mean()

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    if len(gradients) == 0:
        return

    total_squared_norm = sum(
        torch.sum(gradient.detach() ** 2)
        for gradient in gradients
    )

    total_l2_norm = torch.sqrt(total_squared_norm)

    clip_coefficient = (
        max_l2_norm / (total_l2_norm + 1e-6)
    )

    clip_coefficient = torch.clamp(
        clip_coefficient,
        max=1.0,
    )

    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(clip_coefficient)