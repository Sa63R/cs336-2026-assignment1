import torch

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