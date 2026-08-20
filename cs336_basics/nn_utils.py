import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_value = x.max(dim=dim, keepdim=True).values
    shifted_x = x - max_value
    exp_x = torch.exp(shifted_x)
    denominator = exp_x.sum(dim=dim, keepdim=True)

    # 类似于前面的RMSnorm，不难理解
    return exp_x / denominator