import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        if eps < 0:
            raise ValueError(
                f"Invalid epsilon value: {eps}"
            )

        if not 0 <= betas[0] < 1:
            raise ValueError(
                f"Invalid beta1 value: {betas[0]}"
            )

        if not 0 <= betas[1] < 1:
            raise ValueError(
                f"Invalid beta2 value: {betas[1]}"
            )

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight decay: {weight_decay}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )
                # 
                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter
                    )

                state["step"] += 1

                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 解耦权重衰减
                parameter.mul_(
                    1.0 - lr * weight_decay
                )

                # 一阶矩估计
                exp_avg.mul_(beta1).add_(
                    gradient,
                    alpha=1.0 - beta1,
                )

                # 二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                # 偏差修正后的学习率
                corrected_lr = (
                    lr
                    * math.sqrt(1.0 - beta2**step)
                    / (1.0 - beta1**step)
                )

                denominator = (
                    torch.sqrt(exp_avg_sq) + eps
                )

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-corrected_lr,
                )

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 第一阶段：线性预热
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 第二阶段：余弦退火
    if it <= cosine_cycle_iters:
        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        cosine_factor = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_learning_rate
            + cosine_factor
            * (max_learning_rate - min_learning_rate)
        )

    # 第三阶段：保持最小学习率
    return min_learning_rate