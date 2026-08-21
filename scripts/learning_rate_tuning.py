import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
    ) -> None:
        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        defaults = {"lr": lr}
        super().__init__(params, defaults)

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

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                step = state.get("step", 0)

                effective_lr = lr / math.sqrt(step + 1)

                with torch.no_grad():
                    parameter.add_(
                        parameter.grad,
                        alpha=-effective_lr,
                    )

                state["step"] = step + 1

        return loss


def main() -> None:
    torch.manual_seed(0)

    # 三次实验使用完全相同的初始权重。
    initial_weights = 5 * torch.randn((10, 10))

    learning_rates = [1.0, 1e2, 1e3]

    for lr in learning_rates:
        weights = torch.nn.Parameter(
            initial_weights.clone()
        )

        optimizer = SGD(
            [weights],
            lr=lr,
        )

        print(f"\n{'=' * 50}")
        print(f"learning rate = {lr}")
        print(f"{'=' * 50}")

        for step in range(10):
            optimizer.zero_grad()

            loss = (weights**2).mean()

            print(
                f"step={step:2d}  "
                f"loss={loss.item():.8e}"
            )

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()