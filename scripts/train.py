from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cs336_basics.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import (
    cross_entropy,
    gradient_clipping,
)
from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model."
    )

    # 数据与输出
    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
    )

    # 模型参数
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # AdamW
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    # 记录和保存
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def load_token_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        dataset = np.load(
            path,
            mmap_mode="r",
        )
    else:
        dataset = np.memmap(
            path,
            mode="r",
            dtype=np.uint16,
        )

    if dataset.ndim != 1:
        raise ValueError(
            f"{path} must contain a 1D token array, "
            f"but got shape {dataset.shape}"
        )

    if not np.issubdtype(
        dataset.dtype,
        np.integer,
    ):
        raise ValueError(
            f"{path} must contain integer token IDs, "
            f"but got dtype {dataset.dtype}"
        )

    return dataset


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()

    total_loss = 0.0

    for _ in range(eval_iters):
        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(inputs)

        loss = cross_entropy(
            inputs=logits,
            targets=targets,
        )

        total_loss += loss.item()

    model.train()

    return total_loss / eval_iters


def main() -> None:
    args = parse_args()

    if args.warmup_iters >= args.cosine_cycle_iters:
        raise ValueError(
            "warmup_iters must be smaller than "
            "cosine_cycle_iters"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data = load_token_array(args.train_data)
    val_data = load_token_array(args.val_data)

    if len(train_data) <= args.context_length:
        raise ValueError("Training dataset is too short.")

    if len(val_data) <= args.context_length:
        raise ValueError("Validation dataset is too short.")

    print(
        f"train tokens: {len(train_data):,}, "
        f"dtype={train_data.dtype}"
    )
    print(
        f"validation tokens: {len(val_data):,}, "
        f"dtype={val_data.dtype}"
    )
    print(f"device: {args.device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(f"parameters: {parameter_count:,}")

    start_iteration = 0

    if args.resume_from is not None:
        start_iteration = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optimizer,
        )
        print(
            f"resumed from {args.resume_from} "
            f"at iteration {start_iteration}"
        )

    model.train()

    last_log_time = perf_counter()
    last_log_iteration = start_iteration

    for iteration in range(
        start_iteration,
        args.max_iters,
    ):
        learning_rate = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )

        # 调度器返回学习率后，将其写入所有参数组。
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        # 这里优化器能清零梯度，是因为，梯度是保存在parameters里，而parameters已经通过model把可修改的
        # 对象传给了optimizer，而model能存parameters是因为每一个组件的weights都用torch的parameters
        # 注册了
        optimizer.zero_grad(set_to_none=True)

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )

        logits = model(inputs)

        loss = cross_entropy(
            inputs=logits,
            targets=targets,
        )

        loss.backward()

        gradient_clipping(
            parameters=model.parameters(),
            max_l2_norm=args.max_grad_norm,
        )

        optimizer.step()

        # iteration 从 0 开始，completed_steps 表示已完成步数。
        completed_steps = iteration + 1

        should_log = (
            completed_steps == 1
            or completed_steps % args.log_every == 0
        )

        if should_log:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            current_time = perf_counter()
            elapsed = current_time - last_log_time

            tokens_processed = (
                (completed_steps - last_log_iteration)
                * args.batch_size
                * args.context_length
            )

            tokens_per_second = (
                tokens_processed / elapsed
            )

            print(
                f"step={completed_steps:7d} "
                f"train_loss={loss.item():.6f} "
                f"lr={learning_rate:.3e} "
                f"tokens/s={tokens_per_second:,.0f}"
            )

            last_log_time = current_time
            last_log_iteration = completed_steps

        should_evaluate = (
            completed_steps % args.eval_every == 0
            or completed_steps == args.max_iters
        )

        if should_evaluate:
            validation_loss = evaluate(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_iters=args.eval_iters,
            )

            print(
                f"step={completed_steps:7d} "
                f"validation_loss={validation_loss:.6f}"
            )

        should_save = (
            completed_steps % args.save_every == 0
            or completed_steps == args.max_iters
        )

        if should_save:
            checkpoint_path = (
                args.checkpoint_dir
                / f"checkpoint_{completed_steps:07d}.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                out=checkpoint_path,
            )

            print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()