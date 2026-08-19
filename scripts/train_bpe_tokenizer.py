from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    将 bytes 转成十六进制字符串后保存为 JSON。

    bytes 不能直接写入 JSON，但十六进制字符串可以完整、
    无损地表示任意字节序列。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"

    serialized_vocab = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in vocab.items()
    }

    serialized_merges = [
        [
            left_token.hex(),
            right_token.hex(),
        ]
        for left_token, right_token in merges
    ]

    with open(
        vocab_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialized_vocab,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        merges_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialized_merges,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return vocab_path, merges_path


def verify_saved_tokenizer(
    original_vocab: dict[int, bytes],
    original_merges: list[
        tuple[bytes, bytes]
    ],
    vocab_path: Path,
    merges_path: Path,
) -> None:
    """
    重新读取刚刚保存的文件，确认序列化没有破坏数据。
    """

    with open(
        vocab_path,
        encoding="utf-8",
    ) as file:
        serialized_vocab = json.load(file)

    loaded_vocab = {
        int(token_id): bytes.fromhex(
            token_hex
        )
        for token_id, token_hex
        in serialized_vocab.items()
    }

    with open(
        merges_path,
        encoding="utf-8",
    ) as file:
        serialized_merges = json.load(file)

    loaded_merges = [
        (
            bytes.fromhex(left_hex),
            bytes.fromhex(right_hex),
        )
        for left_hex, right_hex
        in serialized_merges
    ]

    if loaded_vocab != original_vocab:
        raise RuntimeError(
            "保存后重新读取的 vocab 不一致"
        )

    if loaded_merges != original_merges:
        raise RuntimeError(
            "保存后重新读取的 merges 不一致"
        )


def save_metadata(
    output_dir: Path,
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
    elapsed_seconds: float,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> Path:
    """
    保存本次实验参数、耗时和最长词元等信息。
    """

    metadata_path = output_dir / "metadata.json"

    longest_tokens = sorted(
        set(vocab.values()),
        key=lambda token: (
            len(token),
            token,
        ),
        reverse=True,
    )[:20]

    parent_usage = resource.getrusage(
        resource.RUSAGE_SELF
    )
    children_usage = resource.getrusage(
        resource.RUSAGE_CHILDREN
    )

    metadata = {
        "input_path": str(
            input_path.resolve()
        ),
        "input_size_bytes": os.path.getsize(
            input_path
        ),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "number_of_merges": len(merges),
        "special_tokens": special_tokens,
        "num_processes": num_processes,
        "elapsed_seconds": elapsed_seconds,
        "max_parent_rss_kib": (
            parent_usage.ru_maxrss
        ),
        "max_child_rss_kib": (
            children_usage.ru_maxrss
        ),
        "longest_tokens": [
            {
                "byte_length": len(token),
                "hex": token.hex(),
                "bytes_repr": repr(token),
                "decoded_text": token.decode(
                    "utf-8",
                    errors="replace",
                ),
            }
            for token in longest_tokens
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and save a byte-level "
            "BPE tokenizer."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="训练语料文件路径",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="最终词表大小",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="词表、merges 和元数据输出目录",
    )

    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help=(
            "特殊词元；需要多个时可重复传入"
        ),
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="预分词进程数，默认 8",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用 cProfile 并保存性能报告",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"找不到输入文件：{args.input}"
        )

    if args.vocab_size <= 0:
        raise ValueError(
            "vocab-size 必须是正整数"
        )

    if args.num_processes <= 0:
        raise ValueError(
            "num-processes 必须是正整数"
        )

    special_tokens = (
        args.special_tokens
        if args.special_tokens is not None
        else ["<|endoftext|>"]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiler = (
        cProfile.Profile()
        if args.profile
        else None
    )

    print(f"训练文件：{args.input}")
    print(
        "文件大小："
        f"{os.path.getsize(args.input) / 1024**3:.2f} GiB"
    )
    print(f"目标词表：{args.vocab_size}")
    print(f"特殊词元：{special_tokens}")
    print(f"预分词进程：{args.num_processes}")

    if profiler is not None:
        profiler.enable()

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    if profiler is not None:
        profiler.disable()

        profile_path = (
            args.output_dir / "profile.prof"
        )
        profiler.dump_stats(profile_path)

        print("\n累计耗时最高的函数：")
        pstats.Stats(
            profiler
        ).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(20)

        print(
            f"完整性能报告：{profile_path}"
        )

    vocab_path, merges_path = save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=args.output_dir,
    )

    verify_saved_tokenizer(
        original_vocab=vocab,
        original_merges=merges,
        vocab_path=vocab_path,
        merges_path=merges_path,
    )

    metadata_path = save_metadata(
        output_dir=args.output_dir,
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
        elapsed_seconds=elapsed_seconds,
        vocab=vocab,
        merges=merges,
    )

    longest_token = max(
        vocab.values(),
        key=lambda token: (
            len(token),
            token,
        ),
    )

    print("\n训练完成")
    print(f"实际词表大小：{len(vocab)}")
    print(f"合并次数：{len(merges)}")
    print(f"训练耗时：{elapsed_seconds:.3f} 秒")
    print(
        "最长词元字节数："
        f"{len(longest_token)}"
    )
    print(
        "最长词元 bytes："
        f"{longest_token!r}"
    )
    print(
        "最长词元文本："
        f"{longest_token.decode('utf-8', errors='replace')!r}"
    )
    print(f"词表文件：{vocab_path}")
    print(f"合并文件：{merges_path}")
    print(f"实验信息：{metadata_path}")
    print("重新读取验证：通过")


if __name__ == "__main__":
    main()