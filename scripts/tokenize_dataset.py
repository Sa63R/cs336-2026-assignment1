from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"
TOKEN_DTYPE = np.dtype(np.uint16)

DEFAULT_NUM_PROCESSES = min(
    8,
    os.cpu_count() or 1,
)

DEFAULT_CHUNKS_PER_PROCESS = 4
READ_BUFFER_SIZE = 4 * 1024 * 1024
WRITE_BUFFER_SIZE = 250_000
COPY_BUFFER_SIZE = 8 * 1024 * 1024


_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_VOCAB_SIZE: int | None = None


def iter_text_lines_in_range(
    input_path: str,
    start: int,
    end: int,
) -> Iterator[str]:
    """
    在 [start, end) 文件区间内流式读取文本。

    每次最多读取 READ_BUFFER_SIZE 字节，并保留尚未
    结束的最后一行，避免一次把整个文件块加载进内存。
    """
    with open(input_path, "rb") as input_file:
        input_file.seek(start)

        remaining_bytes = end - start
        remainder = b""

        while remaining_bytes > 0:
            block = input_file.read(
                min(
                    READ_BUFFER_SIZE,
                    remaining_bytes,
                )
            )

            if block == b"":
                raise EOFError(
                    "Unexpected end of file while reading "
                    f"{input_path}"
                )

            remaining_bytes -= len(block)

            combined = remainder + block
            pieces = combined.split(b"\n")

            # 最后一部分可能是一条尚未结束的行，
            # 留到下一次读取后继续拼接。
            remainder = pieces.pop()

            for line in pieces:
                yield (
                    line + b"\n"
                ).decode("utf-8")

        if remainder:
            yield remainder.decode("utf-8")


def initialize_worker(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: tuple[str, ...],
    vocab_size: int,
) -> None:
    """
    每个子进程启动时只加载一次分词器。
    """
    global _WORKER_TOKENIZER
    global _WORKER_VOCAB_SIZE

    _WORKER_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=list(special_tokens),
    )

    _WORKER_VOCAB_SIZE = vocab_size

    # 预热正则表达式、BPE 和缓存。
    _WORKER_TOKENIZER.encode(
        "Tokenizer worker warmup."
        + SPECIAL_TOKEN
    )


def encode_file_range(
    task: tuple[
        int,
        str,
        int,
        int,
        str,
    ],
) -> tuple[int, int, int, str]:
    """
    编码输入文件的一个安全区间，并写入独立 part 文件。

    返回：
        chunk_index
        processed_bytes
        token_count
        part_path
    """
    if _WORKER_TOKENIZER is None:
        raise RuntimeError(
            "Worker tokenizer is not initialized."
        )

    if _WORKER_VOCAB_SIZE is None:
        raise RuntimeError(
            "Worker vocabulary size is not initialized."
        )

    (
        chunk_index,
        input_path,
        start,
        end,
        part_path,
    ) = task

    token_buffer: list[int] = []
    token_count = 0

    text_segments = iter_text_lines_in_range(
        input_path=input_path,
        start=start,
        end=end,
    )

    with open(part_path, "wb") as output_file:
        for token_id in (
            _WORKER_TOKENIZER.encode_iterable(
                text_segments
            )
        ):
            if not (
                0
                <= token_id
                < _WORKER_VOCAB_SIZE
            ):
                raise ValueError(
                    f"Invalid token ID {token_id} "
                    f"in chunk {chunk_index}"
                )

            token_buffer.append(token_id)

            if (
                len(token_buffer)
                >= WRITE_BUFFER_SIZE
            ):
                token_array = np.asarray(
                    token_buffer,
                    dtype=TOKEN_DTYPE,
                )

                token_array.tofile(output_file)

                token_count += token_array.size
                token_buffer.clear()

        if token_buffer:
            token_array = np.asarray(
                token_buffer,
                dtype=TOKEN_DTYPE,
            )

            token_array.tofile(output_file)
            token_count += token_array.size

    return (
        chunk_index,
        end - start,
        token_count,
        part_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a text dataset into a raw token "
            "array using multiple processes."
        )
    )

    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=DEFAULT_NUM_PROCESSES,
    )
    parser.add_argument(
        "--chunks-per-process",
        type=int,
        default=DEFAULT_CHUNKS_PER_PROCESS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_processes <= 0:
        raise ValueError(
            "num_processes must be positive"
        )

    if args.chunks_per_process <= 0:
        raise ValueError(
            "chunks_per_process must be positive"
        )

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    vocab_path = (
        args.tokenizer_dir / "vocab.json"
    )
    merges_path = (
        args.tokenizer_dir / "merges.json"
    )

    if not vocab_path.is_file():
        raise FileNotFoundError(vocab_path)

    if not merges_path.is_file():
        raise FileNotFoundError(merges_path)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. "
            "Use --overwrite to replace it."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 在父进程中加载一次，用于检查词表大小。
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )

    vocab_size = len(tokenizer.vocab)

    if (
        vocab_size
        > np.iinfo(TOKEN_DTYPE).max + 1
    ):
        raise ValueError(
            f"Vocabulary size {vocab_size} "
            f"does not fit in {TOKEN_DTYPE}"
        )

    input_size = args.input.stat().st_size

    desired_num_chunks = (
        args.num_processes
        * args.chunks_per_process
    )

    print(f"input: {args.input}")
    print(
        f"input size: "
        f"{input_size / 1024**2:.2f} MiB"
    )
    print(f"vocabulary size: {vocab_size}")
    print(f"token dtype: {TOKEN_DTYPE}")
    print(
        f"requested processes: "
        f"{args.num_processes}"
    )

    # 找到位于 <|endoftext|> 处的安全边界。
    with args.input.open("rb") as input_file:
        boundaries = find_chunk_boundaries(
            file=input_file,
            desired_num_chunks=desired_num_chunks,
            split_special_tokens=(
                SPECIAL_TOKEN.encode("utf-8")
            ),
        )

    ranges = [
        (start, end)
        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        )
        if start < end
    ]

    if len(ranges) == 0:
        raise ValueError(
            "No non-empty input chunks were found."
        )

    process_count = min(
        args.num_processes,
        len(ranges),
    )

    print(f"safe chunks: {len(ranges)}")
    print(f"worker processes: {process_count}")

    start_time = perf_counter()

    # 临时目录与最终输出位于同一个父目录，
    # os.replace 时不需要跨文件系统复制。
    with tempfile.TemporaryDirectory(
        dir=args.output.parent,
        prefix=".tokenize-parts-",
    ) as temporary_directory:
        temporary_path = Path(
            temporary_directory
        )

        tasks = [
            (
                chunk_index,
                os.fspath(args.input),
                start,
                end,
                os.fspath(
                    temporary_path
                    / f"part-{chunk_index:05d}.bin"
                ),
            )
            for chunk_index, (start, end)
            in enumerate(ranges)
        ]

        start_methods = (
            mp.get_all_start_methods()
        )

        if "fork" in start_methods:
            context = mp.get_context("fork")
        else:
            context = mp.get_context("spawn")

        results: list[
            tuple[int, int, int, str]
        ] = []

        completed_bytes = 0
        completed_tokens = 0

        with context.Pool(
            processes=process_count,
            initializer=initialize_worker,
            initargs=(
                os.fspath(vocab_path),
                os.fspath(merges_path),
                (SPECIAL_TOKEN,),
                vocab_size,
            ),
        ) as pool:
            partial_results = pool.imap_unordered(
                encode_file_range,
                tasks,
                chunksize=1,
            )

            for result in partial_results:
                (
                    chunk_index,
                    processed_bytes,
                    token_count,
                    _part_path,
                ) = result

                results.append(result)
                completed_bytes += processed_bytes
                completed_tokens += token_count

                elapsed = (
                    perf_counter() - start_time
                )

                token_rate = (
                    completed_tokens / elapsed
                )

                print(
                    f"[{len(results):3d}/"
                    f"{len(tasks):3d}] "
                    f"chunk={chunk_index:3d} "
                    f"tokens={completed_tokens:,} "
                    f"rate={token_rate:,.0f} token/s"
                )

        if completed_bytes != input_size:
            raise RuntimeError(
                "Chunks did not cover the entire input: "
                f"processed {completed_bytes} bytes, "
                f"expected {input_size} bytes"
            )

        # imap_unordered 的结果顺序不固定，
        # 合并前必须按照 chunk_index 排序。
        results.sort(
            key=lambda result: result[0]
        )

        merged_path = (
            temporary_path / "merged.bin"
        )

        with merged_path.open(
            "wb"
        ) as merged_file:
            for (
                _chunk_index,
                _processed_bytes,
                token_count,
                part_path_string,
            ) in results:
                part_path = Path(
                    part_path_string
                )

                expected_part_size = (
                    token_count
                    * TOKEN_DTYPE.itemsize
                )

                actual_part_size = (
                    part_path.stat().st_size
                )

                if (
                    actual_part_size
                    != expected_part_size
                ):
                    raise RuntimeError(
                        f"Invalid part size for "
                        f"{part_path}: "
                        f"expected "
                        f"{expected_part_size}, "
                        f"got {actual_part_size}"
                    )

                with part_path.open(
                    "rb"
                ) as part_file:
                    shutil.copyfileobj(
                        part_file,
                        merged_file,
                        length=COPY_BUFFER_SIZE,
                    )

        expected_output_size = (
            completed_tokens
            * TOKEN_DTYPE.itemsize
        )

        actual_output_size = (
            merged_path.stat().st_size
        )

        if (
            actual_output_size
            != expected_output_size
        ):
            raise RuntimeError(
                "Merged output has incorrect size: "
                f"expected {expected_output_size}, "
                f"got {actual_output_size}"
            )

        # 只有所有编码和检查均成功后，
        # 才将临时结果移动为最终输出。
        os.replace(
            merged_path,
            args.output,
        )

    elapsed = perf_counter() - start_time

    print("\nEncoding complete")
    print(f"output: {args.output}")
    print(f"tokens: {completed_tokens:,}")
    print(
        f"output size: "
        f"{args.output.stat().st_size / 1024**2:.2f} MiB"
    )
    print(f"elapsed: {elapsed:.2f} seconds")
    print(
        f"average rate: "
        f"{completed_tokens / elapsed:,.0f} token/s"
    )
    print(
        f"input throughput: "
        f"{input_size / elapsed / 1024**2:.2f} MiB/s"
    )


if __name__ == "__main__":
    main()
