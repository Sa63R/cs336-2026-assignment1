from __future__ import annotations

import io
import multiprocessing as mp
import os
import random
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPECIAL_TOKEN = "<|endoftext|>"
READ_CHUNK_SIZE = 1024 * 1024
SAMPLE_SIZE = 10
RANDOM_SEED = 336
BENCHMARK_NUM_PROCESSES = min(
    8,
    os.cpu_count() or 1,
)
BENCHMARK_CHUNKS_PER_PROCESS = 2


_WORKER_TOKENIZER: Tokenizer | None = None


def _initialize_encode_worker(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: tuple[str, ...],
) -> None:
    """每个工作进程启动时只加载一次分词器。"""
    global _WORKER_TOKENIZER

    _WORKER_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=list(special_tokens),
    )
    _WORKER_TOKENIZER.encode(
        "Tokenizer worker warmup."
        + SPECIAL_TOKEN
    )


def _count_tokens_in_file_range(
    task: tuple[int, str, int, int],
) -> tuple[int, int, int]:
    """在工作进程中编码一个安全文件区间并统计词元数。"""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError(
            "工作进程中的 Tokenizer 尚未初始化"
        )

    chunk_index, input_path, start, end = task

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8")
    token_count = sum(
        1
        for _token_id in (
            _WORKER_TOKENIZER.encode_iterable(
                io.StringIO(chunk_text)
            )
        )
    )

    return (
        chunk_index,
        len(chunk_bytes),
        token_count,
    )


def iter_documents(
    input_path: Path,
) -> Iterator[str]:
    """
    流式读取数据文件，并按照 <|endoftext|> 分隔文档。

    每次只读取 1 MiB，避免将整个 OWT 验证集一次性
    读入内存。
    """
    remainder = ""

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        while True:
            chunk = file.read(READ_CHUNK_SIZE)

            if chunk == "":
                break

            remainder += chunk

            pieces = remainder.split(
                SPECIAL_TOKEN
            )

            # 最后一段可能只是半篇文档，
            # 留给下一轮继续拼接。
            remainder = pieces.pop()

            for document in pieces:
                if document.strip():
                    # 将分隔符放回文档末尾，
                    # 使样本保持原始语料格式。
                    yield (
                        document
                        + SPECIAL_TOKEN
                    )

    # 文件最后可能有一篇没有结束标记的文档。
    if remainder.strip():
        yield remainder


def sample_documents(
    input_path: Path,
    count: int,
    seed: int,
) -> list[str]:
    """
    使用蓄水池采样，从整个数据文件中均匀抽取文档。

    固定随机种子，使每次运行得到相同的10篇文档。
    """
    random_generator = random.Random(seed)

    samples: list[str] = []
    documents_seen = 0

    for document in iter_documents(input_path):
        documents_seen += 1

        if len(samples) < count:
            samples.append(document)
            continue

        replacement_index = (
            random_generator.randrange(
                documents_seen
            )
        )

        if replacement_index < count:
            samples[replacement_index] = document

    if len(samples) < count:
        raise ValueError(
            f"{input_path} 中只有 "
            f"{len(samples)} 篇非空文档"
        )

    print(
        f"{input_path.name}: "
        f"从 {documents_seen:,} 篇文档中"
        f"抽取 {count} 篇"
    )

    return samples


def load_tokenizer(
    tokenizer_directory: Path,
) -> Tokenizer:
    """从训练后保存的 JSON 文件加载分词器。"""
    return Tokenizer.from_files(
        vocab_filepath=(
            tokenizer_directory / "vocab.json"
        ),
        merges_filepath=(
            tokenizer_directory / "merges.json"
        ),
        special_tokens=[SPECIAL_TOKEN],
    )


def measure(
    label: str,
    tokenizer: Tokenizer,
    documents: list[str],
) -> tuple[int, float, float]:
    """
    编码一组文档，统计总词元数、压缩率和初步吞吐量。

    返回：
        total_tokens
        compression_ratio
        throughput_bytes_per_second
    """
    start_time = perf_counter()

    encoded_documents = [
        tokenizer.encode(document)
        for document in documents
    ]

    elapsed_seconds = (
        perf_counter() - start_time
    )

    # 确认编码和解码没有破坏文本。
    for document, token_ids in zip(
        documents,
        encoded_documents,
        strict=True,
    ):
        assert (
            tokenizer.decode(token_ids)
            == document
        )

    total_bytes = sum(
        len(document.encode("utf-8"))
        for document in documents
    )

    total_tokens = sum(
        len(token_ids)
        for token_ids in encoded_documents
    )

    compression_ratio = (
        total_bytes / total_tokens
    )

    throughput_bytes_per_second = (
        total_bytes / elapsed_seconds
    )

    print(f"\n{label}")
    print(f"  文档数：{len(documents)}")
    print(
        f"  UTF-8 字节数："
        f"{total_bytes:,}"
    )
    print(f"  词元数：{total_tokens:,}")
    print(
        f"  压缩率："
        f"{compression_ratio:.4f} "
        "字节/词元"
    )
    print(
        f"  编码耗时："
        f"{elapsed_seconds:.6f} 秒"
    )
    print(
        f"  初步吞吐量："
        f"{throughput_bytes_per_second / 1024**2:.2f} "
        "MiB/s"
    )
    print(
        "  第一篇文档的前20个ID："
        f"{encoded_documents[0][:20]}"
    )

    return (
        total_tokens,
        compression_ratio,
        throughput_bytes_per_second,
    )

def benchmark_throughput(
    label: str,
    tokenizer_directory: Path,
    input_path: Path,
    num_processes: int,
) -> tuple[float, float]:
    """
    使用单进程或多进程流式编码完整文件，测量端到端
    吞吐量，并估算处理 825 GB 文本需要的时间。

    返回：
        throughput_bytes_per_second
        estimated_days
    """
    if num_processes <= 0:
        raise ValueError(
            "num_processes 必须是正整数"
        )

    input_size_bytes = (
        input_path.stat().st_size
    )

    print(f"\n{label}")
    print(f"  输入文件：{input_path}")
    print(
        "  输入大小："
        f"{input_size_bytes / 1024**2:.2f} MiB"
    )
    print(f"  编码进程数：{num_processes}")

    if num_processes == 1:
        tokenizer = load_tokenizer(
            tokenizer_directory
        )
        tokenizer.encode(
            "Tokenizer benchmark warmup."
            + SPECIAL_TOKEN
        )

        print("  正在进行单进程流式编码……")
        start_time = perf_counter()
        total_tokens = 0

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for _token_id in (
                tokenizer.encode_iterable(file)
            ):
                total_tokens += 1

        processed_bytes = input_size_bytes
        chunk_count = 1
    else:
        desired_num_chunks = (
            num_processes
            * BENCHMARK_CHUNKS_PER_PROCESS
        )

        with input_path.open("rb") as file:
            boundaries = find_chunk_boundaries(
                file=file,
                desired_num_chunks=(
                    desired_num_chunks
                ),
                split_special_tokens=(
                    SPECIAL_TOKEN.encode("utf-8")
                ),
            )

        tasks = [
            (
                chunk_index,
                os.fspath(input_path),
                start,
                end,
            )
            for chunk_index, (start, end)
            in enumerate(
                zip(
                    boundaries[:-1],
                    boundaries[1:],
                )
            )
            if start < end
        ]

        process_count = min(
            num_processes,
            len(tasks),
        )
        chunk_count = len(tasks)

        print(
            f"  安全文本块数：{chunk_count}"
        )
        print("  正在进行多进程流式编码……")

        context = mp.get_context("fork")
        start_time = perf_counter()

        with context.Pool(
            processes=process_count,
            initializer=_initialize_encode_worker,
            initargs=(
                os.fspath(
                    tokenizer_directory
                    / "vocab.json"
                ),
                os.fspath(
                    tokenizer_directory
                    / "merges.json"
                ),
                (SPECIAL_TOKEN,),
            ),
        ) as pool:
            partial_results = pool.imap(
                _count_tokens_in_file_range,
                tasks,
                chunksize=1,
            )
            ordered_results = list(
                partial_results
            )

        processed_bytes = sum(
            chunk_bytes
            for (
                _chunk_index,
                chunk_bytes,
                _token_count,
            ) in ordered_results
        )
        total_tokens = sum(
            token_count
            for (
                _chunk_index,
                _chunk_bytes,
                token_count,
            ) in ordered_results
        )

        if processed_bytes != input_size_bytes:
            raise RuntimeError(
                "多进程切块没有完整覆盖输入文件："
                f"处理了 {processed_bytes} 字节，"
                f"文件共有 {input_size_bytes} 字节"
            )

    elapsed_seconds = (
        perf_counter() - start_time
    )

    throughput_bytes_per_second = (
        input_size_bytes / elapsed_seconds
    )

    throughput_mib_per_second = (
        throughput_bytes_per_second
        / 1024**2
    )

    # 题目中的 GB 按十进制计算：
    # 1 GB = 1,000,000,000 bytes。
    pile_size_bytes = (
        825 * 1_000_000_000
    )

    estimated_seconds = (
        pile_size_bytes
        / throughput_bytes_per_second
    )

    estimated_hours = (
        estimated_seconds / 3600
    )

    estimated_days = (
        estimated_seconds / 86400
    )

    print(f"  产生词元数：{total_tokens:,}")
    print(f"  实际文本块数：{chunk_count}")
    print(
        f"  实际耗时："
        f"{elapsed_seconds:.2f} 秒"
    )
    print(
        "  吞吐量："
        f"{throughput_bytes_per_second:,.2f} "
        "字节/秒"
    )
    print(
        "  吞吐量："
        f"{throughput_mib_per_second:.2f} "
        "MiB/s"
    )
    print("\n  Pile 825 GB 时间估算：")
    print(
        f"  秒：{estimated_seconds:,.2f}"
    )
    print(
        f"  小时：{estimated_hours:.2f}"
    )
    print(
        f"  天：{estimated_days:.2f}"
    )

    return (
        throughput_bytes_per_second,
        estimated_days,
    )


def main() -> None:
    tiny_tokenizer = load_tokenizer(
        PROJECT_ROOT
        / "artifacts/tokenizers/tinystories"
    )

    owt_tokenizer = load_tokenizer(
        PROJECT_ROOT
        / "artifacts/tokenizers/owt"
    )

    tiny_documents = sample_documents(
        input_path=(
            PROJECT_ROOT
            / "data/TinyStoriesV2-GPT4-valid.txt"
        ),
        count=SAMPLE_SIZE,
        seed=RANDOM_SEED,
    )

    owt_documents = sample_documents(
        input_path=(
            PROJECT_ROOT
            / "data/owt_valid.txt"
        ),
        count=SAMPLE_SIZE,
        seed=RANDOM_SEED,
    )

    # 题目 (a)：各自领域的数据使用各自的分词器。
    measure(
        label=(
            "TinyStories 文档 / "
            "TinyStories 分词器"
        ),
        tokenizer=tiny_tokenizer,
        documents=tiny_documents,
    )

    (
        owt_token_count,
        owt_compression_ratio,
        _,
    ) = measure(
        label=(
            "OpenWebText 文档 / "
            "OpenWebText 分词器"
        ),
        tokenizer=owt_tokenizer,
        documents=owt_documents,
    )

    # 题目 (b)：使用 TinyStories 分词器
    # 编码完全相同的 OpenWebText 样本。
    (
        tiny_on_owt_token_count,
        tiny_on_owt_compression_ratio,
        _,
    ) = measure(
        label=(
            "OpenWebText 文档 / "
            "TinyStories 分词器"
        ),
        tokenizer=tiny_tokenizer,
        documents=owt_documents,
    )

    extra_token_percentage = (
        (
            tiny_on_owt_token_count
            / owt_token_count
        )
        - 1
    ) * 100

    compression_decrease_percentage = (
        1
        - (
            tiny_on_owt_compression_ratio
            / owt_compression_ratio
        )
    ) * 100

    print("\n题目 (b) 对比")
    print(
        "  TinyStories 分词器在 OWT 上"
        f"多产生了 {extra_token_percentage:.2f}% "
        "的词元"
    )
    print(
        "  压缩率下降了 "
        f"{compression_decrease_percentage:.2f}%"
    )
    # 题目 (c)：在较大的 OWT 验证集上
    # 测量流式编码吞吐量。
    benchmark_throughput(
        label=(
            "题目 (c)："
            "OpenWebText 分词器吞吐量"
        ),
        tokenizer_directory=(
            PROJECT_ROOT
            / "artifacts/tokenizers/owt"
        ),
        input_path=(
            PROJECT_ROOT
            / "data/owt_valid.txt"
        ),
        num_processes=(
            BENCHMARK_NUM_PROCESSES
        ),
    )


if __name__ == "__main__":
    main()
