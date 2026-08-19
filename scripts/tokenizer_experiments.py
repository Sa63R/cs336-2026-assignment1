from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPECIAL_TOKEN = "<|endoftext|>"
READ_CHUNK_SIZE = 1024 * 1024
SAMPLE_SIZE = 10
RANDOM_SEED = 336


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
    tokenizer: Tokenizer,
    input_path: Path,
) -> tuple[float, float]:
    """
    流式编码完整文件，测量端到端吞吐量，
    并估算处理 825 GB 文本需要的时间。

    返回：
        throughput_bytes_per_second
        estimated_days
    """
    input_size_bytes = (
        input_path.stat().st_size
    )

    # 先运行一个很小的编码，减少首次调用带来的
    # 初始化和缓存开销对 benchmark 的影响。
    tokenizer.encode(
        "Tokenizer benchmark warmup."
        + SPECIAL_TOKEN
    )

    print(f"\n{label}")
    print(f"  输入文件：{input_path}")
    print(
        "  输入大小："
        f"{input_size_bytes / 1024**2:.2f} MiB"
    )
    print(
        "  正在进行流式编码，"
        "这个过程可能需要几分钟……"
    )

    total_tokens = 0

    start_time = perf_counter()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for _token_id in (
            tokenizer.encode_iterable(file)
        ):
            total_tokens += 1

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
        tokenizer=owt_tokenizer,
        input_path=(
            PROJECT_ROOT
            / "data/owt_valid.txt"
        ),
    )


if __name__ == "__main__":
    main()