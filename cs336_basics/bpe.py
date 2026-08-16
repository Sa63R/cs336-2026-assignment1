import os
import regex
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_pair(
        tokens: tuple[bytes,...],
        pair: tuple[bytes,bytes],
)->tuple[bytes,...]:
    result = []
    i = 0
    while i < len(toekns):
        if(
            i<len(tokens)-1
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            result.append(tokens[i]+tokens[i+1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return tuple(result)

        


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])
    for tokens in special_tokens:
        vocab[len(vocab)] = tokens.encode("utf-8")

    # vocabulary is initialized

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 取得了语料text
    if special_tokens:
        special_pattern = "|".join(
            regex.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        chunks = regex.split(special_pattern, text)
    else:
        chunks = [text]

    word_counts = Counter()
    pair_counts = Counter()
    merges = []

    for chunk in chunks:
        for match in regex.finditer(PAT,chunk):
            pretoken = match.group()
            encoded = pretoken.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in encoded) # bytes的本质是一个list[1,2,3,...]
            word_counts[byte_tuple] += 1

    # word_counts是一个Counter对象，key是tuple(bytes([b1]), bytes([b2]), ...)，value是出现次数
    # 下一步开始BPE
    for tokens, freq in word_counts.items():
        # 遍历字典的方法
        # 然后需要遍历取出来的tuple进行计数
        for pair in zip(tokens,tokens[1:]):
            pair_counts[pair] += freq

    # 现在就有了每一个小pair的计数 

    # print(pair_counts.most_common(20)) # 测试当前pre-tokenizer是否正确
    # uv run --locked python -c 'from cs336_basics.bpe import train_bpe; train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])'

    ## 然后是BPEmerge
    best_pair = max(
        pair_counts,
        key=lambda pair:(pair_counts[pair],pair)
    )
    # best_pair 是一个元组

    new_token = best_pair[0]+best_pair[1]
    # 用一个list来记录合并过程
    merges.append(best_pair)

    vocab[len(vocab)] = new_token



    raise NotImplementedError
