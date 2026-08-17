from __future__ import annotations
import os
from collections import Counter,defaultdict
import heapq
import multiprocessing as mp
from typing import BinaryIO

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)

class _HeapItem:
    """让heapq按照频率最大、pair 字典序最大的顺序返回元素。"""

    __slots__ = ("frequency","pair")
    def __init__(
        self,
        frequency: int,
        pair: tuple[bytes,bytes],
    )-> None:
        self.frequency = frequency
        self.pair = pair

    def __lt__(self, other:_HeapItem)-> bool:
        return (self.frequency,self.pair)>(
            other.frequency,
            other.pair,
        )


        
        

def merge_pair(
        tokens: tuple[bytes,...],
        pair: tuple[bytes,bytes],
)->tuple[bytes,...]:
    result = []
    i = 0
    while i < len(tokens):
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

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_tokens: bytes,
)->list[int]:
    """
    先把一个大文件粗略地等分成若干块，
    然后把每个“粗略分界点”向后移动到最近的特殊 token（例如 <|endoftext|>）的位置，
    从而得到适合多进程处理的安全边界。
    """
    assert isinstance(split_special_tokens,bytes,)

    # 计算文件大小file_size
    file.seek(0,os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks+1)
    ]
    boundaries[-1] = file_size

    mini_chunk_size = 4096

    for boundary_index in range(
        1,
        len(boundaries) - 1,
    ):
        position = boundaries[boundary_index]
        file.seek(position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 如果最后读到空了，那么就设置当前分块边界为文件的末尾
            if mini_chunk == b"":
                boundaries[boundary_index] = file_size
                break

            # 在切分的4096块里寻找special token
            found_at = mini_chunk.find(
                split_special_tokens
            )

            if found_at != -1: # 找到了
                boundaries[boundary_index] = position + found_at
                break

            position = position + mini_chunk_size

    return sorted(set(boundaries))

def _compile_special_pattern(
    special_tokens: tuple[str,...],
)-> regex.Pattern | None:
    """构造特殊词元的正则表达式"""
    if not special_tokens:
        return None

    alternatives = "|".join(
        regex.escape(token)
        for token in sorted(
            special_tokens,
            key=len,
            reverse=True,   
        )
    )
    return regex.compile(alternatives)

def _count_ordinary_text(
        text: str,
        counts: Counter[bytes],
        start: int = 0,
        end: int | None = None,
)->None:
    """对指定文本范围执行预分词并统计频率。"""

    for match in PRETOKEN_RE.finditer(
        text,
        start,
        end,
    ):
        pretoken_bytes = (
            match.group().encode("utf-8")
        )
        counts[pretoken_bytes] += 1


# 这里的思想大概是一层层处理，一个处理外层分块，一个函数处理有分词token的快内，再一个函数处理最普通的部分。

# 下面这部分是用来处理一个分块内部的文本

def _count_pretokens_chunk(
        task: tuple[
            str,
            int,
            int,
            tuple[str,...],
        ],
)->Counter[bytes]:
    """
    多进程工作函数。

    每个工作进程自行读取指定文件区间，
    """

    # 解包元组

    (
        input_path,
        start,
        end,
        special_tokens,
    ) = task

    with open(input_path,"rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    text = chunk_bytes.decode("utf-8")
    counts: Counter[bytes] = Counter()

    special_pattern = _compile_special_pattern(
        special_tokens
    )

    if special_pattern is None:
        _count_ordinary_text(
            text,counts
        )
        return counts

    # else

    cursor = 0

    for special_match in special_pattern.finditer(
        text
    ):
    # 手动过虑这一段下的特殊分词符
        _count_ordinary_text(
            text,
            counts,
            cursor,
            special_match.start(),
        )
        cursor = special_match.end()
    _count_ordinary_text(
        text,
        counts,
        cursor,
        len(text),# 文本长度，表示这一段结尾部分的cursor坐标
    )
    # 返回一个计数完这一段的counts统计值
    return counts

def _count_pretokens(
        input_path: str | os.PathLike,
        special_tokens: tuple[str,...],
        num_processes: int,
        parallel_min_bytes: int,
        num_chunks:int,
) -> Counter[bytes]:
    """
    根据文件大小选择串行或并行预分词。
    为处理输入文件的第一步
    """
    path = os.fspath(input_path)
    file_size = os.path.getsize(path)

    if (
        num_processes <= 1
        or file_size < parallel_min_bytes
        or not special_tokens
    ):
        return _count_pretokens_chunk(
            (
            path,
            0,
            file_size,
            special_tokens,
            )
        )

    # 上面是不开多线程，现在开。
    split_token = max(
        special_tokens,
        key=len
    ).encode("utf-8")

    with open(path,"rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            num_chunks,
            split_token,
        )

    # 划分多线程任务
    tasks = [
        (
            path,
            start,
            end,
            special_tokens,
        )
        for start,end in zip(
            boundaries[:-1],
            boundaries[1:],
        )
        if start < end
    ]

    total_counts: Counter[bytes] = Counter()

    context = mp.get_context("fork")

    process_count = min(
        num_processes,
        len(tasks),
    )

    # 类似于打开文件，构建一个线程池

    with context.Pool(
        processes=process_count
    ) as pool:
        partial_results = pool.imap_unordered(
            _count_pretokens_chunk,
            tasks,# 这里是接受一个迭代器/可迭代对象？作为输入参数，非常合适
            chunksize=1,
        )
        for partial_counts in partial_results:
            total_counts.update(partial_counts)

    return total_counts


def _adjacent_pair_counts(
    tokens: tuple[bytes,...],
) -> Counter[tuple[bytes,bytes]]:
    
    return Counter(
        zip(tokens,tokens[1:])
    )

def _pop_best_pair(
    heap: list[_HeapItem],
    pair_counts: Counter[
        tuple[bytes,bytes]
    ],
) -> tuple[bytes,bytes] | None:
    """
    取出当前最高频 pair。

    堆中可能保留旧频率记录，因此需要检查记录是否仍然有效。
    
    """

    while heap:
        item = heapq.heappop(heap)

        current_frequency = pair_counts.get(
            item.pair,
            0,
        )
        
        if (
            item.frequency > 0
            and item.frequency
            == current_frequency
        ):
            return item.pair
    return None


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) ->tuple[
    dict[int,bytes],
    list[tuple[bytes,bytes]],# 返回一个词表和一个合并过程list
]:
    """
    训练字节级 BPE 分词器。

    可选参数：
        num_processes:
            预分词使用的进程数，默认最多为 8。

        parallel_min_bytes:
            文件达到多少字节后启用多进程，
            默认值为 8 MiB。

        num_chunks:
            文件任务分块数量，
            默认值是进程数的 4 倍。
    
    """
    # 把list转元组 
    special_tokens_tuple = tuple(
        special_tokens
    )
    vocab: dict[int,bytes] = {
        byte_value:bytes([byte_value])
        for byte_value in range(256)
    }

    existing_tokens = set(vocab.values())

    for special_token in special_tokens_tuple:
        encoded_special = (
            special_token.encode("utf-8")
        )

        if encoded_special not in existing_tokens:
            vocab[len(vocab)] = encoded_special
            existing_tokens.add(encoded_special)

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller "
            f"than the initial vocabulary size "
            f"{len(vocab)}"
        )
    default_processes = min(
        8,
        os.cpu_count()
    )

    num_processes = max(
        1,
        int(
            kwargs.get(
                "num_processes",
                default_processes,
            )
        )
    )
    parallel_min_bytes = int(
        kwargs.get(
            "parallel_min_bytes",
            8 * 1024 * 1024,
        )
    )

    num_chunks = max(
        num_processes,
        int(
            kwargs.get(
                "num_chunks",
                num_processes * 4,
            )
        ),
    )

    # 上面是一些参数

    pretoken_counts = _count_pretokens(
        input_path=input_path,
        special_tokens=special_tokens_tuple,
        num_processes=num_processes,
        parallel_min_bytes=parallel_min_bytes,
        num_chunks=num_chunks,
    )
    # 现在得到了返回的一个counter记录对应的单词和次数

    # 这个有平替，是用来辅助把bytes拆分成多个bytes的，详见笔记
    one_byte_tokens = tuple(
        bytes([byte_value])
        for byte_value in range(256)
    )


    # 把bytes拆分成多个bytes
    # pretoken_counts = {
    #     b"hi": 3,
    #     b"!": 2,
    # }
    # 最终：
    # words = [
    #     (b"h", b"i"),
    #     (b"!",),
    # ]
    words: list[tuple[bytes,...]] = [
        tuple(
            one_byte_tokens[byte_value]
            for byte_value in pretoken
        )
        for pretoken in pretoken_counts
    ]

    # 和上面words一一对应的list，利用pretoken_counts元组的特性来记录数字
    frequencies: list[int] = [
        pretoken_counts[pretoken]
        for pretoken in pretoken_counts
    ]

    # 举例
    # pair_counts = Counter({
    #     (b"l", b"l"): 5,
    #     (b"h", b"e"): 3,
    #     (b"e", b"l"): 3,
    #     (b"l", b"o"): 3,
    # })
    pair_counts: Counter[
        tuple[bytes,bytes]
    ] = Counter()

    # pair_to_word_ids = {
    #     (b"h", b"e"): {0, 3},
    #     (b"e", b"l"): {0},
    #     (b"l", b"l"): {0, 2},
    # }
    #
    # 表示：
    # (b"h", b"e") 这个 pair 出现在编号 0、3 的 word 中
    # (b"e", b"l") 这个 pair 出现在编号 0 的 word 中
    # (b"l", b"l") 这个 pair 出现在编号 0、2 的 word 中
    #

    pair_to_word_ids: dict[
        tuple[bytes,bytes],
        set[int],
    ] = defaultdict(set)

    for word_id,(
        tokens,
        frequency,
        ) in enumerate(
            zip(words, frequencies)
        ):
        local_counts = (
            _adjacent_pair_counts(tokens)# 详细见函数定义ctrl+鼠标左键，返回一个counter
        )
        # 读取这个计数器里的内容
        # 例：
        # 假设当前：
        # tokens = (b"h", b"e", b"l", b"l", b"o")
        # local_counts = Counter({
        #     (b"h", b"e"): 1,
        #     (b"e", b"l"): 1,
        #     (b"l", b"l"): 1,
        #     (b"l", b"o"): 1,
        # })
        for pair, occurrences in (
            local_counts.items()
        ):
            pair_counts[pair] += (
                occurrences * frequency
            )

            pair_to_word_ids[pair].add(
                word_id
            )

    # 以上就有了pair_counts
    # pair_counts = Counter({
    #     (b"l", b"l"): 5,
    #     (b"h", b"e"): 3,
    #     (b"e", b"l"): 3,
    #     (b"l", b"o"): 3,
    # })

    heap: list[_HeapItem] = [
        _HeapItem(frequency, pair)
        for pair, frequency
        in pair_counts.items()
    ]
    # 把pair_counts构建成了一个堆来方便排序和取出

    # 整理堆
    heapq.heapify(heap)

    # 记录每一次合并
    merges: list[
        tuple[bytes,bytes]
    ] = []

    # 准备进入主循环
    while len(vocab) < vocab_size:
        best_pair = _pop_best_pair(
            heap,
            pair_counts,
        )

        if best_pair is None:
            break

        new_token = (
            best_pair[0]+best_pair[1]
        )

        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        affected_word_ids = tuple(
            pair_to_word_ids.get(
                best_pair,
                (),#所以这里括号其实没什么含义
            )
        )# 这里的返回值是一个set

        # 这个应该是最后用来调整计数的
        pair_deltas: Counter[
            tuple[bytes,bytes]
        ] = Counter()

        # word_id是set里取出来的序号，对每一个受影响的word合并然后重算然后排除
        for word_id in affected_word_ids:
            old_tokens = words[word_id]

            # 合并
            new_tokens = merge_pair(
                old_tokens,
                best_pair,
            )

            if new_tokens == old_tokens:
                continue

            frequency = frequencies[word_id]# 对应的次数

            # 例：
            # 假设：
            # old_tokens = (b"a", b"b", b"a")
            #
            # 如果本轮要合并：
            # best_pair = (b"a", b"b")
            #
            # 那么合并后：
            # new_tokens = (b"ab", b"a")
            #
            # 合并前的相邻 pair：
            # (b"a", b"b")
            # (b"b", b"a")
            #
            # 所以：
            # old_local_counts = Counter({
            #     (b"a", b"b"): 1,
            #     (b"b", b"a"): 1,
            # })

            old_local_counts = (
                _adjacent_pair_counts(
                    old_tokens
                )
            )
            # 合并后的相邻 pair：
            # (b"ab", b"a")
            #
            # 所以：
            # new_local_counts = Counter({
            #     (b"ab", b"a"): 1,
            # })
            new_local_counts = (
                _adjacent_pair_counts(
                    new_tokens
                )
            )
            # 分别计数旧的和新的情况的pair

            old_pairs = set(
                old_local_counts
            )
            # counter转成set，只遍历key，所以类似这样
            # old_pairs = {
            #     (b"a", b"b"),
            #     (b"b", b"a"),
            # }
            #

            new_pairs = set(
                new_local_counts
            )

            for pair in (
                old_pairs - new_pairs # 合并之后消失了哪些 pair
            ):
                pair_to_word_ids[
                    pair
                ].discard(word_id) 

                #for word_id in affected_word_ids:
                # word_id是set里取出来的序号
            
            # 同样的，对于这个单词，合并后，也会受新合并token影响

            for pair in (
                new_pairs - old_pairs
            ):
                pair_to_word_ids[
                    pair
                ].add(word_id)

            for pair in (
                old_pairs | new_pairs
            ):
                old_count = (
                    old_local_counts.get(
                        pair,
                        0,
                    )
                )

                new_count = (
                    new_local_counts.get(
                        pair,
                        0,
                    )
                )

                # new_count - old_count 不一定是 ±1，
                # 因为同一个 pair 在一个 word 中可能出现多次。
                # 例：(a,b,a,b) 中 (a,b) 出现 2 次，
                # 合并后变成 (ab,ab)，出现 0 次，
                # 所以 new_count - old_count = 0 - 2 = -2。
                # 不过大意也就是一个计数

                delta = (
                    new_count - old_count
                ) * frequency

                # frequency仅是这个单词出现的次数，还可能会有其他单词同样对
                # counter产生影响，但是那就在下一个循环里体现了

                if delta:
                    pair_deltas[pair] += delta

            words[word_id] = new_tokens # 把合并后的word放进去

        # 循环每一个受影响的word，终于得到了总的

        for pair, delta in(
            pair_deltas.items()
        ):
            updated_frequency = (
                pair_counts.get(pair, 0)
                + delta
            )
            # 即使更新后也不可能小于零
            if updated_frequency < 0:
                raise RuntimeError(
                    "negative pair count for "
                    f"{pair!r}: "
                    f"{updated_frequency}"
                )
            if updated_frequency == 0:
                pair_counts.pop(
                    pair,
                    None,
                )
                if not pair_to_word_ids.get(
                    pair
                ):
                    pair_to_word_ids.pop(
                        pair,
                        None,
                    )
            else:
                pair_counts[pair] = (
                    updated_frequency
                )
                # 这里还有小巧思，因为pair是只要有delta（更新）的，
                # 就会被遍历到，所以heappush就可以把每一个更新后的pair
                # 给重新塞回堆里
                heapq.heappush(
                    heap,
                    _HeapItem(
                        updated_frequency,
                        pair,
                    ),
                )
    return vocab, merges