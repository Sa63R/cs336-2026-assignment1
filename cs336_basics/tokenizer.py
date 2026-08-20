from __future__ import annotations

import heapq
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)
PRETOKEN_CACHE_MAX_SIZE = 100_000


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int,bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    )-> None:
        # 初始化分词器，手动实现

        self.vocab = dict(vocab)
        self.merges = list(merges)

        # encode 需要执行 bytes → ID 的反向查询。

        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes
            in self.vocab.items()
        }

        self.merge_ranks = {
            pair:rank
            for rank, pair
            in enumerate(self.merges)
        }

        self.one_byte_tokens = tuple(
            bytes([byte_value])
            for byte_value in range(256)
        )

        # 同一个预词元通常会在语料中重复出现很多次。缓存其最终
        # token ID 序列，可以跳过重复的 BPE 合并和 bytes -> ID 查询。
        # 使用有界 LRU 缓存，避免处理大型语料时内存无限增长。
        self._pretoken_id_cache: OrderedDict[
            bytes,
            tuple[int, ...],
        ] = OrderedDict()

        self.special_tokens = list(
            special_tokens or []
        )

        self.special_token_to_id: dict[
            str,
            int,
        ] = {}

        next_token_id = (
            max(self.vocab,default = -1) + 1
        )


        for special_token in self.special_tokens:
            if special_token == "":
                raise ValueError(
                    "special tokens must not be empty"
                )
            special_bytes = special_token.encode(
                "utf-8"
            )
            token_id = self.token_to_id.get(
                special_bytes
            )

            # 特殊词元不在词表时，追加一个新 ID。
            if token_id is None:
                token_id = next_token_id
                next_token_id += 1

            self.vocab[token_id] = (
                special_bytes
            )

            self.token_to_id[
                special_bytes
            ] = token_id


            # 同样，普通的有token到idspecial_token_to_id也要有
            self.special_token_to_id[
                special_token
            ] = token_id

        # 长特殊词元必须排在短特殊词元前面。
        # 例如先匹配两个连续的 <|endoftext|>，
        # 再匹配单个 <|endoftext|>。

        # 类似之前的一个内容，得到一个special)tokens的str"asdfsfdsafd|asfdaasfc|aadsfdsa|dsfa|fad|fsf|(从长到短)"
        if self.special_tokens:
            alternatives = "|".join(
                regex.escape(token)
                for token in sorted(
                    set(self.special_tokens),
                    key=len,
                    reverse=True,
                )
            )

            self.special_pattern = regex.compile(
                alternatives
            )
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        从我们之前保存的十六进制 JSON 文件恢复 Tokenizer。
        """

        with open(
            vocab_filepath,
            encoding="utf-8",
        ) as file:
            serialized_vocab = json.load(file)

        vocab = {
            int(token_id): bytes.fromhex(
                token_hex
            )
            for token_id, token_hex
            in serialized_vocab.items()
        }

        with open(
            merges_filepath,
            encoding="utf-8",
        ) as file:
            serialized_merges = json.load(file)

        merges = [
            (
                bytes.fromhex(left_hex),
                bytes.fromhex(right_hex),
            )
            for left_hex, right_hex
            in serialized_merges
        ]

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _apply_bpe(
        self,
        pretoken_bytes: bytes,
    ) -> tuple[bytes, ...]:
        """
        对一个预词元应用已有的 BPE merges。

        使用最小堆选择 rank 最小的相邻 pair，并用双向链表
        维护仍然存活的 token。每次合并后只更新左右邻居，
        不再反复扫描和重建整个 token 序列。
        """
        tokens = [
            self.one_byte_tokens[byte_value]
            for byte_value in pretoken_bytes
        ]

        token_count = len(tokens)
        if token_count < 2:
            return tuple(tokens)

        previous_indices = [
            index - 1
            for index in range(token_count)
        ]
        next_indices = [
            index + 1
            if index + 1 < token_count
            else -1
            for index in range(token_count)
        ]
        alive = [True] * token_count

        # heap 元素为 (merge rank, 左 token 下标, 右 token 下标)。
        # 下标同时让同一 pair 的重叠出现按从左到右处理。
        merge_heap: list[
            tuple[int, int, int]
        ] = []

        def push_candidate(
            left_index: int,
        ) -> None:
            if left_index == -1:
                return

            right_index = next_indices[left_index]
            if right_index == -1:
                return

            rank = self.merge_ranks.get(
                (
                    tokens[left_index],
                    tokens[right_index],
                )
            )

            if rank is not None:
                heapq.heappush(
                    merge_heap,
                    (
                        rank,
                        left_index,
                        right_index,
                    ),
                )

        for left_index in range(
            token_count - 1
        ):
            push_candidate(left_index)

        while merge_heap:
            selected_rank = merge_heap[0][0]
            selected_occurrences: list[
                tuple[int, int, int]
            ] = []

            # 旧实现会在一轮中从左到右合并当前最佳 pair
            # 的全部非重叠出现，因此先取出这一 rank 的已有项，
            # 再统一处理。合并中新产生的候选留到下一轮。
            while (
                merge_heap
                and merge_heap[0][0]
                == selected_rank
            ):
                selected_occurrences.append(
                    heapq.heappop(merge_heap)
                )

            for (
                rank,
                left_index,
                right_index,
            ) in selected_occurrences:
                if (
                    not alive[left_index]
                    or not alive[right_index]
                    or next_indices[left_index]
                    != right_index
                ):
                    continue

                current_rank = (
                    self.merge_ranks.get(
                        (
                            tokens[left_index],
                            tokens[right_index],
                        )
                    )
                )
                if current_rank != rank:
                    continue

                tokens[left_index] = (
                    tokens[left_index]
                    + tokens[right_index]
                )
                alive[right_index] = False

                next_index = next_indices[
                    right_index
                ]
                next_indices[left_index] = (
                    next_index
                )

                if next_index != -1:
                    previous_indices[next_index] = (
                        left_index
                    )

                # 只有合并结果左右两侧的相邻 pair 发生变化。
                push_candidate(
                    previous_indices[left_index]
                )
                push_candidate(left_index)

        result: list[bytes] = []
        token_index = 0

        while token_index != -1:
            result.append(tokens[token_index])
            token_index = next_indices[token_index]

        return tuple(result)

    def _encode_pretoken(
        self,
        pretoken_bytes: bytes,
    ) -> tuple[int, ...]:
        """将一个预词元编码为 ID，并缓存重复预词元的结果。"""

        cached_token_ids = (
            self._pretoken_id_cache.get(
                pretoken_bytes
            )
        )

        if cached_token_ids is not None:
            # 最近访问的条目移到末尾；缓存满时优先淘汰最久
            # 没有使用的条目。
            self._pretoken_id_cache.move_to_end(
                pretoken_bytes
            )
            return cached_token_ids

        bpe_tokens = self._apply_bpe(
            pretoken_bytes
        )
        token_ids = tuple(
            self.token_to_id[token]
            for token in bpe_tokens
        )

        self._pretoken_id_cache[
            pretoken_bytes
        ] = token_ids

        if (
            len(self._pretoken_id_cache)
            > PRETOKEN_CACHE_MAX_SIZE
        ):
            self._pretoken_id_cache.popitem(
                last=False
            )

        return token_ids

    def _encode_ordinary_text(
        self,
        text: str,
    ) -> Iterator[int]:
        """
        对不含特殊词元的普通文本进行编码。
        """

        for match in PRETOKEN_RE.finditer(text):
            pretoken_bytes = (
                match.group().encode("utf-8")
            )
            yield from self._encode_pretoken(
                pretoken_bytes
            )

    def _encode_text(
        self,
        text: str,
    ) -> Iterator[int]:
        """
        处理一段可能包含特殊词元的文本。
        """
        if self.special_pattern is None:
            yield from self._encode_ordinary_text(
                text
            )
            return

        cursor = 0
        for match in self.special_pattern.finditer(
            text
        ):
            # 特殊词元之前的普通文本。
            ordinary_text = text[
                cursor:match.start()
            ]
            yield from self._encode_ordinary_text(
                ordinary_text
            )
            # 特殊词元本身直接输出一个 ID，
            # 不参与普通预分词和 BPE 合并。
            yield self.special_token_to_id[
                match.group()
            ]
            cursor = match.end()

        # 最后一个特殊词元之后的普通文本。
        yield from self._encode_ordinary_text(
            text[cursor:]
        )

        # 到这里大概理解了为什么不能用之前类似的方法来处理预分词，因为这里的special tokens也是要编码的

        # 返回的是迭代器iterator对象，比较特殊，详见笔记，或者python语法
        
    # 把迭代器转为list
    def encode(
        self,
        text: str,
    ) -> list[int]:
        """将完整字符串编码成词元 ID 列表。"""

        return list(self._encode_text(text))

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        惰性地编码字符串可迭代对象。

        文件句柄每次提供一行，因此不会一次把完整文件
        读入内存。
        """

        for text in iterable:
            yield from self._encode_text(text)


    # 这里的目标应该是先判断文件大小，然后根据大小来决定用哪个encode

    def decode(
        self,
        ids: list[int],
    ) -> str:
        """
        将词元 ID 解码成字符串。

        如果拼接后的 bytes 不是合法 UTF-8，
        errors='replace' 会插入 U+FFFD 替换字符。
        """
        # bytes
        combined_bytes = b"".join(
            (
                self.vocab[token_id]
                for token_id in ids
            )
        )
        
        return combined_bytes.decode(
            "utf-8",
            errors="replace",
        )
