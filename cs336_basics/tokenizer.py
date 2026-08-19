from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = regex.compile(PAT)


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

    # 这个是之前写过的函数
    @staticmethod
    def _merge_pair(
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        从左到右、不重叠地合并指定 pair。
        """

        result: list[bytes] = []
        index = 0

        while index < len(tokens):
            if (
                index + 1 < len(tokens)
                and tokens[index] == pair[0]
                and tokens[index + 1]
                == pair[1]
            ):
                result.append(
                    tokens[index]
                    + tokens[index + 1]
                )
                index += 2
            else:
                result.append(tokens[index])
                index += 1

        return tuple(result)

    # 执行merge的核心函数
    def _apply_bpe(
        self,
        pretoken_bytes: bytes,            
    ) -> tuple[bytes,...]:
        """
        对一个预词元应用已有的 BPE merges。

        编码时选择
        merges 中创建时间最早、rank 最小的可用 pair。
        """
        tokens = tuple(
            self.one_byte_tokens[byte_value]
            for byte_value in pretoken_bytes
        )

        # 下面是一个非常美丽的单token，merge循环，最后返还merge完后的token

        while len(tokens) >= 2:
            best_pair: (
                tuple[bytes,bytes] | None
            ) = None

            best_rank = len(self.merges)

            # 这里最巧妙的地方在于取了所有可能merge的组合一个个查，然后比较得到最高优先级的那个，如果没有那么就没有（不重不漏）
            for pair in zip (
                tokens,
                tokens[1:],
            ):
                rank = self.merge_ranks.get(pair)

                if(
                    rank is not None
                    and rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            # 如果无法合并，则break
            if best_pair is None:
                break

            # else

            tokens = self._merge_pair(
                tokens,
                best_pair,
            )
        return tokens

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

            bpe_tokens = self._apply_bpe(
                pretoken_bytes
            )

            for token in bpe_tokens:
                yield self.token_to_id[token]

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