import math 

import torch
from torch import nn

from cs336_basics.nn_utils import softmax

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2.0 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    # 数学上等价于x @ self.weight.T
    # 两种写法等价。einsum 更明确地表达了张量维度：
    # x:      (..., in_features)
    # weight: (out_features, in_features)
    # output: (..., out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)
    


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # 真正调用的时候你可能这样写：
        # embedding = Embedding(
        #     num_embeddings=50000,
        #     embedding_dim=768,
        #   )
        # 其中最重要的是前两个。

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            x.square().mean(dim=-1,keepdim = True) + self.eps
        )

        result =  (x / rms) * self.weight

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # 建立三个线性W矩阵，很好理解
        self.w1 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.w3 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        # FFN(𝑥) = SwiGLU(𝑥, 𝑊1 , 𝑊2 , 𝑊3 ) = 𝑊2 (SiLU(𝑊1 𝑥) ⊙ 𝑊3 𝑥). (7)
        
        # 我感觉这一块重点是他是如何帮助梯度/残差传播的，这个之后再研究
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)

        hidden = gate * value

        return self.w2(hidden)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        # 这里的theta就是RoPE里的超参数
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )

        inverse_frequencies = theta ** (
            -dimension_indices / d_k
        )


        # 这里本质就是把每一个位置的token乘上对应的维度theta
        # 得到对应的旋转向量，注意两个为一组是除了2的，得到每一组的角度！！
        # 例如：
        # positions = [0, 1, 2]                  shape: (3,)
        # inverse_frequencies = [1.0, 0.1]       shape: (2,)
        #
        # positions[:, None]
        # = [[0],
        #    [1],
        #    [2]]                                shape: (3, 1)
        #
        # inverse_frequencies[None, :]
        # = [[1.0, 0.1]]                         shape: (1, 2)
        #
        # 广播相乘后：
        # angles =
        # [[0*1.0, 0*0.1],
        #  [1*1.0, 1*0.1],
        #  [2*1.0, 2*0.1]]
        # =
        # [[0.0, 0.0],
        #  [1.0, 0.1],
        #  [2.0, 0.2]]                           shape: (3, 2)
        angles = (
            positions[:, None]
            * inverse_frequencies[None, :]
        )
        self.register_buffer(
            "cos_angles",
            torch.cos(angles),# 这里要用torch估计就是因为只有torch才实现了cos的张量实现
            persistent=False,
        )
        self.register_buffer(
            "sin_angles",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        # 这个的大体形状也可以参考前面的angles，也就是[...,tokens,d]
        cos = self.cos_angles[token_positions].to(dtype=x.dtype)
        sin = self.sin_angles[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 仔细看维度，这里的点积确实是对齐的，tokens对tokens，
        # 并且tokens奇偶分开了，cos和sin也已经分开了，或者说角度同组一样
        # 只是取cos则为偶，取sin则为奇数组
        # 最后算出来是[....seq_len,d/2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # 恢复维度：合并，展平
        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:

    # 也就是维度
    d_k = Q.shape[-1]

    attention_scores = torch.einsum(
        "...qd,...kd->...qk",
        Q,
        K,
    )

    attention_scores = attention_scores / math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = softmax(
        attention_scores,
        dim=-1,
    )
    return torch.einsum(
        "...qk,...kv->...qv",
        attention_weights,
        V,
    )


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        if (theta is None) != (max_seq_len is None):
            raise ValueError(
                "theta and max_seq_len must be provided together"
            )
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )


    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None,
        ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.unflatten(
            -1,
            (self.num_heads, self.d_head),
        ).transpose(-3, -2)

        k = k.unflatten(
            -1,
            (self.num_heads, self.d_head),
        ).transpose(-3, -2)

        v = v.unflatten(
            -1,
            (self.num_heads, self.d_head),
        ).transpose(-3, -2)

        sequence_length = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )

            rope_positions = token_positions.unsqueeze(-2)

            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )

        context = scaled_dot_product_attention(
            Q=q,
            K=k,
            V=v,
            mask=causal_mask,
        )

        context = context.transpose(-3, -2).flatten(-2)
        
        return self.output_proj(context)