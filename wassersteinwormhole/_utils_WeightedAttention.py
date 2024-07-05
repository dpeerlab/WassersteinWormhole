from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax.typing import ArrayLike

from wassersteinwormhole.DefaultConfig import DefaultConfig


def scaled_dot_product(
    q,
    k,
    v,
    weights: Optional[ArrayLike] = None,
    scale_weights: float = 1,
    deterministic: bool = False,
    dropout_rng: Optional[ArrayLike] = random.key(0),
    dropout_rate: float = 0.0,
):

    dtype, d_k = (
        q.dtype,
        q.shape[-1],
    )

    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)

    if weights is not None:
        # attn_logits = attn_logits + jnp.tan(math.pi*(jnp.clip(weights, 1e-7, 1-1e-7)-1/2)) - jnp.tan(math.pi*(1/q.shape[-2]-1/2))
        attn_logits = attn_logits + jnp.log(
            weights / scale_weights + jnp.finfo(jnp.float32).tiny
        )
        attn_logits = jnp.where(weights == 0, -9e15, attn_logits)
        attn_logits = jnp.where(weights == 1, 9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis=-1)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep = random.bernoulli(dropout_rng, keep_prob, attention.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention = attention * multiplier

    values = jnp.matmul(attention, v)
    return values, attention


def expand_weights(weights):
    if weights.ndim == 2:
        weights = weights[:, None, None, :]
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    while weights.ndim < 4:
        weights = weights.unsqueeze(0)
    return weights


class WeightedMultiheadAttention(nn.Module):

    config: DefaultConfig
    scale_weights: Optional[float] = 1

    def setup(self):
        config = self.config
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(
            3 * config.emb_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )

    def __call__(
        self,
        x,
        weights: Optional[ArrayLike] = None,
        deterministic: Optional[bool] = True,
        dropout_rng: Optional[ArrayLike] = random.key(0),
    ):

        config = self.config
        scale_weights = self.scale_weights

        batch_size, seq_length, _ = x.shape

        assert x.shape[-1] == config.emb_dim

        if weights is not None:
            weights = expand_weights(weights)

        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, config.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(
            q,
            k,
            v,
            weights=weights,
            scale_weights=scale_weights,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            dropout_rate=config.attention_dropout_rate,
        )
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, config.emb_dim)

        return values
