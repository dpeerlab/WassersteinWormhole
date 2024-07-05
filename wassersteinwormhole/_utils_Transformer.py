from typing import Any, Optional

import jax.numpy as jnp  # type: ignore
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state  # type: ignore
from jax import random

from wassersteinwormhole._utils_WeightedAttention import WeightedMultiheadAttention
from wassersteinwormhole.DefaultConfig import DefaultConfig


class Embedding(nn.Module):
    """Transformer embedding block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        output = nn.Dense(
            config.emb_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        return output


class Unembedding(nn.Module):
    """Transformer embedding block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig
    inp_dim: int

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        output = nn.Dense(
            self.inp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        return output


class Multiplyer(nn.Module):
    """Encoding multiplyer block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig
    out_seq_len: int

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        output = nn.Dense(
            config.emb_dim * self.out_seq_len,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        return output.reshape([inputs.shape[0], self.out_seq_len, config.emb_dim])


class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.relu(x)
        output = (
            nn.Dense(
                inputs.shape[-1],
                dtype=config.dtype,
                kernel_init=config.kernel_init,
                bias_init=config.bias_init,
            )(x)
            + inputs
        )
        return output


class EncoderBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig
    scale_weights: Optional[float] = 1

    @nn.compact
    def __call__(self, inputs, weights, deterministic, dropout_rng):

        config = self.config
        scale_weights = self.scale_weights
        # Attention block.

        x = (
            WeightedMultiheadAttention(config, scale_weights)(
                x=inputs,
                weights=weights,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
            )
            + inputs
        )

        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = FeedForward(config=config)(x)
        output = nn.LayerNorm(dtype=config.dtype)(x)
        return output


class DecoderBlock(nn.Module):
    """Transformer decoder layer.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs, deterministic, dropout_rng):
        config = self.config

        # Attention block.
        x = (
            WeightedMultiheadAttention(config)(
                x=inputs, deterministic=deterministic, dropout_rng=dropout_rng
            )
            + inputs
        )

        # x = nn.Dropout(rate=config.attention_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = FeedForward(config=config)(x)
        output = nn.LayerNorm(dtype=config.dtype)(x)
        return output


class EncoderModel(nn.Module):
    """Transformer encoder network.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig
    scale_weights: Optional[float] = 1

    @nn.compact
    def __call__(self, inputs, weights, deterministic, dropout_rng=random.key(0)):

        config = self.config
        scale_weights = self.scale_weights

        x = inputs  # .astype('int32')
        x = Embedding(config)(x)

        for _ in range(config.num_layers):
            x = EncoderBlock(config, scale_weights)(
                inputs=x,
                weights=weights,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
            )

        x = jnp.sum(x * weights[:, :, None], axis=1) / jnp.sum(
            weights, axis=1, keepdims=True
        )
        output = FeedForward(config)(x)
        #output = nn.LayerNorm(dtype=config.dtype)(x)
        return output


class DecoderModel(nn.Module):
    """Transformer decoder network.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig
    out_seq_len: int
    imp_dim: int

    @nn.compact
    def __call__(self, inputs, deterministic, dropout_rng=random.key(0)):

        config = self.config

        x = inputs  # .astype('int32')
        x = Multiplyer(config, self.out_seq_len)(x)

        for _ in range(config.num_layers):
            x = DecoderBlock(config)(
                inputs=x, deterministic=deterministic, dropout_rng=dropout_rng
            )

        x = FeedForward(config)(x)
        x = nn.LayerNorm(dtype=config.dtype)(x)
        output = Unembedding(config, self.imp_dim)(x)
        return output


class Transformer(nn.Module):
    """Transformer autoencoder model.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig
    out_seq_len: int
    inp_dim: int
    scale_weights: Optional[float] = 1
    scale_out: Optional[bool] = True
    min_val: Optional[Any] = -1
    max_val: Optional[Any] = 1

    def setup(self):
        config = self.config
        out_seq_len = self.out_seq_len
        inp_dim = self.inp_dim
        scale_weights = self.scale_weights

        self.Encoder = EncoderModel(
            config, scale_weights
        )  # (inputs, weights, deterministic=deterministic)
        self.Decoder = DecoderModel(
            config, out_seq_len, inp_dim
        )  # (enc, deterministic=deterministic)

    def __call__(self, inputs, weights, deterministic=True, dropout_rng=random.key(0)):
        scale_out = self.scale_out
        min_val = self.min_val
        max_val = self.max_val

        enc = self.Encoder(
            inputs=inputs,
            weights=weights,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
        )

        dec = self.Decoder(
            inputs=enc, deterministic=deterministic, dropout_rng=dropout_rng
        )

        if scale_out:
            dec = nn.sigmoid(dec) * (max_val - min_val) + min_val
        return (enc, dec)


@struct.dataclass
class Metrics(metrics.Collection):
    enc_loss: metrics.Average
    dec_loss: metrics.Average
    enc_corr: metrics.Average


class TrainState(train_state.TrainState):
    metrics: Metrics
