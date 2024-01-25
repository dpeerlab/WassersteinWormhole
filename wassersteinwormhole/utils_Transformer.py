import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state 
from clu import metrics

import jax
import jax.numpy as jnp
from jax import random


from functools import partial
import scipy.stats
import numpy as np

from typing import Callable, Any, Optional


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int
    output_vocab_size: int
    dtype: Any = jnp.float32
    dist_func_enc: str = 'S2'
    dist_func_dec: str = 'S2'
    eps_enc: float = 0.1
    eps_dec: float = 0.01
    lse_enc: bool = False
    lse_dec: bool = True
    coeff_dec: float = 1
    scale: str = 'min_max_total'
    factor: float = 1.0
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    qkv_dim: int = 128
    mlp_dim: int = 512
    max_len: int = 256
    attention_dropout_rate: float = 0.1
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros_init()

    
class Embedding(nn.Module):
    """Transformer embedding block.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: TransformerConfig

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
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
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
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
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
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: TransformerConfig

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
        x = nn.Dense(
            inputs.shape[-1],
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x) + inputs
        output = nn.LayerNorm(dtype=config.dtype)(x)
        return output

    
class EncoderBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, masks, deterministic):
        config = self.config

        # Attention block.
        # x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
        )(inputs, mask = masks[:, None, None, :]) + inputs

        #x = nn.Dropout(rate=config.attention_dropout_rate)(x, deterministic=deterministic)
        x = x + inputs
        x = nn.LayerNorm(dtype=config.dtype)(x)
        output = FeedForward(config=config)(x)
        return output

class DecoderBlock(nn.Module):
    """Transformer decoder layer.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic):
        config = self.config

        # Attention block.
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=deterministic,
        )(inputs) + inputs

        x = nn.LayerNorm(dtype=config.dtype)(x)
        output = FeedForward(config=config)(x)
        
        return output

    
class Encoder(nn.Module):
    """Transformer encoder network.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig
    
    @nn.compact
    def __call__(self, inputs, masks, deterministic):

        config = self.config

        x = inputs#.astype('int32')
        x = Embedding(config)(x)

        for _ in range(config.num_layers):
            x = EncoderBlock(config)(x, masks = masks, deterministic=deterministic)

        x = jnp.sum(x * masks[:, :, None], axis = 1)/jnp.sum(masks, axis = 1, keepdims = True)
        output = FeedForward(config)(x)
        return output

class Decoder(nn.Module):
    """Transformer decoder network.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig
    out_seq_len: int
    imp_dim: int
    
    @nn.compact
    def __call__(self, inputs, deterministic):

        config = self.config

        x = inputs#.astype('int32')
        x = Multiplyer(config, self.out_seq_len)(x)

        for _ in range(config.num_layers):
            x = DecoderBlock(config)(x, deterministic=deterministic)
        x = FeedForward(config)(x)
        output = Unembedding(config, self.imp_dim)(x)
        return output
    
class Transformer(nn.Module):
    """Transformer autoencoder model.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    """
    
    config: TransformerConfig
    out_seq_len: int
    inp_dim : int
    scale_out: Optional[bool] = True
    min_val: Optional[Any] = -1
    max_val: Optional[Any] = 1
    
    def setup(self):
        config = self.config
        out_seq_len = self.out_seq_len
        inp_dim = self.inp_dim
        scale_out = self.scale_out
        min_val = self.min_val
        max_val = self.max_val
        
        self.Encoder = Encoder(config)#(inputs, masks, deterministic=deterministic)
        self.Decoder = Decoder(config, out_seq_len, inp_dim)#(enc, deterministic=deterministic)
    
    def __call__(self, inputs, masks, deterministic):
        config = self.config
        out_seq_len = self.out_seq_len
        inp_dim = self.inp_dim
        scale_out = self.scale_out
        min_val = self.min_val
        max_val = self.max_val
        
        enc = self.Encoder(inputs, masks, deterministic=deterministic)
        dec = self.Decoder(enc, deterministic=deterministic)
        
        if(scale_out):
            dec = nn.sigmoid(dec) * (max_val - min_val) + min_val        
        return(enc, dec)
    

    
@struct.dataclass
class Metrics(metrics.Collection):
    enc_loss: metrics.Average.from_output('enc_loss')
    dec_loss: metrics.Average.from_output('dec_loss')
    enc_corr: metrics.Average.from_output('enc_corr')
    
class TrainState(train_state.TrainState):
    metrics: Metrics

    

        