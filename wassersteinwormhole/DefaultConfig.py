from flax import struct
from flax import linen as nn
import jax.numpy as jnp

from typing import Callable, Any, Optional

@struct.dataclass
class DefaultConfig:
    
    """
    Object with configuration parameters for Wormhole
    
    
    :param dtype: (data type) float point precision for Wormhole model (default jnp.float32)
    :param dist_func_enc: (str) OT metric used for embedding space (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param dist_func_dec: (str) OT metric used for Wormhole decoder loss (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param eps_enc: (float) entropic regularization for embedding OT (default 0.1)
    :param eps_dec: (float) entropic regularization for Wormhole decoder loss (default 0.1)
    :param lse_enc: (bool) whether to use log-sum-exp mode or kernel mode for embedding OT (default False)
    :param lse_dec: (bool) whether to use log-sum-exp mode or kernel mode for decoder OT (default True)
    :param coeff_dec: (float) coefficient for decoder loss (default 1)
    :param scale: (str) how to scale input point clouds ('min_max_total' and scales all point clouds so values are between -1 and 1)
    :param factor: (float) multiplicative factor applied on point cloud coordinates after scaling (default 1)
    :param emb_dim: (int) Wormhole embedding dimention (defulat 128)
    :param num_heads: (int) number of heads in multi-head attention (default 4)
    :param num_layers: (int) number of layers of multi-head attention for Wormhole encoder and decoder (default 3)
    :param mlp_dim: (int) dimention of hidden layer for fully-connected network after every multi-head attention layer
    :param attention_dropout_rate: (float) dropout rate for attention matrices during training (default 0.1)
    :param kernel_init: (Callable) initializer of kernel weights (default nn.initializers.glorot_uniform())
    :param bias_init: ((Callable) initializer of bias weights (default nn.initializers.zeros_init())
    """ 
    
    dtype: Any = jnp.float32
    dist_func_enc: str = 'S2'
    dist_func_dec: str = 'S2'
    eps_enc: float = 0.1
    eps_dec: float = 0.01
    lse_enc: bool = False
    lse_dec: bool = True
    out_seq_len: int = -1
    num_sinkhorn_iter: int = 200
    coeff_dec: float = 1
    scale: str = 'min_max_total'
    scale_ot: bool = True
    factor: float = 1.0
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    mlp_dim: int = 512
    attention_dropout_rate: float = 0.1
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros_init()

@struct.dataclass
class SpatialDefaultConfig(DefaultConfig):
    """
    Default configuration for SpatialWormhole, inheriting from DefaultConfig.
    
    Adds parameters specific to handling AnnData objects.
    
    :param rep: (str, optional) The key in `adata.obsm` to use as the expression representation. If None, `adata.X` is used. (default None)
    :param batch_key: (str, optional) The key in `adata.obs` that denotes the sample/batch for each cell. If None, all cells are treated as one batch. (default None)
    """
    rep_key: Optional[str] = None
    batch_key: Optional[str] = None
    spatial_key: str = 'spatial'
