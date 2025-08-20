WassersteinWormhole
======================

Embedding point-clouds by preserving Wasserstein distances with the Wormhole.

This implementation is written in Python3 and relies on FLAX, JAX, & JAX-OTT.


To install JAX, simply run the command:

    pip install --upgrade pip install -U "jax[cuda12]” 

And to install WassersteinWormhole along with the rest of the requirements: 

    pip install wassersteinwormhole

And running the Wormhole on your own set of point-clouds is as simple as:
    
    from wassersteinwormhole import Wormhole 
    WormholeModel = Wormhole(point_clouds = point_clouds)
    WormholeModel.train()
    Embeddings = WormholeModel.encode(WormholeModel.point_clouds, WormholeModel.masks)
 
For more details, follow tutorial at [https://wasserstienwormhole.readthedocs.io.](https://wassersteinwormhole.readthedocs.io/en/latest/)
