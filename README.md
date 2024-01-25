WassersteinWormhole for Python3
======================

Embedding point-clouds by presering Wasserstein distancse with the Wormhole.

This implementation is written in Python3 and relies on FLAX, JAX, & JAX-OTT.


To install JAX, simply run the command:

    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

And to install WassersteinWormhole along with the rest of the requirements: 

    pip install WassersteinWormhole

And running the Womrhole on your own set of point-clouds is as simple as:
    
    WormholeModel = Wormhole(point_clouds = point_clouds)
    WormholeModel.train()
    Embeddings = WormholeModel.encode(WormholeModel.point_clouds, WormholeModel.masks)
    
For more details, follow tutorial at https://github.com/dpeerlab/WassersteinWormhole.
    