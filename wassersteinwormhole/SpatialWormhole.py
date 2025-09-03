from functools import partial
import pickle

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.scipy as jsp # type: ignore
import numpy as np # type: ignore
import optax # type: ignore
import scipy.stats # type: ignore
from flax import linen as nn # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange # type: ignore

# Imports needed for the new functionality
import anndata # type: ignore
import sklearn.neighbors # type: ignore
import scipy.sparse # type: ignore

from wassersteinwormhole._utils_Processing import get_max_dist_statistic
from wassersteinwormhole._utils_Transformer import Metrics, TrainState, Transformer
from wassersteinwormhole.DefaultConfig import SpatialDefaultConfig
import wassersteinwormhole.utils_OT as utils_OT


class SpatialWormhole:
    """
    Initializes a memory-efficient Spatial Wormhole model with on-the-fly padding.

    This model pre-computes neighbor indices and assembles niche point clouds dynamically
    during training, padding each batch to the largest niche size found in the dataset.
    This makes it robust to variable neighborhood sizes.

    :param adata_train: (anndata.AnnData) An anndata object for training.
    :param k: (int) The number of nearest neighbors to define the niche.
    :param adata_test: (anndata.AnnData) An optional anndata object for testing.
    :param config: (flax struct.dataclass) Configuration object for Wormhole parameters.
    :param kwargs: Keyword arguments to override settings in the config object.
    """
    def __init__(
        self,
        adata_train,
        k_neighbours,
        adata_test=None,
        config=None,
        **kwargs,
    ):
        
        if not isinstance(adata_train, anndata.AnnData):
            raise TypeError("Input 'adata_train' must be an anndata.AnnData object.")
        if 'spatial' not in adata_train.obsm:
             raise ValueError("Input 'adata_train' must have spatial coordinates in .obsm['spatial'].")

        if config is None:
            config = SpatialDefaultConfig()
        if kwargs:
            config = config.replace(**kwargs)
        self.config = config
        self.k_neighbours = k_neighbours
        
        self.adata_train = adata_train
        print("Pre-computing neighbor indices and caching expression data...")
        self.exp_data_train = self._get_exp_data(self.adata_train, self.config.rep_key)
        self.niche_indices_train = self._get_niche_indices(
            self.adata_train, self.k_neighbours, self.config.spatial_key, self.config.batch_key
        )
        
        # Determine the maximum niche size for padding across the entire dataset
        self.max_niche_size = max(len(indices) for indices in self.niche_indices_train)
        print(f"Largest niche found has {self.max_niche_size} neighbors. This will be the padding size.")

        if adata_test is not None:
            self.adata_test = adata_test
            self.exp_data_test = self._get_exp_data(self.adata_test, self.config.rep_key)
            self.niche_indices_test = self._get_niche_indices(
                self.adata_test, self.k_neighbours, self.config.spatial_key, self.config.batch_key
            )
            self.max_niche_size = max(self.max_niche_size, max(len(indices) for indices in self.niche_indices_test))

        # --- The rest of the init is similar ---
        self.scale_weights = float(self.k_neighbours)
        self.out_seq_len = self.config.out_seq_len if self.config.out_seq_len != -1 else self.max_niche_size
        print("Decoder generating point-clouds of size: ", self.out_seq_len)

        if(self.config.rep_key is None):
            self.inp_dim = self.adata_train.shape[1]
        else:
            self.inp_dim = self.adata_train.obsm[self.config.rep_key].shape[1]    
        self.eps_enc = config.eps_enc
        self.eps_dec = config.eps_dec
        self.lse_enc = config.lse_enc
        self.lse_dec = config.lse_dec
        self.num_sinkhorn_iter = config.num_sinkhorn_iter
        self.coeff_dec = config.coeff_dec
        self.dist_func_enc = config.dist_func_enc
        self.dist_func_dec = config.dist_func_dec

        if self.coeff_dec < 0:
            self.jit_dist_dec = jax.jit(jax.vmap(utils_OT.Zeros, (0, 0, None, None), 0), static_argnums=[2, 3])
            self.coeff_dec = 0.0

        self.scale = config.scale
        self.factor = config.factor
        self._setup_scaling_function()
        
        self.scale_ot = config.scale_ot
        if self.scale_ot:
            print("Calculating OT scale value from a sample of niches...")
            sample_pcs, sample_weights = self._assemble_and_pad_batch(np.random.choice(self.adata_train.shape[0], min(self.adata_train.shape[0], 1000), replace=False))
            self.ot_scale_value = get_max_dist_statistic(sample_pcs, sample_weights, num_rand=100, reduction="max", dist_func_enc = self.dist_func_enc)
        else:
            self.ot_scale_value  = 1.0
        print("Using OT scale value of", "{:.2e}".format(self.ot_scale_value))

        if(self.num_sinkhorn_iter == -1):
            print("Automatically setting num_sinkhorn_iter by testing OT function...")
            sample_pcs, sample_weights = self._assemble_and_pad_batch(np.random.choice(self.adata_train.shape[0], min(self.adata_train.shape[0], 1000), replace=False))
            self.num_sinkhorn_iter = utils_OT.auto_find_num_iter(point_clouds=sample_pcs, 
                                                                 weights=sample_weights, eps=self.eps_enc, 
                                                                 lse_mode = self.lse_enc, ot_scale=self.ot_scale_value,
                                                                 ot_func=self.dist_func_enc)
            print("Setting num_sinkhorn_iter to", self.num_sinkhorn_iter)
        else:
            print("Using num_sinkhorn_iter =", self.num_sinkhorn_iter)
        
        self.jit_dist_enc = jax.jit(jax.vmap(partial(getattr(utils_OT, self.dist_func_enc), eps=self.eps_enc, lse_mode=self.lse_enc, num_iter=self.num_sinkhorn_iter, ot_scale=self.ot_scale_value), (0, 0), 0))
        self.jit_dist_dec = jax.jit(jax.vmap(partial(getattr(utils_OT, self.dist_func_enc), eps=self.eps_dec, lse_mode=self.lse_dec, num_iter=self.num_sinkhorn_iter, ot_scale=self.ot_scale_value), (0, 0), 0))

        self.pc_max_val = self.exp_data_train.max()
        self.pc_min_val = self.exp_data_train.min()
        self.scale_out = True

        self.model = Transformer(
            self.config, out_seq_len=self.out_seq_len, inp_dim=self.inp_dim,
            scale_weights=self.scale_weights, scale_out=self.scale_out,
            min_val=self.pc_min_val, max_val=self.pc_max_val,
        )

    @staticmethod
    def _get_exp_data(adata, rep):
        """Extracts the expression matrix as a dense numpy array for fast indexing."""
        if rep is None:
            return adata.X.toarray().astype('float32') if scipy.sparse.issparse(adata.X) else np.asarray(adata.X).astype('float32')
        else:
            return np.asarray(adata.obsm[rep]).astype('float32')
    
    @staticmethod
    def _get_niche_indices(spatial_data, k, spatial_key, batch_key):
        """
        Computes the k-NN graph and returns a list of neighbor index arrays for each cell.
        :meta private:
        """
        if batch_key == -1 or batch_key not in spatial_data.obs.columns:
            kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=k, mode='connectivity', n_jobs=-1).tocsr()
            return np.split(kNNGraph.indices, kNNGraph.indptr[1:-1])
        else:
            batch = spatial_data.obs[batch_key]
            niche_indices_list = [np.array([], dtype=int)] * spatial_data.shape[0]
            for val in np.unique(batch):
                val_mask = (batch == val)
                original_indices = np.where(val_mask)[0]
                data_batch = spatial_data[val_mask]
                batch_k = min(k, data_batch.shape[0] - 1)
                if batch_k < 1: continue
                
                batch_knn = sklearn.neighbors.kneighbors_graph(data_batch.obsm[spatial_key], n_neighbors=batch_k, mode='connectivity', n_jobs=-1).tocsr()
                split_indices = np.split(batch_knn.indices, batch_knn.indptr[1:-1])
                for i, local_indices in enumerate(split_indices):
                    niche_indices_list[original_indices[i]] = original_indices[local_indices]
            return niche_indices_list

    def _assemble_and_pad_batch(self, cell_indices, is_test=False):
        """
        Assembles point clouds for a batch of cells, pads them to the max niche size,
        and computes correct weights (1/n for real points, 0 for padded).
        :meta private:
        """
        exp_data = self.exp_data_test if is_test else self.exp_data_train
        niche_indices_source = self.niche_indices_test if is_test else self.niche_indices_train
        
        batch_size = len(cell_indices)
        # Get the list of neighbor index arrays for the current batch
        unpadded_indices = [niche_indices_source[i] for i in cell_indices]
        
        # Initialize padded arrays for point clouds and weights
        padded_pcs = np.zeros((batch_size, self.max_niche_size, self.inp_dim), dtype=np.float32)
        padded_weights = np.zeros((batch_size, self.max_niche_size), dtype=np.float32)

        for i, neighbor_idx_array in enumerate(unpadded_indices):
            num_neighbors = len(neighbor_idx_array)
            if num_neighbors == 0:
                continue # Leave as zeros if a cell has no neighbors
            
            # Assemble the point cloud from expression data
            pc = exp_data[neighbor_idx_array]
            
            # Place data and weights into the padded arrays
            padded_pcs[i, :num_neighbors, :] = pc
            padded_weights[i, :num_neighbors] = 1.0 / num_neighbors # Uniform weights

        return jnp.asarray(padded_pcs), jnp.asarray(padded_weights)

    def _setup_scaling_function(self):
        """Defines the scaling function based on the config."""
        if self.scale == "min_max_total" or self.scale == "min_max_total_all_axis":
            self.max_val = self.exp_data_train.max()
            self.min_val = self.exp_data_train.min()
            self.scale_func = lambda x: 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1
        else:
            self.scale_func = lambda x: x

    def train(self, training_steps=10000, batch_size=16, verbose=8, init_lr=0.0001, decay_steps=2000, key=random.key(0)):
        """
        Set up and run the training loop for the Spatial Wormhole model.
        """
        num_train_cells = self.adata_train.shape[0]
        batch_size = min(num_train_cells, batch_size)

        self.tri_u_ind = jnp.stack(jnp.triu_indices(batch_size, 1), axis=1)
        self.pseudo_weights = jnp.ones([batch_size, self.out_seq_len]) / self.out_seq_len

        key, subkey = random.split(key)
        state = self.create_train_state(subkey, init_lr=init_lr, decay_steps=decay_steps)

        self.enc_loss_curve, self.dec_loss_curve = [], []

        print("Starting training loop using adam...")
        tq = trange(training_steps, leave=True, desc="")
        for training_step in tq:
            key, subkey = random.split(key)

            # 1. Sample cell indices
            batch_cell_ind = random.choice(key=subkey, a=num_train_cells, shape=[batch_size], replace=False)
            
            # 2. Assemble, pad, and weigh the batch on-the-fly
            point_clouds_batch, weights_batch = self._assemble_and_pad_batch(batch_cell_ind)
            
            # 3. Apply scaling
            point_clouds_batch = self.scale_func(point_clouds_batch) * self.factor
            
            key, subkey = random.split(key)
            state, loss = self.train_step(state, point_clouds_batch, weights_batch, subkey)
            self.params = state.params

            # --- Logging logic ---
            self.enc_loss_curve.append(loss[1][0])
            self.dec_loss_curve.append(loss[1][1])
            

            if training_step % verbose == 0:
                print_statement = ""
                for metric, value in zip(
                    ["enc_loss", "dec_loss", "enc_corr"],
                    [loss[1][0], loss[1][1], loss[1][2]],
                ):
                    if metric == "enc_corr":
                        print_statement = (
                            print_statement
                            + " "
                            + metric
                            + ": {:.3f}".format(value)
                        )
                    else:
                        print_statement = (
                            print_statement
                            + " "
                            + metric
                            + ": {:.3e}".format(value)
                        )

                # state.replace(metrics=state.metrics.empty())
                tq.set_description(print_statement)
                tq.refresh()  # to show immediately the update


    def encode(self, cell_indices, from_test_set=False, max_batch=256):
        """
        Encode cellular niches for the given cell indices using the trained model.

        :param cell_indices: (np.array) An array of cell indices to encode.
        :param from_test_set: (bool) Whether indices refer to the test set `adata`.
        :param max_batch: (int) Maximum size of batch during inference calls.
        :return enc: An array of per-niche embeddings.
        """
        if not hasattr(self, 'params'):
            raise RuntimeError("Model has not been trained yet. Please run .train() first.")

        num_niches = len(cell_indices)
        all_encodings = []

        for i in trange(0, num_niches, max_batch, desc="Encoding"):
            batch_cell_indices = cell_indices[i:i+max_batch]
            
            # Assemble, pad, and weigh the batch on-the-fly
            pcs_batch, weights_batch = self._assemble_and_pad_batch(batch_cell_indices, is_test=from_test_set)

            # Scale and run encoder
            pcs_batch = self.scale_func(pcs_batch) * self.factor
            enc = self.model.bind({"params": self.params}).Encoder(
                pcs_batch, weights_batch, deterministic=True
            )
            all_encodings.append(enc)
            
        return np.concatenate(all_encodings, axis=0)

    def create_train_state(self, key=random.key(0), init_lr=0.0001, decay_steps=1000):
        """Initializes the training state. :meta private:"""
        # We need a sample batch to initialize the model parameters
        sample_indices = np.random.choice(self.adata_train.shape[0], min(self.adata_train.shape[0], 32), replace=False)
        sample_pcs, sample_weights = self._assemble_and_pad_batch(sample_indices)

        key, subkey = random.split(key)
        params = self.model.init(
            rngs={"params": key}, dropout_rng=subkey, deterministic=False,
            inputs=sample_pcs, weights=sample_weights,
        )["params"]


        lr_sched = optax.exponential_decay(init_lr, decay_steps, 0.98, staircase=False)
        tx = optax.adam(lr_sched)

        return TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, metrics=Metrics.empty()
        )

    # ========================================================================
    # The following core JAX-based methods are unchanged
    # ========================================================================
    
    def decode(self, enc, max_batch=256):
        """Decodes embeddings back into point clouds."""
        # ... (code identical to previous versions)
        if enc.shape[0] < max_batch:
            dec = self.model.bind({"params": self.params}).Decoder(
                enc, deterministic=True
            )
            if self.scale_out:
                dec = (
                    nn.sigmoid(dec) * (self.pc_max_val - self.pc_min_val)
                    + self.pc_min_val
                )
        else:
            num_split = int(enc.shape[0] / max_batch) + 1
            enc_split = np.array_split(enc, num_split)
            dec = np.concatenate(
                [
                    self.model.bind({"params": self.params}).Decoder(
                        enc_split[split_ind], deterministic=True
                    )
                    for split_ind in range(num_split)
                ],
                axis=0,
            )
            if self.scale_out:
                dec_split = np.array_split(dec, num_split)
                dec = np.concatenate(
                    [
                        nn.sigmoid(dec_split[split_ind])
                        * (self.pc_max_val - self.pc_min_val)
                        + self.pc_min_val
                        for split_ind in range(num_split)
                    ],
                    axis=0,
                )
        return dec

    def compute_losses(self, pc, weights, enc, dec):
        """Computes the encoder and decoder losses for a batch. :meta private:"""
        # ... (code identical to previous versions)
        pc_pairwise_dist = self.jit_dist_enc(
            [pc[self.tri_u_ind[:, 0]], weights[self.tri_u_ind[:, 0]]],
            [pc[self.tri_u_ind[:, 1]], weights[self.tri_u_ind[:, 1]]])

        enc_pairwise_dist = jnp.mean(
            jnp.square(enc[self.tri_u_ind[:, 0]] - enc[self.tri_u_ind[:, 1]]), axis=1)
        
        pc_dec_dist = self.jit_dist_dec(
            [pc, weights], [dec, self.pseudo_weights])

        return (pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist)
    
    @partial(jit, static_argnums=(0))
    def train_step(self, state, pc, weights, key=random.key(0)):
        """Performs a single gradient update step. :meta private:"""
        # ... (code identical to previous versions)
        def loss_fn(params):
            enc, dec = state.apply_fn(
                {"params": params}, inputs=pc, weights=weights,
                deterministic=False, dropout_rng=key,
            )
            pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist = self.compute_losses(
                pc, weights, enc, dec
            )
            enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
            dec_loss = jnp.mean(pc_dec_dist)
            enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0, 1]
            return (enc_loss + self.coeff_dec * dec_loss, [enc_loss, dec_loss, enc_corr])

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, loss)
    
    def save(self, file_path):
        """
        Saves the trained model's parameters and configuration to a file.
        The AnnData objects are not saved and must be provided when loading.

        :param file_path: (str) The path to save the model file to (e.g., 'model.pkl').
        """


        if not hasattr(self, 'params'):
            raise RuntimeError("Model has not been trained yet. Cannot save an untrained model.")

        # Prepare a dictionary with all the necessary components to restore the model state.
        # We don't save AnnData objects as they are large and user-provided.
        state_to_save = {
            'params': self.params,
            'config': self.config,
            'k_neighbours': self.k_neighbours,
            'ot_scale_value': self.ot_scale_value,
            'num_sinkhorn_iter': self.num_sinkhorn_iter,
        }

        print(f"Saving model to {file_path}...")
        with open(file_path, 'wb') as f:
            pickle.dump(state_to_save, f)
        print("Model saved successfully.")

    @classmethod
    def load(cls, file_path, adata_train, adata_test=None):
        """
        Loads a trained model from a file and returns an initialized instance.

        Note: The AnnData objects are required to correctly initialize the model's
        data-dependent properties (like neighbor graphs and padding size).

        :param file_path: (str) The path to the saved model file.
        :param adata_train: (anndata.AnnData) The training anndata object used originally.
        :param adata_test: (anndata.AnnData) The testing anndata object, if used.
        :return: An initialized and trained SpatialWormhole instance.
        """
        print(f"Loading model from {file_path}...")
        with open(file_path, 'rb') as f:
            saved_state = pickle.load(f)
        print("Model file loaded. Initializing SpatialWormhole instance...")

        # Initialize a new instance with the saved configuration
        instance = cls(
            adata_train=adata_train,
            k_neighbours=saved_state['k_neighbours'],
            adata_test=adata_test,
            config=saved_state['config']
        )

        # Restore the saved parameters
        instance.params = saved_state['params']
        
        # Restore other important attributes to their trained values
        instance.ot_scale_value = saved_state['ot_scale_value']
        instance.num_sinkhorn_iter = saved_state['num_sinkhorn_iter']
        
        # Re-create the JIT-compiled distance functions with the loaded parameters
        instance.jit_dist_enc = jax.jit(jax.vmap(partial(getattr(utils_OT, instance.dist_func_enc), eps=instance.eps_enc, lse_mode=instance.lse_enc, num_iter=instance.num_sinkhorn_iter, ot_scale=instance.ot_scale_value), (0, 0), 0))
        instance.jit_dist_dec = jax.jit(jax.vmap(partial(getattr(utils_OT, instance.dist_func_dec), eps=instance.eps_dec, lse_mode=instance.lse_dec, num_iter=instance.num_sinkhorn_iter, ot_scale=instance.ot_scale_value), (0, 0), 0))

        print("Trained model state restored successfully.")
        return instance
