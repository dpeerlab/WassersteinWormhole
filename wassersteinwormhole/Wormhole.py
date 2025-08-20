from functools import partial

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.scipy as jsp # type: ignore
import numpy as np # type: ignore
import optax # type: ignore
import scipy.stats # type: ignore
from flax import linen as nn # type: ignore
from jax import jit, random# type: ignore
from tqdm import trange # type: ignore



from wassersteinwormhole._utils_Processing import MaxMinScale, pad_pointclouds, get_max_dist_statistic
from wassersteinwormhole._utils_Transformer import Metrics, TrainState, Transformer
from wassersteinwormhole.DefaultConfig import DefaultConfig
import wassersteinwormhole.utils_OT as utils_OT




class Wormhole:
    """
    Initializes Wormhole model and processes input point clouds


    :param point_clouds: (list of np.array) list of train-set point clouds to train Wormhole on
    :param weights: (list of np.array) list of per point weight for each train-set point cloud (default None, indicating uniform weights)
    :param point_clouds_test: (list of np.array) list of test-set point clouds (default None)
    :param weights_test: (list of np.array)  list of per point weight for each test-set point cloud (default None, indicating uniform weights)
    :param config: (flax struct.dataclass) object with parameters for Wormhole such as OT metric choice, emedding dimention, etc. See docs for 'DefaultConfig.py' and tutorial details.

    :return: initialized Wormhole model
    """

    def __init__(
        self,
        point_clouds,
        weights=None,
        point_clouds_test=None,
        weights_test=None,
        config=None,
        **kwargs,
    ):
        
        if len(point_clouds) < 2:
            raise ValueError("Wormhole requires at least two point clouds for training.")

        # If no config object is provided, start with the default.
        if config is None:
            config = DefaultConfig()

        # Any kwargs provided will override the config settings.
        if kwargs:
            config = config.replace(**kwargs)

        self.config = config
        self.point_clouds = point_clouds

        
        if weights is None:
            self.weights = [
                np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds
            ]
        else:
            self.weights = weights

        if point_clouds_test is None:
            self.point_clouds, self.weights = pad_pointclouds(
                self.point_clouds, self.weights
            )
        else:
            self.point_clouds_test = point_clouds_test

            if weights_test is None:
                self.weights_test = [
                    np.ones(pc.shape[0]) / pc.shape[0] for pc in self.point_clouds_test
                ]
            else:
                self.weights_test = weights_test

            total_point_clouds, total_weights = pad_pointclouds(
                list(self.point_clouds) + list(self.point_clouds_test),
                list(self.weights) + list(self.weights_test),
            )
            self.point_clouds, self.weights = (
                total_point_clouds[: len(list(self.point_clouds))],
                total_weights[: len(list(self.point_clouds))],
            )
            self.point_clouds_test, self.weights_test = (
                total_point_clouds[len(list(self.point_clouds)) :],
                total_weights[len(list(self.point_clouds)) :],
            )

        self.scale_weights = np.exp(
            -jsp.special.xlogy(self.weights, self.weights).sum(axis=1)
        ).mean()

        if(self.config.out_seq_len == -1):
            self.out_seq_len = int(
                jnp.exp(-jsp.special.xlogy(self.weights, self.weights).sum(axis=1)).mean()
            )
        else:
            self.out_seq_len = self.config.out_seq_len

        print("Decoder generating point-clouds of size: ", self.out_seq_len)
        
        self.inp_dim = self.point_clouds.shape[-1]

        self.eps_enc = config.eps_enc
        self.eps_dec = config.eps_dec

        self.lse_enc = config.lse_enc
        self.lse_dec = config.lse_dec
        self.num_sinkhorn_iter = config.num_sinkhorn_iter
        
        self.coeff_dec = config.coeff_dec
  

        self.dist_func_enc = config.dist_func_enc
        self.dist_func_dec = config.dist_func_dec



        if self.coeff_dec < 0:
            self.jit_dist_dec = jax.jit(
                jax.vmap(utils_OT.Zeros, (0, 0, None, None), 0), static_argnums=[2, 3]
            )
            self.coeff_dec = 0.0

        self.scale = config.scale
        self.factor = config.factor
        self.point_clouds = self.scale_func(self.point_clouds) * self.factor
        if point_clouds_test is not None:
            self.point_clouds_test = (
                self.scale_func(self.point_clouds_test) * self.factor
            )

        self.scale_ot = config.scale_ot
        if self.scale_ot:
            self.ot_scale_value = get_max_dist_statistic(self.point_clouds, self.weights, num_rand=1000, reduction="max", dist_func_enc = self.dist_func_enc)
        else:
            self.ot_scale_value  = 1.0

        print("Using OT scale value of", "{:.2e}".format(self.ot_scale_value))

        if(self.num_sinkhorn_iter == -1):
            print("Automatically setting num_sinkhorn_iter by testing OT function")

            self.num_sinkhorn_iter = utils_OT.auto_find_num_iter(point_clouds = self.point_clouds, 
                                                                 weights = self.weights,
                                                                 eps = self.eps_enc, lse_mode = self.lse_enc, ot_scale = self.ot_scale_value,
                                                                 ot_func = self.dist_func_enc)
            print("Setting num_sinkhorn_iter to", self.num_sinkhorn_iter)
        else:
            print("Using num_sinkhorn_iter =", self.num_sinkhorn_iter)

        self.jit_dist_enc = jax.jit(jax.vmap(partial(getattr(utils_OT, self.dist_func_enc), 
            eps = self.eps_enc, 
            lse_mode = self.lse_enc, 
            num_iter = self.num_sinkhorn_iter,
            ot_scale = self.ot_scale_value),
            (0, 0), 0))

        self.jit_dist_dec = jax.jit(jax.vmap(partial(getattr(utils_OT, self.dist_func_enc), 
            eps = self.eps_dec, 
            lse_mode = self.lse_dec, 
            num_iter = self.num_sinkhorn_iter,
            ot_scale = self.ot_scale_value),
            (0, 0), 0))


        self.pc_max_val = np.max(self.point_clouds[self.weights > 0])
        self.pc_min_val = np.min(self.point_clouds[self.weights > 0])
        self.scale_out = False if np.isin(self.dist_func_enc, ["GW", "GS"]) else True

        self.model = Transformer(
            self.config,
            out_seq_len=self.out_seq_len,
            inp_dim=self.inp_dim,
            scale_weights=self.scale_weights,
            scale_out=self.scale_out,
            min_val=self.pc_min_val,
            max_val=self.pc_max_val,
        )

    def scale_func(self, point_clouds):
        """
        :meta private:
        """

        if self.scale == "max_dist_total":
            if not hasattr(self, "max_scale_num"):
                max_dist = np.max(
                    [np.max(scipy.spatial.distance.pdist(pc)) for pc in point_clouds]
                )
                self.max_scale_num = max_dist
            else:
                print("Using Calculated Max Dist Scaling Values")
            return point_clouds / self.max_scale_num
        if self.scale == "max_dist_each":
            print("Using Per Sample Max Dist")
            pc_scale = np.asarray(
                [pc / np.max(scipy.spatial.distance.pdist(pc)) for pc in point_clouds]
            )
            return pc_scale
        if self.scale == "min_max_each":
            print("Scaling Per Sample")
            max_val = point_clouds.max(axis=1, keepdims=True)
            min_val = point_clouds.min(axis=1, keepdims=True)
            return 2 * (point_clouds - min_val) / (max_val - min_val) - 1
        elif self.scale == "min_max_total":
            if not hasattr(self, "max_val"):
                self.max_val = self.point_clouds.max(axis=((0, 1)), keepdims=True)
                self.min_val = self.point_clouds.min(axis=((0, 1)), keepdims=True)
            else:
                print("Using Calculated Min Max Scaling Values")
            return 2 * (point_clouds - self.min_val) / (self.max_val - self.min_val) - 1
        elif self.scale == "min_max_total_all_axis":
            if not hasattr(self, "max_val"):
                self.max_val = self.point_clouds.max(keepdims=True)
                self.min_val = self.point_clouds.min(keepdims=True)
            else:
                print("Using Calculated Min Max Scaling Values")
            return 2 * (point_clouds - self.min_val) / (self.max_val - self.min_val) - 1
        else:
            return point_clouds
    
    def encode(self, pc, weights, max_batch=256):
        """
        Encode point clouds with trained Wormhole model


        :param pc: (np.array) array of point clouds to encode
        :param weights: (np.array) point weigts for input point clouds. Wormhole calculates padding for train and test-set point clouds.
        :param max_batch: (int) maximum size of batch during inference calls to Wormhole (default 256)

        :return enc: per point cloud embeddings
        """

        if pc.shape[0] < max_batch:
            enc = self.model.bind({"params": self.params}).Encoder(
                pc, weights, deterministic=True
            )
        else:  # For when the GPU can't pass all point-clouds at once
            num_split = int(pc.shape[0] / max_batch) + 1
            pc_split = np.array_split(pc, num_split)
            mask_split = np.array_split(weights, num_split)

            enc = np.concatenate(
                [
                    self.model.bind({"params": self.params}).Encoder(
                        pc_split[split_ind], mask_split[split_ind], deterministic=True
                    )
                    for split_ind in range(num_split)
                ],
                axis=0,
            )
        return enc

    def decode(self, enc, max_batch=256):
        """
        Decode embedding back into point clouds using Wormhole decoder


        :param enc: (np.array) embeddings to decode
        :param max_batch: (int) maximum size of batch during inference calls to Wormhole (default 256)

        :return dec: decoded point clouds from embeddings
        """

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

    # @partial(jit, static_argnums=(0,4))
    def call(self, pc, weights, deterministic=True, key=random.key(0)):
        """
        :meta private:
        """

        enc, dec = self.model.apply(
            self.variables,
            inputs=pc,
            weights=weights,
            deterministic=deterministic,
            dropout_rng=key,
        )
        return (enc, dec)

    # @partial(jit, static_argnums=(0,4))
    def compute_losses(self, pc, weights, enc, dec):
        """
        :meta private:
        """

        pc_pairwise_dist = self.jit_dist_enc(
            [pc[self.tri_u_ind[:, 0]], weights[self.tri_u_ind[:, 0]]],
            [pc[self.tri_u_ind[:, 1]], weights[self.tri_u_ind[:, 1]]])

        enc_pairwise_dist = jnp.mean(
            jnp.square(enc[self.tri_u_ind[:, 0]] - enc[self.tri_u_ind[:, 1]]), axis=1)
        
        pc_dec_dist = self.jit_dist_dec(
            [pc, weights], [dec, self.pseudo_weights])

        # pc_dec_dist = 0
        return (pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist)

    def create_train_state(self, key=random.key(0), init_lr=0.0001, decay_steps=2000):
        """
        :meta private:
        """

        sample_size = min(256, self.point_clouds.shape[0])

        key, subkey = random.split(key)
        params = self.model.init(
            rngs={"params": key},
            dropout_rng=subkey,
            deterministic=False,
            inputs=self.point_clouds[0:1][:, :sample_size],
            weights=self.weights[0:1][:, :sample_size],
        )["params"]

        if decay_steps < 0:
            # tx = optax.adam(init_lr)
            tx = optax.rmsprop(init_lr)
        else:
            lr_sched = optax.exponential_decay(
                init_lr, decay_steps, 0.75, staircase=True
            )
            tx = optax.rmsprop(lr_sched)  #

        return TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, metrics=Metrics.empty()
        )

    @partial(jit, static_argnums=(0))
    def train_step(self, state, pc, weights, key=random.key(0)):
        """
        :meta private:
        """

        def loss_fn(params):
            enc, dec = state.apply_fn(
                {"params": params},
                inputs=pc,
                weights=weights,
                deterministic=False,
                dropout_rng=key,
            )

            pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist = self.compute_losses(
                pc, weights, enc, dec
            )

            enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
            dec_loss = jnp.mean(pc_dec_dist)
            enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0, 1]

            return (
                enc_loss + self.coeff_dec * dec_loss,
                [enc_loss, dec_loss, enc_corr],
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, loss)

    @partial(jit, static_argnums=(0,))
    def compute_metrics(self, state, pc, weights, key=random.key(0)):
        """
        :meta private:
        """

        enc, dec = state.apply_fn(
            {"params": state.params},
            inputs=pc,
            weights=weights,
            deterministic=False,
            dropout_rng=key,
        )
        pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist = self.compute_losses(
            pc, weights, enc, dec
        )

        enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
        dec_loss = jnp.mean(pc_dec_dist)
        enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0, 1]

        metric_updates = state.metrics.single_from_model_output(
            enc_loss=enc_loss, dec_loss=dec_loss, enc_corr=enc_corr
        )
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    def augment_single_batch(self, single_batch, single_weights, key):
        # apply a random rotation to the point cloud
        rotation_matrix = jax.random.orthogonal(key, single_batch.shape[-1])
        single_batch = jnp.matmul(single_batch, rotation_matrix)
        return [single_batch, single_weights]
    
    def sample_single_batch(self, single_batch, single_weights, key, n_points):
        indices = jax.random.choice(key, single_batch.shape[0], (n_points,), replace=False)
        sampled_pc = jnp.take(single_batch, indices, axis=0)
        sample_weights = jnp.take(single_weights, indices, axis=0)
        sample_weights = sample_weights / jnp.sum(sample_weights)
        
        return [sampled_pc, sample_weights]
    
    def train(
        self,
        training_steps=10000,
        batch_size=16,
        verbose=8,
        init_lr=0.0001,
        decay_steps=2000,
        shape_sample = None,
        augment = False,
        key=random.key(0),
    ):
        """
        Set up optimization parameters and train the ENVI moodel


        :param training_steps: (int) number of gradient descent steps to train ENVI (default 10000)
        :param batch_size: (int) size of train-set point clouds sampled for each training step  (default 16)
        :param verbose: (int) amount of steps between each loss print statement (default 8)
        :param init_lr: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_steps: (int) number of steps before each learning rate decay (default 2000)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """

        batch_size = min(self.point_clouds.shape[0], batch_size)

        if(shape_sample is not None):
            print(f'Sampling {shape_sample} points from each point cloud')
            sample_points = jax.vmap(self.sample_single_batch, in_axes=(0, 0, 0, None))


        self.tri_u_ind = jnp.stack(jnp.triu_indices(batch_size, 1), axis=1)
        self.pseudo_weights = (
            jnp.ones([batch_size, self.out_seq_len]) / self.out_seq_len
        )

        key, subkey = random.split(key)
        state = self.create_train_state(
            subkey, init_lr=init_lr, decay_steps=decay_steps
        )


        enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0, 0, 0, 0

        self.enc_loss_curve, self.dec_loss_curve = [], []

        augment = augment and np.isin(self.dist_func_enc, ["GW", "GS"])

        if(augment):
            print("Augmentation is ON, applying random rotations to point clouds")
            augment_func = jax.vmap(self.augment_single_batch, in_axes=(0, 0, 0))

        tq = trange(training_steps, leave=True, desc="")
        for training_step in tq:
            key, subkey = random.split(key)

            if batch_size < self.point_clouds.shape[0]:
                batch_ind = random.choice(
                    key=subkey,
                    a=self.point_clouds.shape[0],
                    shape=[batch_size],
                    replace=False,
                )
                point_clouds_batch, weights_batch = (
                    self.point_clouds[batch_ind],
                    self.weights[batch_ind],
                )
            else:
                point_clouds_batch, weights_batch = self.point_clouds, self.weights

                        
            if(shape_sample is not None):
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch = sample_points(point_clouds_batch, weights_batch, keys, shape_sample)
                
            if augment:
                keys = jax.random.split(subkey, batch_size)
                point_clouds_batch, weights_batch = augment_func(
                    point_clouds_batch, weights_batch, keys
                )
            key, subkey = random.split(key)
            state, loss = self.train_step(
                state, point_clouds_batch, weights_batch, subkey
            )
            self.params = state.params

            self.enc_loss_curve.append(loss[1][0])
            self.dec_loss_curve.append(loss[1][1])

            enc_loss_mean, dec_loss_mean, enc_corr_mean, count = (
                enc_loss_mean + loss[1][0],
                dec_loss_mean + loss[1][1],
                enc_corr_mean + loss[1][2],
                count + 1,
            )

            if training_step % verbose == 0:
                print_statement = ""
                for metric, value in zip(
                    ["enc_loss", "dec_loss", "enc_corr"],
                    [enc_loss_mean, dec_loss_mean, enc_corr_mean],
                ):
                    if metric == "enc_corr":
                        print_statement = (
                            print_statement
                            + " "
                            + metric
                            + ": {:.3f}".format(value / count)
                        )
                    else:
                        print_statement = (
                            print_statement
                            + " "
                            + metric
                            + ": {:.3e}".format(value / count)
                        )

                # state.replace(metrics=state.metrics.empty())
                enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0, 0, 0, 0
                tq.set_description(print_statement)
                tq.refresh()  # to show immediately the update
