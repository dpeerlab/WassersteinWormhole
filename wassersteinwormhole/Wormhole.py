
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state 

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial
import scipy.stats
import numpy as np
from tqdm import trange

from wassersteinwormhole.utils_Transformer import * 
from wassersteinwormhole.utils_OT import * 



def MaxMinScale(arr):
    min_arr = arr.min(axis = 0)
    max_arr = arr.max(axis = 0)
    
    arr = 2*(arr - arr.min(axis = 0, keepdims = True))/(arr.max(axis = 0, keepdims = True) - arr.min(axis = 0, keepdims = True))-1
    return(arr)

def pad_pointclouds(point_clouds, max_shape = -1):
    if(max_shape == -1):
        max_shape = np.max([pc.shape[0] for pc in point_clouds])+1
    else:
        max_shape = max_shape + 1
    weight_vec = np.asarray([np.concatenate((np.ones(pc.shape[0]), np.zeros(max_shape - pc.shape[0])), axis = 0) for pc in point_clouds])
    point_clouds_pad = np.asarray([np.concatenate([pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis = 0) for pc in point_clouds])
    

    return(point_clouds_pad[:, :-1].astype('float32'), weight_vec[:, :-1].astype('float32'))

class Wormhole():

    def __init__(self, point_clouds, point_clouds_test = None, key = random.key(0), config = TransformerConfig):
    
        self. config = config
        self.point_clouds = point_clouds
        if(point_clouds_test is None):
            self.point_clouds, self.masks = pad_pointclouds(self.point_clouds)
            self.masks_normalized = self.masks/self.masks.sum(axis = 1, keepdims = True)
        else:
            self.point_clouds_test = point_clouds_test

            total_point_clouds, total_masks = pad_pointclouds(list(self.point_clouds) + list(self.point_clouds_test))
            self.point_clouds, self.masks = total_point_clouds[:len(list(self.point_clouds))], total_masks[:len(list(self.point_clouds))]
            self.point_clouds_test, self.masks_test = total_point_clouds[len(list(self.point_clouds)):], total_masks[len(list(self.point_clouds)):]

            self.masks_normalized = self.masks/self.masks.sum(axis = 1, keepdims = True)
            self.masks_test_normalized = self.masks_test/self.masks_test.sum(axis = 1, keepdims = True)

        self.out_seq_len = int(jnp.median(jnp.sum(self.masks, axis = 1)))
        self.inp_dim = self.point_clouds.shape[-1]



        
        self.eps_enc = config.eps_enc
        self.eps_dec = config.eps_dec

        self.lse_enc = config.lse_enc
        self.lse_dec = config.lse_dec

        self.coeff_dec = config.coeff_dec
        
        self.dist_func_enc = config.dist_func_enc
        if(self.dist_func_enc == 'W1'):
            self.jit_dist_enc = jax.jit(jax.vmap(W1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'W2'):
            self.jit_dist_enc = jax.jit(jax.vmap(W2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_enc == 'W2_norm'):
            self.jit_dist_enc = jax.jit(jax.vmap(W2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'GW'):
            self.jit_dist_enc = jax.jit(jax.vmap(GW, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.dist_func_enc == 'S1'):
            self.jit_dist_enc = jax.jit(jax.vmap(S1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'S2'):
            self.jit_dist_enc = jax.jit(jax.vmap(S2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_enc == 'S2_norm'):
            self.jit_dist_enc = jax.jit(jax.vmap(S2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'GS'):
            self.jit_dist_enc = jax.jit(jax.vmap(GS, (0, 0, None, None), 0), static_argnums=[2,3]) 

        self.dist_func_dec = config.dist_func_dec
        if(self.dist_func_dec == 'W1'):
            self.jit_dist_dec = jax.jit(jax.vmap(W1_grad, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'W2'):
            self.jit_dist_dec = jax.jit(jax.vmap(W2_grad, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_dec == 'W2_norm'):
            self.jit_dist_dec = jax.jit(jax.vmap(W2_norm_grad, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'GW'):
            self.jit_dist_dec = jax.jit(jax.vmap(GW_grad, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.dist_func_dec == 'S1'):
            self.jit_dist_dec = jax.jit(jax.vmap(S1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'S2'):
            self.jit_dist_dec = jax.jit(jax.vmap(S2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_dec == 'S2_norm'):
            self.jit_dist_dec = jax.jit(jax.vmap(S2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'GS'):
            self.jit_dist_dec = jax.jit(jax.vmap(GS, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.coeff_dec < 0):
            self.jit_dist_dec  = jax.jit(jax.vmap(Zeros, (0, 0, None, None), 0), static_argnums=[2,3]) 
            self.coeff_dec = 0.0
      


        self.scale = config.scale
        self.factor = config.factor
        self.point_clouds = self.scale_func(self.point_clouds) * self.factor
        if(point_clouds_test is not None):
            self.point_clouds_test = self.scale_func(self.point_clouds_test)*self.factor
        
      
        self.pc_max_val = np.max(self.point_clouds[self.masks > 0])
        self.pc_min_val = np.min(self.point_clouds[self.masks > 0])
        self.scale_out = not np.isin(self.dist_func_dec, ['GS', 'GW'])
        
        self.model = Transformer(self.config, out_seq_len = self.out_seq_len, inp_dim = self.inp_dim,
                                 scale_out = self.scale_out, min_val = self.pc_min_val, max_val = self.pc_max_val)


    def scale_func(self, point_clouds):
        if(self.scale == 'max_dist_total'):
            if(not hasattr(self, 'max_scale_num')):
                max_dist = 0
                for _ in range(10):
                    i,j = np.random.choice(np.arange(len(self.point_clouds)), 2,replace = False)
                    if(self.dist_func_enc == 'GW' or self.dist_func_enc == 'GS'):
                        max_ij = np.max(scipy.spatial.distance.cdist(self.point_clouds[i], self.point_clouds[i])**2)
                    else:
                        max_ij = np.max(scipy.spatial.distance.cdist(self.point_clouds[i], self.point_clouds[j])**2)
                    max_dist = np.maximum(max_ij, max_dist)
                self.max_scale_num = max_dist
            else:
                print("Using Calculated Max Dist Scaling Values") 
            return(point_clouds/self.max_scale_num)
        if(self.scale == 'max_dist_each'):
            print("Using Per Sample Max Dist") 
            pc_scale = np.asarray([pc/np.max(scipy.spatial.distance.pdist(pc)**2) for pc in point_clouds])
            return(pc_scale)
        if(self.scale == 'min_max_each'):
            print("Scaling Per Sample") 
            max_val = point_clouds.max(axis = 1, keepdims = True)
            min_val = point_clouds.min(axis = 1, keepdims = True)
            return(2 * (point_clouds - min_val)/(max_val - min_val) - 1)
        elif(self.scale == 'min_max_total'):
            if(not hasattr(self, 'max_val')):
                self.max_val = self.point_clouds.max(axis = ((0,1)), keepdims = True)
                self.min_val = self.point_clouds.min(axis = ((0,1)), keepdims = True)
            else:
                print("Using Calculated Min Max Scaling Values") 
            return(2 * (point_clouds - self.min_val)/(self.max_val - self.min_val) - 1)
        elif(isinstance(self.scale, (int, float, complex)) and not isinstance(self.scale, bool)):
            print("Using Constant Scaling Value") 
            return(point_clouds/self.scale)
    
    def encode(self, pc, masks, max_batch = 256):
        if(pc.shape[0] < max_batch):
            enc = self.model.bind({'params': self.params}).Encoder(pc, masks, deterministic = True)
        else: # For when the GPU can't pass all point-clouds at once
            num_split = int(pc.shape[0]/max_batch)+1
            pc_split = np.array_split(pc, num_split)
            mask_split = np.array_split(masks, num_split)
            
            enc = np.concatenate([self.model.bind({'params': self.params}).Encoder(pc_split[split_ind], mask_split[split_ind], deterministic = True) for
                                  split_ind in range(num_split)], axis = 0)
        return enc
    
    def decode(self, enc, max_batch = 256):
        if(enc.shape[0]<max_batch):
            dec = self.model.bind({'params': self.params}).Decoder(enc, deterministic = True)
            if(self.scale_out):
                dec = nn.sigmoid(dec) * (self.pc_max_val - self.pc_min_val) + self.pc_min_val
        else:
            num_split = int(enc.shape[0]/max_batch)+1
            enc_split = np.array_split(enc, num_split) 
            dec = np.concatenate([self.model.bind({'params': self.params}).Decoder(enc_split[split_ind], deterministic = True) 
                                  for split_ind in range(num_split)], axis = 0)
            if(self.scale_out):
                dec_split = np.array_split(dec, num_split) 
                dec = np.concatenate([nn.sigmoid(dec_split[split_ind]) * (self.pc_max_val - self.pc_min_val) + self.pc_min_val for split_ind in range(num_split)], axis = 0)
        return dec
    
    #@partial(jit, static_argnums=(0,4))
    def call(self, pc, masks, deterministic = False, key = random.key(0)):
        enc, dec = self.model.apply(self.variables, rngs = {'dropout': key}, deterministic = deterministic,
                                    inputs = pc, masks = masks)
        return(enc, dec)
    
    #@partial(jit, static_argnums=(0,4))
    def compute_losses(self, pc, masks, enc, dec):
    
        
        
        mask_normalized = masks/jnp.sum(masks, axis = 1,keepdims = True)

        pc_pairwise_dist = self.jit_dist_enc([pc[self.tri_u_ind[:, 0]], mask_normalized[self.tri_u_ind[:, 0]]],
                                             [pc[self.tri_u_ind[:, 1]], mask_normalized[self.tri_u_ind[:, 1]]], 
                                             self.eps_enc, self.lse_enc)
       
        enc_pairwise_dist = jnp.mean(jnp.square(enc[self.tri_u_ind[:, 0]] - enc[self.tri_u_ind[:, 1]]), axis = 1)
        
        
        pc_dec_dist = self.jit_dist_dec([pc, mask_normalized], [dec, self.pseudo_masks], 
                                        self.eps_dec, self.lse_dec)
        
        # pc_dec_dist = 0
        return(pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist)
       
    
    def create_train_state(self, key = random.key(0), init_lr = 0.0001, decay_steps = 2000):
        
        key, subkey = random.split(key)
        params = self.model.init(rngs = {'params': key, 'dropout': subkey}, deterministic = False,
                                         inputs = self.point_clouds[0:1], masks = self.masks[0:1])['params']
        
        lr_sched = optax.exponential_decay(0.0001, decay_steps, 0.9, staircase = True)
        tx = optax.adam(lr_sched)#
        
        return(TrainState.create(
          apply_fn=self.model.apply, params=params, tx=tx,
          metrics=Metrics.empty()))
    
    @partial(jit, static_argnums=(0, ))
    def train_step(self, state, pc, masks, key = random.key(0)):
        """Train for a single step."""
        
        def loss_fn(params):
            enc, dec = state.apply_fn({'params':params}, inputs = pc, masks = masks, deterministic = False, rngs = {'dropout': key})
            pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist = self.compute_losses(pc, masks, enc, dec)
            
            enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
            dec_loss = jnp.mean(pc_dec_dist)
            enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0,1]
            return(enc_loss + self.coeff_dec * dec_loss, [enc_loss, dec_loss, enc_corr])
    
        grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return(state, loss)
    
    @partial(jit, static_argnums=(0, ))
    def compute_metrics(self, state, pc, masks, key = random.key(0)):
        enc, dec  = state.apply_fn({'params': state.params}, inputs = pc, masks = masks, deterministic = False, rngs = {'dropout': key})
        pc_pairwise_dist, enc_pairwise_dist, pc_dec_dist = self.compute_losses(pc, masks, enc, dec)
        
        enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
        dec_loss = jnp.mean(pc_dec_dist)
        enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0,1]
        
    
        metric_updates = state.metrics.single_from_model_output(enc_loss = enc_loss, dec_loss = dec_loss, enc_corr = enc_corr)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return(state)

    def train(self, epochs = 10000, batch_size = 16, verbose = 8, init_lr = 0.0001, decay_steps = 2000, key = random.key(0)):
        batch_size = min(self.point_clouds.shape[0], batch_size)
        
        self.tri_u_ind = jnp.stack(jnp.triu_indices(batch_size, 1), axis =1)
        self.pseudo_masks = jnp.ones([batch_size, self.out_seq_len])/self.out_seq_len

        key, subkey = random.split(key)
        state = self.create_train_state(subkey, init_lr = init_lr, decay_steps = decay_steps)
        
        

        
        tq = trange(epochs, leave=True, desc = "")
        enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0,0,0,0
        for epoch in tq:
            # time.sleep(1)
            key, subkey = random.split(key)



            if(batch_size < self.point_clouds.shape[0]):
                batch_ind = random.choice(key = subkey, a = self.point_clouds.shape[0], shape = [batch_size], replace = False)
                point_clouds_batch, masks_batch = self.point_clouds[batch_ind], self.masks[batch_ind]
            else:
                point_clouds_batch, masks_batch = self.point_clouds, self.masks

            key, subkey = random.split(key)
            state, loss = self.train_step(state, point_clouds_batch, masks_batch, subkey)
            self.params = state.params

            enc_loss_mean, dec_loss_mean, enc_corr_mean, count = enc_loss_mean + loss[1][0], dec_loss_mean + loss[1][1], enc_corr_mean + loss[1][2], count + 1

            if(epoch%verbose==0):
                print_statement = ''
                for metric,value in zip(['enc_loss', 'dec_loss', 'enc_corr'], [enc_loss_mean, dec_loss_mean, enc_corr_mean]):
                    if(metric == 'enc_corr'):
                        print_statement = print_statement + ' ' + metric + ': {:.3f}'.format(value/count)
                    else:
                        print_statement = print_statement + ' ' + metric + ': {:.3e}'.format(value/count)

                # state.replace(metrics=state.metrics.empty())
                enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0,0,0,0
                tq.set_description(print_statement)
                tq.refresh() # to show immediately the update

