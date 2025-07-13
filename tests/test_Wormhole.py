import pytest
import numpy as np
import jax.numpy as jnp

from wassersteinwormhole import Wormhole
from wassersteinwormhole.DefaultConfig import DefaultConfig

# A subset of parameters for faster testing. Add more as needed.
DIST_FUNCS = ['S2', 'W2', 'GW']
SCALING_METHODS = ['min_max_total', 'max_dist_each', 'none']

@pytest.fixture
def wormhole_factory():
    """
    A factory fixture that creates Wormhole models, passing kwargs
    directly to the constructor to set custom configurations.
    """
    def _create_wormhole(num_train=64, num_test=32, **kwargs):
        # 1. Generate random data for testing
        point_cloud_sizes_train = np.random.randint(low=8, high=16, size=num_train)
        point_cloud_sizes_test = np.random.randint(low=8, high=16, size=num_test)
        
        pc_train = [np.random.normal(size=[n, 2]) for n in point_cloud_sizes_train]
        pc_test = [np.random.normal(size=[n, 2]) for n in point_cloud_sizes_test]
            
        # 2. Instantiate and return the Wormhole model, passing kwargs through
        model = Wormhole(
            point_clouds=pc_train, 
            point_clouds_test=pc_test,
            **kwargs  # Pass configuration overrides directly
        )
        return model
        
    return _create_wormhole

# ---

### Core Functionality Tests

def test_initialization(wormhole_factory):
    """Checks if the model initializes correctly with default settings."""
    model = wormhole_factory()
    assert model is not None
    assert model.point_clouds.shape[0] == 64
    assert model.point_clouds_test.shape[0] == 32
    assert model.config.emb_dim == 128 # Check a default value

def test_initialization_with_kwargs(wormhole_factory):
    """Checks if kwargs correctly override default config at initialization."""
    model = wormhole_factory(emb_dim=256, num_layers=5)
    assert model.config.emb_dim == 256
    assert model.config.num_layers == 5


@pytest.mark.parametrize("dist_func", DIST_FUNCS)
@pytest.mark.parametrize("scaling", SCALING_METHODS)
def test_train_configurations(wormhole_factory, dist_func, scaling):
    """
    Tests that training runs for 1 step across various critical configurations.
    """
    # Pass kwargs directly to the factory
    model = wormhole_factory(dist_func_enc=dist_func, scale=scaling)
    
    initial_params_sample = model.create_train_state().params['Encoder_0']['layers_0']['attention']['key']['kernel'][0, 0]
    
    model.train(training_steps=1)
    
    trained_params_sample = model.params['Encoder_0']['layers_0']['attention']['key']['kernel'][0, 0]
    assert not jnp.allclose(initial_params_sample, trained_params_sample)


def test_loss_decreases(wormhole_factory):
    """
    Verifies that the training loss decreases over several steps.
    """
    model = wormhole_factory()
    model.train(training_steps=10, batch_size=8, verbose=11)
    
    initial_loss = model.enc_loss_curve[0]
    final_loss = model.enc_loss_curve[-1]
    
    assert initial_loss > final_loss

# ---

### Encoder/Decoder Logic Tests

def test_encode_decode_logic(wormhole_factory):
    """
    Tests the full encode -> decode pipeline, checking shapes and value ranges.
    """
    model = wormhole_factory(dist_func_enc='S2') # S2 enables scale_out
    model.train(training_steps=1)
    
    train_encodings = model.encode(model.point_clouds, model.weights)
    assert train_encodings.shape == (model.point_clouds.shape[0], model.config.emb_dim)
    
    train_decodings = model.decode(train_encodings)
    assert train_decodings.shape == (model.point_clouds.shape[0], model.out_seq_len, model.inp_dim)

    assert model.scale_out is True
    assert jnp.max(train_decodings) <= model.pc_max_val
    assert jnp.min(train_decodings) >= model.pc_min_val
    
# ---

### Specific Feature Tests

def test_auto_sinkhorn_iter_setup(wormhole_factory):
    """

    Tests the automatic tuning of Sinkhorn iterations.
    """
    model = wormhole_factory(num_sinkhorn_iter=-1)
    assert isinstance(model.num_sinkhorn_iter, int)
    assert model.num_sinkhorn_iter > 0

def test_encoder_only_mode(wormhole_factory):
    """
    Tests the behavior when the decoder loss coefficient is negative.
    """
    model = wormhole_factory(coeff_dec=-1.0)
    
    assert model.coeff_dec == 0.0
    assert model.jit_dist_dec.func.func.__name__ == 'Zeros'
    
# ---

### Edge Case Tests

def test_train_with_single_point_cloud(wormhole_factory):
    """
    Tests that the model can handle a dataset with only one point cloud.
    """
    model = wormhole_factory(num_train=1, num_test=1)
    model.train(training_steps=2)
    
    # Encoding loss should be 0 since there are no pairs.
    assert model.enc_loss_curve[0] == 0.0
    assert model.enc_loss_curve[1] == 0.0