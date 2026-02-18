import pytest
import numpy as np
import jax
import jax.numpy as jnp
import anndata

# Import the classes to be tested
from wassersteinwormhole import Wormhole, SpatialWormhole
from wassersteinwormhole.DefaultConfig import DefaultConfig

# --- Fixtures for Wormhole ---

@pytest.fixture
def wormhole_factory():
    """
    A factory fixture that creates Wormhole models, passing kwargs
    directly to the constructor to set custom configurations.
    """
    def _create_wormhole(num_train=32, num_test=16, **kwargs):
        # 1. Generate random data for testing
        point_cloud_sizes_train = np.random.randint(low=20, high=30, size=num_train)
        point_cloud_sizes_test = np.random.randint(low=20, high=30, size=num_test)
        
        pc_train = [np.random.normal(size=[n, 2]) for n in point_cloud_sizes_train]
        pc_test = [np.random.normal(size=[n, 2]) for n in point_cloud_sizes_test]
            
        # 2. Instantiate and return the Wormhole model, passing kwargs through
        model = Wormhole(
            point_clouds=pc_train, 
            point_clouds_test=pc_test,
            **kwargs
        )
        return model
        
    return _create_wormhole

# --- Tests for Standard Wormhole ---

### Core Functionality Tests

def test_initialization(wormhole_factory):
    """Checks if the model initializes correctly with default settings."""
    model = wormhole_factory()
    assert model is not None
    assert model.point_clouds.shape[0] == 32
    assert model.point_clouds_test.shape[0] == 16
    assert model.config.emb_dim == 128

def test_initialization_with_kwargs(wormhole_factory):
    """Checks if kwargs correctly override default config at initialization."""
    model = wormhole_factory(emb_dim=256, num_layers=5)
    assert model.config.emb_dim == 256
    assert model.config.num_layers == 5

@pytest.mark.parametrize("dist_func", ['S2', 'W2'])
def test_train_configurations(wormhole_factory, dist_func):
    """
    Tests that training runs for 1 step and that model parameters are updated.
    This test is robust to changes in the model's internal layer names.
    """
    model = wormhole_factory(dist_func_enc=dist_func)
    
    initial_state = model.create_train_state()
    model.train(training_steps=1)
    trained_params = model.params

    leaves_are_the_same = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5), initial_state.params, trained_params
    )
    assert not jax.tree_util.tree_all(leaves_are_the_same)


def test_loss_decreases(wormhole_factory):
    """Verifies that the training loss decreases over several steps."""
    model = wormhole_factory()
    model.train(training_steps=15, batch_size=8, verbose=20)
    
    initial_loss = model.enc_loss_curve[0]
    final_loss = model.enc_loss_curve[-1]
    
    assert initial_loss >= final_loss

### Encoder/Decoder Logic Tests

def test_encode_decode_logic(wormhole_factory):
    """Tests the full encode -> decode pipeline, checking shapes and value ranges."""
    model = wormhole_factory(dist_func_enc='S2')
    model.train(training_steps=1)
    
    train_encodings = model.encode(model.point_clouds, model.weights)
    assert train_encodings.shape == (model.point_clouds.shape[0], model.config.emb_dim)
    
    train_decodings = model.decode(train_encodings)
    assert train_decodings.shape == (model.point_clouds.shape[0], model.num_particles_output, model.inp_dim)
    
### Specific Feature Tests

def test_auto_sinkhorn_iter_setup(wormhole_factory):
    """Tests the automatic tuning of Sinkhorn iterations."""
    model = wormhole_factory(num_sinkhorn_iter=-1)
    assert isinstance(model.num_sinkhorn_iter, int)
    assert model.num_sinkhorn_iter > 0

def test_encoder_only_mode(wormhole_factory):
    """Tests behavior when the decoder loss coefficient is negative."""
    model = wormhole_factory(coeff_dec=-1.0)
    assert model.coeff_dec == 0.0
    
    dummy_pc = jnp.ones((2, 10, 2))
    dummy_weights = jnp.ones((2, 10)) / 10
    result = model.jit_dist_dec([dummy_pc, dummy_weights], [dummy_pc, dummy_weights])
    assert jnp.all(result == 0)

### Augmentation and Sampling Tests

def test_shape_sampling_logic(wormhole_factory):
    """Tests the point cloud sub-sampling logic directly."""
    model = wormhole_factory()
    key = jax.random.PRNGKey(42)
    
    original_pc = jnp.ones((25, 3))
    original_weights = jnp.ones(25) / 25
    sample_size = 10
    
    sampled_pc, sampled_weights = model.sample_single_batch(original_pc, original_weights, key, sample_size)
    
    assert sampled_pc.shape == (sample_size, 3)
    assert sampled_weights.shape == (sample_size,)
    assert jnp.isclose(jnp.sum(sampled_weights), 1.0)

def test_train_with_shape_sampling(wormhole_factory):
    """Ensures training runs without errors when `shape_sample` is enabled."""
    model = wormhole_factory()
    sample_size = 10 
    model.train(training_steps=2, batch_size=8, shape_sample=sample_size)
    assert len(model.enc_loss_curve) == 2

# ===================================================================
# --- Fixtures and Tests for SpatialWormhole ---
# ===================================================================

@pytest.fixture
def spatial_wormhole_factory():
    """
    A factory fixture that creates SpatialWormhole models using anndata.
    """
    def _create_spatial_wormhole(num_cells=50, num_genes=10, k=5, **kwargs):
        # 1. Generate random anndata for testing
        adata = anndata.AnnData(np.random.normal(size=(num_cells, num_genes)))
        adata.obsm['spatial'] = np.random.normal(size=(num_cells, 2))
        
        # 2. Instantiate and return the SpatialWormhole model
        model = SpatialWormhole(
            adata_train=adata,
            k_neighbours=k,
            **kwargs
        )
        return model
        
    return _create_spatial_wormhole

### Spatial Core Functionality Tests

def test_spatial_initialization(spatial_wormhole_factory):
    """Checks if the SpatialWormhole model initializes correctly."""
    model = spatial_wormhole_factory(num_cells=50, k=5)
    assert model is not None
    assert len(model.niche_indices_train) == 50
    assert model.max_niche_size <= 5
    assert model.config.emb_dim == 128

def test_spatial_train_runs(spatial_wormhole_factory):
    """Tests that spatial training runs for a few steps without errors."""
    model = spatial_wormhole_factory(num_cells=40, k=4)
    try:
        model.train(training_steps=2, batch_size=8)
    except Exception as e:
        pytest.fail(f"SpatialWormhole training failed: {e}")
    assert len(model.enc_loss_curve) == 2

### On-the-Fly Data Handling Tests

def test_assemble_and_pad_batch(spatial_wormhole_factory):
    """
    Tests the core on-the-fly batch assembly and padding logic.
    """
    k = 6
    model = spatial_wormhole_factory(num_cells=30, k=k)
    
    # Select a batch of cell indices
    batch_indices = np.array([0, 5, 10])
    batch_size = len(batch_indices)
    
    # Call the function to be tested
    padded_pcs, padded_weights = model._assemble_and_pad_batch(batch_indices)
    
    # 1. Check output shapes
    assert padded_pcs.shape == (batch_size, model.max_niche_size, model.inp_dim)
    assert padded_weights.shape == (batch_size, model.max_niche_size)
    
    # 2. Check weight correctness for the first cell in the batch
    first_cell_indices = model.niche_indices_train[batch_indices[0]]
    num_neighbors = len(first_cell_indices)
    
    # Weights for real points should be 1/num_neighbors
    if num_neighbors > 0:
        expected_weight = 1.0 / num_neighbors
        assert jnp.allclose(padded_weights[0, :num_neighbors], expected_weight)
    
    # Weights for padded points should be 0
    assert jnp.allclose(padded_weights[0, num_neighbors:], 0.0)
    
    # 3. Check that sum of weights for each item in batch is close to 1.0 or 0.0
    sum_of_weights = jnp.sum(padded_weights, axis=1)
    assert jnp.all(jnp.isclose(sum_of_weights, 1.0) | jnp.isclose(sum_of_weights, 0.0))

### Spatial Encoder/Decoder Tests

def test_spatial_encode_decode_logic(spatial_wormhole_factory):
    """
    Tests the full encode -> decode pipeline for SpatialWormhole.
    """
    model = spatial_wormhole_factory(num_cells=30, k=5)
    model.train(training_steps=2)
    
    # Encode a subset of the cells
    cell_indices_to_encode = np.arange(10)
    encodings = model.encode(cell_indices_to_encode)
    
    assert encodings.shape == (len(cell_indices_to_encode), model.config.emb_dim)
    
    # Decode the embeddings
    decodings = model.decode(encodings)
    assert decodings.shape == (len(cell_indices_to_encode), model.num_particles_output, model.inp_dim)

### Spatial Edge Case Tests

def test_spatial_with_batch_key(spatial_wormhole_factory):
    """
    Tests initialization and training when a `batch_key` is provided.
    """
    num_cells = 60
    k = 4
    
    # Create anndata with two distinct batches
    adata = anndata.AnnData(np.random.normal(size=(num_cells, 10)))
    adata.obsm['spatial'] = np.random.normal(size=(num_cells, 2))
    adata.obs['sample'] = ['A'] * (num_cells // 2) + ['B'] * (num_cells // 2)
    
    # This will fail if the per-batch kNN logic is broken
    model = SpatialWormhole(adata_train=adata, k_neighbours=k, batch_key='sample')
    
    # Check that neighbor indices were computed for all cells
    assert len(model.niche_indices_train) == num_cells
    
    # Ensure training runs
    model.train(training_steps=2, batch_size=16)
    assert len(model.enc_loss_curve) == 2
