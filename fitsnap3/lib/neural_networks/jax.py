import jax.numpy as jnp
from jax import grad, jit, vmap, lax, nn
from jax.ops import segment_sum
from optax import adam, apply_updates


"""
For a batch
Descriptors needs to be of shape (num_configs, num_atoms, num_descriptors) -> predictions (num_configs, num_atoms)
Targets are of shape (num_configs)
Loss = 0.5*mean(squared(sum(predictions)-targets))
"""


def predict(params, in_descriptors):
    """
    Descriptors here are of shape (num_descriptors,)
    j is input_layer, i is output_layer
    j is number of descriptors
    (i, j) * (j,) -> (i,)
    b is shape (i,)
    (i,) + (i,)
    """
    # First layer is standardization
    activations = jnp.dot(params[0][0], in_descriptors) + params[0][1]

    for w, b in params[1:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = nn.softplus(outputs)

    final_w, final_b = params[-1]
    logit = jnp.dot(final_w, activations) + final_b
    return logit


# Vectorization of predictions which batches over each A matrix row
batched_predict = vmap(predict, in_axes=(None, 0))


def loss(params, descriptors, targets, sum_indices, num_atoms, num_segments):
    """
    Descriptors here are of shape (num_configs*num_atoms_per_config, num_descriptors)
    segment_sum contracts first axis from num_configs*num_atoms_per_config to num_configs
    Predictions are of shape (num_configs*num_atoms_per_config)
    Returns mean of the l2 norm of the differences between predictions and targets.
    """
    predictions = segment_sum(batched_predict(params, descriptors), sum_indices, num_segments=num_segments) / num_atoms
    return jnp.mean(jnp.square(predictions-targets))


def accuracy(params, descriptors, targets, sum_indices, num_atoms):
    predictions = segment_sum(batched_predict(params, descriptors), sum_indices) / num_atoms
    return jnp.sqrt(jnp.mean(jnp.square(predictions-targets)))


def mae(params, descriptors, targets, sum_indices, num_atoms):
    predictions = segment_sum(batched_predict(params, descriptors), sum_indices) / num_atoms
    return jnp.mean(jnp.abs(predictions-targets))