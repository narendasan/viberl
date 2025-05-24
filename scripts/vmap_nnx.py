"""
Using nnx.vmap to vectorize models
"""

import jax
import jax.numpy as jnp
from flax import nnx

import viberl


# Arbitrary model definition, 2 linear layers. We control the initializtion of weights and biases so its easier to see where results come from
class MLP(nnx.Module):
  """
  A simple Multi-Layer Perceptron with two linear layers.

  Args:
    din: Dimension of the input.
    dmid: Dimension of the hidden layer.
    dout: Dimension of the output.
    const: Constant value for bias initialization.
    rngs: Random number generator state for parameter initialization.
  """
  def __init__(self, din: int, dmid: int, dout: int, *, const:int, rngs:nnx.Rngs ):
    # Initialize first linear layer with constant 1 weights and specified constant bias
    self.linear1 = nnx.Linear(din, dmid, kernel_init=jax.nn.initializers.constant(1), bias_init=jax.nn.initializers.constant(const), rngs=rngs)
    # Initialize second linear layer with constant 1 weights and specified constant bias
    self.linear2 = nnx.Linear(dmid, dout, kernel_init=jax.nn.initializers.constant(1), bias_init=jax.nn.initializers.constant(const), rngs=rngs)

  def __call__(self, x: jax.Array):
    """Forward pass of the MLP.

    Args:
      x: Input tensor of shape (din,)

    Returns:
      Output tensor of shape (dout,)
    """
    print("x.shape:", x.shape)
    x = self.linear1(x)
    return self.linear2(x)

# Create a model instance with 10 input dimensions, 20 hidden dimensions, and 30 output dimensions
model = MLP(10, 20, 30, const=0, rngs=nnx.Rngs(0))
# Display the model structure
nnx.display(model)

# Run the model with an input vector of ones, In the case of non vectorized models, this will need to be done iteratively
input = jnp.ones((10,))
print("input:", input)
print("input.shape:", input.shape)
# Run the model on the input
output = model(input)
print("output:", output)
print("output.shape:", output.shape)

# A function to create multiple MLP models vectorized over different constant values for the weights
@nnx.vmap(in_axes=(0), out_axes=0)
def create_models(const):
    """Create multiple MLP models vectorized over different constant values.

    Args:
        const: Array of constant values to use for bias initialization.

    Returns:
        Vectorized MLP models.
    """
    return MLP(10, 20, 30, const=const, rngs=nnx.Rngs(0))

# A function to run multiple models on multiple inputs in a vectorized way, the extra dimension is for the batch size
@nnx.vmap(in_axes=(0,0), out_axes=0)
def run_models(models, inputs):
    """Run multiple models on multiple inputs in a vectorized way.

    Args:
        models: Array of MLP models.
        inputs: Array of input tensors.

    Returns:
        Array of model outputs.
    """
    return models(inputs)

# Create an array of constants from 0 to 9
consts = jnp.arange(0,10)
print("consts:", consts)
# Create 10 models with different constant values
models = create_models(consts)
nnx.display(models)

# Create 10 identical inputs
inputs = jnp.stack([jnp.ones((10,))] * 10)
print("inputs.shape:", inputs.shape)
# Run all models on all inputs
outputs_vec = run_models(models, inputs)
print("outputs_vec:", outputs_vec)
print("outputs_vec.shape:", outputs_vec.shape)

# Run the vectorized model iteratively
model_slices = viberl.utils.unstack_modules(
    MLP,
    nnx.state(models),
    num_slices=10,
    module_init_args=[(10, 20, 30)] * 10,
    module_init_kwargs=[{"const": -1, "rngs": nnx.Rngs(i)} for i in range(10)]
)
outputs = []
for i in range(10):
    output = model_slices[i](input) # Apply each model *once*
    outputs.append(output)
    print(f"Output from model {i+1}: {output}")

outputs_iter = jnp.stack(outputs) # Convert the list to an array
print("Stacked Outputs:", outputs_iter.shape)
# Assert that the vectorized and iterative outputs match
print("Checking if vectorized and iterative outputs match...")
assert jnp.allclose(outputs_vec, outputs_iter), "Vectorized and iterative outputs should be identical"
print("Both outputs match!")


# How to squash model slices into a vectorized model
stacked_state = viberl.utils.tree_stack([nnx.state(model) for model in model_slices])
print(stacked_state)

new_stacked_model = create_models(jnp.full_like(consts, -1))
nnx.update(new_stacked_model, stacked_state)

# Create 10 identical inputs
inputs = jnp.stack([jnp.ones((10,))] * 10)
print("inputs.shape:", inputs.shape)
# Run all models on all inputs
outputs_vec2 = run_models(new_stacked_model, inputs)
print("outputs_vec2:", outputs_vec2)
print("outputs_vec2.shape:", outputs_vec2.shape)

# Assert that the vectorized and iterative outputs match
print("Checking if vectorized and iterative outputs match...")
assert jnp.allclose(outputs_vec2, outputs_iter), "Vectorized and iterative outputs should be identical"
print("Both outputs match!")
