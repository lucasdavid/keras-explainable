from functools import partial
from typing import List, Optional, Tuple

import tensorflow as tf
from keras_explainable import filters
from keras_explainable.inspection import KERNEL_AXIS
from keras_explainable.inspection import SPATIAL_AXIS
from keras_explainable.inspection import gather_units
from keras_explainable.inspection import biases

METHODS = []

def transpose_jacobian(x, spatial_rank=len(SPATIAL_AXIS)):
  dims = [2 + i for i in range(spatial_rank)]

  return tf.transpose(x, [0] + dims + [1])

def gradients(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
):
  """Computes the Grad-CAM Visualization Method.

  This method expects `inputs` to be a batch of positional signals of shape
  `BHWC`, and will return a tensor of shape `BH'W'L`, where `(H', W')` are
  the sizes of the visual receptive field in the explained activation layer
  and `L` is the number of labels represented within the model's output
  logits.

  If `indices` is passed, the specific logits indexed by elements in this
  tensor are selected before the gradients are computed, effectivelly
  reducing the columns in the jacobian, and the size of the output
  explaining map.

  References:

    - Simonyan, K., Vedaldi, A., & Zisserman, A. (2013).
      Deep inside convolutional networks: Visualising image classification
      models and saliency maps. arXiv preprint arXiv:1312.6034.

  """

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(inputs)
    logits = model(inputs, training=False)
    logits = gather_units(logits, indices, indices_axis, indices_batch_dims)

  maps = tape.batch_jacobian(logits, inputs)
  maps = tf.reduce_mean(maps, axis=-1)
  maps = transpose_jacobian(maps, len(spatial_axis))

  return logits, maps

METHODS.extend((gradients,))

def resized_psi_dfx(
    inputs: tf.Tensor,
    outputs: tf.Tensor,
    sizes: tf.Tensor,
    psi: callable = filters.absolute_normalize,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
) -> tf.Tensor:
  t = outputs * inputs
  t = psi(t, spatial_axis)
  t = tf.reduce_mean(t, axis=-1, keepdims=True)
  # t = transpose_jacobian(t, len(spatial_axis))
  t = tf.image.resize(t, sizes)

  return t

def full_gradients(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
    psi: callable = filters.absolute_normalize,
    biases: Optional[List[tf.Tensor]] = None,
    node_index: int = 0,
):
  """Computes the Full-Gradient Visualization Method.

  As described in the article "Full-Gradient Representation forNeural Network
  Visualization", Full-Gradient can be summarized in the following equation:

  ::math::

    f(x) = ψ(∇_xf(x)\odot x) +∑_{l\in L}∑_{c\in c_l} ψ(f^b(x)_c)

  This approach main idea is to add to add the individual contributions of
  each bias factor in the network onto the extracted gradient.

  This method expects `inputs` to be a batch of positional signals of shape
  `BHWC`, and will return a tensor of shape `BH'W'L`, where `(H', W')` are
  the sizes of the visual receptive field in the explained activation layer
  and `L` is the number of labels represented within the model's output
  logits.

  If `indices` is passed, the specific logits indexed by elements in this
  tensor are selected before the gradients are computed, effectivelly
  reducing the columns in the jacobian, and the size of the output
  explaining map.

  References:

  - Srinivas S, Fleuret F. Full-gradient representation for neural network visualization.
    [arXiv preprint arXiv:1905.00780](https://arxiv.org/pdf/1905.00780.pdf), 2019.

  """

  shape = tf.shape(inputs)
  sizes = [shape[a] for a in spatial_axis]

  resized_psi_dfx_ = partial(
    resized_psi_dfx,
    sizes=sizes,
    psi=psi,
    spatial_axis=spatial_axis,
  )

  if biases is None:
    _, biases = biases(
      model,
      node_index=node_index,
      exclude=tf.keras.layers.Dense
    )

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(inputs)
    logits, *intermediates = model(inputs, training=False)
    logits = gather_units(logits, indices, indices_axis, indices_batch_dims)

  maps, *intermediate_maps = tape.gradient(logits, [inputs, *intermediates])

  maps = resized_psi_dfx_(inputs, maps)
  for b, i in zip(biases, intermediate_maps):
    maps += resized_psi_dfx_(b, i)
  # for idx in tf.range(len(biases)):
  #   maps += resized_psi_dfx_(biases[idx], intermediate_maps[idx])

  return logits, maps

METHODS.extend((full_gradients,))
