from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import tensorflow as tf

from keras_explainable import filters
from keras_explainable import inspection
from keras_explainable.inspection import KERNEL_AXIS
from keras_explainable.inspection import SPATIAL_AXIS


def transpose_jacobian(
    x: tf.Tensor, spatial_rank: Tuple[int] = len(SPATIAL_AXIS)
) -> tf.Tensor:
    """Transpose the Jacobian of shape (b,g,...) into (b,...,g).

    Args:
        x (tf.Tensor): the jacobian tensor.
        spatial_rank (Tuple[int], optional): the spatial rank of ``x``.
            Defaults to ``len(SPATIAL_AXIS)``.

    Returns:
        tf.Tensor: the transposed jacobian.
    """
    dims = [2 + i for i in range(spatial_rank)]

    return tf.transpose(x, [0] + dims + [1])


def gradients(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
    gradient_filter: Callable = tf.abs,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the Gradient Back-propagation Visualization Method.

    This method expects `inputs` to be a batch of positional signals of
    shape ``BHW...C``, and will return a tensor of shape ``BH'W'...L``,
    where ``(H', W', ...)`` are the sizes of the visual receptive field
    in the explained activation layer and `L` is the number of labels
    represented within the model's output logits.

    If `indices` is passed, the specific logits indexed by elements in this
    tensor are selected before the gradients are computed, effectively
    reducing the columns in the jacobian, and the size of the output explaining map.

    Args:
        model (tf.keras.Model): the model being explained
        inputs (tf.Tensor): the input data
        indices (Optional[tf.Tensor], optional): indices that should be gathered
            from ``outputs``. Defaults to None.
        indices_axis (int, optional): the axis containing the indices to gather.
            Defaults to ``KERNEL_AXIS``.
        indices_batch_dims (int, optional): the number of dimensions to broadcast
            in the ``tf.gather`` operation. Defaults to ``-1``.
        spatial_axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.
        gradient_filter (Callable, optional): filter before channel combining.
            Defaults to tf.abs.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and saliency maps.

    References:

        - Simonyan, K., Vedaldi, A., & Zisserman, A. (2013).
          Deep inside convolutional networks: Visualising image classification
          models and saliency maps. arXiv preprint
          `arXiv:1312.6034 <https://arxiv.org/abs/1312.6034>`_.

    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        logits = model(inputs, training=False)
        logits = inspection.gather_units(
            logits, indices, indices_axis, indices_batch_dims
        )

    maps = tape.batch_jacobian(logits, inputs)
    maps = gradient_filter(maps)
    maps = tf.reduce_mean(maps, axis=-1)
    maps = transpose_jacobian(maps, len(spatial_axis))

    return logits, maps


def _resized_psi_dfx(
    inputs: tf.Tensor,
    outputs: tf.Tensor,
    sizes: tf.Tensor,
    psi: Callable = filters.absolute_normalize,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
) -> tf.Tensor:
    """Filter and resize the gradient tensor.

    Args:
        inputs (tf.Tensor): the input signal.
        outputs (tf.Tensor): the output signal.
        sizes (tf.Tensor): the expected sizes.
        psi (Callable, optional): the filtering function. Defaults to
            :func:`~keras_explainable.filters.absolute_normalize`.
        spatial_axis (Tuple[int], optional): the spatial axes in the signal.
            Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the resized and processed tensor.
    """
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
    psi: Callable = filters.absolute_normalize,
    biases: Optional[List[tf.Tensor]] = None,
):
    """Computes the Full-Grad Visualization Method.

    This technique adds the individual contributions of each bias factor
    in the model to the extracted gradient, forming the "full gradient"
    representation, and it can be summarized by the following equation:

    .. math::

      f(x) = ψ(∇_xf(x)\\odot x) +∑_{l\\in L}∑_{c\\in c_l} ψ(f^b(x)_c)

    This method expects `inputs` to be a batch of positional signals of
    shape ``BHW...C``, and will return a tensor of shape ``BH'W'...L``,
    where ``(H', W', ...)`` are the sizes of the visual receptive field
    in the explained activation layer and `L` is the number of labels
    represented within the model's output logits.

    If `indices` is passed, the specific logits indexed by elements in this
    tensor are selected before the gradients are computed, effectively
    reducing the columns in the jacobian, and the size of the output explaining map.

    Args:
        model (tf.keras.Model): the model being explained
        inputs (tf.Tensor): the input data
        indices (Optional[tf.Tensor], optional): indices that should be gathered
            from ``outputs``. Defaults to None.
        indices_axis (int, optional): the axis containing the indices to gather.
            Defaults to ``KERNEL_AXIS``.
        indices_batch_dims (int, optional): the number of dimensions to broadcast
            in the ``tf.gather`` operation. Defaults to ``-1``.
        spatial_axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.
        psi (Callable, optional): filter operation before combining the intermediate
            signals. Defaults to ``filters.absolute_normalize``.
        biases: (List[tf.Tensor], optional): list of biases associated with each
            intermediate signal exposed by the model. If none is passed, it will
            be infered from the endpoints (nodes) outputed by the model.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and saliency maps.

    References:
        - Srinivas S, Fleuret F. Full-gradient representation for neural network
          visualization. `arxiv.org/1905.00780 <https://arxiv.org/pdf/1905.00780.pdf>`_,
          2019.

    """
    shape = tf.shape(inputs)
    sizes = [shape[a] for a in spatial_axis]

    resized_psi_dfx_ = partial(
        _resized_psi_dfx,
        sizes=sizes,
        psi=psi,
        spatial_axis=spatial_axis,
    )

    if biases is None:
        _, *intermediates = (i._keras_history.layer for i in model.outputs)
        biases = inspection.biases(intermediates)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        logits, *intermediates = model(inputs, training=False)
        logits = inspection.gather_units(
            logits, indices, indices_axis, indices_batch_dims
        )

    grad_input, *grad_inter = tape.gradient(logits, [inputs, *intermediates])

    maps = resized_psi_dfx_(inputs, grad_input)
    for b, i in zip(biases, grad_inter):
        maps += resized_psi_dfx_(b, i)

    return logits, maps


METHODS = [
    gradients,
    full_gradients,
]
"""Available Gradient-based AI Explaining methods.

This list contains all available methods implemented in this module,
and it is kept and used for introspection and validation purposes.
"""
