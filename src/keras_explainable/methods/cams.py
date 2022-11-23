"""Implementation of various CAM-based AI explaining methods and techniques.
"""
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf
from keras.backend import int_shape
from keras.engine.base_layer import Layer

from keras_explainable.filters import normalize
from keras_explainable.inspection import KERNEL_AXIS
from keras_explainable.inspection import SPATIAL_AXIS
from keras_explainable.inspection import gather_units
from keras_explainable.inspection import get_logits_layer


def cam(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
    logits_layer: Optional[Union[str, Layer]] = None,
):
    """Computes the CAM Visualization Method.

    This method expects `inputs` to be a batch of positional signals of
    shape ``BHW...C``, and will return a tensor of shape ``BH'W'...L``,
    where ``(H', W', ...)`` are the sizes of the visual receptive field
    in the explained activation layer and ``L`` is the number of labels
    represented within the model's output logits.

    If ``indices`` is passed, the specific logits indexed by elements in
    this tensor are selected before the gradients are computed,
    effectively reducing the columns in the jacobian, and the size of
    the output explaining map.

    Usage:

    .. code-block:: python

        import numpy as np
        import keras_explainable as ke

        x = np.random.normal((2, 224, 224, 3))
        y = np.asarray([
            [16, 32],
            [32, 64],
        ])

        model = tf.keras.applications.ResNet50V2(weights=None, include_top=True)
        model = ke.inspection.expose(model)

        scores, cams = ke.methods.cams.cam(model, x, y)

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
        logits_layer (Callable, optional): filter before channel combining.
            Defaults to tf.abs.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and Class Activation Maps (CAMs).

    References:
        - Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016).
          Learning deep features for discriminative localization. In Proceedings
          of the IEEE conference on computer vision and pattern
          recognition (pp. 2921-2929). Available at:
          `arxiv/1512.04150 <https://arxiv.org/pdf/1512.04150.pdf>`_.

    """
    logits, activations = model(inputs, training=False)
    logits = gather_units(logits, indices, indices_axis, indices_batch_dims)

    if isinstance(logits_layer, str) or logits_layer is None:
        logits_layer = get_logits_layer(model, name=logits_layer)

    weights = gather_units(
        tf.squeeze(logits_layer.kernel), indices, axis=-1, batch_dims=0
    )

    dims = "kc" if indices is None else "kbc"
    maps = tf.einsum(f"b...k,{dims}->b...c", activations, weights)

    return logits, maps


def gradcam(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
):
    """Computes the Grad-CAM Visualization Method.

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

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and Class Activation Maps (CAMs).

    References:
        - Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
          (2017). Grad-CAM: Visual explanations from deep networks via gradient-based
          localization. In Proceedings of the IEEE international conference on computer
          vision (pp. 618-626).
          Available at: `arxiv/1610.02391 <https://arxiv.org/abs/1610.02391>`_.

    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        logits, activations = model(inputs, training=False)
        logits = gather_units(logits, indices, indices_axis, indices_batch_dims)

    dlda = tape.batch_jacobian(logits, activations)
    weights = tf.reduce_mean(dlda, axis=spatial_axis)
    maps = tf.einsum("b...k,bck->b...c", activations, weights)

    return logits, maps


def gradcampp(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
):
    """Computes the Grad-CAM++ Visualization Method.

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

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and Class Activation Maps (CAMs).

    References:
        - Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
          (2018, March). Grad-cam++: Generalized gradient-based visual explanations
          for deep convolutional networks. In 2018 IEEE winter conference on
          applications of computer vision (WACV) (pp. 839-847). IEEE.
        - Grad-CAM++'s official implementation. Github. Available at:
          `adityac94/Grad-CAM++ <github.com/adityac94/Grad_CAM_plus_plus>`_

    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(inputs)
        logits, activations = model(inputs, training=False)
        logits = gather_units(logits, indices, indices_axis, indices_batch_dims)

    dlda = tape.batch_jacobian(logits, activations)

    dyda = tf.einsum("bc,bc...k->bc...k", tf.exp(logits), dlda)
    d2 = dlda**2
    d3 = dlda**3
    aab = tf.reduce_sum(activations, axis=spatial_axis)  # (BK)
    akc = tf.math.divide_no_nan(
        d2,
        2.0 * d2
        + tf.einsum("bk,bc...k->bc...k", aab, d3),  # (2*(BUHWK) + (BK)*BUHWK)
    )

    # Tensorflow has a glitch that doesn't allow this form:
    # weights = tf.einsum('bc...k,bc...k->bck', akc, tf.nn.relu(dyda))  # w: buk
    # So we use this one instead:
    weights = tf.reduce_sum(akc * tf.nn.relu(dyda), axis=spatial_axis)

    maps = tf.einsum(
        "bck,b...k->b...c", weights, activations
    )  # a: bhwk, m: buhw

    return logits, maps


def scorecam(
    model: tf.keras.Model,
    inputs: tf.Tensor,
    indices: Optional[tf.Tensor] = None,
    indices_axis: int = KERNEL_AXIS,
    indices_batch_dims: int = -1,
    spatial_axis: Tuple[int] = SPATIAL_AXIS,
):
    """Computes the Score-CAM Visualization Method.

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

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and Class Activation Maps (CAMs).

    References:
        - Score-CAM: Score-Weighted Visual Explanations for Convolutional
          Neural Networks. Available at:
          `arxiv/1910.01279 <https://arxiv.org/abs/1910.01279>`_

    """
    scores, activations = model(inputs, training=False)
    scores = gather_units(scores, indices, indices_axis, indices_batch_dims)

    classes = int_shape(scores)[-1] or tf.shape(scores)[-1]
    kernels = int_shape(activations)[-1] or tf.shape(activations)[-1]

    shape = tf.shape(inputs)
    sizes = [shape[a] for a in spatial_axis]
    maps = tf.zeros([shape[0]] + sizes + [classes])

    for i in tf.range(kernels):
        mask = activations[..., i : i + 1]
        mask = normalize(mask, axis=spatial_axis)
        mask = tf.image.resize(mask, sizes)

        si, _ = model(inputs * mask, training=False)
        si = gather_units(si, indices, indices_axis, indices_batch_dims)
        si = tf.einsum("bc,bhw->bhwc", si, mask[..., 0])
        maps += si

    return scores, maps


METHODS = [
    cam,
    gradcam,
    gradcampp,
    scorecam,
]
"""Available CAM-based AI Explaining methods.

This list contains all available methods implemented in this module,
and it is kept and used for introspection and validation purposes.
"""
