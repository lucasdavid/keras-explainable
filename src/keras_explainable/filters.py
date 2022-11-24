"""Shortcuts for commonly used signal filters used in literature.

These filters can be used as post or mid processing for explaining
methods and techniques. For example, to account for the absolute
pixel contribution, when computing the gradient of a score unit
with respect to the input image:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import numpy as np
    import keras_explainable as ke

.. jupyter-execute::

    x = 5 * np.random.normal(size=(4, 16, 16, 3))
    y = ke.filters.absolute_normalize(x).numpy()
    print(f"[{x.min()}, {x.max()}] -> [{y.min()}, {y.max()}]")

"""

from typing import Tuple

import tensorflow as tf

from keras_explainable.inspection import SPATIAL_AXIS


def normalize(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Normalize the signal into the interval [0, 1].

    Args:
        x (tf.Tensor): the input signal to be normalized.
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the normalized signal.
    """
    x = tf.convert_to_tensor(x)
    x -= tf.reduce_min(x, axis=axis, keepdims=True)

    return tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))


def positive(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Retain only positive values of the input signal.

    Args:
        x (tf.Tensor): the input signal.
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the filtered signal.
    """
    return tf.nn.relu(x)


def negative(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Retain only negative values of the input signal.

    Args:
        x (tf.Tensor): the input
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the filtered signal.
    """
    return tf.maximum(x, 0)


def positive_normalize(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Retain only positive values of the input signal and normalize it between 0 and 1.

    Args:
        x (tf.Tensor): the input signal.
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the filtered signal.
    """
    return normalize(positive(x, axis=axis), axis=axis)


def absolute_normalize(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Absolute values of the input signal and normalize it between 0 and 1.

    Args:
        x (tf.Tensor): the input signal.
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the filtered signal.
    """
    return normalize(tf.abs(x), axis=axis)


def negative_normalize(x: tf.Tensor, axis: Tuple[int] = SPATIAL_AXIS) -> tf.Tensor:
    """Retain only negative values of the input signal and normalize it between 0 and 1.

    Args:
        x (tf.Tensor): the input signal.
        axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.

    Returns:
        tf.Tensor: the filtered signal.
    """
    return normalize(negative(x), axis=axis)


__all__ = [
    "normalize",
    "positive",
    "negative",
    "positive_normalize",
    "absolute_normalize",
    "negative_normalize",
]
