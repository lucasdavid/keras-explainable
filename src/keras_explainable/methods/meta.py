"""Implementation of various Meta techniques.

These can be used conjointly with CAM and Gradient-based methods,
providing cleaner and more robust results.
"""

from functools import partial
from typing import Callable
from typing import List
from typing import Tuple

import tensorflow as tf

from keras_explainable.inspection import SPATIAL_AXIS


def smooth(
    method: Callable,
    repetitions: int = 20,
    noise: int = 0.1,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Smooth Meta Explaining Method.

    This technique consists of repeatedly applying an AI explaining method, considering
    small variations of the input signal each time (tempered with gaussian noise).

    Usage:

    .. code-block:: python

        x = np.random.normal((1, 224, 224, 3))
        y = np.asarray([[16, 32]])

        model = tf.keras.applications.ResNet50V2(classifier_activation=None)

        smoothgrad = ke.methods.meta.smooth(
            ke.methods.gradient.gradients,
            repetitions=20,
            noise=0.1,
        )

        scores, maps = smoothgrad(model, x, y)

    References:

      - Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
        SmoothGrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.
        Available at: [arxiv/1706.03825](https://arxiv.org/abs/1706.03825)

    Args:
      method (Callable): the explaining method to be smoothed
      repetitions (int, optional): number of repetitions. Defaults to 20.
      noise (int, optional): standard deviation of the gaussian noise
        added to the input signal. Defaults to 0.1.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and explaining maps.

    """
    def apply(
        model: tf.keras.Model,
        inputs: tf.Tensor,
        *args,
        **params,
    ):
        logits, maps = method(model, inputs, *args, **params)
        shape = tf.concat(([repetitions - 1], tf.shape(inputs)), axis=0)

        noisy_inputs = inputs + tf.random.normal(shape, 0, noise, dtype=inputs.dtype)

        with tf.control_dependencies([logits, maps]):
            for step in tf.range(repetitions - 1):
                batch_inputs = noisy_inputs[step]
                batch_logits, batch_maps = method(model, batch_inputs, *args, **params)

                logits += batch_logits
                maps += batch_maps

        return (
            logits / repetitions,
            maps / repetitions,
        )

    apply.__name__ = f"{method.__name__}_smooth"
    return apply


def tta(
    method: Callable,
    scales: List[float] = [0.5, 1.5, 2.0],
    hflip: bool = True,
    resize_method: str = "bilinear",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes the TTA version of a visualization method.

    Usage:

    .. code-block:: python

        x = np.random.normal((1, 224, 224, 3))
        y = np.asarray([[16, 32]])

        model = tf.keras.applications.ResNet50V2(classifier_activation=None)

        scores, maps = ke.explain(
            methods.gradient.gradients,
            rn50,
            inputs,
            explaining_units,
            postprocessing=filters.absolute_normalize,
        )

    Args:
        method (Callable): the explaining method to be augmented
        scales (List[float], optional): a list of coefs to scale the inputs by.
            Defaults to [0.5, 1.5, 2.0].
        hflip (bool, optional): wether or not to flip horizontally the inputs.
            Defaults to True.
        resize_method (str, optional): the resizing method used. Defaults to "bilinear".

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and explaining maps.
    """
    scales = tf.convert_to_tensor(scales, dtype=tf.float32)

    def apply(
        model: tf.keras.Model,
        inputs: tf.Tensor,
        spatial_axis: Tuple[int] = SPATIAL_AXIS,
        **params,
    ):
        method_ = partial(method, spatial_axis=spatial_axis, **params)

        shapes = tf.shape(inputs)
        sizes = shapes[1:-1]

        logits, maps = _forward(method_, model, inputs, sizes, None, False, resize_method)

        if hflip:
            with tf.control_dependencies([logits, maps]):
                logits_r, maps_r = _forward(
                    method_, model, inputs, sizes, None, True, resize_method
                )
                logits += logits_r
                maps += maps_r

        for idx in tf.range(scales.shape[0]):
            scale = scales[idx]
            logits_r, maps_r = _forward(
                method_, model, inputs, sizes, scale, False, resize_method
            )
            logits += logits_r
            maps += maps_r

            if hflip:
                logits_r, maps_r = _forward(
                    method_, model, inputs, sizes, scale, True, resize_method
                )
                logits += logits_r
                maps += maps_r

        repetitions = scales.shape[0]
        if hflip:
            repetitions *= 2

        logits /= repetitions
        maps /= repetitions

        return logits, maps

    def _forward(method, model, inputs, sizes, scale, hflip, resize_method):
        if hflip:
            inputs = tf.image.flip_left_right(inputs)

        if scale is not None:
            resizes = tf.cast(sizes, tf.float32)
            resizes = tf.cast(scale * resizes, tf.int32)
            inputs = tf.image.resize(inputs, resizes, method=resize_method)

        logits, maps = method(model, inputs)

        if hflip:
            maps = tf.image.flip_left_right(maps)

        maps = tf.image.resize(maps, sizes, method=resize_method)

        return logits, maps

    apply.__name__ = f"{method.__name__}_tta"
    return apply


__all__ = [
    "smooth",
    "tta",
]
