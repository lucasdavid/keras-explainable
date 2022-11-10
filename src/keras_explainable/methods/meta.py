from functools import partial
from typing import Callable, List, Tuple

import tensorflow as tf

from keras_explainable.inspection import SPATIAL_AXIS

def smooth(
  method: Callable,
  repetitions: int = 20,
  noise: int = 0.1,
):
  """Smooth Meta Explaining Method.

  Args:
    method (Callable): the explaining method to be smoothed
    repetitions (int, optional): number of repetitions. Defaults to 20.
    noise (int, optional): standard deviation of the gaussian noise
      added to the input signal. Defaults to 0.1.

  References:

    - Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
      Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.
      Available at: https://arxiv.org/abs/1706.03825
  """
  def _smooth(
      model: tf.keras.Model,
      inputs: tf.Tensor,
      *args,
      **params,
  ):
    """Computes the Smooth-Grad Visualization Method.

    """

    outputs = method(model, inputs, *args, **params)
    shape = tf.shape(inputs)

    for step in tf.range(repetitions - 1):
      noisy_inputs = inputs + tf.random.normal(shape, 0, noise, inputs.dtype)
      noisy_outputs = method(model, noisy_inputs, *args, **params)

      for o, n in zip(outputs, noisy_outputs):
        o += n

    for o in outputs:
      o /= repetitions

    return outputs

  _smooth.__name__ = f'{method.__name__}_smooth'
  return _smooth

def tta(
  method: Callable,
  scales: List[float] = [0.5, 1.5, 2.],
  hflip: bool = True,
  resize_method: str = 'bilinear',
):
  """Computes the TTA version of a visualization method.

  """
  scales = tf.convert_to_tensor(scales, dtype=tf.float32)

  def _tta(
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
        logits_r, maps_r = _forward(method_, model, inputs, sizes, None, True, resize_method)
        logits += logits_r
        maps += maps_r

    for idx in tf.range(scales.shape[0]):
      scale = scales[idx]
      logits_r, maps_r = _forward(method_, model, inputs, sizes, scale, False, resize_method)
      logits += logits_r
      maps += maps_r

      if hflip:
        logits_r, maps_r = _forward(method_, model, inputs, sizes, scale, True, resize_method)
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

  _tta.__name__ = f'{method.__name__}_tta'
  return _tta

__all__ = [
  "smooth",
  "tta",
]
