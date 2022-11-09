import tensorflow as tf

from keras_explainable.inspection import SPATIAL_AXIS

def normalize(x, axis=SPATIAL_AXIS):
  """Normalize a positional signal between 0 and 1."""
  x = tf.convert_to_tensor(x)
  x -= tf.reduce_min(x, axis=axis, keepdims=True)

  return tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))

def positive(x, axis=SPATIAL_AXIS):
  return tf.nn.relu(x)

def negative(x, axis=SPATIAL_AXIS):
  return tf.nn.relu(-x)

def positive_normalize(x, axis=SPATIAL_AXIS):
  return normalize(positive(x, axis=axis), axis=axis)

def absolute_normalize(x, axis=SPATIAL_AXIS):
  return normalize(tf.abs(x), axis=axis)

def negative_normalize(x, axis=SPATIAL_AXIS):
  return normalize(negative(x), axis=axis)
