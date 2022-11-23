import tensorflow as tf
import numpy as np

import keras_explainable as ke


class InspectionTest(tf.test.TestCase):
  BATCH = 2
  SHAPE = [64, 64, 3]
  RUN_EAGERLY = False

  def _compile(self, model, run_eagerly=False, jit_compile=False):
    model.compile(
      optimizer='sgd',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
      run_eagerly=run_eagerly,
      jit_compile=jit_compile,
    )

    return model
  
  def _assert_valid_results(self, model):
    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    logits, maps = ke.gradcam(model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

    self.assertGreater(maps.max(), 0.)

  def test_functional_backbone(self):
    model = tf.keras.applications.ResNet50V2(
      input_shape=[64, 64, 3],
      classifier_activation=None,
      weights=None,
      classes=10,
    )
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)

  def test_sequential_nested_backbone(self):
    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
    )
    model = tf.keras.Sequential([
      tf.keras.Input([None, None, 3]),
      rn50,
      tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
      tf.keras.layers.Dense(10, name="logits"),
      tf.keras.layers.Activation("softmax", name="predictions"),
    ])
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)

  def test_sequential_nested_backbone_with_pooling(self):
    self.skipTest(
      "Nodes are not being properly appended when a model is nested. "
      "Skipping this test while this is not fixed. "
      "See #34977 and #16123 for more information."
    )

    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
      pooling="avg",
    )
    model = tf.keras.Sequential([
      tf.keras.Input([None, None, 3]),
      rn50,
      tf.keras.layers.Dense(10, name="logits"),
      tf.keras.layers.Activation("softmax", name="predictions"),
    ])
    self._compile(model)

    exposed = ke.inspection.expose(
      model,
      {"name": ("resnet50v2", "avg_pool"), "link": "input", "node": 1}
    )
    self._assert_valid_results(exposed)

  def test_functional_nested_backbone(self):
    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
    )
    x = tf.keras.Input([None, None, 3])
    y = rn50(x)
    y = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(y)
    y = tf.keras.layers.Dense(10, name="logits")(y)
    y = tf.keras.layers.Activation("softmax", name="predictions")(y)
    model = tf.keras.Model(x, y)
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)

  def test_functional_nested_backbone_with_pooling(self):
    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
      pooling="avg",
    )
    x = tf.keras.Input([None, None, 3])
    y = rn50(x)
    y = tf.keras.layers.Dense(10, name="logits")(y)
    y = tf.keras.layers.Activation("softmax", name="predictions")(y)
    model = tf.keras.Model(x, y)
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)

  def test_functional_flatten_backbone(self):
    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
    )
    y = rn50.output
    y = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(y)
    y = tf.keras.layers.Dense(10, name="logits")(y)
    y = tf.keras.layers.Activation("softmax", name="predictions")(y)
    model = tf.keras.Model(rn50.input, y)
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)

  def test_functional_nested_backbone_with_pooling(self):
    rn50 = tf.keras.applications.ResNet50V2(
      input_shape=[None, None, 3],
      include_top=False,
      weights=None,
      pooling="avg",
    )
    y = rn50.output
    y = tf.keras.layers.Dense(10, name="logits")(y)
    y = tf.keras.layers.Activation("softmax", name="predictions")(y)
    model = tf.keras.Model(rn50.input, y)
    self._compile(model)

    exposed = ke.inspection.expose(model)
    self._assert_valid_results(exposed)
