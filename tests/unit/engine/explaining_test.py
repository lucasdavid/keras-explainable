from parameterized import parameterized

import tensorflow as tf
import numpy as np

import keras_explainable as ke

TEST_EXPLAIN_SANITY_GRADIENTS_EXCLUDE = (
  ke.methods.gradient.full_gradients,
)

class ExplainTest(tf.test.TestCase):
  BATCH = 2
  SHAPE = [64, 64, 3]
  RUN_EAGERLY = False

  def _build_model(self, run_eagerly=False, jit_compile=False):
    input_tensor = tf.keras.Input([None, None, 3], name='inputs')
    model = tf.keras.applications.ResNet50V2(
      weights=None,
      input_tensor=input_tensor,
      classifier_activation=None,
    )
    model.compile(
      optimizer='sgd',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
      run_eagerly=run_eagerly,
      jit_compile=jit_compile,
    )

    return model

  def _build_model_with_activations(self, run_eagerly=False, jit_compile=False):
    model = self._build_model(run_eagerly, jit_compile)

    return tf.keras.Model(
      inputs=model.inputs,
      outputs=[model.output, model.get_layer('avg_pool').input]
    )

  @parameterized.expand([(m,) for m in ke.methods.cams.METHODS])
  def test_explain_sanity_cams(self, explaining_method):
    model = self._build_model_with_activations()

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    logits, maps = ke.explain(explaining_method, model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  @parameterized.expand([
    (False, True),
    (True, False),
  ])
  def test_explain_cams_jit_compile(self, run_eagerly, jit_compile):
    model = self._build_model_with_activations(run_eagerly, jit_compile)

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    logits, maps = ke.explain(ke.methods.cams.gradcam, model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  @parameterized.expand([
    (m,)
    for m in ke.methods.gradient.METHODS
    if m not in TEST_EXPLAIN_SANITY_GRADIENTS_EXCLUDE
  ])
  def test_explain_sanity_gradients(self, explaining_method):
    model = self._build_model()

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    logits, maps = ke.explain(explaining_method, model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  def test_explain_tta_cam(self):
    model = self._build_model_with_activations()

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    explaining_method = ke.methods.meta.tta(
      ke.methods.cams.cam,
      scales=[0.5],
      hflip=True,
    )
    logits, maps = ke.explain(explaining_method, model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  def test_explain_smoothgrad(self):
    model = self._build_model(run_eagerly=True)

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    explaining_method = ke.methods.meta.smooth(
      ke.methods.gradient.gradients,
      repetitions=3,
      noise=0.1,
    )
    logits, maps = ke.explain(explaining_method, model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  def test_explain_sanity_fullgradients(self):
    model = self._build_model()
    logits = ke.inspection.get_logits_layer(model)
    inters, biases = ke.inspection.layers_with_biases(model, exclude=[logits])

    model_exposed = ke.inspection.expose(model, inters, logits)

    x, y = (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    )

    logits, maps = ke.explain(
      ke.methods.gradient.full_gradients,
      model_exposed,
      x,
      y,
      biases=biases,
    )

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))
