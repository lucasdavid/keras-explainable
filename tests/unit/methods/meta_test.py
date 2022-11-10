import numpy as np
import tensorflow as tf

import keras_explainable as ke

class MetaTest(tf.test.TestCase):
  BATCH = 2
  SHAPE = [64, 64, 3]
  RUN_EAGERLY = False

  def _build_model(self, run_eagerly=RUN_EAGERLY):
    input_tensor = tf.keras.Input([None, None, 3], name='inputs')
    model = tf.keras.applications.ResNet50V2(
      weights=None,
      input_tensor=input_tensor,
      classifier_activation=None,
    )
    model.run_eagerly = run_eagerly

    return model

  def _build_model_with_activations(self, run_eagerly=RUN_EAGERLY):
    model = self._build_model(run_eagerly)

    return tf.keras.Model(
      inputs=model.inputs,
      outputs=[model.output, model.get_layer('avg_pool').input]
    )

  def test_sanity_tta_cam(self):
    model = self._build_model_with_activations()

    x, y = map(tf.convert_to_tensor, (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    ))

    tta = ke.methods.meta.tta(
      ke.methods.cams.cam,
      scales=[0.5],
      hflip=True,
    )
    logits, maps = tta(model, x, indices=y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))

  def test_sanity_smooth_grad(self):
    model = self._build_model(run_eagerly=False)

    x, y = map(tf.convert_to_tensor, (
      np.random.rand(self.BATCH, *self.SHAPE),
      np.random.randint(10, size=(self.BATCH, 1))
    ))

    smoothgrad = ke.methods.meta.smooth(
      ke.methods.gradient.gradients,
      repetitions=5,
      noise=0.2,
    )
    logits, maps = smoothgrad(model, x, y)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (self.BATCH, 1))

    self.assertIsNotNone(maps)
    self.assertEqual(maps.shape, (self.BATCH, *self.SHAPE[:2], 1))
