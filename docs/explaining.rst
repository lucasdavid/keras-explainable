==============================
Explaining Model's Predictions
==============================

This library has the function :func:`~keras_explainable.explain` as core
component, which is used to execute any AI explaining method and technique.
Think of it as the :meth:`keras.Model.fit` or :meth:`keras.Model.predict`
loops of Keras' models, in which the execution graph of the operations
contained in a model is compiled (conditioned to :attr:`Model.run_eagerly`
and :attr:`Model.jit_compile`) and the explaining maps are computed
according to the method's strategy.

Just like in :meth:`keras.model.predict`, :func:`~keras_explainable.explain`
allows various types of input data and retrieves the Model's associated
distribute strategy in order to distribute the workload across multiple
GPUs and/or workers.


.. jupyter-execute::
  :hide-code:
  :hide-output:

  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from keras.utils import load_img, img_to_array

  import keras_explainable as ke

  SOURCE_DIRECTORY = 'docs/_static/images/singleton/'
  SAMPLES = 8
  SIZES = (224, 224)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]

We demonstrate bellow how predictions can be explained using the
ResNet101 network trained over ImageNet, using a few image samples.
Firstly, we load the network:

.. jupyter-execute::

  rn101 = tf.keras.applications.ResNet101V2(
    classifier_activation=None,
    weights='imagenet',
  )

  prec = tf.keras.applications.resnet_v2.preprocess_input
  decode_predictions = tf.keras.applications.resnet_v2.decode_predictions

  print(f"Spatial map sizes: {rn101.get_layer('avg_pool').input.shape}")

We can feed-foward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputing the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  inputs = prec(images.astype("float").copy())
  logits = rn101.predict(inputs, verbose=0)

  indices = np.argsort(logits, axis=-1)[:, ::-1]
  probs = tf.nn.softmax(logits).numpy()
  predictions = decode_predictions(probs, top=1)

  ke.utils.visualize(
    images=images,
    titles=[
      ', '.join(f"{klass} {prob:.0%}" for code, klass, prob in p)
      for p in predictions
    ]
  )

Finally, we can simply run all available explaining methods:

.. jupyter-execute::

  explaining_units = indices[:, :1]  # First most likely class.

  # Gradient Back-propagation
  _, g_maps = ke.gradients(rn101, inputs, explaining_units)

  # Full-Gradient
  logits = ke.inspection.get_logits_layer(rn101)
  inters, biases = ke.inspection.layers_with_biases(rn101, exclude=[logits])
  rn101_exp = ke.inspection.expose(rn101, inters, logits)
  _, fg_maps = ke.full_gradients(rn101_exp, inputs, explaining_units, biases=biases)

  # CAM-Based
  rn101_exp = ke.inspection.expose(rn101)
  _, c_maps = ke.cam(rn101_exp, inputs, explaining_units)
  _, gc_maps = ke.gradcam(rn101_exp, inputs, explaining_units)
  _, gcpp_maps = ke.gradcampp(rn101_exp, inputs, explaining_units)
  _, sc_maps = ke.scorecam(rn101_exp, inputs, explaining_units)

Following the original Grad-CAM paper, we only consider the positive contributing regions
in the creation of the CAMs, crunching negatively contributing and non-related regions together:

.. jupyter-execute::

  all_maps = (g_maps, fg_maps, c_maps, gc_maps, gcpp_maps, sc_maps)

  _images = images.repeat(1 + len(all_maps), axis=0)
  _titles = 'original Gradients Full-Grad CAM Grad-CAM Grad-CAM++ Score-CAM'.split()
  _overlays = sum(zip([None] * len(images), *all_maps), ())

  ke.utils.visualize(_images, _titles, _overlays, cols=1 + len(all_maps))
