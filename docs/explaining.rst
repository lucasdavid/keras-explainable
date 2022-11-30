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
  SIZES = (299, 299)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]

We demonstrate bellow how predictions can be explained using the
Xception network trained over ImageNet, using a few image samples.
Firstly, we load the network:

.. jupyter-execute::

  model = tf.keras.applications.Xception(
    classifier_activation=None,
    weights='imagenet',
  )

  print(f"Spatial map sizes: {model.get_layer('avg_pool').input.shape}")

We can feed-forward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputting the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

  inputs = images / 127.5 - 1
  logits = model.predict(inputs, verbose=0)

  indices = np.argsort(logits, axis=-1)[:, ::-1]
  probs = tf.nn.softmax(logits).numpy()
  predictions = decode_predictions(probs, top=1)

  ke.utils.visualize(
    images=images,
    titles=[
      ", ".join(f"{klass} {prob:.0%}" for code, klass, prob in p)
      for p in predictions
    ]
  )

Finally, we can simply run all available explaining methods:

.. jupyter-execute::
  :hide-output:

  explaining_units = indices[:, :1]  # First most likely class.

  # Gradient Back-propagation
  _, g_maps = ke.gradients(model, inputs, explaining_units)

  # Full-Gradient
  logits = ke.inspection.get_logits_layer(model)
  inters, biases = ke.inspection.layers_with_biases(model, exclude=[logits])
  model_exp = ke.inspection.expose(model, inters, logits)
  _, fg_maps = ke.full_gradients(model_exp, inputs, explaining_units, biases=biases)

  # CAM-Based
  model_exp = ke.inspection.expose(model)
  _, c_maps = ke.cam(model_exp, inputs, explaining_units)
  _, gc_maps = ke.gradcam(model_exp, inputs, explaining_units)
  _, gcpp_maps = ke.gradcampp(model_exp, inputs, explaining_units)
  _, sc_maps = ke.scorecam(model_exp, inputs, explaining_units)

.. jupyter-execute::
  :hide-code:

  all_maps = (g_maps, fg_maps, c_maps, gc_maps, gcpp_maps, sc_maps)

  _images = images.repeat(1 + len(all_maps), axis=0)
  _titles = 'original Gradients Full-Grad CAM Grad-CAM Grad-CAM++ Score-CAM'.split()
  _overlays = sum(zip([None] * len(images), *all_maps), ())

  ke.utils.visualize(_images, _titles, _overlays, cols=1 + len(all_maps))

The functions above are simply shortcuts for
:func:`~keras_explainable.engine.explaining.explain`, using their conventional
hyper-parameters and post processing functions.
For more flexibility, you can use the regular form:

.. code-block:: python

  logits, cams = ke.explain(
    ke.methods.cam.gradcam,
    model_exp,
    inputs,
    explaining_units,
    batch_size=32,
    postprocessing=ke.filters.positive_normalize,
  )

While the :func:`~keras_explainable.engine.explaining.explain` function is a convenient
wrapper, transparently distributing the workload based on the distribution strategy
associated with the model, it is not a necessary component in the overall functioning
of the library. Alternatively, one can call any explaining method directly:

.. code-block:: python

  gradcam =  ke.methods.cams.gradcam
  # Uncomment the following to compile the explaining pass:
  # gradcam = tf.function(ke.methods.cams.gradcam, reduce_retracing=True, jit_compile=True)

  logits, cams = gradcam(model, inputs, explaining_units)

  cams = ke.filters.positive_normalize(cams)
  cams = tf.image.resize(cams, (299, 299)).numpy()
