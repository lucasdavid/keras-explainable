==================
Gradient Back-prop
==================

In this page, we describe how to obtain *saliency maps* from a trained
Convolutional Neural Network (CNN) with respect to an input signal (an image,
in this case) using the Gradient backprop AI explaining method.
Said maps can be used to explain the model's predictions, determining regions
which most contributed to its effective output. 

Gradient Back-propagation (or Gradient Backprop, for short) is an early
form of visualizing and explaining the salient and contributing features
considered in the decision process of a neural network, being first
described in the following article:

Simonyan, K., Vedaldi, A., & Zisserman, A. (2013).
Deep inside convolutional networks: Visualising image classification
models and saliency maps. arXiv preprint arXiv:1312.6034.
Available at: https://arxiv.org/abs/1312.6034

Briefly, this can be achieved with the following template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = build_model(...)
  model.layers[-1].activation = 'linear'  # Usually softmax or sigmoid.

  logits, maps = ke.gradients(model, x, y, batch_size=32)

We describe bellow these lines in detail.

.. jupyter-execute::
  :hide-code:
  :hide-output:

  import os
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from keras.utils import load_img, img_to_array

  import keras_explainable as ke

  SOURCE_DIRECTORY = '_static/images/'
  SAMPLES = 8
  SIZES = (299, 299)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f)
                 for f in file_names
                 if f != '_links.txt']

  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])

  print('Images shape =', images.shape[1:])
  print('Images avail =', len(images))
  print('Images used  =', SAMPLES)

  images = images[:SAMPLES]

Firstly, we employ the :py:class:`ResNet101` network pre-trained over the
ImageNet dataset:

.. jupyter-execute::

  WEIGHTS = 'imagenet'

  input_tensor = tf.keras.Input(shape=(*SIZES, 3), name='inputs')

  rn101 = tf.keras.applications.ResNet101V2(
    input_tensor=input_tensor,
    classifier_activation=None,
    weights=WEIGHTS
  )
  rn101.trainable = False
  rn101.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
  )

  prec = tf.keras.applications.resnet_v2.preprocess_input
  decode_predictions = tf.keras.applications.resnet_v2.decode_predictions

  print(f'ResNet101 with {WEIGHTS} pre-trained weights loaded.')
  print(f"Spatial map sizes: {rn101.get_layer('avg_pool').input.shape}")

We can feed-foward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputing the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  inputs = prec(images.copy())
  logits = rn101.predict(inputs, verbose=0)

  indices = np.argsort(logits, axis=-1)[:, ::-1]
  probs = tf.nn.softmax(logits).numpy()
  predictions = decode_predictions(probs, top=1)

.. jupyter-execute::
  :hide-code:

  pd.DataFrame(sum(predictions, []), columns=['code', 'class', 'confidence'])

Gradient Backprop can be obtained by computing the differential of a function
(usually expressing the logit score for a given class) with respect to pixels
contained in the input signal (usually expressing an image):

.. jupyter-execute::

  explaining_units = indices[:, :1]  # First most likely class.

  logits, maps = ke.gradients(rn101, inputs, explaining_units)

  ke.utils.visualize(sum(zip(images.astype(np.uint8), maps), ()), cols=4)

.. note::

  If the parameter ``indices`` in ``gradients`` is not set, an
  explanation for each unit in the explaining layer will be provided,
  possibly resuting in *OOM* errors for models containing many units.

  To increase efficiency, we sub-select only the top :math:`K` scoring
  classification units to explain. The jacobian will only be computed
  for these :math:`NK` outputs.

Inside the hood, :func:`keras_explainable.gradients` is simply
executing the following call to the
:func:`explain` function:

.. code-block:: python

  logits, maps = ke.explain(
    methods.gradient.gradients,
    rn101,
    inputs,
    explaining_units,
    postprocessing=filters.absolute_normalize,
  )

Following Gradient Backprop paper, we consider the positive and
negative contributing regions in the creation of the saliency maps
by computing their individual absolute contributions before
normalizing them. Different strategies can be employed by
changing the :python:`postprocessing` parameter.

.. note::

  For more information on the :func:`explain` function,
  check its documentation or its own examples page.

Of course, we can obtain the same result by directly
calling the :func:`methods.gradient.gradients` function (though it will
not laverage the model's inner distributed strategy and data optimizations
implemented in :func:`explaining.explain`):

.. jupyter-execute::

  gradients = tf.function(ke.methods.gradient.gradients, jit_compile=True, reduce_retracing=True)
  _, direct_maps = gradients(rn101, inputs, explaining_units)

  direct_maps = ke.filters.absolute_normalize(maps)
  direct_maps = tf.image.resize(direct_maps, inputs.shape[1:-1])
  direct_maps = direct_maps.numpy()

  np.testing.assert_array_almost_equal(maps, direct_maps)
  print('Maps computed with `explain` and `methods.gradient.gradients` are the same!')

  del logits, direct_maps
