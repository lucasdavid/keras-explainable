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
Available at: `arxiv/1312.6034 <https://arxiv.org/abs/1312.6034>`_.

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

  SOURCE_DIRECTORY = 'docs/_static/images/singleton/'
  SAMPLES = 8
  SIZES = (224, 224)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]

Firstly, we employ the :class:`ResNet50V2` network pre-trained over the
ImageNet dataset:

.. jupyter-execute::

  rn50 = tf.keras.applications.ResNet50V2(
    classifier_activation=None,
    weights='imagenet',
  )

  print(f'ResNet50 pretrained over ImageNet was loaded.')
  print(f"Spatial map sizes: {rn50.get_layer('avg_pool').input.shape}")

We can feed-forward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputting the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  from tensorflow.keras.applications.imagenet_utils import preprocess_input

  inputs = images / 127.5 - 1
  logits = rn50.predict(inputs, verbose=0)
  indices = np.argsort(logits, axis=-1)[:, ::-1]
  explaining_units = indices[:, :1]  # First most likely class.

Gradient Backprop can be obtained by computing the differential of a function
(usually expressing the logit score for a given class) with respect to pixels
contained in the input signal (usually expressing an image):

.. jupyter-execute::

  logits, maps = ke.gradients(rn50, inputs, explaining_units)

  ke.utils.visualize(sum(zip(images, maps), ()), cols=4)

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
    rn50,
    inputs,
    explaining_units,
    postprocessing=filters.absolute_normalize,
  )

Following Gradient Backprop paper, we consider the positive and
negative contributing regions in the creation of the saliency maps
by computing their individual absolute contributions before
normalizing them. Different strategies can be employed by
changing the ``postprocessing`` parameter.

.. note::

  For more information on the :func:`~keras_explainable.explain` function,
  check its documentation or its own examples page.

Of course, we can obtain the same result by directly calling the
:func:`~keras_explainable.methods.gradient.gradients` function
(though it will not leverage the model's inner distributed strategy
and data optimizations implemented in :func:`~keras_explainable.explain`):

.. jupyter-execute::

  gradients = tf.function(ke.methods.gradient.gradients, jit_compile=True, reduce_retracing=True)
  _, direct_maps = gradients(rn50, inputs, explaining_units)

  direct_maps = ke.filters.absolute_normalize(maps)
  direct_maps = tf.image.resize(direct_maps, inputs.shape[1:-1])
  direct_maps = direct_maps.numpy()

  np.testing.assert_array_almost_equal(maps, direct_maps)
  print('Maps computed with `explain` and `methods.gradient.gradients` are the same!')
