===========
Smooth-Grad
===========

In this page, we describe how to obtain *saliency maps* from a trained
Convolutional Neural Network (CNN) with respect to an input signal (an image,
in this case) using the Smooth-Grad AI explaining method.

Smooth-Grad is the variant of the Gradient Backprop algorithm first described
in the following paper:

Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017).
Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.
Available at: `arxiv/1706.03825 <https://arxiv.org/abs/1706.03825>`_.

It consists of consecutive repetitions of the Gradient Backprop method,
each of which is applied over the original sample tempered with
some gaussian noise.
Finally, averaging the resulting explaining maps results in cleaner
visualization results, robust against marginal noise.

Briefly, this can be achieved with the following template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = build_model(...)
  model.layers[-1].activation = 'linear'  # Usually softmax or sigmoid.

  smoothgrad = ke.methods.meta.smooth(
    ke.methods.gradient.gradients,
    repetitions=10,
    noise=0.1
  )

  logits, maps = ke.explain(
    smoothgrad, model, x, y,
    batch_size=32,
    postprocessing=ke.filters.normalize,
  )

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
  SIZES = (299, 299)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]

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

  print(f'ResNet101 with {WEIGHTS} pre-trained weights loaded.')
  print(f"Spatial map sizes: {rn101.get_layer('avg_pool').input.shape}")

We can feed-foward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputing the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  prec = tf.keras.applications.resnet_v2.preprocess_input

  inputs = prec(images.astype("float").copy())
  logits = rn101.predict(inputs, verbose=0)
  indices = np.argsort(logits, axis=-1)[:, ::-1]
  explaining_units = indices[:, :1]  # First most likely class.

keras-explainable implements the Smooth-Grad with the meta explaining function
:func:`keras_explainable.methods.meta.smooth`, which means it wraps any
explaining method and smooths out its outputs. For example:

.. smoothgrad = ke.methods.meta.smooth(
..   ke.methods.gradient.gradients,
..   repetitions=20,
..   noise=0.1,
.. )
.. _, smoothed_maps = ke.explain(
..   smoothgrad(
..   rn101,
..   inputs,
..   explaining_units,
..   postprocessing=ke.filters.normalize,
.. )

.. jupyter-execute::

  smoothgrad = ke.methods.meta.smooth(
    tf.function(ke.methods.gradient.gradients, jit_compile=True),
    repetitions=20,
    noise=0.1,
  )
  _, smoothed_maps = smoothgrad(rn101, inputs, explaining_units)
  smoothed_maps = ke.filters.absolute_normalize(smoothed_maps).numpy()

For comparative purposes, we also compute the vanilla gradients method:

.. jupyter-execute::

  _, maps = ke.gradients(rn101, inputs, explaining_units)

  ke.utils.visualize(
    sum(zip(images, maps, smoothed_maps), ()),
    cols=3
  )
