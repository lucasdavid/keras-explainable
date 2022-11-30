==============
Full Gradients
==============

In this page, we describe how to obtain *saliency maps* from a trained
Convolutional Neural Network (CNN) with respect to an input signal (an image,
in this case) using the Full Gradients AI explaining method.
Said maps can be used to explain the model's predictions, determining regions
which most contributed to its effective output. 

FullGrad (short for Full Gradients) extends Gradient Back-propagation by adding the
individual biases contributions to the gradient signal, forming the "full" explaining
maps. This technique is fully described in the paper "Full-gradient representation for
neural network visualization", published in Advances in neural information processing
systems, 32 by Srinivas, S., & Fleuret, F. (2019),
`arxiv.org/1905.00780v4 <https://arxiv.org/abs/1905.00780v4>`_.

Briefly, this can be achieved with the following template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = build_model(...)
  model.layers[-1].activation = 'linear'  # Usually softmax or sigmoid.

  logits = ke.inspection.get_logits_layer(model)
  inters, biases = ke.inspection.layers_with_biases(model, exclude=[logits])
  model = ke.inspection.expose(model, inters, logits)

  x, y = (
    np.random.rand(32, 512, 512, 3),
    np.random.randint(10, size=[32, 1])
  )

  logits, maps = ke.full_gradients(
    model,
    x,
    y,
    biases=biases,
  )

We describe bellow these lines in detail.
Firstly, we employ the :class:`Xception` network pre-trained over the
ImageNet dataset:

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

.. jupyter-execute::

  model = tf.keras.applications.Xception(
    classifier_activation=None,
    weights="imagenet",
  )

  print(f'Xception pretrained over ImageNet was loaded.')
  print(f"Spatial map sizes: {model.get_layer('avg_pool').input.shape}")

We can feed-forward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputting the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  from tensorflow.keras.applications.imagenet_utils import preprocess_input

  inputs = preprocess_input(images.astype("float").copy(), mode="tf")
  logits = model.predict(inputs, verbose=0)
  indices = np.argsort(logits, axis=-1)[:, ::-1]
  explaining_units = indices[:, :1]  # First-most likely classes.

The FullGrad algorithm, implemented through the
:func:`keras_explainable.methods.gradient.full_gradients`,
expects a model that exposes all layers containing biases (besides the output).
Thus, we must first expose them. The most efficient way to do so is
by collecting the layers directly:

.. jupyter-execute::

  logits = ke.inspection.get_logits_layer(model)
  inters, biases = ke.inspection.layers_with_biases(model, exclude=[logits])
  model = ke.inspection.expose(model, inters, logits)

Now we can obtain FullGrad by simply calling to the :func:`explain` function:

.. jupyter-execute::

  _, maps = ke.full_gradients(
    model,
    inputs,
    explaining_units,
    biases=biases,
  )

  ke.utils.visualize(
    images=[*images, *maps, *images],
    overlays=[None] * (2 * len(images)) + [*maps],
  )

.. note::

  Passing the list of ``biases`` as a parameter to the
  :func:`~keras_explainable.full_gradients` function is not required, but it
  is generally a good idea, as it avoids unnecessary recollection of those.
