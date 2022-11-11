==============
Full Gradients
==============

In this page, we describe how to obtain *saliency maps* from a trained
Convolutional Neural Network (CNN) with respect to an input signal (an image,
in this case) using the Full Gradients AI explaining method.
Said maps can be used to explain the model's predictions, determining regions
which most contributed to its effective output. 

FullGrad (short for Full Gradients) extends Gradient Back-propagation by
adding the individual biases contributions to the gradient signal,
forming the "full" explaining maps. This fully described in the paper

Srinivas, S., & Fleuret, F. (2019). Full-gradient representation for
neural network visualization. Advances in neural information processing
systems, 32. `arxiv.org/1905.00780v4 <https://arxiv.org/abs/1905.00780v4>`_.

Briefly, this can be achieved with the following template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = build_model(...)
  model.layers[-1].activation = 'linear'  # Usually softmax or sigmoid.

  logits = ke.inspection.get_logits_layer(model)
  inters, biases = ke.inspection.layers_with_biases(model, exclude=[logits])
  model_exposed = ke.inspection.expose(model, inters, logits)

  x, y = (
    np.random.rand(self.BATCH, *self.SHAPE),
    np.random.randint(10, size=(self.BATCH, 1))
  )

  logits, maps = ke.full_gradients(
    model_exposed,
    x,
    y,
    biases=biases,
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
  explaining_units = indices[:, :1]  # Firstmost likely classes.

The FullGrad algorithm, implemented through the
:func:`keras_explainable.methods.gradient.full_gradients`,
expects a model that exposes all layers containing biases (besides the output).
Thus, we must first expose them. The most efficient way to do so is
by collecting the layers directly:

.. jupyter-execute::

  logits = ke.inspection.get_logits_layer(rn101)
  inters, biases = ke.inspection.layers_with_biases(rn101, exclude=[logits])
  model_exposed = ke.inspection.expose(rn101, inters, logits)

Now we can obtain FullGrad by simply calling to the :func:`explain` function:

.. jupyter-execute::

  _, maps = ke.full_gradients(
    model_exposed,
    inputs,
    explaining_units,
    biases=biases,
    postprocessing=ke.filters.normalize,
  )

  ke.utils.visualize(sum(zip(images, maps), ()), cols=4)

.. note::

  The parameter ``biases`` is not required, and will be inferred if not passed.
  Of course, you should pass it to the :func:`full_gradients` function,
  if it is known, as it avoids unnecessary digging/assumptions over the
  model's topology.
