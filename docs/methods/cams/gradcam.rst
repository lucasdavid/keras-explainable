========
Grad-CAM
========

This example illustrate how to explain predictions of a Convolutional Neural
Network (CNN) using Grad-CAM. This can be easily achieved with the following
code template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = tf.keras.applications.ResNet50V2(...)
  model = ke.inspection.expose(model)

  scores, cams = ke.gradcam(model, x, y, batch_size=32)

In this page, we describe how to obtain *Class Activation Maps* (CAMs) from a
trained Convolutional Neural Network (CNN) with respect to an input signal
(an image, in this case) using the Grad-CAM visualization method.
Said maps can be used to explain the model's predictions, determining regions
which most contributed to its effective output.

Grad-CAM is a form of visualizing regions that most contributed to the output
of a given logit unit of a neural network, often times associated with the
prediction of the occurrence of a class in the problem domain. This method
is first described in the following article:

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
(2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).

Briefly, this can be achieved with the following template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = build_model(...)
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

Firstly, we employ the :class:`ResNet50` network pre-trained over the
ImageNet dataset:

.. jupyter-execute::

  rn50 = tf.keras.applications.ResNet50V2(
    classifier_activation=None,
    weights='imagenet'
  )

  print(f'ResNet50 pretrained over ImageNet was loaded.')
  print(f"Spatial map sizes: {rn50.get_layer('avg_pool').input.shape}")

We can feed-foward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputing the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  from tensorflow.keras.applications.imagenet_utils import preprocess_input

  inputs = preprocess_input(images.astype("float").copy(), mode="tf")
  logits = rn50.predict(inputs, verbose=0)
  indices = np.argsort(logits, axis=-1)[:, ::-1]

  explaining_units = indices[:, :1]  # First most likely class of each sample.

Grad-CAM works by computing the differential of an activation function,
usually associated with the prediction of a given class, with respect to pixels
contained in the activation map retrieved from an intermediate convolutional
signal (oftentimes advent from the last convolutional layer).

CAM-based methods implemented here expect the model to output both logits and
activation signal, so their respective representative tensors are exposed and
the jacobian can be computed from the former with respect to the latter.
Hence, we modify the current `rn50` model --- which only output logits at this
time --- to expose both activation maps and logits signals:

.. jupyter-execute::

  rn101_exposed = ke.inspection.expose(rn50)
  _, cams = ke.gradcam(rn101_exposed, inputs, explaining_units)

  ke.utils.visualize(
    images,
    overlays=cams.clip(0., 1.).transpose((3, 0, 1, 2)).reshape(-1, *SIZES, 1),
    cols=4
  )

.. note::

  To increase efficiency, we sub-select only the top :math:`K` scoring
  classification units to explain. The jacobian will only be computed for
  these :math:`NK` outputs.

Breakdown of Model Exposure and Grad-CAM
""""""""""""""""""""""""""""""""""""""""

The function :py:func:`keras_explainable.inspection.expose` will take a
:class:`keras.Model` as argument and instantiate a new model that outputs
both logits and the activation signal immediately before the
*Global Average Pooling* layer.

Under the hood of our example,
:func:`keras_explainable.inspection.expose` is simply
collecting the input and output signals of the global pooling
and predictions layer, respectively:

.. code-block:: python

  activations = rn50.get_layer('avg_pool').input
  scores = rn50.get_layer('predictions').output

  rn101_exposed = tf.keras.Model(rn50.inputs, [scores, activations])

You can also provide hints regarding the argument and output signals, if
your model's topology is more complex or if you simply wish to compute the
Grad-CAM with respect to other layer than the last convolutional one:

.. code-block:: python

  rn101_exposed = ke.inspection.expose(rn50, 'conv5_out', 'predictions')

For nested models that were created from different Input objects, you can
further specify which nodes to access within each layer, which maintains
the computation graph connected:

.. code-block:: python

  from keras import Input, Sequential
  from keras.layers import Dense, Activation
  from keras.applications import ResNet50V2

  inputs = Input(shape=[None, None, 3])
  backbone = ResNet50V2(include_top=False, pooling='avg')
  model = Sequential([
    inputs,
    backbone,
    Dense(10, name='logits'),
    Activation('softmax', dtype='float32'),
  ])

  rn101_exposed = ke.inspection.expose(
    rn50,
    arguments={
      'name': 'rn50.avg_pool',
      'link': 'input',
      'index': 1
    },
    outputs='predictions'
  )

As for the :py:func:`ke.gradcam` function, it is only a shortcut for
``ke.explain(ke.methods.cams.gradcam, model, inputs, ...)``.

All explaining methods can also be called directly:

.. code-block:: python

  gradcam = tf.function(ke.methods.cams.gradcam, reduce_retracing=True)
  logits, cams = gradcam(model, inputs, explaining_units)

  cams = ke.filters.positive_normalize(cams)
  cams = tf.image.resize(cams, SIZES).numpy()

Following the original Grad-CAM paper, we only consider the positive
contributing regions in the creation of the CAMs, crunching negatively
contributing and non-related regions together.
This is done automatically by :py:func:`ke.gradcam`, which assigns
the default value :py:func:`filters.positive_normalize` to the
``postprocessing`` parameter.
