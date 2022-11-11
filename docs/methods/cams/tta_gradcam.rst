============
TTA Grad-CAM
============

Test-time augmentation (TTA) is a commonly employed strategy in Saliency
detection and Weakly Supervised Segmentation tasks order to obtain smoother
and more stable explaining maps.

We illustrate in this example how to apply TTA to AI explaining methods using
``keras-explainable``. This can be easily achieved with the following code
template snippet:

.. code-block:: python

  import keras_explainable as ke

  model = tf.keras.applications.ResNet50V2(...)
  model = ke.inspection.expose(model)

  tta_gradcam = ke.methods.meta.tta(
    ke.methods.cams.gradcam,
    scales=[0.5, 1.0, 1.5, 2.],
    hflip=True
  )
  _, cams = ke.explain(tta_gradcam, model, inputs)

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

.. jupyter-execute::

  rn101_exposed = ke.inspection.expose(rn101)

  tta_gradcam = ke.methods.meta.tta(
    ke.methods.cams.gradcam,
    scales=[0.5, 1.0, 1.5, 2.],
    hflip=True
  )
  _, cams = ke.explain(tta_gradcam, rn101_exposed, inputs, explaining_units)

  ke.utils.visualize(
    images,
    overlays=cams.clip(0., 1.).transpose((3, 0, 1, 2)).reshape(-1, *SIZES, 1),
    cols=4
  )
