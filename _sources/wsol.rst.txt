=====================================
Weakly Supervised Object Localization
=====================================

Object localization cues can be extracted from models trained over
multi-label datasets in a weakly supervised setup.

.. jupyter-execute::
  :hide-code:
  :hide-output:

  import os
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from keras.utils import load_img, img_to_array

  import keras_explainable as ke

  SOURCE_DIRECTORY = 'docs/_static/images/voc12/'
  SAMPLES = 8
  SIZES = (299, 299)

  file_names = os.listdir(SOURCE_DIRECTORY)
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]

  def pascal_voc_colors(classes):
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
      for channel in range(3):
        colormap[:, channel] |= ((ind >> channel) & 1) << shift
      ind >>= 3

    return np.concatenate((
      colormap[:classes], colormap[255:]
    ))

  def pascal_voc_classes():
    return np.asarray((
      "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable"
      "dog horse motorbike person pottedplant sheep sofa train tvmonitor"
    ).split())

Firstly, we employ the :py:class:`ResNet38-d` network pre-trained over the
Pascal VOC 2012 dataset:

.. jupyter-execute::

  WEIGHTS = 'voc2012'
  CLASSES = pascal_voc_classes()
  COLORS = pascal_voc_colors(classes=len(CLASSES) + 1)

  ! wget -nc https://raw.githubusercontent.com/lucasdavid/resnet38d-tf/main/resnet38d.py

  from resnet38d import ResNet38d

  input_tensor = tf.keras.Input(shape=(*SIZES, 3), name="inputs")
  rn38d = ResNet38d(input_tensor=input_tensor, weights=WEIGHTS)

  print(f"ResNet38-d with {WEIGHTS} pre-trained weights loaded.")
  print(f"Spatial map sizes: {rn38d.get_layer('s5/ac').input.shape}")

  ! rm resnet38d.py

We can feed-foward the samples once and get the predicted classes for each sample.
Besides making sure the model is outputing the expected classes, this step is
required in order to determine the most activating units in the *logits* layer,
which improves performance of the explaining methods.

.. jupyter-execute::

  prec = tf.keras.applications.imagenet_utils.preprocess_input

  inputs = prec(images.astype("float").copy(), mode='torch')
  probs = rn38d.predict(inputs, verbose=0)

Finally, we can simply run all available explaining methods:

.. jupyter-execute::

  rn38d = ke.inspection.expose(rn38d, "s5/ac", 'avg_pool')
  _, maps = ke.cam(rn38d, inputs)

Explaining maps can be converted into color maps,
respecting the conventional Pascal color mapping:

.. jupyter-execute::

  @tf.function(reduce_retracing=True, jit_compile=True)
  def cams_to_colors(maps, probabilities, threshold=0.5):
    detected = tf.cast(probabilities > threshold, maps.dtype)

    return tf.einsum('bhwl,bl,lc->bwhc', maps, detected, COLORS[:20])

  overlays = cams_to_colors(maps, probs)

  ke.utils.visualize(images, overlays=overlays)
