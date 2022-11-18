===============================================================
Weakly Supervised Object Localization and Semantic Segmentation
===============================================================

Object localization and segmentation cues can be extracted from models
trained over multi-label datasets in a weakly supervised setup.

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
  SIZES = (384, 384)

  file_names = sorted(os.listdir(SOURCE_DIRECTORY))
  image_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in file_names if f != '_links.txt']
  images = np.stack([img_to_array(load_img(ip).resize(SIZES)) for ip in image_paths])
  images = images.astype("uint8")[:SAMPLES]
  label_indices = [[13], [12, 13], [6, 12, 13], [13], [11, 13],
            [4, 13], [1, 13], [8, 13], [13], [13]]
  labels = np.zeros((len(label_indices), 20))
  for i, l in enumerate(label_indices):
    labels[i, l] = 1.

  def pascal_voc_classes():
    return np.asarray((
      "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable"
      "dog horse motorbike person pottedplant sheep sofa train tvmonitor"
    ).split())

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

Firstly, we employ the :py:class:`ResNet38-d` network pre-trained over the
Pascal VOC 2012 dataset:

.. jupyter-execute::

  WEIGHTS = 'voc2012'
  CLASSES = pascal_voc_classes()
  COLORS = pascal_voc_colors(classes=len(CLASSES) + 1)

  ! wget -q -nc https://raw.githubusercontent.com/lucasdavid/resnet38d-tf/main/resnet38d.py

  from resnet38d import ResNet38d

  input_tensor = tf.keras.Input(shape=(384, 384, 3), name="inputs")
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

  def cams_to_colors(labels, maps, colors):
    overlays = []
    labels = labels.astype(bool)

    for i in range(8):
      l = labels[i]        # L
      c = colors[l]        # LC
      m = maps[i][..., l]  # HWL
      o = (m @ c).clip(0, 1)
      overlays.append(o)

    return overlays

  overlays = cams_to_colors(labels, maps, COLORS[:20])

  ke.utils.visualize(images, overlays=overlays, rows=2)
