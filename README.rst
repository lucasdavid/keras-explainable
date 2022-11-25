=================
keras Explainable
=================

.. image:: https://github.com/lucasdavid/keras-explainable/actions/workflows/ci.yml/badge.svg?branch=release
  :alt: Travis build status
  :target: https://github.com/lucasdavid/keras-explainable/actions/workflows/ci.yml

.. image:: https://img.shields.io/badge/docs-0.0.2-blue
  :alt: Documentation status
  :target: https://lucasdavid.github.io/keras-explainable

Efficient explaining AI algorithms for Keras models.

.. image:: _static/images/cover.jpg
   :alt: Examples of explaining methods employed to explain outputs from various example images.

Installation
------------

.. code-block:: shell

  pip install tensorflow
  pip install git+https://github.com/lucasdavid/keras-explainable.git

Usage
-----

This example illustrate how to explain predictions of a Convolutional Neural
Network (CNN) using Grad-CAM. This can be easily achieved with the following
example:

.. code-block:: python

  import keras_explainable as ke

  model = tf.keras.applications.ResNet50V2(...)
  model = ke.inspection.expose(model)

  scores, cams = ke.gradcam(model, x, y, batch_size=32)

Implemented Explaining Methods
------------------------------

.. table::
   :widths: auto
   :align: left

   ===========================  =========  ======================================================  ==================
   Method                       Kind       Description                                             Reference                                                                                
   ===========================  =========  ======================================================  ==================
   Gradient Back-propagation    gradient   Computes the gradient of the output activation unit     `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.gradient.gradients>`_
                                           being explained with respect to each unit in the input  `paper <https://arxiv.org/abs/1312.6034>`_
                                           signal.
   Full-Gradient                gradient   Adds the individual contributions of each bias factor   `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.gradient.full_gradients>`_
                                           in the model to the extracted gradient, forming the     `paper <https://arxiv.org/abs/1905.00780>`_
                                           "full gradient" representation.
   CAM                          CAM        Creates class-specific maps by linearly combining the   `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.cams.cam>`_
                                           activation maps advent from the last convolutional      `paper <https://arxiv.org/abs/1512.04150>`_
                                           layer, scaled by their contributions to the unit of
                                           interest.
   Grad-CAM                     CAM        Linear combination of activation maps, weighted by      `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.cams.gradcam>`_
                                           the gradient of the output unit with respect to the     `paper <https://arxiv.org/abs/1610.02391>`_
                                           maps themselves.
   Grad-CAM++                   CAM        Weights pixels in the activation maps in order to       `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.cams.gradcampp>`_
                                           counterbalance, resulting in similar activation         `paper <https://arxiv.org/abs/1710.11063>`_
                                           intensity over multiple instances of objects.
   Score-CAM                    CAM        Combines activation maps considering their              `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.cams.scorecam>`_
                                           contribution towards activation, when used to mask      `paper <https://arxiv.org/abs/1910.01279>`_
                                           Activation maps are used to mask the input signal,
                                           which is feed-forwarded and activation intensity is
                                           computed for the new . Maps are combined weighted by
                                           their relative activation retention.
   SmoothGrad                   Meta       Consecutive applications of an AI explaining method,    `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.meta.smooth>`_
                                           adding Gaussian noise to the input signal each time.    `paper <https://arxiv.org/abs/1706.03825>`_
   TTA                          Meta       Consecutive applications of an AI explaining method,    `docs <https://lucasdavid.github.io/keras-explainable/api/keras_explainable.methods.html#keras_explainable.methods.meta.tta>`_
                                           applying augmentation to the input signal each time.    `paper <https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0/>`_
   ===========================  =========  ======================================================  ==================
