=================
keras Explainable
=================

.. |Travis build| image:: https://img.shields.io/travis/lucasdavid/keras-explainable?style=for-the-badge
  :target: https://app.travis-ci.com/github/lucasdavid/keras-explainable

.. |Documentation| image:: https://img.shields.io/badge/docs-0.0.1-blue
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

   ===========================  =========  ========================================================================================
   Method                       Kind       Reference                                                                                
   ===========================  =========  ========================================================================================
   Gradient Back-propagation    gradient   `paper <https://arxiv.org/abs/1312.6034>`_                                                
   FullGrad                     gradient   `paper <https://arxiv.org/abs/1905.00780>`_                                               
   CAM                          CAM        `paper <https://arxiv.org/abs/1512.04150>`_                                               
   Grad-CAM                     CAM        `paper <https://arxiv.org/abs/1610.02391>`_                                               
   Grad-CAM++                   CAM        `paper <https://arxiv.org/abs/1710.11063>`_                                               
   Score-CAM                    CAM        `paper <https://arxiv.org/abs/1910.01279>`_                                               
   SmoothGrad                   Meta       `paper <https://arxiv.org/abs/1706.03825>`_                                               
   TTA                          Meta       `paper <https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0/>`_  
   ===========================  =========  ========================================================================================
