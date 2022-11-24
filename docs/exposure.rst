=============================
Exposing Intermediate Signals
=============================

This page details the exposure procedure, necessary for most AI explaining
methods, and which can be easened with the help of the
:func:`~keras_explainable.inspection.expose` function.

Simple Exposition Examples
--------------------------

Many explaining techniques require us to expose the intermediate tensors
so their respective signals can be used, or so the gradient of the output
can be computed with respect to their signals.
For example, Grad-CAM computes the gradient of an output unit with respect
to the activation signal advent from the last positional layer in the model:

.. code-block:: python

  with tf.GradientTape() as tape:
    logits, activations = model(x)

  gradients = tape.batch_jacobian(logits, activations)

Which evidently means the ``activations`` signal, a tensor of
shape ``(batch, height, width, ..., kernels)`` must be available at runtime.
For that to happen, we must redefine the model, setting its outputs
to contain the :class:`KerasTensor`'s objects that reference both
``logits`` and ``activations`` tensors:

.. jupyter-execute::

  import numpy as np
  import tensorflow as tf
  from keras import Input, Model, Sequential
  from keras.applications import ResNet50V2
  from keras.layers import Activation, Dense, GlobalAveragePooling2D

  import keras_explainable as ke

  rn50 = ResNet50V2(weights=None, classifier_activation=None)
  # activations_tensor = rn50.get_layer("avg_pool").input  # or...
  activations_tensor = rn50.get_layer("post_relu").output

  model = Model(rn50.input, [rn50.output, activations_tensor])
  print("Network's outputs:")
  for o in model.outputs:
    print(f"  {o}")

Which can be simplified with:

.. code-block:: python

  model = ke.inspection.expose(rn50)

The :func:`~keras_explainable.inspection.expose` function inspects the model,
seeking for the *logits* layer (the last containing a kernel property) and the
*global pooling* layer, an instance of a :class:`GlobalPooling` or
:class:`Flatten` layer classes. The output of the former and the input of the
latter are collected and a new model is defined.

You can also manually indicate the name of the argument and output layers.
All options bellow are equivalent:

.. code-block:: python

  model = ke.inspection.expose(rn50, "post_relu", "predictions")
  model = ke.inspection.expose(
    rn50,
    {"name": "post_relu", "link": "output"},
    {"name": "predictions"},
  )
  model = ke.inspection.expose(
    rn50,
    {"name": "post_relu", "link": "output", "node": 0},
    {"name": "predictions", "link": "output", "node": 0},
  )
  model = ke.inspection.expose(
    rn50,
    {"name": "avg_pool", "link": "input"},
    "predictions",
  )

Grad-CAM (or Grad-CAM++) can be called immediately after that:

.. jupyter-execute::

  inputs = np.random.normal(size=(4, 224, 224, 3))
  indices = np.asarray([[13], [9], [32], [164]])

  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

Exposing Nested Models
----------------------

Unfortunately, some model's topologies can make exposition a little tricky.

Empty Activation and Explaining Maps
""""""""""""""""""""""""""""""""""""

Collecting loose endpoints from nested models can inadvertently result in
constant being used as arguments, producing in a zero gradient signal:

.. jupyter-execute::

  rn50 = ResNet50V2(weights=None, include_top=False)

  x = Input([224, 224, 3], name="input_images")
  y = rn50(x)
  y = GlobalAveragePooling2D(name="avg_pool")(y)
  y = Dense(10, name="logits")(y)
  y = Activation("softmax", name="predictions", dtype="float32")(y)

  rn50_clf = Model(x, y, name='resnet50v2_clf')
  rn50_clf.summary()

  model = ke.inspection.expose(rn50_clf, "resnet50v2", "predictions")
  scores, cams = ke.gradcam(model, inputs, indices)
  
  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

Notice all pixels in the ``cams`` variable are equal to zero. This can be
fixed by accessing the actual input node from the global pooling layer:

.. jupyter-execute::

  model = ke.inspection.expose(
    rn50_clf,
    {"name": "avg_pool", "link": "input"},
    "predictions"
  )
  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

.. jupyter-execute::
  :hide-code:
  :hide-output:

  del rn50, rn50_clf, model

.. note::
  
  Looking for the *input* the *Global Pooling* layer is the default
  behavior of the :func:`~keras_explainable.inspection.expose` function.

Access Nested Layer Signals
"""""""""""""""""""""""""""

Another problem occurs when the global pooling layer is not part of layers set
of the out-most model. While you can still collect its output using a name 
composition, we get a ``ValueError: Graph disconnected``:

.. jupyter-execute::
  :raises: ValueError

  rn50 = ResNet50V2(weights=None, include_top=False, pooling="avg")
  rn50_clf = Sequential([
    Input([224, 224, 3], name="input_images"),
    rn50,
    Dense(10, name="logits"),
    Activation("softmax", name="predictions", dtype="float32"),
  ])

  model = ke.inspection.expose(
    rn50_clf,
    {"name": ("resnet50v2", "avg_pool"), "link": "input"},
    "predictions"
  )
  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

This problem occurs because the identifier ``"ResNet50V2"`` passed
in the function parameter ``arguments`` will be expanded into
``{"name": "ResNet50V2", "link": "output", "node": 0}``, and
result in the collection of the :class:`KerasTensor`
``rn50_clf.get_layer("resnet50v2").get_output_at(0)``, or, equivalently,
``rn50_clf.get_layer("resnet50v2").output``.

As ``rn50_clf``'s layers are associated to two execution graphs at once
(the one created when invoking ``ResNet50V2(...)``, and the other created
when instantiating ``Sequential([...])``), they contain two output nodes. At
the same time, the GAP, Dense and Activation layers are associated with
exactly one :class:`Node` (these layers did not appear in the first model, and
thus, are not included in the first execution graph).

By exposing the first :class:`Node` associated with the *ResNet50V2* layer and
the first :class:`Node` associated with the *logits* layer, we redefined the
model to output an :class:`KerasTensor` which is not connected to the input
``rn50_clf.input``. Thus, the exception is raised.

.. warning::

  Since TensorFlow 2, nodes are no longer being stacked in ``_inbound_nodes``
  for layers in nested models, which obstructs the access to intermediate
  signals contained in a nested model, and makes the remaining of this
  document obsolete.
  To avoid this problem, it is recommended to "flat out" the model before
  explaining it, or avoiding nesting models altogether.
  
  For more information, see the GitHub issue
  `#16123 <https://github.com/keras-team/keras/issues/16123>`_.

To solve this problem, we must collect the second node:

.. jupyter-execute::
  :raises: ValueError

  model = ke.inspection.expose(
    rn50_clf,
    {"name": ("resnet50v2", "avg_pool"), "link": "input", "node": 1},
    "predictions"
  )
  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

Another example, this time using the functional API:

.. jupyter-execute::
  :raises: ValueError

  rn50 = ResNet50V2(weights=None, include_top=False, pooling="avg")

  x = Input([224, 224, 3], name="input_images")
  y = rn50(x)
  y = Dense(10, name="logits")(y)
  y = Activation("softmax", name="predictions", dtype="float32")(y)

  rn50_clf = Model(x, y, name='resnet50v2_clf')
  rn50_clf.summary()

  model = ke.inspection.expose(rn50_clf, ("resnet50v2", "post_relu"), "predictions")
  scores, cams = ke.gradcam(model, inputs, indices)
  
  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

Which can be correctly accessed with:

.. jupyter-execute::
  :raises: ValueError

  rn50 = ResNet50V2(weights=None, include_top=False, pooling="avg")

  x = Input([224, 224, 3], name="input_images")
  y = rn50(x)
  y = Dense(10, name="logits")(y)
  y = Activation("softmax", name="predictions", dtype="float32")(y)

  rn50_clf = Model(x, y, name='resnet50v2_clf')
  rn50_clf.summary()

  model = ke.inspection.expose(
    rn50_clf,
    {"name": ("resnet50v2", "post_relu"), "node": 1},
    "predictions"
  )
  scores, cams = ke.gradcam(model, inputs, indices)
  
  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

.. note::

  The following would also have worked:

  .. code-block:: python

    model = ke.inspection.expose(
      rn50_clf,
      {"name": ("resnet50v2", "avg_pool"), "link": "input", "node": 1},
      "predictions"
    )
