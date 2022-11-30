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

  print(model.name)
  print(f"  input: {model.input}")
  print("  outputs:")
  for o in model.outputs:
    print(f"    {o}")

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
  indices = np.asarray([[4], [9], [0], [2]])

  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

Exposing Nested Models
----------------------

Unfortunately, some model's topologies can make exposition a little tricky.
An example of this is when nesting multiple models, producing more than one
``Input`` object and multiple conceptual graphs at once.
Then, if one naively collects ``KerasTensor``'s from the model, disconnected
nodes may be retrieved, resulting in the exception ``ValueError: Graph disconnected``
being raised:

.. jupyter-execute::
  :raises: ValueError

  rn50 = ResNet50V2(weights=None, include_top=False)

  x = Input([224, 224, 3], name="input_images")
  y = rn50(x)
  y = GlobalAveragePooling2D(name="avg_pool")(y)
  y = Dense(10, name="logits")(y)
  y = Activation("softmax", name="predictions", dtype="float32")(y)

  rn50_clf = Model(x, y, name="resnet50v2_clf")
  rn50_clf.summary()

  logits = rn50_clf.get_layer("logits").output
  activations = rn50_clf.get_layer("resnet50v2").output

  model = tf.keras.Model(rn50_clf.input, [logits, activations])
  scores, cams = ke.gradcam(model, inputs, indices)
  
  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

The operations in ``rn50`` appear in two conceptual graphs. The first, defined
when ``ResNet50V2(...)`` was invoked, contains all operations associated with the layers
in the ResNet50 architecture. The second one, on the other hand, is defined when
invoking :meth:`Layer.__call__` of each layer (``rn50``, ``GAP``, ``Dense`` and
``Activation``).

When calling ``rn50_clf.get_layer("resnet50v2").output`` (which is equivalent
to ``rn50_clf.get_layer("resnet50v2").get_output_at(0)``), the :class:`Node`
from the first graph is retrieved.
This ``Node`` is not associated with ``rn50_clf.input`` or ``logits``, and thus
the error is raised.

There are multiple ways to correctly access the Node from the second graph. One of them
is to retrieve the input from the ``GAP`` layer, as it only appeared in one graph:

.. jupyter-execute::

  model = ke.inspection.expose(
    rn50_clf, {"name": "avg_pool", "link": "input"}, "predictions"
  )
  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")

.. jupyter-execute::
  :hide-code:
  :hide-output:

  del rn50, rn50_clf, model

.. note::

  The alternatives ``ke.inspection.expose(rn50_clf, "resnet50v2", "predictions")``
  and ``ke.inspection.expose(rn50_clf)`` would work as well.
  In the former, the **last** output node is retrieved.
  In the latter, the **last** input node (there's only one) associated
  with the ``GAP`` layer is retrieved.

Access Nested Layer Signals
"""""""""""""""""""""""""""

Another problem occurs when the global pooling layer is not part of layers set
of the out-most model. While you can still collect its output using a name 
composition, we get a ``ValueError: Graph disconnected``.

This problem occurs because Keras does not create ``Nodes`` for inner layers in a nested
model, when that model is reused. Instead, the model is treated as a single operation
in the conceptual graph, with a single new ``Node`` being created to represent it.
Calling :func:`keras_explainable.inspection.expose` over the model will expand the
parameter ``arguments`` into ``{"name": ("ResNet50V2", "avg_pool"), "link": "input", "node": "last"}``,
but because no new nodes were created for the ``GAP`` layer, the :class:`KerasTensor`
associated with the first conceptual graph is retrieved, and the error ensues.

.. jupyter-execute::
  :raises: ValueError

  rn50 = ResNet50V2(weights=None, include_top=False, pooling="avg")
  rn50_clf = Sequential([
    Input([224, 224, 3], name="input_images"),
    rn50,
    Dense(10, name="logits"),
    Activation("softmax", name="predictions", dtype="float32"),
  ])

  model = ke.inspection.expose(rn50_clf)
  scores, cams = ke.gradcam(model, inputs, indices)

  print(f"scores:{scores.shape} in [{scores.min()}, {scores.max()}]")
  print(f"cams:{cams.shape} in [{cams.min()}, {cams.max()}]")


.. warning::

  Since TensorFlow 2, nodes are no longer being stacked in ``_inbound_nodes``
  for layers in nested models, which obstructs the access to intermediate
  signals contained in a nested model, and makes the remaining of this
  document obsolete.
  To avoid this problem, it is recommended to "flat out" the model before
  explaining it, or avoiding nesting models altogether.

  For more information, see the GitHub issue
  `#16123 <https://github.com/keras-team/keras/issues/16123>`_.

If you are using TensorFlow < 2.0, nodes are created for each operation
in the inner model, and you may collect their internal signal by simply:

.. code-block:: python

  model = ke.inspection.expose(rn50_clf)
  # ... or: ke.inspection.expose(rn50_clf, ("resnet50v2", "post_relu"))
  # ... or: ke.inspection.expose(
  #  rn50_clf, {"name": ("resnet50v2", "avg_pool"), "link": "input"}
  # )
  
  scores, cams = ke.gradcam(model, inputs, indices)

.. note::

  The above works because :func:`~keras_explainable.inspection.expose`
  will recursively seek for a ``GAP`` layer within the nested models.
