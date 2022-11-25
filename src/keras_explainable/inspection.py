"""Inspection utils for models and layers.
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.engine.keras_tensor import KerasTensor
from keras.engine.training import Model
from keras.layers.normalization.batch_normalization import BatchNormalizationBase
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.layers.pooling.base_global_pooling1d import GlobalPooling1D
from keras.layers.pooling.base_global_pooling2d import GlobalPooling2D
from keras.layers.pooling.base_global_pooling3d import GlobalPooling3D
from keras.layers.reshaping.flatten import Flatten

from keras_explainable.utils import tolist

E = Union[str, int, tf.Tensor, KerasTensor, Dict[str, Union[str, int]]]

KERNEL_AXIS = -1
SPATIAL_AXIS = (-3, -2)

NORMALIZATION_LAYERS = (
    BatchNormalizationBase,
    LayerNormalization,
)

POOLING_LAYERS = (
    Flatten,
    GlobalPooling1D,
    GlobalPooling2D,
    GlobalPooling3D,
)


def get_nested_layer(
    model: Model,
    name: Union[str, List[str]],
) -> Layer:
    """Retrieve a nested layer in the model.

    Args:
        model (Model): the model containing the nested layer.
        name (Union[str, List[str]]): a string (or list of string) containing
            the name of the layer (or a list of names, each of which references
            a recursively nested module up to the layer of interest).

    Example:

    .. code-block:: python

        model = tf.keras.Sequential([
            tf.keras.applications.ResNet101V2(include_top=False, pooling='avg'),
            tf.keras.layers.Dense(10, activation='softmax', name='predictions')
        ])

        pooling_layer = get_nested_layer(model, ('resnet101v2', 'avg_pool'))

    Raises:
        ValueError: if ``name`` is not a nested member of ``model``.

    Returns:
        tf.keras.layer.Layer: the retrieved layer.
    """
    for n in tolist(name):
        model = model.get_layer(n)

    return model


def get_logits_layer(
    model: Model,
    name: str = None,
) -> Layer:
    """Retrieve the "logits" layer.

    Args:
        model (Model): the model containing the logits layer.
        name (str, optional): the name of the layer, if known. Defaults to None.

    Raises:
        ValueError: if a logits layer cannot be found

    Returns:
        Layer: the retrieved logits layer
    """
    return find_layer_with(model, name, properties=["kernel"])


def get_global_pooling_layer(
    model: Model,
    name: str = None,
) -> Layer:
    """Retrieve the last global pooling layer.

    Args:
        model (Model): the model containing the pooling layer.
        name (str, optional): the name of the layer, if known. Defaults to None.

    Raises:
        ValueError: if a pooling layer cannot be found

    Returns:
        Layer: the retrieved pooling layer
    """
    return find_layer_with(model, name, klass=POOLING_LAYERS)


def find_layer_with(
    model: Model,
    name: Optional[str] = None,
    properties: Optional[Tuple[str]] = None,
    klass: Optional[Tuple[Type[Layer]]] = None,
    search_reversed: bool = True,
) -> Layer:
    """Find a layer within a model that satisfies all required properties.

    Args:
        model (Model): the container model.
        name (Optional[str], optional): the name of the layer, if known.
          Defaults to None.
        properties (Optional[Tuple[str]], optional): a list of properties that
          should be visible from the searched layer. Defaults to None.
        klass (Optional[Tuple[Type[Layer]]], optional): a collection of classes
          allowed for the searched layer. Defaults to None.
        search_reversed (bool, optional): wether to search from last-to-first.
          Defaults to True.

    Raises:
        ValueError: if no search parameters are passed.
        ValueError: if no valid layer can be found with the specified search
          parameters.

    Returns:
        Layer: the layer satisfying all search parameters.
    """
    search_params = (name, properties, klass)
    if all(p is None for p in search_params):
        raise ValueError(
            "At least one of the search search parameters must "
            "be set when calling `get_layer`, indicating the "
            "necessary properties for the layer being retrieved."
        )

    if name is not None:
        return get_nested_layer(model, name)

    layers = model._flatten_layers(include_self=False)
    if search_reversed:
        layers = reversed(list(layers))
    for layer in layers:
        if klass and not isinstance(layer, klass):
            continue
        if properties and not all(hasattr(layer, p) for p in properties):
            continue

        return layer  # `layer` matches all conditions.

    raise ValueError(
        f"A valid layer couldn't be inferred from the name=`{name}`, "
        f"klass=`{klass}` and properties=`{properties}`. Make sure these "
        "attributes correctly reflect a layer in the model."
    )


def endpoints(model: Model, endpoints: List[E]) -> List[KerasTensor]:
    """Collect intermediate endpoints in a model based on structured descriptors.

    Args:
        model (Model): the model containing the endpoints to be collected.
        endpoints (List[E]): descriptors of endpoints that should be collected.

    Raises:
        ValueError: raised whenever one of the endpoint descriptors is invalid
          or it does not describe a nested layer in the `model`.

    Returns:
        List[KerasTensor]: a list containing the endpoints of interest.
    """
    endpoints_ = []

    for ep in endpoints:
        if isinstance(ep, int):
            ep = {"layer": model.layers[ep]}
        elif isinstance(ep, Layer):
            ep = {"layer": ep}
        elif isinstance(ep, str):
            ep = {"name": ep}

        if not isinstance(ep, dict):
            raise ValueError(
                f"Illegal type {type(ep)} for endpoint {ep}. Expected a "
                "layer index (`int`), layer name (`str`), a layer "
                "(`keras.layers.Layer`) or a dictionary with "
                "`name`/`layer`, `link` and `node` keys."
            )

        if "layer" in ep:
            layer = ep["layer"]
        else:
            layer = get_nested_layer(model, ep["name"])

        link = ep.get("link", "output")
        node = ep.get("node", "last")

        if node == "last":
            node = len(layer._inbound_nodes) - 1

        endpoint = (
            layer.get_input_at(node) if link == "input" else layer.get_output_at(node)
        )

        endpoints_.append(endpoint)

    return endpoints_


def expose(
    model: Model,
    arguments: Optional[E] = None,
    outputs: Optional[E] = None,
) -> Model:
    """Creates a new model that exposes all endpoints described by
    ``arguments`` and ``outputs``.

    Args:
        model (Model): The model being explained.
        arguments (Optional[E], optional): Name of the argument layer/tensor in
            the model. The jacobian of the output explaining units will be computed
            with respect to the input signal of this layer. This argument can also
            be an integer, a dictionary representing the intermediate signal or
            the pooling layer itself. If None is passed, the penultimate layer
            is assumed to be a GAP layer. Defaults to None.
        outputs (Optional[E], optional): Name of the output layer in the model.
            The jacobian will be computed for the activation signal of units in this
            layer. This argument can also be an integer, a dictionary representing
            the output signal and the logits layer itself. If None is passed,
            the last layer is assumed to be the logits layer. Defaults to None.

    Returns:
        Model: the exposed model, whose outputs contain the intermediate and
        output tensors.
    """
    if outputs is None:
        outputs = get_logits_layer(model)
    if isinstance(arguments, (str, tuple)):
        arguments = {"name": arguments}
    if arguments is None:
        gpl = get_global_pooling_layer(model)
        arguments = {"layer": gpl, "link": "input"}

    outputs = tolist(outputs)
    arguments = tolist(arguments)

    tensors = endpoints(model, outputs + arguments)

    return Model(
        inputs=model.inputs,
        outputs=tensors,
    )


def gather_units(
    tensor: tf.Tensor,
    indices: Optional[tf.Tensor],
    axis: int = -1,
    batch_dims: int = -1,
) -> tf.Tensor:
    """Gather units (in the last axis) from a tensor.

    Args:
        tensor (tf.Tensor): the input tensor.
        indices (tf.Tensor, optional): the indices that should be gathered.
        axis (int, optional): the axis from which indices should be taken,
          used to fine control gathering. Defaults to -1.
        batch_dims (int, optional): the number of batch dimensions, used to
          fine control gathering. Defaults to -1.

    Returns:
        tf.Tensor: the gathered units
    """
    if indices is None:
        return tensor

    return tf.gather(tensor, indices, axis=axis, batch_dims=batch_dims)


def layers_with_biases(
    model: Model,
    exclude: Tuple[Layer] = (),
    return_biases: bool = True,
) -> List[Layer]:
    """Extract layers containing biases from a model.

    Args:
        model (Model): the model inspected.
        exclude (Tuple[Layer], optional): a list of layers to ignore. Defaults to ().
        return_biases (bool, optional): wether or not to return the biases as well.
            Defaults to True.

    Returns:
        List[Layer]: a list of layers.
        List[Layer], List[tf.Tensor]: a list of layers and biases.
    """
    layers = [
        layer
        for layer in model._flatten_layers(include_self=False)
        if (
            layer not in exclude
            and (
                isinstance(layer, NORMALIZATION_LAYERS)
                or hasattr(layer, "bias")
                and layer.bias is not None
            )
        )
    ]

    if return_biases:
        return layers, biases(layers)

    return layers


def biases(
    layers: List[Layer],
) -> List[tf.Tensor]:
    """Recursively retrieve the biases from layers.

    Layers containing implicit bias are unrolled before returned. For
    instance, the Batch Normalization layer, whose equation is defined by
    :math:`y(x) = \\frac{x - \\mu}{\\sigma} w + b`, will have bias equals to:

    .. math::

        \\frac{-\\mu w}{s} + b

    Args:
        layers (List[Layer]): a list of layers from which
            biases should be extracted.

    Returns:
        List[tf.Tensor]: a list of all biases retrieved.
    """
    biases = []

    for layer in layers:
        if isinstance(layer, NORMALIZATION_LAYERS):
            # Batch norm := ((x - m)/s)*w + b
            # Hence bias factor is -m*w/s + b.
            biases.append(
                -layer.moving_mean
                * layer.gamma
                / tf.sqrt(layer.moving_variance + 1e-07)  # might be variance here.
                + layer.beta
            )

        elif hasattr(layer, "bias") and layer.bias is not None:
            biases.append(layer.bias)

    return biases
