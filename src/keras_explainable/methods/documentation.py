"""Docs base module.
"""

_INPUTS_DESCR = """
    This method expects `inputs` to be a batch of positional signals of
    shape ``BHW...C``, and will return a tensor of shape ``BH'W'...L``,
    where ``(H', W', ...)`` are the sizes of the visual receptive field
    in the explained activation layer and `L` is the number of labels
    represented within the model's output logits.

    If `indices` is passed, the specific logits indexed by elements in this
    tensor are selected before the gradients are computed, effectivelly
    reducing the columns in the jacobian, and the size of the output explaining map.
"""

_DEFAULT_DOCS = """Computes the {method} Visualization Method.

{lead}
{description}

    References:
{references}

    Args:
        model (tf.keras.Model): the model being explained
        inputs (tf.Tensor): the input data
        indices (Optional[tf.Tensor], optional): indices that should be gathered
            from ``outputs``. Defaults to None.
        indices_axis (int, optional): the axis containing the indices to gather.
            Defaults to ``KERNEL_AXIS``.
        indices_batch_dims (int, optional): the number of dimensions to broadcast
            in the ``tf.gather`` operation. Defaults to ``-1``.
        spatial_axis (Tuple[int], optional): the dimensions containing positional
            information. Defaults to ``SPATIAL_AXIS``.
{more_args}

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: the logits and {map_type} maps tensors.
"""


def generate(
    method,
    lead="",
    description="",
    more_args="",
    references="",
    map_type="saliency",
    inputs_descr=True,
    **kwargs,
):
    if inputs_descr:
        description = f"{description}{_INPUTS_DESCR}"

    return _DEFAULT_DOCS.format(
        method=method,
        lead=lead,
        description=description,
        more_args=more_args,
        references=references,
        map_type=map_type,
        **kwargs,
    )


def docstring(
    method,
    lead="",
    description="",
    more_args="",
    references="",
    map_type="saliency",
    inputs_descr=True,
    **kwargs,
):
    docstring = generate(
        method,
        lead,
        description,
        more_args,
        references,
        map_type,
        inputs_descr,
        **kwargs,
    )

    def wrapper(func):
        func.__doc___ = docstring
        return func

    return wrapper
