import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from keras import callbacks as callbacks_module
from keras.callbacks import Callback
from keras.engine import data_adapter
from keras.engine.training import _is_tpu_multi_host
from keras.engine.training import _minimum_control_deps
from keras.engine.training import potentially_ragged_concat
from keras.engine.training import reduce_per_replica
from keras.utils import tf_utils
from tensorflow.python.eager import context

from keras_explainable.inspection import SPATIAL_AXIS


def explain_step(
    model: tf.keras.Model,
    method: Callable,
    data: Tuple[tf.Tensor],
    spatial_axis: Tuple[int, int] = SPATIAL_AXIS,
    postprocessing: Callable = None,
    resizing: Optional[Union[bool, tf.Tensor]] = True,
    **params,
) -> Tuple[tf.Tensor, tf.Tensor]:
    inputs, indices, _ = data_adapter.unpack_x_y_sample_weight(data)
    logits, maps = method(
        model=model,
        inputs=inputs,
        indices=indices,
        spatial_axis=spatial_axis,
        **params,
    )

    if postprocessing is not None:
        maps = postprocessing(maps, axis=spatial_axis)

    if resizing is not None and resizing is not False:
        if resizing is True:
            resizing = tf.shape(inputs)[1:-1]
        maps = tf.image.resize(maps, resizing)

    return logits, maps


def make_explain_function(
    model: tf.keras.Model,
    method: Callable,
    params: Dict[str, Any],
    force: bool = False,
):
    explain_function = getattr(model, "explain_function", None)

    if explain_function is not None and not force:
        return explain_function

    def explain_function(iterator):
        """Runs a single explain step."""

        def run_step(data):
            outputs = explain_step(model, method, data, **params)
            # Ensure counter is updated only if `test_step` succeeds.
            with tf.control_dependencies(_minimum_control_deps(outputs)):
                model._explain_counter.assign_add(1)
            return outputs

        if model._jit_compile:
            run_step = tf.function(
                run_step, jit_compile=True, reduce_retracing=True
            )

        data = next(iterator)
        outputs = model.distribute_strategy.run(run_step, args=(data,))
        outputs = reduce_per_replica(
            outputs, model.distribute_strategy, reduction="concat"
        )
        return outputs

    if not model.run_eagerly:
        explain_function = tf.function(explain_function, reduce_retracing=True)

    model.explain_function = explain_function

    return explain_function


def make_data_handler(
    model,
    x,
    y,
    batch_size=None,
    steps=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
):
    dataset_types = (tf.compat.v1.data.Dataset, tf.data.Dataset)
    if (
        model._in_multi_worker_mode()
        or _is_tpu_multi_host(model.distribute_strategy)
    ) and isinstance(x, dataset_types):
        try:
            opts = tf.data.Options()
            opts.experimental_distribute.auto_shard_policy = (
                tf.data.experimental.AutoShardPolicy.DATA
            )
            x = x.with_options(opts)
        except ValueError:
            warnings.warn(
                "Using evaluate with MultiWorkerMirroredStrategy "
                "or TPUStrategy and AutoShardPolicy.FILE might lead to "
                "out-of-order result. Consider setting it to "
                "AutoShardPolicy.DATA.",
                stacklevel=2,
            )

    return data_adapter.get_data_handler(
        x=x,
        y=y,
        batch_size=batch_size,
        steps_per_epoch=steps,
        initial_epoch=0,
        epochs=1,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        model=model,
        steps_per_execution=model._steps_per_execution,
    )


def explain(
    method: Callable,
    model: tf.keras.Model,
    x: Union[np.ndarray, tf.Tensor, tf.data.Dataset],
    y: Optional[Union[np.ndarray, tf.Tensor]] = None,
    batch_size: Optional[int] = None,
    verbose: Union[str, int] = "auto",
    steps: Optional[int] = None,
    callbacks: List[Callback] = None,
    max_queue_size: int = 10,
    workers: int = 1,
    use_multiprocessing: bool = False,
    force: bool = True,
    **method_params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explain the outputs of ``model`` with respect to the inputs or an intermediate
    signal, using an AI explaining method.

    Args:
        method (Callable): An AI explaining function, as the ones contained in
            `methods` module.
        model (tf.keras.Model): The model whose predictions should be explained.
        x (Union[np.ndarray, tf.Tensor, tf.data.Dataset]): the input data for the model.
        y (Optional[Union[np.ndarray, tf.Tensor]], optional): the indices in the output
            tensor that should be explained. If none, an activation map is computed
            for each unit. Defaults to None.
        batch_size (Optional[int], optional): the batch size used by ``method``.
            Defaults to 32.
        verbose (Union[str, int], optional): wether to show a progress bar during
            the calculation of the explaining maps. Defaults to "auto".
        steps (Optional[int], optional): the number of steps, if ``x`` is a
          ``tf.data.Dataset`` of unknown cardinallity. Defaults to None.
        callbacks (List[Callback], optional): list of callbacks called during the
            explaining procedure. Defaults to None.
        max_queue_size (int, optional): the queue size when retrieving inputs.
            Used if ``x`` is a generator. Defaults to 10.
        workers (int, optional): the number of workers used when retrieving inputs.
            Defaults to 1.
        use_multiprocessing (bool, optional): wether to employ multi-process or
            multi-threading when retrieving inputs, when ``x`` is a generator.
            Defaults to False.
        force (bool, optional): to force the creation of the explaining function.
            Can be set to False if the same function is always applied to a model,
            avoiding retracing. Defaults to True.

    Besides the parameters described above, any named parameters passed to this function
    will be collected into ``methods_params`` and passed onto the :func:`explain_step`
    and ``method`` functions. Common ones are:

    - indices_batch_dims (int): The dimensions marked as ``batch`` when gathering
      units described by ``y``. Ignore if ``y`` is None.
    - indices_axis: The axes from which to gather units described by ``y``.
      Ignore if ``y`` is None.
    - spatial_axis: The axes containing the positional visual info. We assume `inputs`
      to contain 2D images or videos in the shape `(B1, B2, ..., BN, H, W, 3)`.
      For 3D image data, set `spatial_axis` to `(1, 2, 3)` or `(-4, -3, -2)`.
    - postprocessing: A function to process the activation maps before normalization
      (most commonly adopted being `maximum(x, 0)` and `abs`).

    Raises:
        ValueError: the explaining method produced in an unexpected.

    Returns:
        Tuple[np.ndarray, np.ndarray]: logits and explaining maps tensors.
    """

    if not hasattr(model, "_explain_counter"):
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        model._explain_counter = tf.Variable(0, dtype="int64", aggregation=agg)

    outputs = None
    with model.distribute_strategy.scope():
        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = make_data_handler(
            model,
            x,
            y,
            batch_size=batch_size,
            steps=steps,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=model,
                verbose=verbose,
                epochs=1,
                steps=data_handler.inferred_steps,
            )

        explain_function = make_explain_function(
            model, method, method_params, force
        )
        model._explain_counter.assign(0)
        callbacks.on_predict_begin()
        batch_outputs = None
        for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_predict_batch_begin(step)
                    tmp_batch_outputs = explain_function(iterator)
                    if data_handler.should_sync:
                        context.async_wait()
                    batch_outputs = (
                        tmp_batch_outputs  # No error, now safe to assign.
                    )
                    if outputs is None:
                        outputs = tf.nest.map_structure(
                            lambda batch_output: [batch_output],
                            batch_outputs,
                        )
                    else:
                        tf.__internal__.nest.map_structure_up_to(
                            batch_outputs,
                            lambda output, batch_output: output.append(
                                batch_output
                            ),
                            outputs,
                            batch_outputs,
                        )
                    end_step = step + data_handler.step_increment
                    callbacks.on_predict_batch_end(
                        end_step, {"outputs": batch_outputs}
                    )
        if batch_outputs is None:
            raise ValueError(
                "Unexpected result of `explain_function` "
                "(Empty batch_outputs). Please use "
                "`Model.compile(..., run_eagerly=True)`, or "
                "`tf.config.run_functions_eagerly(True)` for more "
                "information of where went wrong, or file a "
                "issue/bug to `keras-explainable`."
            )
        callbacks.on_predict_end()
    all_outputs = tf.__internal__.nest.map_structure_up_to(
        batch_outputs, potentially_ragged_concat, outputs
    )
    return tf_utils.sync_to_numpy_or_python_type(all_outputs)


def partial_explain(method: Callable, **default_params):
    """Wrapper for explaining methods.

    Args:
        method (Callable): the explaining method being wrapped by ``explain``.
    """

    def _partial_method_explain(*args, **params):
        params = {**default_params, **params}
        return explain(method, *args, **params)

    _partial_method_explain.__name__ = f"{method.__name__}_explain"

    return _partial_method_explain
