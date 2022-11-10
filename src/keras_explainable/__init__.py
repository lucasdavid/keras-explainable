import sys

from keras_explainable import filters
from keras_explainable import inspection
from keras_explainable import methods
from keras_explainable import utils
from keras_explainable.engine import explaining
from keras_explainable.engine.explaining import explain
from keras_explainable.engine.explaining import partial_explain

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError  # pragma: no cover
    from importlib.metadata import version
else:
    from importlib_metadata import PackageNotFoundError  # pragma: no cover
    from importlib_metadata import version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "keras-explainable"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

cam = partial_explain(
    methods.cams.cam, postprocessing=filters.positive_normalize
)
gradcam = partial_explain(
    methods.cams.gradcam, postprocessing=filters.positive_normalize
)
gradcampp = partial_explain(
    methods.cams.gradcampp, postprocessing=filters.positive_normalize
)
scorecam = partial_explain(
    methods.cams.scorecam, postprocessing=filters.positive_normalize
)

gradients = partial_explain(
    methods.gradient.gradients,
    postprocessing=filters.normalize,
)

__all__ = [
    "methods",
    "inspection",
    "filters",
    "utils",
    "explaining",
    "explain",
    "cam",
    "gradcam",
    "gradcampp",
    "scorecam",
    "gradients",
]
