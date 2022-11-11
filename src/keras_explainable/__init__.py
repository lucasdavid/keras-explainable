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

_SHORTCUTS_DOCS = """Shortcut for ``{method}``,
filtering {filter} contributing regions
"""

f"""{_SHORTCUTS_DOCS.format(method='methods.cams.cam', filter='positively',)}"""
cam = partial_explain(
    methods.cams.cam, postprocessing=filters.positive_normalize
)

f"""{_SHORTCUTS_DOCS.format(method='methods.cams.gradcam', filter='positively',)}"""
gradcam = partial_explain(
    methods.cams.gradcam, postprocessing=filters.positive_normalize
)

f"""{_SHORTCUTS_DOCS.format(method='methods.cams.gradcampp', filter='positively',)}"""
gradcampp = partial_explain(
    methods.cams.gradcampp, postprocessing=filters.positive_normalize
)

f"""{_SHORTCUTS_DOCS.format(method='methods.cams.scorecam', filter='positively',)}"""
scorecam = partial_explain(
    methods.cams.scorecam,
    postprocessing=filters.positive_normalize,
    resizing=False,
)

f"""{_SHORTCUTS_DOCS.format(method='methods.gradient.gradients', filter='absolutely',)}"""
gradients = partial_explain(
    methods.gradient.gradients,
    postprocessing=filters.normalize,
    resizing=False,
)

f"""{_SHORTCUTS_DOCS.format(
    method='methods.gradient.full_gradients', filter='absolutely',
)}"""
full_gradients = partial_explain(
    methods.gradient.full_gradients,
    postprocessing=filters.normalize,
    resizing=False,
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
