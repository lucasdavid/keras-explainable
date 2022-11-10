import io
from math import ceil
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

# region Generics


def tolist(item):
    if isinstance(item, list):
        return item

    if isinstance(item, (tuple, set)):
        return list(item)

    return [item]


# endregion

# region Visualization


def get_dims(image):
    if hasattr(image, "shape"):
        return image.shape
    return (len(image), *get_dims(image[0]))


def visualize(
    images,
    title=None,
    overlay: Optional[List[np.ndarray]] = None,
    overlay_alpha: float = 0.75,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    figsize: Tuple[float, float] = None,
    cmap: str = None,
    overlay_cmap: str = None,
    to_file: str = None,
    to_buffer: io.BytesIO = None,
    subplots_ws: float = 0.0,
    subplots_hs: float = 0.0,
):
    import matplotlib.pyplot as plt

    dims = get_dims(images)
    rank = len(dims)

    if isinstance(images, tf.Tensor):
        images = images.numpy()

    if isinstance(images, (list, tuple)) or rank > 3:
        images = images
    else:
        images = [images]

    if rows is None and cols is None:
        cols = min(8, len(images))
        rows = ceil(len(images) / cols)
    elif rows is None:
        rows = ceil(len(images) / cols)
    else:
        cols = ceil(len(images) / rows)

    plt.figure(figsize=figsize or (4 * cols, 4 * rows))

    for ix, image in enumerate(images):
        plt.subplot(rows, cols, ix + 1)

        if image is not None:
            if isinstance(image, tf.Tensor):
                image = image.numpy()

            if len(image.shape) > 2 and image.shape[-1] == 1:
                image = image[..., 0]

            plt.imshow(image, cmap=cmap)

            if (
                overlay is not None
                and len(overlay) > ix
                and overlay[ix] is not None
            ):
                oi = overlay[ix]
                if len(oi.shape) > 2 and oi.shape[-1] == 1:
                    oi = oi[..., 0]
                plt.imshow(oi, overlay_cmap, alpha=overlay_alpha)
        if title is not None and len(title) > ix:
            plt.title(title[ix])
        plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=subplots_ws, hspace=subplots_hs)

    if to_buffer:
        plt.savefig(to_buffer)
        return Image.open(to_buffer)

    if to_file is not None:
        plt.savefig(to_file)


# endregion
