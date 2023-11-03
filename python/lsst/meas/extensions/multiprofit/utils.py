# This file is part of meas_extensions_multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import defaultdict
import numpy as np

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.geom as geom


def defaultdictNested():
    """Get a nested defaultdict with defaultdict default value.

    Returns
    -------
    defaultdict : `defaultdict`
        A `defaultdict` with `defaultdict` default values.
    """
    return defaultdict(defaultdictNested)


# TODO: Allow addition to existing image
def get_spanned_image(
    exposure: afwImage.Exposure,
    footprint: afwDetect.Footprint = None,
    bbox: geom.Box2I = None,
    spans: np.ndarray = None,
    get_sig_inv: bool = False,
):
    """Get an image masked by its spanset.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        An exposure to extract arrays from.
    footprint : `lsst.afw.detection`
        The footprint to get spans/bboxes from. Not needed if both of
        `bbox` and `spans` are provided.
    bbox : `lsst.geom.Box2I`
        The bounding box to subset the exposure with.
        Defaults to the footprint's bbox.
    spans : `np.ndarray`
        A spanset array (inverse mask/selection).
        Defaults to the footprint's spans.
    get_sig_inv : bool
        Whether to get the inverse variance and return its square root.

    Returns
    -------
    image : `np.ndarray`
        The image array, with masked pixels set to zero.
    bbox : `lsst.geom.Box2I`
        The bounding box used to subset the exposure.
    sig_inv : `np.ndarray`
        The inverse sigma array, with masked pixels set to zero.
        Set to None if `get_sig_inv` is False.
    """
    bbox_is_none = bbox is None
    if bbox_is_none:
        bbox = footprint.getBBox()
    if not (bbox.getHeight() > 0 and bbox.getWidth() > 0):
        return None, bbox
    if spans is None:
        spans = footprint.getSpans().asArray()
    sig_inv = None
    img = afwImage.Image(bbox, dtype="D")
    maskedIm = exposure.photoCalib.calibrateImage(exposure.maskedImage.subset(bbox))
    img.array[spans] = maskedIm.image.array[spans]
    if get_sig_inv:
        sig_inv = afwImage.Image(bbox, dtype="D").array
        sig_inv[spans] = 1 / np.sqrt(maskedIm.variance.array[spans])
    return img.array, bbox, sig_inv


def join_and_filter(separator, items, exclusion=None):
    """Join an iterable of items by a separator, filtering out an exclusion.

    Parameters
    ----------
    separator : `string`
        The separator to join items by.
    items : iterable of `str`
        Items to join.
    exclusion : `str`, optional
        The pattern to exclude; default None.

    Returns
    -------
    joined : `str`
        The joined string.
    """
    return separator.join(filter(exclusion, items))
