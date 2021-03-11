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
import lsst.afw.image as afwImage


def defaultdictNested():
    """Get a nested defaultdict with defaultdict default value.

    Returns
    -------
    defaultdict : `defaultdict`
        A `defaultdict` with `defaultdict` default values.
    """
    return defaultdict(defaultdictNested)


# TODO: Allow addition to existing image
def get_spanned_image(footprint, bbox=None):
    spans = footprint.getSpans()
    bbox_is_none = bbox is None
    if bbox_is_none:
        bbox = footprint.getBBox()
    if not (bbox.getHeight() > 0 and bbox.getWidth() > 0):
        return None, bbox
    bbox_fp = bbox if bbox_is_none else footprint.getBBox()
    img = afwImage.Image(bbox_fp, dtype='D')
    spans.setImage(img, 1)
    img.array[img.array == 1] = footprint.getImageArray()
    if not bbox_is_none:
        img = img.subset(bbox)
    return img.array, bbox


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
