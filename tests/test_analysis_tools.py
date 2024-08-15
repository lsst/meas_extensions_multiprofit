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

from lsst.meas.extensions.multiprofit.analysis_tools import make_size_magnitude_tools
import pytest


@pytest.fixture(scope="module")
def kwargs_plot():
    kwargs = dict(
        xLims=(18, 25),
        yLims=(-3, 4),
    )
    return kwargs


@pytest.fixture(scope="module")
def tool_sersic(kwargs_plot):
    (atool,) = make_size_magnitude_tools(
        name_model="ser",
        label_model="MPF Ser",
        components=(("ser", "Sersic"),),
        kwargs_plot=kwargs_plot,
    )
    atool.finalize()
    return atool


@pytest.fixture(scope="module")
def data_sersic(tool_sersic):
    schema = tool_sersic.getInputSchema()
    data = {key: [] for key in schema}
    return data


def test_psf_fits(tool_sersic, data_sersic):
    assert tool_sersic is not None
    assert len(data_sersic) > 0
