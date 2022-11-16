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

from lsst.afw.image import ExposureF
from lsst.afw.table import SourceCatalog
from lsst.meas.extensions.multiprofit.fit_coadd_psf import (
    CatalogExposure, MultiProFitPsfConfig, MultiProFitPsfTask,
)

import numpy as np
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
filename_cat = os.path.join(ROOT, "data", "deepCoadd_meas_0_24_r_2k_ci_imsim.fits")
filename_exp = os.path.join(ROOT, "data", "deepCoadd_calexp_0_24_r_2k_ci_imsim.fits")


def test_fit_psf():
    catalog = SourceCatalog.readFits(filename_cat)
    good = (catalog['detect_isPrimary'] == 1) & (catalog['merge_peak_sky'] == 0)
    n_test = 5
    good[np.where(good)[0][n_test:]] = False
    catalog = catalog[good]
    exposure = ExposureF.readFits(filename_exp)
    catexp = CatalogExposure(dataId={'band': 'r'}, catalog=catalog, exposure=exposure)
    task = MultiProFitPsfTask(
        config=MultiProFitPsfConfig(
            flag_errors={'InvalidParameterError': 'no_inputs_flag'},
            sigmas=[3.0],
        ),
    )
    results = task.run(catexp).output.to_pandas()
    assert len(results) == n_test
    assert np.all(results.values >= -np.Inf)
