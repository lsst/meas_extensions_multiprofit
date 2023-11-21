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

from pydantic.dataclasses import dataclass

from lsst.daf.butler.formatters.parquet import astropy_to_arrow
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_psf as fitCP
import lsst.utils.timer as utilsTimer

from lsst.multiprofit.fit_psf import CatalogExposurePsfABC, CatalogPsfFitter, CatalogPsfFitterConfig


@dataclass(frozen=True, kw_only=True, config=fitCP.CatalogExposureConfig)
class CatalogExposure(fitCP.CatalogExposurePsf, CatalogExposurePsfABC):
    """A CatalogExposure for PSF fitting."""


class MultiProFitPsfConfig(CatalogPsfFitterConfig, fitCP.CoaddPsfFitSubConfig):
    """Configuration for the MultiProFit Gaussian mixture PSF fitter."""

    fit_linear = pexConfig.Field[bool](default=True, doc="Fit linear parameters to initialize")
    prefix_column = pexConfig.Field[str](default="mpf_psf_", doc="Column name prefix")

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {"no_inputs_flag": "InvalidParameterError"}


class MultiProFitPsfTask(CatalogPsfFitter, fitCP.CoaddPsfFitSubTask):
    """Fit a Gaussian mixture PSF model at cataloged locations.

    This task uses MultiProFit to fit a PSF model to the coadd PSF,
    evaluated at the centroid of each source in the corresponding
    catalog.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info.
    """

    ConfigClass = MultiProFitPsfConfig
    _DefaultName = "multiProFitPsf"

    def __init__(self, **kwargs):
        errors_expected = {} if "errors_expected" not in kwargs else kwargs.pop("errors_expected")
        if InvalidParameterError not in errors_expected:
            # Cannot compute CoaddPsf at point (x, y)
            errors_expected[InvalidParameterError] = "no_inputs_flag"
        CatalogPsfFitter.__init__(self, errors_expected=errors_expected)
        fitCP.CoaddPsfFitSubTask.__init__(self, **kwargs)

    @utilsTimer.timeMethod
    def run(
        self,
        catexp: CatalogExposure,
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit PSF task on a catalog-exposure pair.

        Parameters
        ----------
        catexp : `CatalogExposure`
            An exposure to fit a model PSF at the position of all
            sources in the corresponding catalog.
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `astropy.Table`
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        catalog = self.fit(catexp=catexp, config=self.config, **kwargs)
        return pipeBase.Struct(output=astropy_to_arrow(catalog))
