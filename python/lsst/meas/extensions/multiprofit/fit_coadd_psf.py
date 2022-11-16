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


from dataclasses import dataclass

from lsst.daf.butler.formatters.parquet import astropy_to_arrow
import lsst.pex.config as pexConfig
from lsst.pex.exceptions import InvalidParameterError
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_psf as fitCP
import lsst.utils.timer as utilsTimer

from multiprofit.fit_psf import CatalogExposureABC, CatalogPsfFitter, CatalogPsfFitterConfig


@dataclass(frozen=True)
class CatalogExposure(fitCP.CatalogExposure, CatalogExposureABC):
    def get_catalog(self):
        return self.catalog

    def get_psf_image(self, source):
        bbox = source.getFootprint().getBBox()
        center = bbox.getCenter()
        return self.exposure.getPsf().computeKernelImage(center).array


class MultiProFitPsfConfig(CatalogPsfFitterConfig, fitCP.CoaddPsfFitSubConfig):
    """Configuration for the MultiProFit profile fitter."""
    fit_linear = pexConfig.Field[bool](default=True, doc="Fit linear parameters to initialize")
    prefix_column = pexConfig.Field[str](default="mpf_psf_", doc="Column name prefix")
    sigmas = pexConfig.ListField[float](default=[1.5, 3], doc="Number of Gaussian components in PSF")


class MultiProFitPsfTask(CatalogPsfFitter, fitCP.CoaddPsfFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit a PSF model to the coadd PSF,
    evaluated at the centroid of each source in the corresponding
    catalog.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info.
    """
    ConfigClass = MultiProFitPsfConfig
    _DefaultName = "multiProFitPsf"


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

    def __init__(self, **kwargs):
        super().__init__(errors_expected={"no_inputs_flag": InvalidParameterError}, **kwargs)
