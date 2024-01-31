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

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_psf as fitCP
import lsst.utils.timer as utilsTimer
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
from lsst.multiprofit.fit_psf import CatalogPsfFitter, CatalogPsfFitterConfig, CatalogPsfFitterConfigData
from lsst.pex.exceptions import InvalidParameterError

from .errors import IsParentError


class MultiProFitPsfConfig(CatalogPsfFitterConfig, fitCP.CoaddPsfFitSubConfig):
    """Configuration for the MultiProFit Gaussian mixture PSF fitter."""

    fit_parents = pexConfig.Field[bool](default=False, doc="Whether to fit parent object PSFs")
    prefix_column = pexConfig.Field[str](default="mpf_psf_", doc="Column name prefix")

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {"no_inputs_flag": "InvalidParameterError"}


class MultiProFitPsfTask(CatalogPsfFitter, fitCP.CoaddPsfFitSubTask):
    """Fit a Gaussian mixture PSF model at cataloged locations.

    This task uses MultiProFit to fit a PSF model to the coadd PSF,
    evaluated at the centroid of each source in the corresponding
    catalog.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to CoaddPsfFitSubTask.__init__.

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

    def check_source(self, source, config):
        if (
            config
            and hasattr(config, "fit_parents")
            and not config.fit_parents
            and (source["parent"] == 0)
            and (source["deblend_nChild"] > 1)
        ):
            raise IsParentError(
                f"{source['id']=} is a parent with nChild={source['deblend_nChild']}" f" and will be skipped"
            )

    @utilsTimer.timeMethod
    def run(
        self,
        catexp: fitCP.CatalogExposurePsf,
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit PSF task on a catalog-exposure pair.

        Parameters
        ----------
        catexp
            An exposure to fit a model PSF at the position of all
            sources in the corresponding catalog.
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        is_parent_name = IsParentError.column_name()
        if not self.config.fit_parents:
            if is_parent_name not in self.errors_expected:
                self.errors_expected[IsParentError] = is_parent_name
            if "IsParentError" not in self.config.flag_errors.values():
                self.config._frozen = False
                self.config.flag_errors[is_parent_name] = "IsParentError"
                self.config._frozen = True
        elif is_parent_name in self.errors_expected:
            del self.errors_expected[is_parent_name]
            if is_parent_name in self.config.flag_errors.keys():
                self.config._frozen = False
                del self.config.flag_errors[is_parent_name]
                self.config._frozen = True

        config_data = CatalogPsfFitterConfigData(config=self.config)
        catalog = self.fit(catexp=catexp, config_data=config_data, **kwargs)
        return pipeBase.Struct(output=astropy_to_arrow(catalog))
