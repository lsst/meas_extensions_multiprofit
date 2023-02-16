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

import math
import numpy as np
import pydantic
from pydantic.dataclasses import dataclass
from typing import Sequence

import gauss2d as g2
import gauss2d.fit as g2f

from lsst.daf.butler.formatters.parquet import astropy_to_arrow
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_multiband as fitMB
import lsst.utils.timer as utilsTimer

from multiprofit.fit_psf import CatalogPsfFitterConfig
from multiprofit.fit_source import CatalogExposureSourcesABC, CatalogSourceFitter, CatalogSourceFitterConfig

from .utils import get_spanned_image


class NotPrimaryError(RuntimeError):
    pass


@dataclass(frozen=True, kw_only=True, config=fitMB.CatalogExposureConfig)
class CatalogExposurePsfs(fitMB.CatalogExposureInputs, CatalogExposureSourcesABC):
    """ Input data from lsst pipelines, parsed for MultiProFit.
    """
    channel: g2f.Channel = pydantic.Field(title="Channel for the image's band")
    config_fit: CatalogSourceFitterConfig = pydantic.Field(title="Channel for the image's band")

    def get_psfmodel(self, source):
        match = np.argwhere(
            (self.table_psf_fits[self.config_fit_psf.column_id] == source[self.config_fit.column_id])
        )[0][0]
        return self.config_fit_psf.rebuild_psfmodel(self.table_psf_fits[match])

    def get_source_observation(self, source) -> g2f.Observation:
        if (not source['detect_isPrimary']) or source['merge_peak_sky']:
            raise NotPrimaryError(f'source {source[self.config_fit.column_id]} has invalid flags for fit')
        footprint = source.getFootprint()
        bbox = footprint.getBBox()
        bitmask = 0
        mask = self.exposure.mask[bbox]
        spans = footprint.spans.asArray()
        for bitname in self.config_fit.mask_names_zero:
            bitval = mask.getPlaneBitMask(bitname)
            bitmask |= bitval
        mask = ((mask.array & bitmask) != 0) & (spans != 0)
        mask = ~ mask
        img, _, sigma_inv = get_spanned_image(
            exposure=self.exposure, bbox=bbox, spans=spans, get_sig_inv=True,
        )
        sigma_inv[~mask] = 0

        obs = g2f.Observation(image=g2.ImageD(img), sigma_inv=g2.ImageD(sigma_inv),
                              mask_inv=g2.ImageB(mask), channel=self.channel)
        return obs

    def __post_init__(self):
        # Regular standard library dataclasses require this hideous workaround
        # due to https://github.com/python/cpython/issues/83315 : super(
        #  fitMB.CatalogExposureInputs, self).__thisclass__.__post_init__(self)
        # ... but pydantic dataclasses do not seem to, and also don't pass self
        super().__post_init__()
        object.__setattr__(self, 'config_fit_psf',
                           CatalogPsfFitterConfig(**self.table_psf_fits.meta['config']))


class MultiProFitSourceConfig(CatalogSourceFitterConfig, fitMB.CoaddMultibandFitSubConfig):
    """Configuration for the MultiProFit profile fitter."""
    bands_fit = pexConfig.ListField(dtype=str, default=[], doc="list of bandpass filters to fit",
                                    listCheck=lambda x: len(set(x)) == len(x))
    fit_linear = pexConfig.Field[bool](default=True, doc="Fit linear parameters to initialize")
    mask_names_zero = pexConfig.ListField[str](default=['BAD', 'EDGE', 'SAT', 'NO_DATA'],
                                               doc="Mask bits to mask out")
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")

    def bands_read_only(self) -> set[str]:
        # TODO: Re-implement determination of prior-only bands once priors
        # are re-implemented (DM-3xxxx)
        return set()

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {"not_primary_flag": "NotPrimaryError"}


class MultiProFitSourceTask(CatalogSourceFitter, fitMB.CoaddMultibandFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit a single model to all sources in a coadd,
    using a previously-fit PSF model for each exposure. The task may also use
    prior measurements from single- or merged multiband catalogs for
    initialization.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info.
    """
    ConfigClass = MultiProFitSourceConfig
    _DefaultName = "multiProFitSource"

    def __init__(self, **kwargs):
        errors_expected = {} if 'errors_expected' not in kwargs else kwargs.pop('errors_expected')
        if NotPrimaryError not in errors_expected:
            errors_expected[NotPrimaryError] = 'not_primary_flag'
        CatalogSourceFitter.__init__(self, errors_expected=errors_expected)
        fitMB.CoaddMultibandFitSubTask.__init__(self, **kwargs)

    @staticmethod
    def _init_component(
        component: g2f.Component, flux: float | None = None,
        size_x: float | None = None, size_y: float | None = None,
        rho: float | None = None, cenx: float | None = None, ceny: float | None = None,
    ):
        """Initialize component parameter values.

        Parameters
        ----------
        component : `gauss2d.fit.Component`
            The component to initialize.
        flux : float
            The flux (integral) value, if any.
        size_x : float
            The x-axis size value, if any.
        size_y : float
            The y-axis size value, if any.
        rho : float
            The rho (correlation coefficient) value, if any.
        cenx : float
            The x centroid value, if any.
        ceny : float
            The y centroid value, if any.
        """
        has_flux = flux is not None
        has_cenx = cenx is not None
        has_ceny = ceny is not None
        for parameter in component.parameters():
            if has_flux and isinstance(parameter, g2f.IntegralParameterD):
                parameter.value = flux
            elif size_x is not None and isinstance(parameter, g2f.SizeXParameterD):
                parameter.value = size_x
            elif size_y is not None and isinstance(parameter, g2f.SizeYParameterD):
                parameter.value = size_y
            elif rho is not None and isinstance(parameter, g2f.RhoParameterD):
                parameter.value = rho
            elif has_cenx and isinstance(parameter, g2f.CentroidXParameterD):
                parameter.value = cenx
            elif has_ceny and isinstance(parameter, g2f.CentroidYParameterD):
                parameter.value = ceny

    def initialize_model(self, model: g2f.Model, source: g2f.Source,
                         limits_x: g2f.LimitsD, limits_y: g2f.LimitsD):
        comps = model.sources[0].components
        sig_x = math.sqrt(source['base_SdssShape_xx'])
        sig_y = math.sqrt(source['base_SdssShape_yy'])
        rho = np.clip(source['base_SdssShape_xy']/(sig_x*sig_y), -0.5, 0.5)
        if not np.isfinite(rho):
            sig_x, sig_y, rho = 0.5, 0.5, 0
        if not source['base_SdssShape_flag']:
            flux = source['base_SdssShape_instFlux']
        else:
            flux = source['base_GaussianFlux_instFlux']
            if not (flux > 0):
                flux = source['base_PsfFlux_instFlux']
                if not (flux > 0):
                    flux = 1
        n_psfs = self.config.n_pointsources
        n_extended = len(self.config.sersics)
        observation = model.data[0]
        limits_x.max = float(observation.image.n_cols)
        limits_y.max = float(observation.image.n_rows)
        try:
            cenx_img, ceny_img = self.catexps[0].exposure.wcs.skyToPixel(
                geom.SpherePoint(source['coord_ra'], source['coord_dec'])
            )
            bbox = source.getFootprint().getBBox()
            beginx, beginy = bbox.beginX, bbox.beginY
            # multiprofit bottom left corner coords are 0, 0, not -0.5, -0.5
            cenx = cenx_img - beginx + 0.5
            ceny = ceny_img - beginy + 0.5
        # TODO: determine which exceptions can occur above
        except Exception:
            cenx = observation.image.n_cols/2.
            ceny = observation.image.n_rows/2.
        flux = flux/(n_psfs + n_extended)
        for comp in comps[:n_psfs]:
            self._init_component(comp, cenx=cenx, ceny=ceny, flux=flux)
        for comp in comps[n_psfs:]:
            self._init_component(comp, cenx=cenx, ceny=ceny, flux=flux, size_x=sig_x, size_y=sig_y, rho=rho)

    @utilsTimer.timeMethod
    def run(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit source fit task on catalog-exposure pairs.

        Parameters
        ----------
        catalog_multi : `typing.Sequence`
            A multi-band, indexable source catalog.
        catexps : list[`CatalogExposureInputs`]
            Catalog-exposure-PSF model tuples to fit source models for.
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `astropy.Table`
            A table with fit parameters for the PSF model at the location
            of each source.
        """

        catexps_conv = [None]*len(catexps)
        for idx, catexp in enumerate(catexps):
            if isinstance(catexp, CatalogExposurePsfs):
                catexps_conv[idx] = catexp
            else:
                catexps_conv[idx] = CatalogExposurePsfs(
                    # dataclasses.asdict(catexp)_ makes a recursive deep copy and must be avoided
                    **{key: getattr(catexp, key) for key in catexp.__dataclass_fields__.keys()},
                    channel=g2f.Channel.get(catexp.band),
                    config_fit=self.config,
                )
        self.catexps = catexps
        catalog = self.fit(
            catalog_multi=catalog_multi,
            catexps=catexps_conv,
            config=self.config,
            **kwargs
        )
        return pipeBase.Struct(output=astropy_to_arrow(catalog))
