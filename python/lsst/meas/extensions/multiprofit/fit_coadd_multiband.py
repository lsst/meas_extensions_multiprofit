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
from typing import Any, Mapping, Sequence, Type

import gauss2d as g2
import gauss2d.fit as g2f
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_multiband as fitMB
import lsst.utils.timer as utilsTimer
import numpy as np
import pydantic
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
from lsst.multiprofit.config import set_config_from_dict
from lsst.multiprofit.fit_psf import CatalogPsfFitterConfig, PsfRebuildFitFlagError
from lsst.multiprofit.fit_source import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterABC,
    CatalogSourceFitterConfig,
)
from lsst.multiprofit.utils import get_params_uniq
from pydantic.dataclasses import dataclass

from .utils import get_spanned_image


class NotPrimaryError(RuntimeError):
    """RuntimeError for objects that are not primary and shouldn't be fit."""


@dataclass(frozen=True, kw_only=True, config=fitMB.CatalogExposureConfig)
class CatalogExposurePsfs(fitMB.CatalogExposureInputs, CatalogExposureSourcesABC):
    """Input data from lsst pipelines, parsed for MultiProFit."""

    channel: g2f.Channel = pydantic.Field(title="Channel for the image's band")
    config_fit: CatalogSourceFitterConfig = pydantic.Field(title="Channel for the image's band")

    def get_psfmodel(self, source):
        match = np.argwhere(
            (self.table_psf_fits[self.config_fit_psf.column_id] == source[self.config_fit.column_id])
        )[0][0]
        return self.config_fit_psf.rebuild_psfmodel(self.table_psf_fits[match])

    def get_source_observation(self, source) -> g2f.Observation:
        if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
            raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")
        footprint = source.getFootprint()
        bbox = footprint.getBBox()
        bitmask = 0
        mask = self.exposure.mask[bbox]
        spans = footprint.spans.asArray()
        for bitname in self.config_fit.mask_names_zero:
            bitval = mask.getPlaneBitMask(bitname)
            bitmask |= bitval
        mask = ((mask.array & bitmask) != 0) & (spans != 0)
        mask = ~mask
        img, _, sigma_inv = get_spanned_image(
            exposure=self.exposure,
            bbox=bbox,
            spans=spans,
            get_sig_inv=True,
        )
        sigma_inv[~mask] = 0

        obs = g2f.Observation(
            image=g2.ImageD(img),
            sigma_inv=g2.ImageD(sigma_inv),
            mask_inv=g2.ImageB(mask),
            channel=self.channel,
        )
        return obs

    def __post_init__(self):
        # Regular standard library dataclasses require this hideous workaround
        # due to https://github.com/python/cpython/issues/83315 : super(
        #  fitMB.CatalogExposureInputs, self).__thisclass__.__post_init__(self)
        # ... but pydantic dataclasses do not seem to, and also don't pass self
        super().__post_init__()
        config = CatalogPsfFitterConfig()
        set_config_from_dict(config, self.table_psf_fits.meta["config"])
        object.__setattr__(self, "config_fit_psf", config)


class MultiProFitSourceConfig(CatalogSourceFitterConfig, fitMB.CoaddMultibandFitSubConfig):
    """Configuration for the MultiProFit profile fitter."""

    bands_fit = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="list of bandpass filters to fit",
        listCheck=lambda x: len(set(x)) == len(x),
    )
    mask_names_zero = pexConfig.ListField[str](
        default=["BAD", "EDGE", "SAT", "NO_DATA"], doc="Mask bits to mask out"
    )
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")

    def bands_read_only(self) -> set[str]:
        # TODO: Re-implement determination of prior-only bands once
        # data-driven priors are re-implemented (DM-4xxxx)
        return set()

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {
            "not_primary_flag": "NotPrimaryError",
            "psf_fit_flag": "PsfRebuildFitFlagError",
        }


class MultiProFitSourceTask(CatalogSourceFitterABC, fitMB.CoaddMultibandFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit a single model to all sources in a coadd,
    using a previously-fit PSF model for each exposure. The task may also use
    prior measurements from single- or merged multiband catalogs for
    initialization.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to CoaddMultibandFitSubTask.__init__.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info.
    """

    ConfigClass = MultiProFitSourceConfig
    _DefaultName = "multiProFitSource"

    def __init__(self, **kwargs: Any):
        errors_expected = {} if "errors_expected" not in kwargs else kwargs.pop("errors_expected")
        if NotPrimaryError not in errors_expected:
            errors_expected[NotPrimaryError] = "not_primary_flag"
        if PsfRebuildFitFlagError not in errors_expected:
            errors_expected[PsfRebuildFitFlagError] = "psf_fit_flag"
        CatalogSourceFitterABC.__init__(self, errors_expected=errors_expected)
        fitMB.CoaddMultibandFitSubTask.__init__(self, **kwargs)

    @staticmethod
    def _init_component(
        component: g2f.Component,
        values_init: dict[Type[g2f.ParameterD], float] = None,
        limits_init: dict[Type[g2f.ParameterD], g2f.LimitsD] = None,
    ):
        """Initialize component parameter values.

        Parameters
        ----------
        component : `gauss2d.fit.Component`
            The component to initialize.
        values_init
            Initial values to set per parameter type.
        limits_init
            Initial limits to set per parameter type.
        """
        # These aren't necessarily all free - should set cen_x, y
        # even if they're fixed, for example
        params_init = get_params_uniq(component)
        for param in params_init:
            type_param = type(param)
            if (value := values_init.get(type_param)) is not None:
                if param.limits.check(value):
                    param.value = value
            if (limits := limits_init.get(type_param)) is not None:
                value = value if value is not None else param.value
                if not limits.check(value):
                    param.value = (limits.max + limits.min) / 2.0
                param.limits = limits

    def get_model_cens(self, source: Mapping[str, Any]):
        cenx_img, ceny_img = self.catexps[0].exposure.wcs.skyToPixel(
            geom.SpherePoint(source["coord_ra"], source["coord_dec"])
        )
        bbox = source.getFootprint().getBBox()
        begin_x, begin_y = bbox.beginX, bbox.beginY
        # multiprofit bottom left corner coords are 0, 0, not -0.5, -0.5
        cen_x = cenx_img - begin_x + 0.5
        cen_y = ceny_img - begin_y + 0.5
        return cen_x, cen_y

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        bbox = source.getFootprint().getBBox()
        begin_x, begin_y = bbox.beginX, bbox.beginY
        # multiprofit bottom left corner coords are 0, 0, not -0.5, -0.5
        cen_x_img = cen_x + begin_x - 0.5
        cen_y_img = cen_y + begin_y - 0.5
        ra, dec = self.catexps[0].exposure.wcs.pixelToSky(cen_x_img, cen_y_img)
        return ra.asDegrees(), dec.asDegrees()

    def initialize_model(
        self, model: g2f.Model, source: Mapping[str, Any], limits_x: g2f.LimitsD, limits_y: g2f.LimitsD
    ):
        comps = model.sources[0].components
        sig_x = math.sqrt(source["base_SdssShape_xx"])
        sig_y = math.sqrt(source["base_SdssShape_yy"])
        # There is a sign convention difference
        rho = np.clip(-source["base_SdssShape_xy"] / (sig_x * sig_y), -0.5, 0.5)
        if not np.isfinite(rho):
            sig_x, sig_y, rho = 0.5, 0.5, 0
        if not source["base_SdssShape_flag"]:
            flux = source["base_SdssShape_instFlux"]
        else:
            flux = source["base_GaussianFlux_instFlux"]
            if not (flux > 0):
                flux = source["base_PsfFlux_instFlux"]
                if not (flux > 0):
                    flux = 1
        n_psfs = self.config.n_pointsources
        n_extended = len(self.config.sersics)
        observation = model.data[0]
        x_max = float(observation.image.n_cols)
        y_max = float(observation.image.n_rows)
        limits_x.max = x_max
        limits_y.max = y_max
        try:
            cenx, ceny = self.get_model_cens(source)
        # TODO: determine which exceptions can occur above
        except Exception:
            cenx = observation.image.n_cols / 2.0
            ceny = observation.image.n_rows / 2.0
        flux = flux / (n_psfs + n_extended)
        values_init = {
            g2f.IntegralParameterD: flux,
            g2f.CentroidXParameterD: cenx,
            g2f.CentroidYParameterD: ceny,
            g2f.ReffXParameterD: sig_x,
            g2f.ReffYParameterD: sig_y,
            g2f.SigmaXParameterD: sig_x,
            g2f.SigmaYParameterD: sig_y,
            g2f.RhoParameterD: -rho,
        }
        # Do not initialize PSF size/rho: they'll all stay zero
        params_psf_init = (g2f.IntegralParameterD, g2f.CentroidXParameterD, g2f.CentroidYParameterD)
        values_init_psf = {key: values_init[key] for key in params_psf_init}
        size_major = g2.EllipseMajor(g2.Ellipse(sigma_x=sig_x, sigma_y=sig_y, rho=rho)).r_major
        # An R_eff larger than the box size is problematic. This should also
        # stop unreasonable size proposals; a log10 transform isn't enough.
        # TODO: Try logit for r_eff?
        flux_max = 5 * max([np.sum(np.abs(datum.image.data)) for datum in model.data])
        flux_min = 1 / flux_max
        limits_flux = g2f.LimitsD(flux_min, flux_max, "unreliable flux limits")

        limits_init = {
            g2f.IntegralParameterD: limits_flux,
            g2f.ReffXParameterD: g2f.LimitsD(1e-5, x_max),
            g2f.ReffYParameterD: g2f.LimitsD(1e-5, y_max),
            g2f.SigmaXParameterD: g2f.LimitsD(1e-5, x_max),
            g2f.SigmaYParameterD: g2f.LimitsD(1e-5, y_max),
        }
        limits_init_psf = {g2f.IntegralParameterD: limits_init[g2f.IntegralParameterD]}

        for comp in comps[:n_psfs]:
            self._init_component(comp, values_init=values_init_psf, limits_init=limits_init_psf)
        for comp, config_comp in zip(comps[n_psfs:], self.config.sersics.values()):
            if config_comp.sersicindex.fixed:
                if g2f.SersicIndexParameterD in values_init:
                    del values_init[g2f.SersicMixComponentIndexParameterD]
            else:
                values_init[g2f.SersicMixComponentIndexParameterD] = config_comp.sersicindex.value_initial
            self._init_component(comp, values_init=values_init, limits_init=limits_init)
        for prior in model.priors:
            if isinstance(prior, g2f.GaussianPrior):
                # TODO: Add centroid prior
                pass
            elif isinstance(prior, g2f.ShapePrior):
                prior.prior_size.mean_parameter.value = size_major

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
        catexps_conv = [None] * len(catexps)
        for idx, catexp in enumerate(catexps):
            if isinstance(catexp, CatalogExposurePsfs):
                catexps_conv[idx] = catexp
            else:
                catexps_conv[idx] = CatalogExposurePsfs(
                    # dataclasses.asdict(catexp)_makes a recursive deep copy.
                    # That must be avoided.
                    **{key: getattr(catexp, key) for key in catexp.__dataclass_fields__.keys()},
                    channel=g2f.Channel.get(catexp.band),
                    config_fit=self.config,
                )
        self.catexps = catexps
        catalog = self.fit(catalog_multi=catalog_multi, catexps=catexps_conv, config=self.config, **kwargs)
        return pipeBase.Struct(output=astropy_to_arrow(catalog))
