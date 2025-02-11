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

__all__ = (
    "PsfFitSuccessActionBase",
    "PsfComponentsActionBase",
    "SourceTablePsfFitSuccessAction",
    "SourceTablePsfComponentsAction",
    "MagnitudeDependentSizePriorConfig",
    "ModelInitializer",
    "MakeInitializerActionBase",
    "BasicModelInitializer",
    "CachedBasicModelInitializer",
    "InitialInputData",
    "MakeBasicInitializerAction",
    "MakeCachedBasicInitializerAction",
    "MultiProFitSourceConfig",
    "CatalogExposurePsfs",
    "MultiProFitSourceFitter",
    "MultiProFitSourceTask",
)

from abc import abstractmethod, ABC
from functools import cached_property
import logging
import math
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from astropy.table import Table
import astropy.units as u
import lsst.afw.geom
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
from lsst.multiprofit.errors import NoDataError, PsfRebuildFitFlagError
from lsst.multiprofit.fitting.fit_psf import CatalogPsfFitterConfig, CatalogPsfFitterConfigData
from lsst.multiprofit.fitting.fit_source import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterABC,
    CatalogSourceFitterConfig,
    CatalogSourceFitterConfigData,
)
from lsst.multiprofit.modeller import Model
from lsst.multiprofit.utils import frozen_arbitrary_allowed_config, get_params_uniq, set_config_from_dict
import lsst.pex.config as pexConfig
from lsst.pex.config.configurableActions import ConfigurableAction, ConfigurableActionField
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_multiband as fitMB
import lsst.utils.timer as utilsTimer
import numpy as np
import pydantic

from .errors import IsParentError, NotPrimaryError
from .input_config import InputConfig
from .utils import get_spanned_image

_LOG = logging.getLogger(__name__)
TWO_SQRT_PI = 2 * math.sqrt(np.pi)


class PsfFitSuccessActionBase(ConfigurableAction):
    """Base action to return whether a source had a succesful PSF fit."""

    def get_schema(self) -> list[str]:
        """Return the list of columns required to call this action."""
        raise NotImplementedError("This method must be overloaded in subclasses")

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError("This method must be overloaded in subclasses")


class PsfComponentsActionBase(ConfigurableAction):
    """Base action to return a list of Gaussians from a source mapping."""

    def get_schema(self) -> list[str]:
        """Return the list of columns required to call this action."""
        raise NotImplementedError("This method must be overloaded in subclasses")

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> list[g2.Gaussian]:
        raise NotImplementedError("This method must be overloaded in subclasses")


class SourceTablePsfFitSuccessAction(PsfFitSuccessActionBase):
    """Action to return PSF fit status from a SourceTable row."""

    flag_format = pexConfig.Field[str](
        doc="Format for the flag field; flag_prefix, flag_suffix and flag_sub are substituted",
        default="{flag_prefix}{flag_suffix}{flag_sub}",
    )
    flag_prefix = pexConfig.Field[str](
        doc="Prefix for the key for the summed flag field",
        default="modelfit_DoubleShapeletPsfApprox",
    )
    flag_suffix = pexConfig.Field[str](
        doc="Suffix for all flag fields",
        default="_flag",
    )
    flags_sub = pexConfig.ListField[str](
        doc="Suffixes for specific flag fields that must not be true",
        default=["_invalidPointForPsf", "_invalidMoments", "_maxIterations"],
    )

    def _format(self, flag_sub: str) -> str:
        return self.flag_format.format(
            flag_prefix=self.flag_prefix,
            flag_sub=flag_sub,
            flag_suffix=self.flag_suffix,
        )

    def get_schema(self) -> Iterable[str]:
        for flag_sub in self.flags_sub:
            yield self._format(flag_sub=flag_sub)

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> bool:
        good = True
        for flag_sub in self.flags_sub:
            good &= not source[self._format(flag_sub=flag_sub)]
        return good


class SourceTablePsfComponentsAction(PsfComponentsActionBase):
    """Action to return PSF components from a SourceTable.

    This is anticipated to be a deepCoadd_meas with PSF fit parameters from a
    measurement plugin returning covariance matrix terms.
    """

    action_source = ConfigurableActionField[PsfFitSuccessActionBase](
        doc="Action to return whether the PSF fit was successful for a single source row",
        default=SourceTablePsfFitSuccessAction,
    )
    format = pexConfig.Field[str](
        doc="Format for the field names, where {idx_comp} is the index of the component and {moment}"
        "is the name of the moment (xx, xy or yy, integral)",
        default="modelfit_DoubleShapeletPsfApprox_{idx_comp}_{moment}",
    )
    name_moment_xx = pexConfig.Field[str](doc="Name of the xx (2nd x-axis) moment", default="xx")
    name_moment_xy = pexConfig.Field[str](doc="Name of the xy (covariance term) moment", default="xy")
    name_moment_yy = pexConfig.Field[str](doc="Name of the yy (2nd y-axis) moment", default="yy")
    name_moment_integral = pexConfig.Field[str](doc="Name of the integral (zeroth) moment", default="0")
    n_components = pexConfig.Field[int](
        doc="Number of Gaussian components",
        default=2,
        check=lambda x: x >= 2,
    )

    @staticmethod
    def get_integral(moment_zero) -> float:
        """Get the total integrated flux from a zeroth moment value.

        The zeroth moment is simply the integrated flux divided by a
        constant value of 2*sqrt(pi).

        Parameters
        ----------
        moment_zero
            The zeroth moment value.

        Returns
        -------
        integral
            The total integrated weight (flux).
        """
        return moment_zero * TWO_SQRT_PI

    def get_schema(self) -> list[str]:
        names_moments = (
            self.name_moment_xx,
            self.name_moment_yy,
            self.name_moment_xy,
            self.name_moment_integral,
        )
        columns = [
            column
            for idx_comp in range(self.n_components)
            for column in (
                self.format.format(name_moment=name_moment, idx_comp=idx_comp)
                for name_moment in names_moments
            )
        ] + self.action_source.get_schema()
        return columns

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> list[g2.Gaussian]:
        if not self.action_source(source):
            raise PsfRebuildFitFlagError(
                f"PSF fit failed due to action based on schema: {self.action_source.get_schema()}"
            )
        gaussians = [None] * self.n_components
        for idx_comp in range(self.n_components):
            gaussian = g2.Gaussian(
                ellipse=g2.Ellipse(
                    g2.Covariance(
                        sigma_x_sq=source[self.format.format(moment=self.name_moment_xx, idx_comp=idx_comp)],
                        sigma_y_sq=source[self.format.format(moment=self.name_moment_yy, idx_comp=idx_comp)],
                        cov_xy=source[self.format.format(moment=self.name_moment_xy, idx_comp=idx_comp)],
                    )
                ),
                integral=g2.GaussianIntegralValue(
                    value=self.get_integral(
                        source[self.format.format(moment=self.name_moment_integral, idx_comp=idx_comp)]
                    )
                ),
            )
            gaussians[idx_comp] = gaussian
        return gaussians


class MagnitudeDependentSizePriorConfig(pexConfig.Config):
    """Configuration for a magnitude-dependent size prior.

    Defaults are for ugrizy total mag and log10(r_eff/arcsec).
    """

    intercept_mag = pexConfig.Field[float](
        doc="The magnitude at which no adjustment is applied",
        default=18.0,
    )
    slope_median_per_mag = pexConfig.Field[float](
        doc="The slope in the median size, in dex per mag",
        default=-0.15,
    )
    slope_stddev_per_mag = pexConfig.Field[float](
        doc="The slope in the standard deviation of the size, in dex per mag",
        default=0.,
    )


class ModelInitializer(ABC, pydantic.BaseModel):
    """An interface for a configurable model initializer based on priors
    and optional external data.
    """

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    inputs: dict[str, Any] = pydantic.Field(
        title="Additional external inputs used in initialization",
        default_factory=dict,
    )
    priors_shape_mag: dict = pydantic.Field(
        title="Magnitude-dependent shape prior configurations",
        default_factory=dict,
    )

    @abstractmethod
    def initialize_model(
        self,
        model: Model,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        **kwargs,
    ):
        """Initialize a MultiProFit model for a single object corresponding
        to a row in a catalog.

        Parameters
        ----------
        model
            The model to initialize parameter values for.
        source
            A mapping with fields expected to be populated in the
            corresponding source catalog for initialization.
        catexps
            Per-band catalog-exposure pairs.
        config_data
            Fitter configuration and data.
        values_init
            Default initial values for parameters.
        kwargs
            Additional keyword arguments for any purpose.
        """
        raise NotImplementedError(f"{self.__name__} must implement initialize_model")


class MakeInitializerActionBase(ConfigurableAction):
    """An interface for an action that creates an initializer."""

    def __call__(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
        **kwargs
    ) -> ModelInitializer:
        """Make a ModelInitializer object that can initialize model
           parameter values for a given object in a catalog.

        Parameters
        ----------
        catalog_multi
            The multiband catalog with one row per object to fit.
        catexps
            Per-band catalog-exposure pairs.
        config_data
            Fitter configuration and data.
        kwargs
            Additional arguments to pass to add to ModelInitializer.inputs.

        Returns
        -------
        initializer
            The configured ModelInitializer.
        """
        raise NotImplementedError(f"{self.__name__} must implement __call__")


class BasicModelInitializer(ModelInitializer):
    """A generic model initializer that should work on most kinds of models
    with a single source.
    """

    def _get_params_init(self, model_sources: tuple[g2f.Source]) -> tuple[g2f.ParameterD]:
        """Return an ordered set of free parameters from a model's sources.

        Parameters
        ----------
        model_sources
            The sources in the model.

        Returns
        -------
        params_init
            The parameter objects for sources in the model.

        Notes
        -----
        Only free and/or centroid parameters are returned (centroids are
        always needed even if they are fixed).
        """
        # TODO: There ought to be a better way to not get the PSF centroids
        # (those are part of model.data's fixed parameters)
        params_init = (
            tuple(
                (
                    param
                    for param in get_params_uniq(model_sources[0])
                    if param.free
                    or (
                        isinstance(param, g2f.CentroidXParameterD)
                        or isinstance(param, g2f.CentroidYParameterD)
                    )
                )
            )
            if (len(model_sources) == 1)
            else tuple(
                {
                    param: None
                    for source in model_sources
                    for param in get_params_uniq(source)
                    if param.free
                    or (
                        isinstance(param, g2f.CentroidXParameterD)
                        or isinstance(param, g2f.CentroidYParameterD)
                    )
                }.keys()
            )
        )
        return params_init

    def _get_priors_type(
        self, priors: tuple[g2f.Prior],
    ) -> tuple[tuple[g2f.GaussianPrior], tuple[g2f.ShapePrior]]:
        """Return the list of priors of known type, by type.

        Parameters
        ----------
        priors
            A list of priors of any type, typically from a model.

        Returns
        -------
        priors_gauss
            A list of all of the Gaussian priors, in the order they occurred.
        priors_shape
            A list of all of the shape priors, in the order they occurred.
        """
        priors_gauss: list[g2f.GaussianPrior] = []
        priors_shape: list[g2f.ShapePrior] = []
        for prior in priors:
            if isinstance(prior, g2f.GaussianPrior):
                priors_gauss.append(prior)
            elif isinstance(prior, g2f.ShapePrior):
                priors_shape.append(prior)
        return tuple(priors_gauss), tuple(priors_shape)

    def get_centroid_and_shape(
        self,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float, float]]:
        """Get the centroid and shape for a source.

        Parameters
        ----------
        source
            A mapping with fields expected to be populated in the
            corresponding source catalog for initialization.
        catexps
            A list of (source and psf) catalog-exposure pairs.
        config_data
            Configuration settings and data for fitting and output.
        values_init
            Initial parameter values from the model configuration.

        Returns
        -------
        centroid
            x- and y-axis centroid values.
        sig_x, sig_y, rho
            The x- and y-axis Gaussian sigma and rho values defining the
            estimated elliptical shape of the source.
        """
        centroid = source["slot_Centroid_x"], source["slot_Centroid_y"]
        sig_x = math.sqrt(source["slot_Shape_xx"])
        sig_y = math.sqrt(source["slot_Shape_yy"])
        # TODO: Verify if there's a sign difference here
        rho = np.clip(source["slot_Shape_xy"] / (sig_x * sig_y), -0.5, 0.5)
        shape = sig_x, sig_y, rho
        return centroid, shape

    def get_params_init(self, model: Model) -> tuple[g2f.ParameterD]:
        """Return the free and/or centroid parameters for a model.

        Parameters
        ----------
        model
            The model to return parameters for.

        Returns
        -------
        parameters
            The ordered list of parameters for the model.
        """
        return self._get_params_init(model_sources=model.sources)

    def get_priors_type(self, model: Model) -> tuple[tuple[g2f.GaussianPrior], tuple[g2f.ShapePrior]]:
        """Return the list of priors of known type, by type.

        Parameters
        ----------
        model
            The model to return priors for.

        Returns
        -------
        priors_gauss
            A list of all of the Gaussian priors, in the order they occurred.
        priors_shape
            A list of all of the shape priors, in the order they occurred.
        """
        return self._get_priors_type(model.priors)

    def initialize_model(
        self,
        model: Model,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        **kwargs,
    ):
        if values_init is None:
            values_init = {}
        set_flux_limits = kwargs.pop("set_flux_limits") if "set_flux_limits" in kwargs else True
        if kwargs:
            raise ValueError(f"Unexpected {kwargs=}")
        centroid_pixel_offset = config_data.config.centroid_pixel_offset
        (cen_x, cen_y), (sig_x, sig_y, rho) = self.get_centroid_and_shape(
            source, catexps, config_data, values_init=values_init,
        )
        # If we couldn't get a shape at all, make it small and roundish
        if not np.isfinite(rho):
            # Note rho=0 (circular) is generally disfavoured by shape priors
            # However, setting it to a non-zero value seems to make scipy
            # fail to move off initial conditions, as do sizes below 2 pixels
            sig_x, sig_y, rho = 2.0, 2.0, 0.0

        # Make restrictive centroid limits (intersection, not union)
        x_min, y_min, x_max, y_max = -np.inf, -np.inf, np.inf, np.inf

        fluxes_init = {}
        fluxes_limits = {}

        # This is the maximum number of potential observations
        # They might not all have made it into the data
        n_catexps = len(catexps)
        n_components = len(model.sources[0].components)

        # If not true, some bands must have no data to fit
        if len(catexps) != len(model.data):
            catexps_obs = []
            for catexp in catexps:
                fluxes_init[catexp.channel] = 0
                fluxes_limits[catexp.channel] = (0, np.inf)
                # No associated catalog means we can't fit (and should be
                # because there's no exposure for this band in this patch)
                if len(catexp.get_catalog()) > 0:
                    catexps_obs.append(catexp)
        else:
            catexps_obs = catexps

        for idx_obs, observation in enumerate(model.data):
            coordsys = observation.image.coordsys
            catexp = catexps_obs[idx_obs]
            band = catexp.band

            x_min = max(x_min, coordsys.x_min)
            y_min = max(y_min, coordsys.y_min)
            x_max = min(x_max, coordsys.x_min + float(observation.image.n_cols))
            y_max = min(y_max, coordsys.y_min + float(observation.image.n_rows))

            flux_total = np.nansum(observation.image.data[observation.mask_inv.data])

            column_ref = f"merge_measurement_{band}"
            if column_ref in source.schema.getNames() and source[column_ref]:
                row = source
            else:
                row = catexp.catalog.find(source["id"])

            if not row["base_SdssShape_flag"]:
                flux_init = row["base_SdssShape_instFlux"]
            else:
                flux_init = row["slot_GaussianFlux_instFlux"]
                if not (flux_init > 0):
                    flux_init = row["slot_PsfFlux_instFlux"]

            calib = catexp.exposure.photoCalib
            flux_init = calib.instFluxToNanojansky(flux_init) if (flux_init > 0) else max(flux_total, 1.0)
            if set_flux_limits:
                flux_max = 10 * max((flux_init, flux_total))
                flux_min = min(1e-12, flux_max / 1000)
            else:
                flux_min, flux_max = 0, np.inf
            fluxes_init[observation.channel] = flux_init / n_components
            fluxes_limits[observation.channel] = (flux_min, flux_max)

        if not np.isfinite(cen_x):
            cen_x = observation.image.n_cols / 2.0
        else:
            cen_x -= centroid_pixel_offset
        if not np.isfinite(cen_y):
            # TODO: Add bbox coords or remove
            cen_y = observation.image.n_rows / 2.0
        else:
            cen_y -= centroid_pixel_offset

        # An R_eff larger than the box size is problematic. This should also
        # stop unreasonable size proposals; a log10 transform isn't enough.
        # TODO: Try logit for r_eff?
        size_major = g2.EllipseMajor(g2.Ellipse(sigma_x=sig_x, sigma_y=sig_y, rho=rho)).r_major
        limits_size = max(5.0 * size_major, 2.0 * np.hypot(x_max - x_min, y_max - y_min))
        limits_xy = (1e-5, limits_size)
        params_limits_init = {
            g2f.CentroidXParameterD: (cen_x, (x_min, x_max)),
            g2f.CentroidYParameterD: (cen_y, (y_min, y_max)),
            g2f.ReffXParameterD: (sig_x, limits_xy),
            g2f.ReffYParameterD: (sig_y, limits_xy),
            g2f.SigmaXParameterD: (sig_x, limits_xy),
            g2f.SigmaYParameterD: (sig_y, limits_xy),
            g2f.RhoParameterD: (rho, None),
            # TODO: get guess from configs?
            g2f.SersicMixComponentIndexParameterD: (1.0, None),
        }

        fluxes_init_tuple = tuple(fluxes_init.values())
        fluxes_limits_tuple = tuple(fluxes_limits.values())
        idx_obs = 0
        for param in self.params_init:
            if param.linear:
                value_init = fluxes_init_tuple[idx_obs]
                limits_new = fluxes_limits_tuple[idx_obs]
                idx_obs += 1
                if idx_obs == n_catexps:
                    idx_obs = 0
            else:
                type_param = type(param)
                value_init, limits_new = params_limits_init.get(type_param, (values_init.get(param), None))
            if limits_new:
                param.limits = g2f.LimitsD(limits_new[0], limits_new[1])
            if value_init is not None:
                param.value = value_init

        priors_shape_mag = self.priors_shape_mag
        has_priors_mag = len(priors_shape_mag) > 0
        if has_priors_mag:
            mag_total = u.nJy.to(u.ABmag, np.nansum(fluxes_init_tuple))

        # TODO: Add centroid prior
        priors_gauss, priors_shape = self.get_priors_type(model)
        for prior in priors_shape:
            if has_priors_mag and ((prior_adjustments := priors_shape_mag.get(prior)) is not None):
                mag_dep_prior, prior_shape_new = prior_adjustments
                prior_size_new = prior_shape_new.prior_size
                # the size-apparent mag relation probably flattens
                # for very bright/faint objects - maybe not so
                # sharply, but clipping a broad mag range ought to be fine
                prior.prior_size.mean_parameter.value = prior_size_new.mean_parameter.value * 10**(
                    mag_dep_prior.slope_median_per_mag*np.clip(
                        mag_total - mag_dep_prior.intercept_mag, -12.5, 12.5,
                    )
                )
                # it's uncertain how the intrinsic scatter behaves
                # educated guess is it doesn't change much, also
                # one runs out of bright galaxies to measure it anyway
                prior.prior_size.stddev_parameter.value = prior_size_new.stddev_parameter.value * 10**(
                    mag_dep_prior.slope_stddev_per_mag*np.clip(
                        mag_total - mag_dep_prior.intercept_mag, -12.5, 12.5,
                    )
                )
            else:
                prior.prior_size.mean_parameter.value = size_major


class CachedBasicModelInitializer(BasicModelInitializer):
    """A basic initializer with a cached list of model sources and priors."""

    priors: tuple[g2f.Prior, ...] = pydantic.Field(title="The gauss2d_fit model sources")
    sources: tuple[g2f.Source, ...] = pydantic.Field(title="The gauss2d_fit model sources")

    @cached_property
    def params_init(self) -> tuple[g2f.ParameterD]:
        """Return a cached reference to the result of _get_params_init."""
        return self._get_params_init(model_sources=self.sources)

    @cached_property
    def priors_type(self) -> tuple[tuple[g2f.GaussianPrior], tuple[g2f.ShapePrior]]:
        """Return a cached reference to the result of _get_priors_type."""
        return self._get_priors_type(self.priors)

    def get_params_init(self, model: Model) -> tuple[g2f.ParameterD]:
        assert tuple(model.sources) == self.sources
        return self.params_init

    def get_priors_type(self, model: Model) -> tuple[tuple[g2f.GaussianPrior], tuple[g2f.ShapePrior]]:
        assert tuple(model.priors) == self.priors
        return self.priors_type


class InitialInputData(pydantic.BaseModel):
    """A configurable wrapper to retrieve formatted columns from a catalog.

    This provides a common interface to typical MultiProFit table outputs.
    """

    model_config: ClassVar[pydantic.ConfigDict] = frozen_arbitrary_allowed_config

    column_id: str | None = pydantic.Field(
        title="Override for id column specified in config_input",
        default=None,
    )
    config_input: InputConfig = pydantic.Field(title="Configuration for the data table")
    data: Table = pydantic.Field(title="The data table")
    name_model: str = pydantic.Field(title="The name of the model in columns")
    prefix_column: str = pydantic.Field(title="The prefix for all fitted column names")
    size_column: str = pydantic.Field(title="The name of the size column", default="reff")

    def get_column_id(self):
        """Return the name of the object ID column."""
        return self.column_id or self.config_input.column_id

    def get_column(self, name_column: str, data=None):
        """Get the values from a column.

        Parameters
        ----------
        name_column
            The name of the column to retrieve.
        data
            The catalog to retrieve the column from. Default is self.data.

        Returns
        -------
        values
            The column values.
        """
        if data is None:
            data = self.data
        return data[f"{self.prefix_column}{name_column}"]

    def model_post_init(self, __context: Any) -> None:
        # Initialize a mapping of the row number for a given object ID value
        # This is implemented in afw catalogs but not most other tabular types
        id_index = {idnum: idx for idx, idnum in enumerate(self.data[self.get_column_id()])}
        object.__setattr__(self, "id_index", id_index)


class MakeBasicInitializerAction(MakeInitializerActionBase):
    """An action to construct an initializer for a single-component,
    single-source model.
    """

    def _make_initializer(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
    ) -> ModelInitializer:
        return BasicModelInitializer()

    def __call__(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
        **kwargs
    ) -> ModelInitializer:
        initializer = self._make_initializer(
            catalog_multi=catalog_multi, catexps=catexps, config_data=config_data,
        )
        for name, (config_input, data) in kwargs.items():
            if not isinstance(data, Table) and hasattr(data, "meta"):
                _LOG.warning(
                    f"Ignoring extra input {name=} because it is of type {type(data)} and is either not an"
                    f" astropy.table.Table or missing a 'meta' attr"
                )
            config_data = data.meta["config"]
            prefix_column = config_data["prefix_column"]
            config_source = next(iter(config_data["config_model"]["sources"].values()))
            config_group = next(iter(config_source["component_groups"].values()))
            is_sersic = len(config_group["components_sersic"]) > 0
            name_model = next(iter(
                config_group["components_sersic"] if is_sersic else config_group["components_gaussian"]
            ))
            initializer.inputs[name] = InitialInputData(
                column_id=config_data.get("column_id"),
                config_input=config_input,
                data=data,
                name_model=name_model,
                prefix_column=prefix_column,
                size_column="reff" if is_sersic else "sig",
            )
        return initializer


class MakeCachedBasicInitializerAction(MakeBasicInitializerAction):
    """A MakeBasicInitializerAction that caches references to the source
    and prior objects of the model.

    This is solely a performance optimization and should be favored over
    MakeBasicInitializerAction unless the caching is shown to be slower.
    """

    def _make_initializer(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
    ) -> ModelInitializer:
        sources, priors = config_data.sources_priors
        return CachedBasicModelInitializer(priors=priors, sources=sources)


class MultiProFitSourceConfig(CatalogSourceFitterConfig, fitMB.CoaddMultibandFitSubConfig):
    """Configuration for the MultiProFit profile fitter."""

    action_initializer = ConfigurableActionField[MakeInitializerActionBase](
        doc="The action to return an initializer",
        default=MakeCachedBasicInitializerAction,
    )
    action_psf = ConfigurableActionField[PsfComponentsActionBase](
        doc="The action to return PSF component values from catalogs, if implemented",
        default=None,
    )
    columns_copy = pexConfig.DictField[str, str](
        doc="Mapping of input/output column names to copy from the input"
        "multiband catalog to the output fit catalog.",
        default={
            "base_ClassificationExtendedness_value": "refExtendedness",
            "base_ClassificationExtendedness_flag": "refExtendedness_flag",
            "detect_isPatchInner": "detect_isPatchInner",
            "detect_isPrimary": "detect_isPrimary",
        },
        dictCheck=lambda x: len(set(x.values())) == len(x.values()),
    )
    mask_names_zero = pexConfig.ListField[str](
        doc="Mask bits to mask out",
        default=["BAD", "EDGE", "SAT", "NO_DATA"],
    )
    psf_sigma_subtract = pexConfig.Field[float](
        doc="PSF x/y sigma value to subtract in quadrature from best-fit values",
        default=0.1,
        check=lambda x: np.isfinite(x) and (x >= 0),
    )
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")
    size_priors = pexConfig.ConfigDictField[str, MagnitudeDependentSizePriorConfig](
        doc="Per-component magnitude-dependent size prior configurations."
            " Will be added to component with existing configs.",
        default={},
    )

    def bands_read_only(self) -> set[str]:
        # TODO: Re-implement determination of prior-only bands once
        # data-driven priors are re-implemented (DM-4xxxx)
        return set()

    def requires_psf(self):
        """Return whether the PSF action is not None."""
        return type(self.action_psf) is PsfComponentsActionBase

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {
            IsParentError.column_name(): "IsParentError",
            NoDataError.column_name(): "NoDataError",
            NotPrimaryError.column_name(): "NotPrimaryError",
            PsfRebuildFitFlagError.column_name(): "PsfRebuildFitFlagError",
        }
        self.centroid_pixel_offset = -0.5
        self.naming_scheme = "lsst"
        self.prefix_column = ""
        self.suffix_error = "Err"


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, config=fitMB.CatalogExposureConfig)
class CatalogExposurePsfs(fitMB.CatalogExposureInputs, CatalogExposureSourcesABC):
    """Input data from lsst pipelines, parsed for MultiProFit."""

    channel: g2f.Channel = pydantic.Field(title="Channel for the image's band")
    config_fit: MultiProFitSourceConfig = pydantic.Field(title="Config for fitting options")

    @cached_property
    def _psf_flux_params(self) -> tuple[list[g2f.ParameterD], bool]:
        psf_model = self.psf_model_data.psf_model
        n_comps = len(psf_model.components)
        params_flux = [None] * n_comps
        is_frac = [False] * n_comps
        for idx_comp, comp in enumerate(psf_model.components):
            # TODO: Change to comp.integralmodel when DM-44344 is fixed
            # integralmodels will still need to be handled differently
            params_all = get_params_uniq(comp)
            params_frac = [param for param in params_all if isinstance(param, g2f.ProperFractionParameterD)]
            if params_frac:
                is_last = idx_comp == (n_comps - 1)
                if len(params_frac) != (idx_comp + 1 - is_last):
                    raise RuntimeError(
                        f"Got unexpected {params_frac=} for"
                        f" {self.psf_model_data.psf_model.components[idx_comp]=} ({idx_comp=});"
                        f" len should be idx_comp+1"
                    )
                params_flux[idx_comp] = None if is_last else params_frac[idx_comp]
                is_frac[idx_comp] = True
            else:
                params_integral = [param for param in params_all if isinstance(param, g2f.IntegralParameterD)]
                if len(params_integral != 1):
                    raise RuntimeError(
                        f"Got unexpected {params_integral=} != 1 for"
                        f" {self.psf_model_data.psf_model.components[idx_comp]=} ({idx_comp=})"
                    )
                params_flux[idx_comp] = params_integral[0]
        is_frac_any = any(is_frac)
        if is_frac_any and not all(is_frac):
            # TODO: This should work by iterating through componentgroups
            # But that's not trivial or supported now
            raise RuntimeError("Got PSF model with a mix of fractional and linear models; cannot initialize")

        return params_flux, is_frac_any

    def get_psf_model(self, source):
        psf_model = self.psf_model_data.psf_model
        # PsfComponentsActionBase is an abstract class, so check if the action
        # is a subclass that needs to be called
        if not self.config_fit.requires_psf():
            try:
                gaussians = self.config_fit.action_psf(source)
            except PsfRebuildFitFlagError:
                return None
            n_comps = len(psf_model.components)
            fluxes = [0.0] * n_comps
            params_flux, is_frac = self._psf_flux_params
            for idx_comp, (comp, gaussian) in enumerate(zip(psf_model.components, gaussians)):
                ellipse_out = comp.ellipse
                ellipse_in = gaussian.ellipse
                ellipse_out.sigma_x = ellipse_in.sigma_x
                ellipse_out.sigma_y = ellipse_in.sigma_y
                ellipse_out.rho = ellipse_in.rho
                fluxes[idx_comp] = gaussian.integral.value
            # Apparently negative fluxes are possible. Not much can be done to
            # fix that but set them to a tiny value (zero might work)
            fluxes = np.clip(fluxes, 1e-3, np.inf)
            flux_total = sum(fluxes)
            if is_frac:
                flux_remaining = 1.0
                for flux, param_frac in zip(fluxes, params_flux[:-1]):
                    flux_component = flux / flux_total
                    param_frac.value = flux_component / flux_remaining
                    flux_remaining -= flux_component
            else:
                for flux, param_flux in zip(fluxes, params_flux):
                    param_flux.value = flux / flux_total
        else:
            # TODO: this should probably use .index or something
            match = np.argwhere(
                self.table_psf_fits[self.psf_model_data.config.column_id] == source[self.config_fit.column_id]
            )[0][0]
            psf_model = self.psf_model_data.psf_model
            self.psf_model_data.init_psf_model(self.table_psf_fits[match])

        sigma_subtract = self.config_fit.psf_sigma_subtract
        if sigma_subtract > 0:
            sigma_subtract_sq = sigma_subtract * sigma_subtract
            for param in self.psf_model_data.parameters.values():
                if isinstance(
                    param,
                    g2f.SigmaXParameterD | g2f.SigmaYParameterD | g2f.ReffXParameterD | g2f.ReffYParameterD,
                ):
                    param.value = math.sqrt(param.value**2 - sigma_subtract_sq)
        return psf_model

    def get_source_observation(self, source, **kwargs) -> g2f.ObservationD:
        if not kwargs.get("skip_flags"):
            if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
                raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")
        footprint = source.getFootprint()
        bbox = footprint.getBBox()
        if not (bbox.getArea() > 0):
            return None
        bitmask = 0
        mask = self.exposure.mask[bbox]
        spans = footprint.spans.asArray()
        for bitname in self.config_fit.mask_names_zero:
            bitval = mask.getPlaneBitMask(bitname)
            bitmask |= bitval
        mask = ((mask.array & bitmask) != 0) & (spans != 0)
        mask = ~mask

        is_deblended_child = source["parent"] != 0

        img, _, sigma_inv = get_spanned_image(
            exposure=self.exposure,
            footprint=footprint if is_deblended_child else None,
            bbox=bbox,
            spans=spans,
            get_sig_inv=True,
        )
        x_min_bbox, y_min_bbox = bbox.beginX, bbox.beginY
        # Crop to tighter box for deblended model if edges are unusable
        # ... this rarely ever seems to happen though
        if is_deblended_child:
            coords = np.argwhere(np.isfinite(img) & (sigma_inv > 0) & np.isfinite(sigma_inv))
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            x_max += 1
            y_max += 1

            if (x_min > 0) or (y_min > 0) or (x_max < img.shape[0]) or (y_max < img.shape[1]):
                # Ensure the nominal centroid is still inside the box
                # ... although it's a bad sign if that row/column is all bad
                x_cen = source["slot_Centroid_x"] - x_min_bbox
                y_cen = source["slot_Centroid_y"] - y_min_bbox
                x_min = min(x_min, int(np.floor(x_cen)))
                x_max = max(x_max, int(np.ceil(x_cen)))
                y_min = min(y_min, int(np.floor(y_cen)))
                y_max = max(y_max, int(np.ceil(y_cen)))
                x_min_bbox += x_min
                y_min_bbox += y_min
                img = img[x_min:x_max, y_min:y_max]
                sigma_inv = sigma_inv[x_min:x_max, y_min:y_max]
                mask = mask[x_min:x_max, y_min:y_max]

        sigma_inv[~mask] = 0

        coordsys = g2.CoordinateSystem(1.0, 1.0, x_min_bbox, y_min_bbox)

        obs = g2f.ObservationD(
            image=g2.ImageD(img, coordsys),
            sigma_inv=g2.ImageD(sigma_inv, coordsys),
            mask_inv=g2.ImageB(mask, coordsys),
            channel=self.channel,
        )
        return obs

    def __post_init__(self):
        # TODO: Can/should this be the derived type (MultiProFitPsfConfig)?
        config = CatalogPsfFitterConfig()
        config_dict = self.table_psf_fits.meta.get("config")
        if config_dict:
            set_config_from_dict(config, config_dict)
        else:
            # TODO: How should this be set?
            # If using external PSF fits, it needs to be configured normally
            pass
        config_data = CatalogPsfFitterConfigData(config=config)
        object.__setattr__(self, "psf_model_data", config_data)


class MultiProFitSourceFitter(CatalogSourceFitterABC):
    """A MultiProFit source fitter.

    Parameters
    ----------
    wcs
        A WCS solution that applies to all exposures.
    errors_expected
        A dictionary of exceptions that are expected to sometimes be raised
        during processing (e.g. for missing data) keyed by the name of the
        flag column used to record the failure.
    add_missing_errors
        Whether to add all of the standard MultiProFit errors with default
        column names to errors_expected, if not already present.
    **kwargs
        Keyword arguments to pass to the superclass constructor.
    """

    initializer: ModelInitializer = pydantic.Field(
        title="The model parameter initializer",
        default_factory=lambda: BasicModelInitializer(),
    )
    wcs: lsst.afw.geom.SkyWcs = pydantic.Field(
        title="The WCS object to use to convert pixel coordinates to RA/dec",
    )

    def __init__(
        self,
        wcs: lsst.afw.geom.SkyWcs,
        errors_expected: dict[str, Exception] | None = None,
        add_missing_errors: bool = True,
        **kwargs: Any,
    ):
        if errors_expected is None:
            errors_expected = {}
        if add_missing_errors:
            for error_catalog in (IsParentError, NoDataError, NotPrimaryError, PsfRebuildFitFlagError):
                if error_catalog not in errors_expected:
                    errors_expected[error_catalog] = error_catalog.column_name()
        super().__init__(wcs=wcs, errors_expected=errors_expected, **kwargs)

    def copy_centroid_errors(
        self,
        columns_cenx_err_copy: tuple[str],
        columns_ceny_err_copy: tuple[str],
        results: Table,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
    ):
        for column in columns_cenx_err_copy:
            results[column] = catalog_multi["slot_Centroid_xErr"]
        for column in columns_ceny_err_copy:
            results[column] = catalog_multi["slot_Centroid_yErr"]

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        # no extra conversions are needed here - cen_x, cen_y are in catalog
        # coordinates already
        ra, dec = self.wcs.pixelToSky(cen_x, cen_y)
        return ra.asDegrees(), dec.asDegrees()

    def initialize_model(
        self,
        model: g2f.ModelD,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        **kwargs,
    ):
        self.initializer.initialize_model(
            model=model, source=source, catexps=catexps, config_data=config_data, values_init=values_init,
            **kwargs
        )

    def make_CatalogExposurePsfs(
        self,
        catexp: fitMB.CatalogExposureInputs,
        config: MultiProFitSourceConfig,
    ) -> CatalogExposurePsfs:
        """Make a CatalogExposurePsfs from a list of inputs and a fit config.

        Parameters
        ----------
        catexp
            The input catalog-exposure pairs.
        config
            The MultiProFit source fitting config.

        Returns
        -------
        catexp_psf
            The resulting CatalogExposurePsfs.
        """
        catexp_psf = CatalogExposurePsfs(
            # dataclasses.asdict(catexp)_makes a recursive deep copy.
            # That must be avoided.
            **{key: getattr(catexp, key) for key in catexp.__dataclass_fields__.keys()},
            channel=g2f.Channel.get(catexp.band),
            config_fit=config,
        )
        return catexp_psf

    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposurePsfs],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        errors = []
        for idx, catexp in enumerate(catexps):
            if not isinstance(catexp, CatalogExposurePsfs):
                errors.append(f"catexps[{idx=} {type(catexp)=} !isinstance(CatalogExposurePsfs)")
        # Pre-validate the model
        config_sources = config_data.config.config_model.sources
        model_sources, priors = config_data.sources_priors
        priors_shape = [prior for prior in priors if isinstance(prior, g2f.ShapePrior)]

        if len(config_sources.keys()) > 1:
            errors.append(f"model config has multiple sources: {list(config_sources.keys())}")
        elif len(priors_shape) > 0:
            idx_prior_found = 0
            name_source, config_source = next(iter(config_sources.items()))
            source = model_sources[0]
            config_groups = config_source.component_groups
            if len(config_groups.keys()) > 1:
                errors.append(f"model {name_source=} has multiple groups: {list(config_source.keys())}")
            else:
                name_group, config_group = next(iter(config_groups.items()))
                for idx_comp, (name_comp, config_comp) in enumerate(
                    config_group.get_component_configs().items()
                ):
                    ellipse = source.components[idx_comp].ellipse
                    # component.ellipse returns a const ref and must be copied
                    # The ellipse classes might need copy constructors
                    ellipse_copy = type(ellipse)(
                        size_x=ellipse.size_x, size_y=ellipse.size_y, rho=ellipse.rho,
                    )
                    prior_shape_new = config_comp.make_shape_prior(ellipse_copy)
                    if prior_shape_new is not None:
                        if idx_prior_found == len(priors_shape):
                            errors.append(
                                f"Could not validate prior for {name_source=} {name_group=} {name_comp=}"
                            )
                            break
                        prior_shape_old = priors_shape[idx_prior_found]
                        ll_new, ll_old = (
                            prior.evaluate().loglike for prior in (prior_shape_new, prior_shape_old)
                        )
                        # The necessary tolerance for this check is uncertain
                        if not np.isclose(ll_new, ll_old):
                            logger.warning(
                                f"shape prior for {name_comp=} got inconsistent {ll_new=} vs {ll_old}"
                            )
                        if (prior_shape_mod := config_data.config.size_priors.get(name_comp)) is not None:
                            self.initializer.priors_shape_mag[prior_shape_old] = (
                                prior_shape_mod, prior_shape_new,
                            )

        if errors:
            raise RuntimeError("\n".join(errors))


class MultiProFitSourceTask(fitMB.CoaddMultibandFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit a single model to all sources in a coadd,
    using a previously-fit PSF model for each exposure. The task may also use
    prior measurements from single- or merged multiband catalogs for
    initialization.
    """

    ConfigClass: ClassVar = MultiProFitSourceConfig
    _DefaultName: ClassVar = "multiProFitSource"

    def make_default_fitter(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
        **kwargs
    ) -> MultiProFitSourceFitter:
        """Make a default MultiProFitSourceFitter.

        Parameters
        ----------
        catalog_multi
            A multi-band, indexable source catalog.
        catexps
            Catalog-exposure-PSF model tuples to fit source models for.
        config_data
            Configuration and data for the initalizer.
        kwargs
            Additional keyword arguments to pass to
            self.config.action_initializer.

        Returns
        -------
        fitter
            A MultiProFitSourceFitter using the first catexp's wcs.
        """
        initializer = self.config.action_initializer(
            catalog_multi=catalog_multi, catexps=catexps, config_data=config_data,
            **kwargs
        )
        # Look for the first WCS - they ought to be identical
        # If they are not, the patch coadd data model must have changed
        wcs = None
        for catexp in catexps:
            if catexp.exposure is not None:
                wcs = catexp.exposure.wcs
                break
        if wcs is None:
            raise RuntimeError(f"Could not find valid wcs in any of {catexps=}")
        fitter = MultiProFitSourceFitter(wcs=wcs, initializer=initializer)
        return fitter

    @utilsTimer.timeMethod
    def run(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        fitter: MultiProFitSourceFitter | None = None,
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit source fit task on catalog-exposure pairs.

        Parameters
        ----------
        catalog_multi
            A multi-band, indexable source catalog.
        catexps
            Catalog-exposure-PSF model tuples to fit source models for.
        fitter
            The fitter instance to use. Default-initialized if not provided.
        kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `astropy.Table`
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        n_catexps = len(catexps)
        if n_catexps == 0:
            raise ValueError("Must provide at least one catexp")
        catexps_conv: list[CatalogExposurePsfs] = [None] * n_catexps
        channels = [g2f.Channel.get(catexp.band) for catexp in catexps]
        config_data = CatalogSourceFitterConfigData(channels=channels, config=self.config)
        if fitter is None:
            inputs_init = kwargs.get("inputs_init")
            if inputs_init:
                del kwargs["inputs_init"]
            else:
                inputs_init = {}
            fitter = self.make_default_fitter(
                catalog_multi=catalog_multi, catexps=catexps, config_data=config_data,
                **inputs_init
            )
        for idx, catexp in enumerate(catexps):
            if not isinstance(catexp, CatalogExposurePsfs):
                catexp = fitter.make_CatalogExposurePsfs(catexp, config=self.config)
            catexps_conv[idx] = catexp
        catalog = fitter.fit(
            catalog_multi=catalog_multi, catexps=catexps_conv, config_data=config_data, **kwargs
        )
        for name_in, name_out in self.config.columns_copy.items():
            catalog[name_out] = catalog_multi[name_in]
            catalog[name_out].description = catalog_multi.schema.find(name_in).field.getDoc()
        return pipeBase.Struct(output=astropy_to_arrow(catalog))
