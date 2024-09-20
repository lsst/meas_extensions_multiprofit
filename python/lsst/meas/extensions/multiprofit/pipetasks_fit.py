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
    "component_names_default",
    "model_names_default",
    "MultiProFitCoaddPsfFitConfig",
    "MultiProFitCoaddPsfFitTask",
    "MultiProFitCoaddFitConfig",
    "MultiProFitCoaddPointFitConfig",
    "MultiProFitCoaddSersicFitConfig",
    "MultiProFitCoaddSersicFitTask",
    "MultiProFitCoaddGaussFitConfig",
    "MultiProFitCoaddGaussFitTask",
    "MultiProFitCoaddExpFitConfig",
    "MultiProFitCoaddExpFitTask",
    "MultiProFitCoaddDeVFitConfig",
    "MultiProFitCoaddDeVFitTask",
    "MultiProFitCoaddExpDeVFitConfig",
    "MultiProFitCoaddExpDeVFitTask",
)

from abc import abstractmethod
from types import SimpleNamespace

from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.pex.config import Field
from lsst.pipe.tasks.fit_coadd_multiband import (
    CoaddMultibandFitConfig,
    CoaddMultibandFitConnections,
    CoaddMultibandFitTask,
)
from lsst.pipe.tasks.fit_coadd_psf import CoaddPsfFitConfig, CoaddPsfFitConnections, CoaddPsfFitTask

from .fit_coadd_multiband import MultiProFitSourceTask, SourceTablePsfComponentsAction
from .fit_coadd_psf import MultiProFitPsfTask

component_names_default = SimpleNamespace(
    point="point",
    gauss="gauss",
    exp="exp",
    deV="deV",
    sersic="sersic",
)

model_names_default = SimpleNamespace(
    point="Point",
    gauss="Gauss",
    exp="Exp",
    deV="DeV",
    sersic="Sersic",
    fixed_cen="FixedCen",
    shapelet_psf="ShapeletPsf",
)


class MultiProFitCoaddPsfFitConfig(
    CoaddPsfFitConfig,
    pipelineConnections=CoaddPsfFitConnections,
):
    """MultiProFit PSF fit task config."""

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_psf.retarget(MultiProFitPsfTask)
        self.fit_coadd_psf.config_fit.eval_residual = False


class MultiProFitCoaddPsfFitTask(CoaddPsfFitTask):
    """MultiProFit PSF fit task."""

    ConfigClass = MultiProFitCoaddPsfFitConfig
    _DefaultName = "multiProFitCoaddPsfFit"


class MultiProFitCoaddFitConfig(
    CoaddMultibandFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """Generic MultiProFit source fit task config."""

    # This needs to be set, ideally in setDefaults of subclasses
    name_model = Field[str](doc="The name of the model", default=None)

    def _get_source(self):
        return next(iter(self.fit_coadd_multiband.config_model.sources))

    def _get_component_group(self, source: SourceConfig | None = None):
        if source is None:
            source = self._get_source()
        return source.next(iter(source.component_groups))

    def add_point_source(self, name: str | None = None):
        if name is None:
            name = component_names_default.pointsource
        source = self._get_source()
        group = self._get_component_group(source=source)
        if name in group.components_gauss:
            raise RuntimeError(f"{name=} component already exists in {source=}")
        group.components_gauss[name] = self.make_point_source_component()
        self.connections.name_table += name.pointsource

    def finalize(
        self,
        add_point_source: bool = False,
        fix_centroid: bool = False,
        use_shapelet_psf: bool = False,
    ):
        if add_point_source:
            self.add_point_source()
        if fix_centroid:
            self.fix_centroid()
        if use_shapelet_psf:
            self.use_shapelet_psf()

    def fix_centroid(self):
        group = self._get_component_group()
        centroids = group.centroids["default"]
        centroids.x.fixed = True
        centroids.y.fixed = True
        self.connections.name_table += model_names_default.fixed_cen

    @classmethod
    @abstractmethod
    def get_model_name_default(cls) -> str:
        """Return the default name for this model in table columns."""
        raise NotImplementedError("Subclasses must implement get_model_name_default")

    @classmethod
    def get_model_name_full(cls) -> str:
        """Return a longer, more descriptive name for the model."""
        return cls.get_model_name_default()

    @abstractmethod
    def make_default_model_config(self) -> ModelConfig:
        """Make a default configuration object for this model."""
        raise NotImplementedError("Subclasses must implement make_default_model_config")

    @staticmethod
    def make_point_source_component():
        return GaussianComponentConfig(
            size_x=ParameterConfig(value_initial=0.0, fixed=True),
            size_y=ParameterConfig(value_initial=0.0, fixed=True),
            rho=ParameterConfig(value_initial=0.0, fixed=True),
        )

    @staticmethod
    def make_sersic_component(**kwargs):
        return SersicComponentConfig(
            prior_axrat_stddev=0.8,
            prior_size_stddev=0.3,
            sersic_index=SersicIndexParameterConfig(**kwargs),
        )

    @staticmethod
    def make_single_model_config(group: ComponentGroupConfig):
        return ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": group,
                    }
                )
            }
        )

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_multiband.retarget(MultiProFitSourceTask)
        self.fit_coadd_multiband.action_psf = SourceTablePsfComponentsAction()
        self.fit_coadd_multiband.bands_fit = ("u", "g", "r", "i", "z", "y")

        self.fit_coadd_multiband.config_model = self.make_default_model_config()
        self.name_model = self.get_model_name_default()
        self.connections.name_table = self.name_model

    def use_shapelet_psf(self):
        self.fit_coadd_multiband.action_psf = SourceTablePsfComponentsAction()
        self.drop_psf_connection = True
        self.connections.name_table += model_names_default.shapelet_psf


class MultiProFitCoaddPointFitConfig(
    MultiProFitCoaddFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Sersic model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.point

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Point Source"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig()
        self.add_point_source()
        return self.make_single_model_config(group=config_group)


class MultiProFitCoaddSersicFitConfig(
    MultiProFitCoaddFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Sersic model fit task config."""

    def _rename_defaults(
        self,
        name_new: str,
        name_model: str | None = None,
        name_old: str | None = None,
        index_new: float | None = None,
        fix_index: bool = False,
    ):
        """Rename the default Sersic component to something more specific.

        This is intended for fixed index models such as exponential and
        deVaucouleurs.

        Parameters
        ----------
        name_new
            The new name for the component.
        name_model
            The new name of the model. Default is to capitalize name_new.
        name_old
            The old name of the component. Default is to set to
            component_names_default.sersic.
        index_new
            The initial value for the Sersic index.
        fix_index
            Whether the fix the index to the new value.
        """
        if name_old is None:
            name_old = component_names_default.sersic
        if name_model is None:
            name_model = name_new.capitalize()
        group = self._get_component_group()
        comps_sersic = group.components_sersic

        if name_new in comps_sersic:
            raise RuntimeError(f"{name_new=} is already in {comps_sersic=}")

        comp_sersic = comps_sersic[name_old]
        del comps_sersic[name_old]
        if index_new is not None:
            comp_sersic.sersic_index.value_initial = index_new
        if fix_index:
            comp_sersic.sersic_index.fixed = True
        comp_sersic[name_new] = comp_sersic

        self.name_model = name_model
        self.connections.name_table = name_model

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.sersic

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Sersic"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig(
            components_sersic={
                component_names_default.sersic: self.make_sersic_component(),
            },
        )
        return self.make_single_model_config(group=config_group)


class MultiProFitCoaddSersicFitTask(CoaddMultibandFitTask):
    """MultiProFit single Sersic model fit task."""

    ConfigClass = MultiProFitCoaddSersicFitConfig
    _DefaultName = "multiProFitCoaddSersicFit"


class MultiProFitCoaddGaussFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Gaussian model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.gauss

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Gaussian"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.gauss,
            name_model=model_names_default.gauss,
            index_new=0.5,
            fix_index=True,
        )


class MultiProFitCoaddGaussFitTask(CoaddMultibandFitTask):
    """MultiProFit single Gaussian model fit task."""

    ConfigClass = MultiProFitCoaddGaussFitConfig
    _DefaultName = "multiProFitCoaddGaussFit"


class MultiProFitCoaddExpFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single exponential model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.exp

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Exponential"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.exp,
            name_model=model_names_default.exp,
            index_new=1.0,
            fix_index=True,
        )


class MultiProFitCoaddExpFitTask(CoaddMultibandFitTask):
    """MultiProFit single exponential model fit task."""

    ConfigClass = MultiProFitCoaddExpFitConfig
    _DefaultName = "multiProFitCoaddExpFit"


class MultiProFitCoaddDeVFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single DeVaucouleurs model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.deV

    @classmethod
    def get_model_name_full(cls) -> str:
        return "de Vaucouleurs"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.deV,
            name_model=model_names_default.deV,
            index_new=4.0,
            fix_index=True,
        )


class MultiProFitCoaddDeVFitTask(CoaddMultibandFitTask):
    """MultiProFit single DeVaucouleurs model fit task."""

    ConfigClass = MultiProFitCoaddDeVFitConfig
    _DefaultName = "multiProFitCoaddDeVFit"


class MultiProFitCoaddExpDeVFitConfig(
    MultiProFitCoaddFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Exponential+DeVaucouleurs model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return f"{model_names_default.exp}{model_names_default.deV}"

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Exponential + de Vaucouleurs"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig(
            components_sersic={
                component_names_default.exp: self.make_sersic_component(value_initial=1.0, fixed=True),
                component_names_default.deV: self.make_sersic_component(value_initial=4.0, fixed=True),
            },
        )
        return self.make_single_model_config(group=config_group)

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_multiband.config_model = self.make_default_model_config()
        self.name_model = self.get_model_name_default()
        self.connections.name_table = self.name_model


class MultiProFitCoaddExpDeVFitTask(CoaddMultibandFitTask):
    """MultiProFit single ExpDeV model fit task."""

    ConfigClass = MultiProFitCoaddExpDeVFitConfig
    _DefaultName = "multiProFitCoaddExpDeVFit"
