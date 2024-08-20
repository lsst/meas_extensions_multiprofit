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
    "MultiProFitCoaddPsfFitConfig",
    "MultiProFitCoaddPsfFitTask",
    "MultiProFitCoaddFitConfig",
    "MultiProFitCoaddSersicFitConfig",
    "MultiProFitCoaddSersicFitTask",
    "MultiProFitCoaddGaussianFitConfig",
    "MultiProFitCoaddGaussianFitTask",
    "MultiProFitCoaddExponentialFitConfig",
    "MultiProFitCoaddExponentialFitTask",
    "MultiProFitCoaddDeVaucFitConfig",
    "MultiProFitCoaddDeVaucFitTask",
)


from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.pipe.tasks.fit_coadd_multiband import (
    CoaddMultibandFitConfig,
    CoaddMultibandFitConnections,
    CoaddMultibandFitTask,
)
from lsst.pipe.tasks.fit_coadd_psf import CoaddPsfFitConfig, CoaddPsfFitConnections, CoaddPsfFitTask

from .fit_coadd_multiband import MultiProFitSourceTask, SourceTablePsfComponentsAction
from .fit_coadd_psf import MultiProFitPsfTask


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

    def add_pointsource(self):
        group = self.fit_coadd_multiband.config_model.sources[""].component_groups[""]
        group.components_gauss["ps"] = self.get_pointsource_component()
        self.connections.name_table += "_ps"

    def finalize(
        self,
        add_pointsource: bool = False,
        fix_centroid: bool = False,
        use_shapelet_psf: bool = False,
    ):
        if add_pointsource:
            self.add_pointsource()
        if fix_centroid:
            self.fix_centroid()
        if use_shapelet_psf:
            self.use_shapelet_psf()

    def fix_centroid(self):
        group = self.fit_coadd_multiband.config_model.sources[""].component_groups[""]
        centroids = group.centroids["default"]
        centroids.x.fixed = True
        centroids.y.fixed = True
        self.connections.name_table += "_fixedcen"

    @staticmethod
    def get_pointsource_component():
        return GaussianComponentConfig(
            size_x=ParameterConfig(value_initial=0.0, fixed=True),
            size_y=ParameterConfig(value_initial=0.0, fixed=True),
            rho=ParameterConfig(value_initial=0.0, fixed=True),
        )

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_multiband.retarget(MultiProFitSourceTask)
        self.fit_coadd_multiband.action_psf = SourceTablePsfComponentsAction()
        self.fit_coadd_multiband.bands_fit = ("u", "g", "r", "i", "z", "y")

    def use_shapelet_psf(self):
        self.fit_coadd_multiband.action_psf = SourceTablePsfComponentsAction()
        self.drop_psf_connection = True
        self.connections.name_table += "_shapelet"


class MultiProFitCoaddSersicFitConfig(
    MultiProFitCoaddFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Sersic model fit task config."""

    def _rename_defaults(self, name_new: str, index_new: float | None = None, fix_index: bool = False):
        comps = self.fit_coadd_multiband.config_model.sources[""].component_groups[""].components_sersic
        comp_sersic = comps["ser"]
        del comps["ser"]
        if index_new is not None:
            comp_sersic.sersic_index.value_initial = index_new
        if fix_index:
            comp_sersic.sersic_index.fixed = True
        comps[name_new] = comp_sersic
        self.connections.name_table = name_new

    def _set_model_config(self, add_ps: bool = False):
        self.fit_coadd_multiband.config_model = ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_sersic={
                                "ser": SersicComponentConfig(
                                    prior_axrat_stddev=0.8,
                                    prior_size_stddev=0.3,
                                ),
                            },
                            components_gauss={"ps": self.get_pointsource_component()} if add_ps else {},
                        )
                    }
                )
            }
        )

    def setDefaults(self):
        super().setDefaults()
        self.connections.name_table = "ser"
        self._set_model_config()


class MultiProFitCoaddSersicFitTask(CoaddMultibandFitTask):
    """MultiProFit single Sersic model fit task."""

    ConfigClass = MultiProFitCoaddSersicFitConfig
    _DefaultName = "multiProFitCoaddSersicFit"


class MultiProFitCoaddGaussianFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Gaussian model fit task config."""

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(name_new="gauss", index_new=0.5, fix_index=True)


class MultiProFitCoaddGaussianFitTask(CoaddMultibandFitTask):
    """MultiProFit single Gaussian model fit task."""

    ConfigClass = MultiProFitCoaddGaussianFitConfig
    _DefaultName = "multiProFitCoaddGaussianFit"


class MultiProFitCoaddExponentialFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single exponential model fit task config."""

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(name_new="exp", index_new=1.0, fix_index=True)


class MultiProFitCoaddExponentialFitTask(CoaddMultibandFitTask):
    """MultiProFit single exponential model fit task."""

    ConfigClass = MultiProFitCoaddExponentialFitConfig
    _DefaultName = "multiProFitCoaddExponentialFit"


class MultiProFitCoaddDeVaucFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single DeVaucouleurs model fit task config."""

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(name_new="dev", index_new=4.0, fix_index=True)


class MultiProFitCoaddDeVaucFitTask(CoaddMultibandFitTask):
    """MultiProFit single DeVaucouleurs model fit task."""

    ConfigClass = MultiProFitCoaddDeVaucFitConfig
    _DefaultName = "multiProFitCoaddDeVaucFit"


class MultiProFitCoaddExpDevFitConfig(
    MultiProFitCoaddFitConfig,
    pipelineConnections=CoaddMultibandFitConnections,
):
    """MultiProFit single Exp-Dev model fit task config."""

    def _set_model_config(self, add_ps: bool = False):
        self.fit_coadd_multiband.config_model = ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_sersic={
                                "exp": SersicComponentConfig(
                                    prior_axrat_stddev=0.8,
                                    prior_size_stddev=0.3,
                                    sersic_index=SersicIndexParameterConfig(value_initial=1.0, fixed=True),
                                ),
                                "dev": SersicComponentConfig(
                                    prior_axrat_stddev=0.8,
                                    prior_size_stddev=0.3,
                                    sersic_index=SersicIndexParameterConfig(value_initial=4.0, fixed=True),
                                ),
                            },
                            components_gauss={"ps": self.get_pointsource_component()} if add_ps else {},
                        )
                    }
                )
            }
        )

    def setDefaults(self):
        super().setDefaults()
        self._set_model_config()
        self.connections.name_table = "expdev"


class MultiProFitCoaddExpDevFitTask(CoaddMultibandFitTask):
    """MultiProFit single Exp-Dev model fit task."""

    ConfigClass = MultiProFitCoaddExpDevFitConfig
    _DefaultName = "multiProFitCoaddExpDevFit"
