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

__all__ = [
    "MultiProFitConsolidateTablesConfig",
    "MultiProFitConsolidateTablesTask",
]

from .consolidate_astropy_table import (
    ConsolidateAstropyTableConfig,
    ConsolidateAstropyTableConnections,
    ConsolidateAstropyTableTask,
    InputConfig,
    MergeMultibandFluxes,
)


class MultiProFitConsolidateTablesConfig(
    ConsolidateAstropyTableConfig,
    pipelineConnections=ConsolidateAstropyTableConnections,
):
    """PipelineTaskConfig for MultiProFitConsolidateTablesTask."""

    names_long = {
        "ser": "Sersic",
        "gauss": "Gaussian",
        "exp": "Exponential",
        "dev": "deVauc.",
        "expdev": "Exponential + deVauc.",
    }

    def add_model(
        self,
        name: str,
        description: str | None = None,
        has_pointsource: bool = False,
        is_centroid_fixed: bool = False,
        is_psf_shapelet: bool = False,
    ):
        description = self.names_long.get(name, description or "Unspecified")
        if has_pointsource:
            description = f"Point Source + {description}"
        if is_centroid_fixed:
            description = f"{description} fixed centroid"
        if is_psf_shapelet:
            description = f"{description} shapelet PSF"
        self.inputs[f"deepCoadd_{name}_multiprofit"] = InputConfig(
            doc=f"{description} object fit parameters",
            action=MergeMultibandFluxes(name_model=name),
            column_id="id",
            is_multiband=True,
        )

    def setDefaults(self):
        super().setDefaults()
        inputs = {
            "deepCoadd_psfs_multiprofit": InputConfig(
                doc="PSF fit parameters",
                column_id="id",
            ),
        }
        self.inputs = inputs
        self.connections.cat_output = "objectTable_tract_multiprofit"


class MultiProFitConsolidateTablesTask(ConsolidateAstropyTableTask):
    """Consolidate MultiProFit PSF and object fit tables."""

    _DefaultName = "multiProFitConsolidateTables"
    ConfigClass = MultiProFitConsolidateTablesConfig


class MultiProFitConsolidateTablesSersicConfig(
    MultiProFitConsolidateTablesConfig,
    pipelineConnections=ConsolidateAstropyTableConnections,
):
    def setDefaults(self):
        super().setDefaults()
        self.add_model("ser")


class MultiProFitConsolidateTablesSersicTask(MultiProFitConsolidateTablesTask):
    """Consolidate MultiProFit PSF and object fit tables."""

    _DefaultName = "multiProFitConsolidateTablesSersic"
    ConfigClass = MultiProFitConsolidateTablesSersicConfig
