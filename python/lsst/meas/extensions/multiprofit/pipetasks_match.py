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
    "MultiProFitMatchTractCatalogConfig",
    "MultiProFitMatchDC2CatalogConfig",
    "MultiProFitMatchDC2CatalogTask",
)

from lsst.pipe.tasks.match_tract_catalog import (
    MatchTractCatalogConfig,
    MatchTractCatalogConnections,
    MatchTractCatalogTask,
)
from lsst.pipe.tasks.match_tract_catalog_probabilistic import MatchTractCatalogProbabilisticTask


class MultiProFitMatchTractCatalogConfig(
    MatchTractCatalogConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """Generic MultiProFit reference source match task config."""

    def setDefaults(self):
        self.connections.name_input_cat_target = "objectTable_tract_multiprofit"
        self.match_tract_catalog.retarget(MatchTractCatalogProbabilisticTask)

        self.match_tract_catalog.columns_ref_copy = ["id", "truth_type"]
        self.match_tract_catalog.columns_target_copy = ["objectId"]
        # Override detect_isPrimary default because MultiProFit doesn't fit
        # non-primary rows anyway
        self.match_tract_catalog.columns_target_select_true = []


class MultiProFitMatchDC2CatalogConfig(
    MultiProFitMatchTractCatalogConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """PipelineTaskConfig for MultiProFitMatchDC2CatalogTask."""

    def finalize(
        self,
        model_prefix: str,
        bands_match: list[str] | None = None,
    ):
        """Finalize configuration for a given model.

        Parameters
        ----------
        model_prefix
            The model column prefix, e.g. mpf_ser.
        bands_match
            List of bands to match fluxes on.
        """
        if bands_match is None:
            bands_match = ["u", "g", "r", "i", "z", "y"]
        fluxes_ref = [f"flux_{band}" for band in bands_match]
        print(fluxes_ref)
        self.match_tract_catalog.columns_ref_flux = fluxes_ref
        self.match_tract_catalog.columns_ref_meas = ["ra", "dec"] + fluxes_ref
        fluxes_meas = [f"{model_prefix}_{band}_flux" for band in bands_match]
        columns_meas = [f"{model_prefix}_cen_ra", f"{model_prefix}_cen_dec"] + fluxes_meas
        self.match_tract_catalog.columns_target_meas = columns_meas
        self.match_tract_catalog.columns_target_err = [f"{col}_err" for col in columns_meas]
        self.match_tract_catalog.coord_format.column_target_coord1 = f"{model_prefix}_cen_ra"
        self.match_tract_catalog.coord_format.column_target_coord2 = f"{model_prefix}_cen_dec"
        self.match_tract_catalog.columns_target_select_false = [
            f"{model_prefix}_not_primary_flag",
        ]

    def setDefaults(self):
        super().setDefaults()
        self.match_tract_catalog.mag_faintest_ref = 27.0
        self.match_tract_catalog.columns_ref_select_true = ["is_unique_truth_entry"]


class MultiProFitMatchDC2CatalogTask(MatchTractCatalogTask):
    """Match DC2 truth_summary to a single model from an
    objectTable_tract_multiprofit.
    """

    _DefaultName = "multiProFitMatchDC2Catalog"
    ConfigClass = MultiProFitMatchDC2CatalogConfig


class MultiProFitMatchDC2CatalogSersicConfig(
    MultiProFitMatchDC2CatalogConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """PipelineTaskConfig for MultiProFitMatchDC2SersicCatalogTask."""

    def setDefaults(self):
        super().setDefaults()
        self.finalize(model_prefix="mpf_ser")


class MultiProFitMatchDC2CatalogSersicTask(MultiProFitMatchDC2CatalogTask):
    """Match DC2 truth_summary to the single Sersic model from an
    objectTable_tract_multiprofit.
    """

    _DefaultName = "multiProFitMatchDC2Catalog"
    ConfigClass = MultiProFitMatchDC2CatalogConfig
