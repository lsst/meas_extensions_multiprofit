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
    "CatalogAction",
    "MergeMultibandFluxes",
)

from collections import defaultdict

import lsst.pex.config as pexConfig
from lsst.pex.config.configurableActions import ConfigurableAction
import numpy as np


class CatalogAction(ConfigurableAction):
    """Configurable action to return a catalog."""

    def __call__(self, data, **kwargs):
        return data


class MergeMultibandFluxes(CatalogAction):
    """Configurable action to merge single-band flux tables into one."""

    name_model = pexConfig.Field[str](doc="The name of the model that fluxes are measured from", default="")

    def __call__(self, data, **kwargs):
        datasetType = kwargs.get("datasetType")
        prefix_model = self.name_model + ("_" if self.name_model else "")
        if (
            self.name_model
            and hasattr(data, "meta")
            and datasetType
            and (config := data.meta.get(datasetType))
        ):
            prefix = config.get("config", {}).get("prefix_column", "")
        else:
            prefix = ""
        columns_rest = []
        columns_flux_band = defaultdict(list)
        for column in data.columns:
            if not prefix or column.startswith(prefix):
                if column.endswith("_flux"):
                    band = column.split("_")[-2]
                    columns_flux_band[band].append(column)
            else:
                columns_rest.append(column)

        for band, columns_band in columns_flux_band.items():
            column_flux = f'{columns_band[0].partition("_")[0]}_{band}_flux'
            flux = np.nansum([data[column] for column in columns_band], axis=0)
            data[column_flux] = flux

            columns_band_err = [f"{column}_err" for column in columns_band]
            errors = [data[column] ** 2 for column in columns_band_err if column in data.columns]
            if errors:
                flux_err = np.sqrt(np.nansum(errors, axis=0))
                flux_err[flux_err == 0] = np.nan
                column_flux_err = f"{column_flux}_err"
                data[column_flux_err] = flux_err

        if prefix_model:
            colnames = [
                col if (col in columns_rest) else f"{prefix}{prefix_model}{col.split(prefix, 1)[1]}"
                for col in data.columns
            ]
            if hasattr(data, "rename_columns"):
                data.rename_columns([x for x in data.columns], colnames)
            else:
                data.columns = colnames

        return data
