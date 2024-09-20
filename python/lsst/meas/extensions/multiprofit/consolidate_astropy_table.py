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

from collections import defaultdict

import astropy.table as apTab
import lsst.pex.config as pexConfig
from lsst.pex.config.configurableActions import ConfigurableAction, ConfigurableActionField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
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


class InputConfig(pexConfig.Config):
    """Config for inputs to ConsolidateAstropyTableTask."""

    doc = pexConfig.Field[str](doc="Doc for connection", optional=False)
    action = ConfigurableActionField[CatalogAction](
        doc="Action to modify the input table",
        default=None,
    )
    columns = pexConfig.ListField[str](
        doc="Column names to copy; default of None copies all", optional=True, default=None
    )
    column_id = pexConfig.Field[str](doc="ID column to merge", optional=False, default="objectId")
    is_multiband = pexConfig.Field[bool](doc="Whether the dataset is multiband or not", default=False)
    is_multipatch = pexConfig.Field[bool](doc="Whether the dataset is multipatch or not", default=False)
    join_column = pexConfig.Field[str](
        doc="Column to join on if unequal length instead of stacking", default=None, optional=True
    )
    storageClass = pexConfig.Field[str](doc="Storage class for DatasetType", default="ArrowAstropy")

    def get_connection(self, name: str) -> connectionTypes.Input:
        dimensions = ["skymap", "tract"]
        if not self.is_multipatch:
            dimensions.append("patch")
        if not self.is_multiband:
            dimensions.append("band")
        connection = connectionTypes.Input(
            doc=self.doc,
            name=name,
            storageClass=self.storageClass,
            dimensions=dimensions,
            multiple=not (self.is_multiband and self.is_multipatch),
            deferLoad=self.columns is not None,
        )
        return connection


class ConsolidateAstropyTableConfigBase(pexConfig.Config):
    """Config for ConsolidateAstropyTableTask."""

    inputs = pexConfig.ConfigDictField(
        doc="Mapping of input dataset type config by name",
        keytype=str,
        itemtype=InputConfig,
        default={},
    )


class ConsolidateAstropyTableConnections(
    # Ignore the undocumented inherited config arg in __init__
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),  # numpydoc ignore=PR01
):
    """Connections for ConsolidateAstropyTableTask."""

    cat_output = connectionTypes.Output(
        doc="Per-tract horizontal concatenation of the input AstropyTables",
        name="objectAstropyTable_tract",
        storageClass="ArrowTable",
        dimensions=("tract", "skymap"),
    )

    def __init__(self, *, config: ConsolidateAstropyTableConfigBase):
        for name, config_input in config.inputs.items():
            if hasattr(self, name):
                raise ValueError(
                    f"{config_input=} {name=} is invalid, due to being an existing attribute" f" of {self=}"
                )
            connection = config_input.get_connection(name)
            setattr(self, name, connection)


class ConsolidateAstropyTableConfig(
    pipeBase.PipelineTaskConfig,
    ConsolidateAstropyTableConfigBase,
    pipelineConnections=ConsolidateAstropyTableConnections,
):
    """PipelineTaskConfig for ConsolidateAstropyTableTask."""

    join_type = pexConfig.ChoiceField[str](
        doc="Type of join to perform in the final hstack",
        allowed={
            "inner": "Inner join",
            "outer": "Outer join",
        },
        default="inner",
        optional=False,
    )


class ConsolidateAstropyTableTask(pipeBase.PipelineTask):
    """Write patch-merged astropy tables to a tract-level astropy table."""

    _DefaultName = "consolidateAstropyTable"
    ConfigClass = ConsolidateAstropyTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        bands_ref, patches_ref = None, None
        band_null, patch_null = "", -1
        bands_null, patches_null = {band_null}, {patch_null: None}
        data = dict()
        bands_sorted = None

        for name, inputRef_list in inputRefs:
            inputConfig = self.config.inputs[name]
            bands, patches = set(), dict()
            data_name = defaultdict(dict)
            inputs_name = inputs[name]
            if not hasattr(inputRef_list, "__len__"):
                inputRef_list = tuple((inputRef_list,))
                inputs_name = tuple((inputs_name,))

            for dataRef, data_in in zip(inputRef_list, inputs_name):
                dataId = dataRef.dataId
                band = dataId.band.name if not inputConfig.is_multiband else band_null

                if inputConfig.columns is not None:
                    columns = inputConfig.columns
                    data_in = data_in.get(parameters={"columns": columns})
                else:
                    columns = tuple(data_in.columns)

                if inputConfig.storageClass == "DataFrame":
                    data_in = apTab.Table.from_pandas(data_in.reset_index(drop=False))
                elif inputConfig.storageClass == "ArrowAstropy":
                    data_in.meta = {name: data_in.meta}

                if not inputConfig.is_multiband:
                    columns_new = [
                        column if column == inputConfig.column_id else f"{band}_{column}"
                        for column in columns
                    ]
                    data_in.rename_columns(columns, columns_new)
                if inputConfig.action is not None:
                    data_in = inputConfig.action(data_in, datasetType=name)

                if inputConfig.is_multipatch:
                    patch = patch_null
                    patches[patch] = None
                else:
                    patch = dataId.patch.id
                    patches[patch] = min(data_in[inputConfig.column_id])
                data_name[patch][band] = data_in
                bands.add(band)

            if inputConfig.is_multiband:
                if bands != bands_null:
                    raise RuntimeError(f"multiband {inputConfig=} has non-trivial {bands=}")
            else:
                if bands_ref is None:
                    bands_ref = bands
                    bands_sorted = tuple(band for band in sorted(bands_ref))
                else:
                    if bands != bands_ref:
                        raise RuntimeError(f"{inputConfig=} {bands=} != {bands_ref=}")
            if inputConfig.is_multipatch:
                if patches != patches_null:
                    raise RuntimeError(f"{inputConfig=} {patches=} != {patches_null=}")
            else:
                column_id = inputConfig.column_id
                if patches_ref is None:
                    bands = tuple(bands) if inputConfig.is_multiband else bands_sorted
                    for patch in patches:
                        data_patch = data_name[patch]
                        tab = data_patch[bands[0]]
                        tab.add_column(np.full(len(tab), patch), name="patch", index=1)
                        tab.rename_column(column_id, "objectId")
                        for band in bands[1:]:
                            tab = data_patch[band]
                            del tab[column_id]
                    patches_objid = {objid: patch for patch, objid in patches.items()}
                    patches_ref = {patch: objid for objid, patch in sorted(patches_objid.items())}
                elif {patch: patches[patch] for patch in patches_ref.keys()} != patches_ref:
                    raise RuntimeError(f"{inputConfig=} {patches=} != {patches_ref=}")
                else:
                    for data_patch in data_name.values():
                        for tab in data_patch.values():
                            del tab[column_id]

            data[name] = data_name

        self.log.info("Concatenating %s per-patch astropy Tables", len(patches))

        for name, data_name in data.items():
            config_input = self.config.inputs[name]
            tables = [
                (
                    apTab.hstack([data_name[patch][band] for band in bands_sorted], join_type="exact")
                    if not config_input.is_multiband
                    else data_name[patch][band_null]
                )
                for patch in (patches_ref if not config_input.is_multipatch else patches_null)
            ]
            data[name] = tables[0] if (len(tables) == 1) else apTab.vstack(tables, join_type="exact")
        # This will break if all tables have config.join_column
        # ... but that seems unlikely.
        table = apTab.hstack(
            [data[name] for name, config in self.config.inputs.items() if config.join_column is None],
            join_type=self.config.join_type,
        )
        for name, config in self.config.inputs.items():
            if config.join_column:
                table = apTab.join(
                    table,
                    data[name],
                    join_type=self.config.join_type,
                    keys=config.join_column,
                )

        butlerQC.put(pipeBase.Struct(cat_output=table), outputRefs)
