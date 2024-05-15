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

from typing import Any, Iterable

from lsst.analysis.tools.atools.genericBuild import FluxConfig, MomentsConfig, SizeConfig
from lsst.analysis.tools.atools.sizeMagnitude import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext

moments_sersic = MomentsConfig(xx="reff_x", yy="reff_y", xy="rho")


def make_size_magnitude_tools(
    name_model: str,
    label_model: str,
    components: Iterable[tuple[str, str]],
    kwargs_plot: dict[str, Any],
) -> list[SizeMagnitudePlot]:
    """Make a size-magnitude plot analysis_tool for a given model magnitude.

    Parameters
    ----------
    name_model
        The name of the model in keys (column names).
    label_model
        A descriptive label for the model.
    components
        A list of name-label pairs for model components.
    kwargs_plot
        Keyword arguments to set as attributes for the plot action.

    Returns
    -------
    plots
        A list of plot tools for each of the input components.
    """
    tools = []
    for name_comp, label_comp in components:
        name_full = f"{name_model}_{name_comp}"
        flux_config = FluxConfig(
            key_flux=f"mpf_{name_full}_{{band}}_flux",
            key_flux_error=f"mpf_{name_full}_{{band}}_flux_err",
            name_flux=label_model,
        )
        flags_false = (
            f"mpf_{name_model}_{flag}_flag" for flag in ("unknown", "is_parent", "not_primary", "psf_fit")
        )
        size_config = SizeConfig(
            key_size=f"mpf_{name_full}_{{suffix}}",
            name_size=f"{label_comp} {'$R_{eff}$'}",
        )
        atool = SizeMagnitudePlot(
            fluxes={name_full: flux_config},
            mag_x=name_full,
            sizes={name_full: size_config},
            size_y=name_full,
            config_moments=moments_sersic,
            size_type="determinantRadius",
            is_covariance=False,
        )
        atool.applyContext(CoaddContext)
        atool.prep.selectors.flagSelector.selectWhenFalse = flags_false
        atool.prep.selectors.flagSelector.selectWhenTrue = []
        for name_attr, value in kwargs_plot.items():
            setattr(atool.produce.plot, name_attr, value)
        tools.append(atool)
    return tools
