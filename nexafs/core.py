"""NEXAFS Data Processing and Normalization Module.

This module provides a pandas DataFrame accessor for analyzing, normalizing and
visualizing NEXAFS (Near-Edge X-ray Absorption Fine Structure) spectroscopy data. The
accessor enables users to perform common NEXAFS data processing tasks and create
publication-quality visualizations directly from pandas DataFrames containing
spectroscopy measurements.
"""

from functools import lru_cache as cache
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from periodictable import xsf
from scipy.constants import physical_constants

# Physical constants
h = physical_constants["Planck constant in eV/Hz"][0]  # eV*s
c = physical_constants["speed of light in vacuum"][0]  # m/s
# hc in cm
hc = h * c * 1e7  # cm*eV


@pd.api.extensions.register_dataframe_accessor("nexafs")
class NEXAFSAccessor:
    """
    Custom pandas DataFrame accessor for NEXAFS spectroscopy data analysis.

    This accessor provides methods for processing, normalizing, and visualizing
    Near-Edge X-ray Absorption Fine Structure (NEXAFS) spectroscopy data stored
    in pandas DataFrames. It enables easy calculation of atomic absorption coefficients,
    normalization of experimental data, and creation of publication-quality plots.

    The accessor is designed to work with DataFrames containing multiple spectra,
    supporting groupby operations for comparative analysis across different samples,
    elements, or experimental conditions.

    Examples
    --------
    >>> import pandas as pd
    >>> import nexafs
    >>>
    >>> # Load data into DataFrame
    >>> df = pd.read_csv("nexafs_data.csv")
    >>>
    >>> # Calculate bare atom absorption for carbon
    >>> df["beta_atomic"] = df.nexafs.beta_bare_atom("C", "Energy", density=2.2)
    >>>
    >>> # Normalize experimental data
    >>> df["normalized"] = df.nexafs.normalize(
    ...     chemical_formula="C",
    ...     nexafs_column="Raw_Intensity",
    ...     energy_column="Energy",
    ...     density=2.2,
    ...     nexafs_type="electron-yield",
    ...     pre_edge_range=(280, 282),
    ...     post_edge_range=(320, 325),
    ... )
    >>>
    >>> # Create a normalized plot
    >>> df.nexafs.plot_spectrum(
    ...     x="Energy",
    ...     y="normalized",
    ...     reference="beta_atomic",
    ...     title="Carbon K-edge NEXAFS",
    ... )
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """
        Initialize the NEXAFSAccessor with a pandas DataFrame.

        Parameters
        ----------
        pandas_obj : pd.DataFrame
            The pandas DataFrame containing NEXAFS data.
        """
        self._obj = pandas_obj
        self._chemical_formula: str = ""
        self._density: float = 1.0

    @property
    def chemical_formula(self) -> str | None:
        """Get the chemical formula for the NEXAFS data."""
        return self._chemical_formula

    @chemical_formula.setter
    def chemical_formula(self, value: str) -> None:
        """Set the chemical formula for the NEXAFS data."""
        if not isinstance(value, str):
            msg = (
                "Chemical formula must be a string. "
                "Please provide a valid chemical formula (e.g., 'C', 'H2O')."
            )
            raise TypeError(msg)
        self._chemical_formula = value

    @property
    def density(self) -> float | None:
        """Get the density for the NEXAFS data."""
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        """Set the density for the NEXAFS data."""
        if not isinstance(value, (int | float)):
            msg = (
                "Density must be a numeric value. "
                "Please provide a valid density (e.g., 1.0 for g/cm³)."
            )
            raise TypeError(msg)
        self._density = value

    @cache  # noqa: B019
    def bare_atom(
        self,
        energy: str,
        chemical_formula: str | None = None,
        density: float | None = None,
    ) -> pd.Series:
        """
        Calculate the complex index of refraction for a given chemical formula.

        This method computes the complex index of refraction (n = 1-delta+i*beta)
        for a specified material using the periodictable library, which implements
        atomic scattering factors based on X-ray absorption data.

        Parameters
        ----------
        energy : str
            Column name containing energy values in eV.
        chemical_formula : str | None, optional
            Chemical formula of the compound (e.g., 'C', 'H2O', 'C8H8O2').
            If None, uses the accessor's stored chemical_formula property.
        density : float | None, optional
            Material density in g/cm³. If None, uses the accessor's density property.

        Returns
        -------
        pd.Series
            Complex Series containing the index of refraction values.

        Raises
        ------
        ValueError
            If the energy column is not found in the DataFrame or is not numeric.
            If no chemical formula is provided or stored in the accessor.
        """
        if energy in self._obj.columns and pd.api.types.is_numeric_dtype(
            self._obj[energy]
        ):
            energy_ev = self._obj[energy].astype(float)
        else:
            msg = (
                f"Column '{energy}' not found in DataFrame or is not numeric. "
                "Please provide a valid column name with numeric energy values."
            )
            raise ValueError(msg)

        # Use provided values or fall back to class properties
        formula = chemical_formula or self._chemical_formula
        if not formula:
            msg = (
                "No chemical formula provided. "
                "Please provide a chemical formula or set it using the "
                "chemical_formula property."
            )
            raise ValueError(msg)

        mat_density = density if density is not None else self._density

        # periodictable expects energy in keV
        energy_kev = energy_ev * 1e-3

        # Store chemical formula and density in class properties
        if chemical_formula:
            self.chemical_formula = chemical_formula
        if density is not None:
            self.density = density

        # Calculate complex index of refraction
        ba = xsf.index_of_refraction(formula, density=mat_density, energy=energy_kev)

        # Return as pandas Series with same index as energy column
        # Note: beta is negative of imaginary part in the convention used by
        # periodictable
        return pd.Series([complex(b.real, -b.imag) for b in ba], index=energy_ev.index)

    @cache  # noqa: B019
    def beta_bare_atom(
        self,
        chemical_formula: str | None = None,
        energy: str | None = None,
        density: float | None = None,
    ) -> pd.Series:
        """
        Calculate the atomic absorption coefficient (beta) for a given material.

        The absorption coefficient beta represents the imaginary part of the complex
        index of refraction and is directly related to X-ray absorption.

        Parameters
        ----------
        chemical_formula : str | None, optional
            Chemical formula of the compound (e.g., 'C', 'H2O', 'C8H8O2').
            If None, uses the accessor's stored chemical_formula property.
        energy : str, optional
            Column name containing energy values in eV.
        density : float | None, optional
            Material density in g/cm³. If None, uses the accessor's density property.

        Returns
        -------
        pd.Series
            Series containing the beta values for the specified material.

        Examples
        --------
        >>> df = pd.DataFrame({"Energy": np.linspace(280, 320, 100)})
        >>> # Calculate beta for graphite with density 2.26 g/cm³
        >>> df["beta_carbon"] = df.nexafs.beta_bare_atom("C", "Energy", density=2.26)
        >>>
        >>> # Set properties first, then calculate
        >>> df.nexafs.chemical_formula = "C"
        >>> df.nexafs.density = 2.26
        >>> df["beta_carbon"] = df.nexafs.beta_bare_atom(energy="Energy")
        """
        # Use provided values or fall back to class properties
        formula = chemical_formula or self._chemical_formula
        if not formula:
            msg = (
                "No chemical formula provided. "
                "Please provide a chemical formula or set it using the "
                "chemical_formula property."
            )
            raise ValueError(msg)

        if energy is None:
            msg = "Energy column name must be provided."
            raise ValueError(msg)

        mat_density = density if density is not None else self._density

        # Store the values as properties if provided
        if chemical_formula:
            self.chemical_formula = chemical_formula
        if density is not None:
            self.density = density

        return self.bare_atom(
            chemical_formula=formula,
            energy=energy,
            density=mat_density,
        ).apply(lambda x: x.imag)

    @cache  # noqa: B019
    def delta_bare_atom(
        self,
        chemical_formula: str | None = None,
        energy: str | None = None,
        density: float | None = None,
    ) -> pd.Series:
        """
        Calculate the phase shift coefficient (delta) for a given material.

        The phase shift coefficient delta represents the real part decrement of the
        complex index of refraction (n = 1-delta+i*beta) and is related to refraction.

        Parameters
        ----------
        chemical_formula : str | None, optional
            Chemical formula of the compound (e.g., 'C', 'H2O', 'C8H8O2').
            If None, uses the accessor's stored chemical_formula property.
        energy : str, optional
            Column name containing energy values in eV.
        density : float | None, optional
            Material density in g/cm³. If None, uses the accessor's density property.

        Returns
        -------
        pd.Series
            Series containing the delta values for the specified material.

        Examples
        --------
        >>> df = pd.DataFrame({"Energy": np.linspace(280, 320, 100)})
        >>> # Calculate delta for silicon with density 2.33 g/cm³
        >>> df["delta_si"] = df.nexafs.delta_bare_atom("Si", "Energy", density=2.33)
        >>>
        >>> # Set properties first, then calculate
        >>> df.nexafs.chemical_formula = "Si"
        >>> df.nexafs.density = 2.33
        >>> df["delta_si"] = df.nexafs.delta_bare_atom(energy="Energy")
        """
        # Use provided values or fall back to class properties
        formula = chemical_formula or self._chemical_formula
        if not formula:
            msg = (
                "No chemical formula provided. "
                "Please provide a chemical formula or set it using the "
                "chemical_formula property."
            )
            raise ValueError(msg)

        if energy is None:
            msg = "Energy column name must be provided."
            raise ValueError(msg)

        mat_density = density if density is not None else self._density

        # Store the values as properties if provided
        if chemical_formula:
            self.chemical_formula = chemical_formula
        if density is not None:
            self.density = density

        return self.bare_atom(
            chemical_formula=formula,
            energy=energy,
            density=mat_density,
        ).apply(lambda x: x.real)

    def convert_absorption(
        self,
        absorption_coefficient: str,
        energy_column: str,
    ) -> pd.Series:
        """
        Convert optical density or absorption measurements to beta coefficient.

        This method converts various forms of absorption measurements (optical density,
        absorbance, or mass attenuation coefficient) to the beta coefficient using
        the wavelength derived from the energy values.

        Parameters
        ----------
        absorption_coefficient : str
            Column name for the absorption measurement to convert.
            This could be optical density, absorbance, or any measurement proportional
            to the Beer-Lambert absorption.
        energy_column : str
            Column name for the energy values in eV.

        Returns
        -------
        pd.Series
            Series with the converted absorption coefficient (beta) values.

        Notes
        -----
        The conversion uses the relationship: β = A·λ/(4π) where:
        - A is the absorption measurement
        - λ is the wavelength (derived from energy in eV: λ = hc/E)
        - The result is the imaginary part of the refractive index

        Raises
        ------
        ValueError
            If either column is not found in the DataFrame or is not numeric.
        """
        # Validate absorption coefficient column
        if (
            absorption_coefficient in self._obj.columns
            and pd.api.types.is_numeric_dtype(self._obj[absorption_coefficient])
        ):
            A = self._obj[absorption_coefficient].astype(float)
        else:
            msg = (
                f"Column '{absorption_coefficient}' not found in DataFrame or is not "
                "numeric. Please provide a valid column name with measurements."
            )
            raise ValueError(msg)

        # Validate energy column
        if energy_column in self._obj.columns and pd.api.types.is_numeric_dtype(
            self._obj[energy_column]
        ):
            energy = self._obj[energy_column].astype(float)
            # Calculate wavelength in cm from energy in eV using hc constant
            wavelength_cm = hc / energy
        else:
            msg = (
                f"Column '{energy_column}' not found in DataFrame or is not numeric. "
                "Please provide a valid column name with numeric energy values."
            )
            raise ValueError(msg)

        # Apply Beer-Lambert conversion: beta = A * lambda / (4 * pi)
        return A * wavelength_cm / (4 * np.pi)

    def normalize(
        self,
        nexafs_column: str,
        energy_column: str,
        chemical_formula: str | None = None,
        density: float | None = None,
        nexafs_type: Literal["electron-yield", "absorption"] = "electron-yield",
        pre_edge_range: tuple[float, float] | None = None,
        post_edge_range: tuple[float, float] | None = None,
        group_by: str | list[str] | None = None,
    ) -> pd.Series:
        """
        Normalize NEXAFS data to atomic beta using pre-edge and post-edge regions.

        This method performs a two-point normalization of experimental NEXAFS data
        to align with theoretical atomic absorption coefficients. It can handle both
        electron yield and absorption measurements and supports group-wise normalization
        when multiple spectra are present in the DataFrame.

        Parameters
        ----------
        nexafs_column : str
            Column name containing the raw NEXAFS data to normalize.
        energy_column : str
            Column name containing energy values in eV.
        chemical_formula : str | None, optional
            Chemical formula of the compound (e.g., 'C', 'H2O', 'C8H8O2').
            If None, uses the accessor's stored chemical_formula property.
        density : float | None, optional
            Material density in g/cm³. If None, uses the accessor's density property.
        nexafs_type : Literal["electron-yield", "absorption"], optional
            Type of NEXAFS measurement:
            - "electron-yield": TEY or PEY measurements (default)
            - "absorption": Transmission or fluorescence yield measurements
        pre_edge_range : Tuple[float, float], optional
            Energy range (min, max) in eV to use for pre-edge normalization.
            If None, the lowest 5% of energy values will be used.
        post_edge_range : Tuple[float, float], optional
            Energy range (min, max) in eV to use for post-edge normalization.
            If None, the highest 5% of energy values will be used.
        group_by : Optional[Union[str, List[str]]], optional
            Column name(s) to group by for normalizing multiple spectra separately.

        Returns
        -------
        pd.Series
            Series containing the normalized NEXAFS data aligned with atomic beta.

        Examples
        --------
        >>> # Basic normalization
        >>> df["normalized"] = df.nexafs.normalize(
        ...     chemical_formula="C",
        ...     nexafs_column="Raw_Intensity",
        ...     energy_column="Energy",
        ...     pre_edge_range=(280, 282),
        ...     post_edge_range=(320, 325),
        ... )

        >>> # Group-wise normalization by sample using preset properties
        >>> df.nexafs.chemical_formula = "C"
        >>> df.nexafs.density = 2.26
        >>> df["normalized"] = df.nexafs.normalize(
        ...     nexafs_column="Raw_Intensity",
        ...     energy_column="Energy",
        ...     group_by="Sample",
        ... )

        Raises
        ------
        ValueError
            If columns are not found, or if pre/post ranges don't contain data points.
        """
        # Use provided formula or fall back to class property
        formula = chemical_formula or self._chemical_formula
        if not formula:
            msg = (
                "No chemical formula provided. "
                "Please provide a chemical formula or set it using the "
                "chemical_formula property."
            )
            raise ValueError(msg)

        mat_density = density if density is not None else self._density

        # Store values in class properties if provided
        if chemical_formula:
            self.chemical_formula = chemical_formula
        if density is not None:
            self.density = density

        # If grouping is specified, apply normalization to each group
        if group_by is not None:
            if isinstance(group_by, str):
                group_by = [group_by]

            # Create an empty Series with the same index as the DataFrame
            result = pd.Series(index=self._obj.index)

            # Apply normalization to each group
            for _, group in self._obj.groupby(group_by):
                group_accessor = NEXAFSAccessor(group)
                # Copy the chemical formula and density to the group accessor
                group_accessor.chemical_formula = formula
                group_accessor.density = mat_density

                normalized = group_accessor.normalize(
                    nexafs_column=nexafs_column,
                    energy_column=energy_column,
                    nexafs_type=nexafs_type,
                    pre_edge_range=pre_edge_range,
                    post_edge_range=post_edge_range,
                )
                # Store the results back in the result Series
                result.loc[group.index] = normalized.values

            return result

        # Get energy and raw intensity data from DataFrame
        energy = self._obj[energy_column]
        raw_intensity = self._obj[nexafs_column]

        # Auto-determine pre-edge and post-edge ranges if not provided
        if pre_edge_range is None:
            # Use lowest 5% of energy range as pre-edge
            min_e = energy.min()
            pre_edge_width = 0.05 * (energy.max() - min_e)
            pre_edge_range = (min_e, min_e + pre_edge_width)

        if post_edge_range is None:
            # Use highest 5% of energy range as post-edge
            max_e = energy.max()
            post_edge_width = 0.05 * (max_e - energy.min())
            post_edge_range = (max_e - post_edge_width, max_e)

        # Create masks for pre-edge and post-edge regions
        pre_mask = (energy >= pre_edge_range[0]) & (energy <= pre_edge_range[1])
        post_mask = (energy >= post_edge_range[0]) & (energy <= post_edge_range[1])

        # Ensure there are data points in the pre-edge and post-edge regions
        if not pre_mask.any() or not post_mask.any():
            msg = (
                f"Pre-edge or post-edge energy range did not match any data points. "
                f"Adjust ranges: {pre_edge_range} and {post_edge_range}."
            )
            raise ValueError(msg)

        # Convert to beta if absorption data
        if nexafs_type == "absorption":
            raw_intensity = self.convert_absorption(
                absorption_coefficient=nexafs_column,
                energy_column=energy_column,
            )

        # Calculate average values in the pre-edge and post-edge regions
        avg_raw_pre = raw_intensity[pre_mask].mean()
        avg_raw_post = raw_intensity[post_mask].mean()

        # Get atomic beta values and align with the experimental data
        beta_atomic_aligned = (
            self.beta_bare_atom(energy=energy_column)
            .reindex(raw_intensity.index)
            .fillna(0)
        )

        # Calculate average atomic beta values in the pre-edge and post-edge regions
        avg_beta_pre = beta_atomic_aligned[pre_mask].mean()
        avg_beta_post = beta_atomic_aligned[post_mask].mean()

        # Calculate the scaling parameters for normalization
        delta_raw = avg_raw_post - avg_raw_pre

        # Check for sufficient signal change
        if np.abs(delta_raw) < 1e-9:
            msg = (
                f"Raw NEXAFS data has nearly identical pre/post-edge averages. "
                f"Check data or ranges: {avg_raw_pre:.4g} vs {avg_raw_post:.4g}"
            )
            raise ValueError(msg)

        delta_beta = avg_beta_post - avg_beta_pre

        # Calculate normalization coefficients
        A = delta_beta / delta_raw
        B = avg_beta_pre - A * avg_raw_pre

        # Apply normalization
        normalized_intensity = A * raw_intensity + B
        return normalized_intensity.astype(float)

    def plot_spectrum(
        self,
        x: str,
        y: str,
        reference: str | None = None,
        pre_edge_range: tuple[float, float] | None = None,
        post_edge_range: tuple[float, float] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        color: str | pd.Series | None = "red",
        color_column: str | None = None,
        palette: str | list | None = "viridis",
        ref_color: str = "black",
        figsize: tuple[float, float] = (8, 6),
        group_by: str | list[str] | None = None,
        style: str | None = None,
        save_path: str | None = None,
        *,
        grid: bool = True,
        legend: bool = True,
        legend_title: str | None = None,
        colorbar: bool = False,
        show: bool = True,
        **kwargs,
    ) -> Figure | dict[str, Figure]:
        """
        Create publication-quality NEXAFS spectrum plot with optional reference data.

        This versatile plotting function creates high-quality visualizations of NEXAFS
        spectra with options for comparison to reference data highlighting normalization
        regions, and group-wise plotting for comparative analysis.

        Parameters
        ----------
        x : str
            Column name for the x-axis data (typically energy in eV).
        y : str
            Column name for the spectrum data to plot.
        reference : Optional[str], optional
            Column name for reference data to plot alongside the spectrum.
        pre_edge_range : Optional[Tuple[float, float]], optional
            Energy range (min, max) to highlight as pre-edge region.
        post_edge_range : Optional[Tuple[float, float]], optional
            Energy range (min, max) to highlight as post-edge region.
        title : Optional[str], optional
            Plot title. If None, a title will be generated based on column names.
        xlabel : Optional[str], optional
            X-axis label. If None, the x column name will be used.
        ylabel : Optional[str], optional
            Y-axis label. If None, the y column name will be used.
        color : Union[str, pd.Series, None], optional
            Color for the primary spectrum. Can be:
            - A string color name (e.g., "red")
            - A pandas Series with colors for each data point
            - None to use color_column for coloring
        color_column : Optional[str], optional
            Column name to use for coloring data points when color=None.
            Useful for highlighting trends in a third variable.
        palette : Union[str, list, None], optional
            Color palette to use when color_column is specified.
            Can be a seaborn/matplotlib colormap name or a list of colors.
        ref_color : str, optional
            Color for the reference spectrum, by default "black".
        figsize : Tuple[float, float], optional
            Figure dimensions (width, height) in inches, by default (8, 6).
        group_by : Optional[Union[str, List[str]]], optional
            Column name(s) to group data by, creating separate plots for each group.
        style : Optional[str], optional
            Matplotlib style to use for the plot.
        legend_title : Optional[str], optional
            Title for the legend. If None, no title is used.
        grid : bool, optional
            Whether to show grid lines, by default True.
        legend : bool, optional
            Whether to show a legend, by default True.
        colorbar : bool, optional
            Whether to show a colorbar when using color_column, by default False.
        save_path : Optional[str], optional
            Path to save the figure(s). If group_by is used, group name will be added.
        show : bool, optional
            Whether to display the plot, by default True.
        **kwargs
            Additional keyword arguments passed to pandas plot function.

        Returns
        -------
        Union[plt.Figure, Dict[str, plt.Figure]]
            If group_by is None, returns the matplotlib Figure object.
            If group_by is specified, returns a dictionary of {group_name: Figure}.

        Examples
        --------
        >>> # Basic spectrum plot
        >>> fig = df.nexafs.plot_spectrum(
        ...     x="Energy", y="Normalized", title="Carbon K-edge NEXAFS"
        ... )

        >>> # Use a color column to show a third variable
        >>> fig = df.nexafs.plot_spectrum(
        ...     x="Energy",
        ...     y="Intensity",
        ...     color_column="Temperature",
        ...     palette="plasma",
        ...     colorbar=True,
        ... )

        >>> # Compare multiple samples
        >>> figs = df.nexafs.plot_spectrum(
        ...     x="Energy",
        ...     y="Normalized",
        ...     reference="Theoretical",
        ...     group_by="Sample",
        ...     save_path="nexafs_plots/sample",
        ... )
        """
        # Apply matplotlib style if specified
        if style:
            plt.style.use(style)

        # Group-wise plotting
        if group_by is not None:
            if isinstance(group_by, str):
                group_by = [group_by]

            figures = {}

            # Create separate plot for each group
            for name, group in self._obj.groupby(group_by):
                group_accessor = NEXAFSAccessor(group)

                # Generate group-specific title if not provided
                group_title = title
                if group_title is None:
                    if isinstance(name, tuple):
                        group_name = "_".join(str(n) for n in name)
                    else:
                        group_name = str(name)
                    group_title = f"{y} vs {x} for {group_name}"

                # Generate plot for this group
                fig = group_accessor.plot_spectrum(
                    x=x,
                    y=y,
                    reference=reference,
                    pre_edge_range=pre_edge_range,
                    post_edge_range=post_edge_range,
                    title=group_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    color=color,
                    color_column=color_column,
                    palette=palette,
                    ref_color=ref_color,
                    figsize=figsize,
                    grid=grid,
                    legend=legend,
                    legend_title=legend_title,
                    colorbar=colorbar,
                    show=False,  # Don't show individual plots
                    **kwargs,
                )

                # Save if path is provided
                if save_path:
                    if isinstance(name, tuple):
                        group_name = "_".join(str(n) for n in name)
                    else:
                        group_name = str(name)
                    group_save_path = f"{save_path}_{group_name}.png"
                    if isinstance(fig, Figure):
                        fig.savefig(group_save_path, dpi=300, bbox_inches="tight")

                # Add to dictionary of figures
                figures[name] = fig

            # Show all figures if requested
            if show:
                plt.show()

            return figures

        # Create a copy of the plotting data
        plot_data = self._obj[[x, y]].copy()

        # Add reference column if needed
        if reference and reference in self._obj.columns:
            plot_data[reference] = self._obj[reference]

        # Prepare figure
        fig, ax = plt.subplots(figsize=figsize)

        # Handle coloring based on a column
        if color_column is not None and color_column in self._obj.columns:
            # Use pandas scatter plot with color mapping
            plot_data[color_column] = self._obj[color_column]

            # Create a colormap
            if isinstance(palette, list):
                cmap = mcolors.LinearSegmentedColormap.from_list("custom", palette)
            else:
                cmap = plt.get_cmap(palette or "viridis")

            # Use pandas scatter plot with color mapping (leveraging pandas backend)
            scatter = plot_data.plot.scatter(
                x=x,
                y=y,
                c=color_column,
                colormap=cmap,
                title=title or f"{y} vs {x}",
                xlabel=xlabel or x,
                ylabel=ylabel or y,
                ax=ax,
                **kwargs,
            )

            # Add colorbar if requested
            if colorbar:
                plt.colorbar(scatter.collections[0], ax=ax, label=color_column)

        else:
            # Use pandas plot for line
            plot_data.plot(
                x=x,
                y=y,
                color=color,
                title=title or f"{y} vs {x}",
                xlabel=xlabel or x,
                ylabel=ylabel or y,
                label=y,
                ax=ax,
                linewidth=2,
                legend=legend,
                **kwargs,
            )

            # Add reference line if provided
            if reference and reference in self._obj.columns:
                ax.plot(
                    self._obj[x],
                    self._obj[reference],
                    color=ref_color,
                    linestyle="--",
                    linewidth=2,
                    label=reference,
                )

                if legend:
                    ax.legend(title=legend_title)

        # Highlight pre-edge region if provided
        if pre_edge_range:
            ax.axvspan(
                pre_edge_range[0],
                pre_edge_range[1],
                color="lightgrey",
                alpha=0.3,
                label="Pre-edge Region",
            )

        # Highlight post-edge region if provided
        if post_edge_range:
            ax.axvspan(
                post_edge_range[0],
                post_edge_range[1],
                color="darkgrey",
                alpha=0.3,
                label="Post-edge Region",
            )

        # Add grid if requested
        if grid:
            ax.grid(True, linestyle=":", alpha=0.6)

        # Ensure legend has all elements
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if pre_edge_range or post_edge_range:
                # Recreate legend to include regions
                ax.legend(handles, labels, title=legend_title, loc="best")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_heatmap(
        self,
        x: str,
        y: str,
        z: str,
        colormap: str | list = "viridis",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        figsize: tuple[float, float] = (10, 8),
        interpolation: str = "nearest",
        vmin: float | None = None,
        vmax: float | None = None,
        save_path: str | None = None,
        pivot_kwargs: dict | None = None,
        *,  # Force keyword-only arguments after this point
        show: bool = True,
        **kwargs,
    ) -> Figure:
        """
        Create a 2D heatmap visualization of NEXAFS data.

        This method is useful for visualizing NEXAFS data that varies across two
        dimensions, such as angle-resolved measurements or time-series data.

        Parameters
        ----------
        x : str
            Column name for the x-axis data (typically energy).
        y : str
            Column name for the y-axis data (e.g., angle, time, position).
        z : str
            Column name for the intensity data to be displayed as color.
        colormap : Union[str, list], optional
            Colormap for the heatmap. Can be:
            - A string with a matplotlib/seaborn colormap name (e.g., "viridis")
            - A list of colors to create a custom colormap
            By default "viridis".
        title : Optional[str], optional
            Plot title. If None, a title will be generated.
        xlabel : Optional[str], optional
            X-axis label. If None, the x column name will be used.
        ylabel : Optional[str], optional
            Y-axis label. If None, the y column name will be used.
        zlabel : Optional[str], optional
            Colorbar label. If None, the z column name will be used.
        figsize : Tuple[float, float], optional
            Figure dimensions (width, height) in inches, by default (10, 8).
        interpolation : str, optional
            Interpolation method for the heatmap, by default "nearest".
        vmin : Optional[float], optional
            Minimum value for colormap scaling.
        vmax : Optional[float], optional
            Maximum value for colormap scaling.
        save_path : Optional[str], optional
            Path to save the figure.
        pivot_kwargs : Optional[dict], optional
            Additional keyword arguments for pandas pivot_table.
        show : bool, optional
            Whether to display the plot, by default True.
        **kwargs
            Additional keyword arguments passed to plt.pcolormesh.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object.

        Examples
        --------
        >>> # Create heatmap of angle-resolved NEXAFS
        >>> fig = df.nexafs.plot_heatmap(
        ...     x="Energy",
        ...     y="Angle",
        ...     z="Intensity",
        ...     title="Angle-Resolved NEXAFS",
        ...     colormap="plasma",
        ... )

        >>> # Create heatmap with custom color palette and aggregation
        >>> fig = df.nexafs.plot_heatmap(
        ...     x="Energy",
        ...     y="Angle",
        ...     z="Intensity",
        ...     colormap=["blue", "green", "yellow", "red"],
        ...     pivot_kwargs={"aggfunc": "max"},
        ... )

        Notes
        -----
        This method works best with gridded data. If your data is not already
        on a regular grid, consider specifying a different aggregation function
        in pivot_kwargs (default is "mean").
        """
        from typing import Any

        # Create a pivot table with the data - only use safe defaults
        safe_pivot_args: dict[str, Any] = {
            "index": y,
            "columns": x,
            "values": z,
            "aggfunc": "mean",
        }

        # If additional pivot args provided, carefully handle them
        if pivot_kwargs:
            # Only allow non-boolean args or explicitly boolean values
            for k, v in pivot_kwargs.items():
                if k not in ["margins", "dropna", "observed", "sort"] or isinstance(
                    v, bool
                ):
                    safe_pivot_args[k] = v

        # Create the pivot table with only safe arguments
        pivot = self._obj.pivot_table(**safe_pivot_args)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare colormap
        if isinstance(colormap, list):
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", colormap)
        else:
            cmap = colormap

        # Use matplotlib's pcolormesh directly
        pcm = ax.pcolormesh(
            pivot.columns,  # x values
            pivot.index,  # y values
            pivot.values,  # z values
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Set labels and title
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title or f"{z} as a function of {x} and {y}")

        # Add colorbar with label
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(zlabel or z)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_normalized_nexafs(
        self,
        normalized_intensity: str,
        energy_column: str,
        chemical_formula: str | None = None,
        pre_edge_range: tuple[float, float] | None = None,
        post_edge_range: tuple[float, float] | None = None,
        raw_intensity: str | None = None,
        title: str | None = None,
        edge_label: str | None = None,
        figsize: tuple[float, float] = (8, 6),
        save_path: str | None = None,
        *,  # Force keyword-only arguments after this point
        show: bool = True,
        **kwargs,
    ) -> Figure:
        """
        Generate a plot comparing normalized NEXAFS data to atomic beta.

        This is a specialized plotting function that shows normalized experimental data
        alongside theoretical atomic absorption, with optional highlighting of
        normalization regions and raw data display.

        Parameters
        ----------
        normalized_intensity : str
            Column name containing normalized experimental data.
        energy_column : str
            Column name containing energy values in eV.
        chemical_formula : str | None, optional
            Chemical formula for calculating atomic beta reference.
            If None, no reference line is shown.
        pre_edge_range : tuple[float, float] | None, optional
            Energy range (min, max) to highlight as pre-edge normalization region.
        post_edge_range : tuple[float, float] | None, optional
            Energy range (min, max) to highlight as post-edge normalization region.
        raw_intensity : str | None, optional
            Column name for raw experimental data to display on secondary y-axis.
        title : str | None, optional
            Plot title. If None, a title will be generated.
        edge_label : str | None, optional
            Label for the absorption edge (e.g., "C K-edge").
        figsize : tuple[float, float], optional
            Figure dimensions (width, height) in inches, by default (8, 6).
        save_path : str | None, optional
            Path to save the figure.
        show : bool, optional
            Whether to display the plot, by default True.
        **kwargs
            Additional keyword arguments passed to pandas plotting functions.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object.

        Examples
        --------
        >>> # Create normalized NEXAFS plot
        >>> fig = df.nexafs.plot_normalized_nexafs(
        ...     normalized_intensity="Normalized",
        ...     energy_column="Energy",
        ...     chemical_formula="C",
        ...     pre_edge_range=(280, 282),
        ...     post_edge_range=(320, 325),
        ...     raw_intensity="Raw_TEY",
        ...     edge_label="C K-edge",
        ... )
        """
        # Create a copy of the plotting data
        plot_data = self._obj[[energy_column, normalized_intensity]].copy()

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot normalized data using pandas plot
        plot_data.plot(
            x=energy_column,
            y=normalized_intensity,
            label="NEXAFS (Normalized)",
            color="red",
            linewidth=2,
            ax=ax,
            **kwargs,
        )

        # Calculate and plot atomic beta reference if formula provided
        if chemical_formula or self._chemical_formula != "":
            beta_atomic = self.beta_bare_atom(
                chemical_formula=chemical_formula,
                energy=energy_column,
            )

            # Add beta atomic to the plotting data
            beta_col = f"beta_atomic_{chemical_formula}"
            plot_data[beta_col] = beta_atomic

            # Plot beta atomic reference
            ax.plot(
                plot_data[energy_column],
                plot_data[beta_col],
                label=r"$\beta$ (atomic)",
                color="black",
                linewidth=2,
                linestyle="--",
            )

        # Highlight normalization regions if provided
        if pre_edge_range:
            ax.axvspan(
                pre_edge_range[0],
                pre_edge_range[1],
                color="lightgrey",
                alpha=0.2,
                label="Pre-edge Region",
            )

        if post_edge_range:
            ax.axvspan(
                post_edge_range[0],
                post_edge_range[1],
                color="darkgrey",
                alpha=0.3,
                label="Post-edge Region",
            )

        # Set labels and title
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"Absorption Coefficient $\beta$ Scale")

        # Generate title if not provided
        if title is None:
            edge_info = f" ({edge_label})" if edge_label else ""
            title_formula = chemical_formula or ""
            title_text = "NEXAFS Normalization to Atomic $\\beta$ "
            title_text += f"for {title_formula}{edge_info}"
            ax.set_title(title_text)
        else:
            ax.set_title(title)

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # Add raw intensity on secondary axis if provided
        if raw_intensity and raw_intensity in self._obj.columns:
            # Create secondary y-axis
            ax2 = ax.twinx()

            # Add raw intensity to the plot data
            plot_data[raw_intensity] = self._obj[raw_intensity]

            # Plot raw intensity on secondary axis
            ax2.plot(
                plot_data[energy_column],
                plot_data[raw_intensity],
                label="Raw Intensity",
                color="blue",
                alpha=0.4,
                linestyle=":",
            )

            ax2.set_ylabel("Raw Intensity (arb. units)", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

            # Create combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            # Standard legend
            ax.legend(loc="best")

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig
