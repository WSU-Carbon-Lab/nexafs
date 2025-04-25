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

from nexafs.core import NEXAFSAccessor

__all__ = [
    "NEXAFSAccessor",
]
