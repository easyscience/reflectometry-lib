# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import os
import warnings
from typing import Optional
from typing import Sequence
from typing import TextIO
from typing import Union

import numpy as np
import scipp as sc

from easyreflectometry.data import DataSet1D
from easyreflectometry.model.resolution_functions import DEFAULT_RESOLUTION_FWHM_PERCENTAGE
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.model.resolution_functions import Pointwise
from easyreflectometry.model.resolution_functions import ResolutionFunction
from easyreflectometry.orso_utils import is_orso_file
from easyreflectometry.orso_utils import load_data_from_orso_file


def load(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load data from an ORSO file (.ort/.orb) or a plain text file.

    The discriminator is the ORSO banner line (or the HDF5 magic for binary
    files), **not** the file extension: a file carrying the banner that fails
    to parse raises instead of being silently re-read as plain text (which
    would drop the entire header, including polarization).

    For a plain text file the columns are read by position in the ORSO order
    Qz, R, sR, sQz -- further numeric columns are ignored -- and the error
    columns must hold **standard deviations**, not variances. See
    :func:`_load_txt`.

    Parameters
    ----------
    fname : Union[TextIO, str]
        The file to be read.

    Returns
    -------
    sc.DataGroup
        The loaded data.
    """
    if is_orso_file(str(fname)):
        return load_data_from_orso_file(fname)
    return _load_txt(fname)


def dataset_from_datagroup(data_group: sc.DataGroup, data_key: Optional[str] = None) -> DataSet1D:
    """Build a DataSet1D from one dataset of a loaded DataGroup.

    The ORSO header (when present) is attached to the returned dataset as the
    ``orso_header`` attribute (a plain dict), so exporters can reuse the
    original ``data_source``/``reduction`` provenance.

    Parameters
    ----------
    data_group : sc.DataGroup
        A DataGroup as returned by :func:`load`.
    data_key : Optional[str], optional
        The data entry to use (e.g. ``'R_0'``). By default, the first entry.

    Returns
    -------
    DataSet1D
        The dataset.
    """
    if data_key is None:
        data_key = list(data_group['data'].keys())[0]
    coords_key = 'Qz_' + data_key[len('R_') :]
    if coords_key not in data_group['coords']:
        coords_key = list(data_group['coords'].keys())[0]
    dataset = DataSet1D(
        x=data_group['coords'][coords_key].values,
        y=data_group['data'][data_key].values,
        ye=data_group['data'][data_key].variances,
        xe=data_group['coords'][coords_key].variances,
    )
    header = None
    if 'attrs' in data_group and data_key in data_group['attrs']:
        try:
            header = data_group['attrs'][data_key]['orso_header'].values
        except (KeyError, AttributeError):
            header = None
    dataset.orso_header = header
    return dataset


def load_as_dataset(fname: Union[TextIO, str], data_group: Optional[sc.DataGroup] = None) -> DataSet1D:
    """Load data from an ORSO .ort file as a DataSet1D.

    Parameters
    ----------
    fname : Union[TextIO, str]
        The file to be read.
    data_group : Optional[sc.DataGroup], optional
        Pre-loaded DataGroup for *fname* (avoids re-parsing the file).
        By default, None.

    Returns
    -------
    DataSet1D
        The (first) dataset in the file.
    """
    if data_group is None:
        data_group = load(fname)
    basename = os.path.splitext(os.path.basename(fname))[0]
    data_name = 'R_' + basename
    data_name = list(data_group['data'].keys())[0] if data_name not in data_group['data'] else data_name
    return dataset_from_datagroup(data_group, data_key=data_name)


def extract_orso_title(data_group: sc.DataGroup, data_name: str) -> str | None:
    """Extract orso title."""
    try:
        header = data_group['attrs'][data_name]['orso_header']
        title = header.values.get('data_source', {}).get('experiment', {}).get('title')
    except (AttributeError, KeyError, TypeError):
        return None
    if title is None:
        return None
    title_str = str(title).strip()
    return title_str or None


def _load_txt(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load data from a simple txt file.

    Columns are read by position, following the ORSO order: Qz, R, sR, sQz.
    Any further **numeric** columns (e.g. wavelength) are ignored -- the whole
    file is still parsed as numbers before the leading columns are taken, so a
    trailing text column, or rows of differing width, remain an error.

    The error columns are taken to be **standard deviations** (sigma), matching
    the ORSO default, and are squared to obtain the stored variances; a file
    carrying variances instead would be mis-scaled. Plain text carries no
    convention marker, so this is a requirement on the caller and not something
    the loader can check -- unlike an ORSO file, which declares ``value_is`` and
    whose FWHM errors are converted to sigma on load.

    Parameters
    ----------
    fname : Union[TextIO, str]
        The path for the file to be read.
    """
    # fname can have either a space or a comma as delimiter
    # Determine the delimiter used in the file
    delimiter = None
    with open(fname, 'r') as f:
        # find first non-comment and non-empty line
        for line in f:
            if line.strip() and not line.startswith('#'):
                break
        first_line = line
    if ',' in first_line:
        delimiter = ','

    basename = os.path.splitext(os.path.basename(fname))[0]

    try:
        # ndmin=2 keeps a single-row file two-dimensional, so columns are indexable
        data = np.loadtxt(fname, delimiter=delimiter, comments='#', ndmin=2)
        num_columns = data.shape[1]

        # Verify minimum column requirement
        if num_columns < 3:
            raise ValueError(f'File must contain at least 3 columns (found {num_columns})')

        # Take the leading columns by position; any extra columns are ignored
        if num_columns >= 4:
            x, y, e, xe = data[:, :4].T
        else:  # 3 columns
            x, y, e = data[:, :3].T
            xe = np.zeros_like(x)

    except (ValueError, IOError) as error:
        # Re-raise with more descriptive message
        raise ValueError(f'Failed to load data from {fname}: {str(error)}') from error

    data_name = 'R_' + basename
    coords_name = 'Qz_' + basename
    data = {data_name: sc.array(dims=[coords_name], values=y, variances=np.square(e))}
    coords = {
        data[data_name].dims[0]: sc.array(
            dims=[coords_name],
            values=x,
            variances=np.square(xe),
            unit=sc.Unit('1/angstrom'),
        )
    }
    return sc.DataGroup(data=data, coords=coords)


def resolution_from_dataset(dataset: DataSet1D) -> Optional[Pointwise]:
    """The measured q-resolution of a dataset, or ``None`` when it carries none.

    A dataset carries its resolution as the ``xe`` column, the variance of
    ``Qz`` (sQz²) at every point. When any of it is positive a
    :class:`Pointwise` resolution is built from it; an absent, empty, all-zero
    or all-NaN column gives ``None``, meaning the model's resolution applies.

    Parameters
    ----------
    dataset : DataSet1D
        The dataset.

    Returns
    -------
    Optional[Pointwise]
        The per-point resolution, or ``None``.
    """
    xe = getattr(dataset, 'xe', None)
    if xe is None or len(xe) == 0:
        return None
    xe = np.asarray(xe, dtype=float)
    # nan-robust gate: nan is truthy for np.any, but a nan-carrying xe must
    # not build a Pointwise (np.interp would propagate the nan everywhere).
    if not np.any(np.nan_to_num(xe) > 0):
        return None
    return Pointwise(q_data_points=[np.asarray(dataset.x, dtype=float), np.asarray(dataset.y, dtype=float), xe])


def merge_datasets(
    datasets: Sequence[DataSet1D],
    name: Optional[str] = None,
    fill_resolution: Optional[ResolutionFunction] = None,
) -> DataSet1D:
    """Combine several measurements of one contrast into a single dataset.

    A contrast is often measured as several curves -- one per incident angle
    or wavelength band -- each with its own q-resolution. This concatenates
    them, sorted by ``Qz``, into one dataset that keeps every point's own
    resolution, so the fit smears each point with the width it was measured
    with. Overlapping q ranges are kept as they are: the merged dataset may
    hold the same ``Qz`` twice with different widths, which
    :meth:`Pointwise.smearing` honours when evaluated at the data points.

    Parameters
    ----------
    datasets : Sequence[DataSet1D]
        The measurements to combine; at least one. The model, labels and ORSO
        header of the first one are kept.
    name : Optional[str], optional
        Name of the merged dataset. By default the input names joined by ' + '.
    fill_resolution : Optional[ResolutionFunction], optional
        Resolution assumed for the points of a dataset that carries no
        q-resolution when others do, so the merged ``xe`` column is complete.
        By default the 5% FWHM default resolution. A warning is emitted in
        that case. When no dataset carries a resolution, ``xe`` stays zero
        and the merged dataset has no resolution of its own.

    Returns
    -------
    DataSet1D
        The merged dataset. Its ``resolution_function`` is a
        :class:`Pointwise` over all points when any input carried one.

    Raises
    ------
    ValueError
        No datasets were given.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError('At least one dataset is required to merge.')

    measured = [resolution_from_dataset(dataset) is not None for dataset in datasets]
    if any(measured) and not all(measured):
        missing = [dataset.name for dataset, has in zip(datasets, measured) if not has]
        if fill_resolution is None:
            fill_resolution = PercentageFwhm(DEFAULT_RESOLUTION_FWHM_PERCENTAGE)
        warnings.warn(
            f'Dataset(s) {missing} carry no q-resolution while the others do; their points are '
            f'given the {type(fill_resolution).__name__} resolution so the merged dataset has one.',
            UserWarning,
            stacklevel=2,
        )

    xs, ys, yes, xes = [], [], [], []
    for dataset, has in zip(datasets, measured):
        x = np.asarray(dataset.x, dtype=float)
        xe = np.zeros_like(x) if dataset.xe is None else np.asarray(dataset.xe, dtype=float)
        if any(measured) and not has:
            xe = np.square(fill_resolution.smearing(x))
        xs.append(x)
        ys.append(np.asarray(dataset.y, dtype=float))
        yes.append(np.zeros_like(x) if dataset.ye is None else np.asarray(dataset.ye, dtype=float))
        xes.append(xe)

    x = np.concatenate(xs)
    order = np.argsort(x, kind='stable')
    first = datasets[0]
    merged = DataSet1D(
        name=' + '.join(dataset.name for dataset in datasets) if name is None else name,
        x=x[order],
        y=np.concatenate(ys)[order],
        ye=np.concatenate(yes)[order],
        xe=np.concatenate(xes)[order],
        model=first.model,
        x_label=first.x_label,
        y_label=first.y_label,
        auto_background=False,
    )
    merged.orso_header = getattr(first, 'orso_header', None)
    merged.resolution_function = resolution_from_dataset(merged)
    return merged


def merge_datagroups(*data_groups: sc.DataGroup) -> sc.DataGroup:
    """Merge multiple DataGroups into a single DataGroup."""
    merged_data = {}
    merged_coords = {}
    merged_attrs = {}

    for group in data_groups:
        for key, value in group['data'].items():
            if key not in merged_data:
                merged_data[key] = value
            else:
                merged_data[key] = sc.concat([merged_data[key], value], dim=merged_data[key].dims[0])

        for key, value in group['coords'].items():
            if key not in merged_coords:
                merged_coords[key] = value
            else:
                merged_coords[key] = sc.concat([merged_coords[key], value], dim=merged_coords[key].dims[0])

        if 'attrs' not in group:
            continue
        for key, value in group['attrs'].items():
            if key not in merged_attrs:
                merged_attrs[key] = value
            else:
                merged_attrs[key] = {**merged_attrs[key], **value}

    return sc.DataGroup(data=merged_data, coords=merged_coords, attrs=merged_attrs)
