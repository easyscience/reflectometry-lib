# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from matplotlib.gridspec import GridSpec

from easyreflectometry.sample.volume_fraction import VolumeFractionProfile

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

VOLUME_FRACTION_STYLES = ('area', 'line')


def plot_volume_fraction(
    profile: VolumeFractionProfile,
    ax: Optional[plt.Axes] = None,
    *,
    style: str = 'area',
    colors: Optional[Mapping[str, str]] = None,
) -> plt.Axes:
    """Draw a volume fraction (occupancy) profile, one trace per component.

    Parameters
    ----------
    profile : VolumeFractionProfile
        The profile, from :meth:`easyreflectometry.sample.Sample.volume_fraction_profile`
        or :func:`easyreflectometry.sample.volume_fraction_profile`.
    ax : plt.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    style : str, optional
        ``'area'`` (default) draws each component as a line with a translucent
        fill down to zero, the traces overlapping rather than stacked; ``'line'``
        draws lines only.
    colors : Mapping[str, str], optional
        Colour per component key or label; the matplotlib colour cycle is used
        for the rest.

    Returns
    -------
    plt.Axes
        The axes drawn on. The lower y-limit drops below zero when a fraction
        does, so an inadmissible profile is visible rather than clipped.

    Raises
    ------
    ValueError
        ``style`` is not one of :data:`VOLUME_FRACTION_STYLES`.
    """
    if style not in VOLUME_FRACTION_STYLES:
        raise ValueError(f'Unknown style {style!r}; choose one of {VOLUME_FRACTION_STYLES}.')
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    colors = colors or {}
    for i, (key, values) in enumerate(profile.fractions.items()):
        label = profile.labels.get(key, key)
        color = colors.get(key, colors.get(label, color_cycle[i % len(color_cycle)]))
        ax.plot(profile.z, values, linestyle='-', color=color, label=label)
        if style == 'area':
            ax.fill_between(profile.z, 0.0, values, color=color, alpha=0.3, linewidth=0)
    lowest = min((float(np.min(values)) for values in profile.fractions.values()), default=0.0)
    ax.set_ylim(min(0.0, lowest) - 0.02, 1.05)
    ax.set_xlabel('z (Å)')
    ax.set_ylabel('Volume fraction')
    if profile.fractions:
        ax.legend()
    return ax


def plot(data: sc.DataGroup) -> None:
    """A general plotting function for easyreflectometry.

    Parameters
    ----------
    data : sc.DataGroup
        The DataGroup to be plotted.
    """
    if len([i for i in list(data.keys()) if 'SLD' in i]) == 0:
        plot_sld = False
        fig = plt.figure(figsize=(5, 3))
        gs = GridSpec(1, 1, figure=fig)
    else:
        plot_sld = True
        fig = plt.figure(figsize=(5, 6))
        gs = GridSpec(2, 1, figure=fig)
        ax2 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[0, 0])
    refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
    for i, refl_num in enumerate(refl_nums):
        plot_data = sc.DataArray(
            name=f'R_{refl_num}',
            data=data['data'][f'R_{refl_num}'].copy(),
            coords={f'Qz_{refl_num}': data['coords'][f'Qz_{refl_num}'].copy()},
        )
        plot_data.data *= sc.scalar(10.0**i, unit=plot_data.unit)
        plot_data.coords[f'Qz_{refl_num}'].variances = None
        sc.plot(plot_data, ax=ax1, norm='log', linestyle='', marker='.', color=color_cycle[i])
        try:
            plot_model_data = sc.DataArray(
                name=f'R_{refl_num}_model',
                data=data[f'R_{refl_num}_model'].copy(),
                coords={f'Qz_{refl_num}': data['coords'][f'Qz_{refl_num}'].copy()},
            )
            plot_model_data.data *= sc.scalar(10.0**i, unit=plot_model_data.unit)
            plot_model_data.coords[f'Qz_{refl_num}'].variances = None
            sc.plot(
                plot_model_data,
                ax=ax1,
                norm='log',
                linestyle='--',
                color=color_cycle[i],
                marker='',
            )
        except KeyError:
            pass
    ax1.autoscale(True)
    ax1.relim()
    ax1.autoscale_view()

    if plot_sld:
        for i, refl_num in enumerate(refl_nums):
            plot_sld_data = sc.DataArray(
                name=f'SLD_{refl_num}',
                data=data[f'SLD_{refl_num}'].copy(),
                coords={f'z_{refl_num}': data['coords'][f'z_{refl_num}'].copy()},
            )
            sc.plot(plot_sld_data, ax=ax2, linestyle='-', color=color_cycle[i], marker='')
        ax2.autoscale(True)
        ax2.relim()
        ax2.autoscale_view()
