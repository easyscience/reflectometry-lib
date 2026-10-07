# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the volume fraction plot helper."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from easyreflectometry.plot import plot_volume_fraction  # noqa: E402
from easyreflectometry.sample import VolumeFractionProfile  # noqa: E402


@pytest.fixture
def profile():
    z = np.linspace(-10, 30, 41)
    phi_a = 0.5 * (1 + np.tanh(-z / 3))
    return VolumeFractionProfile(z=z, fractions={'a': phi_a, 'b': 1 - phi_a}, labels={'a': 'Air', 'b': 'D2O'})


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_area_style_draws_a_line_and_a_fill_per_component(profile):
    ax = plot_volume_fraction(profile)
    assert len(ax.lines) == 2
    assert len([c for c in ax.collections if isinstance(c, PolyCollection)]) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Air', 'D2O']
    assert ax.get_ylim() == pytest.approx((-0.02, 1.05))
    assert ax.get_xlabel() == 'z (Å)'


def test_line_style_and_colours(profile):
    _, ax = plt.subplots()
    returned = plot_volume_fraction(profile, ax, style='line', colors={'Air': 'red', 'b': 'blue'})
    assert returned is ax
    assert len(ax.lines) == 2
    assert not ax.collections
    assert ax.lines[0].get_color() == 'red'
    assert ax.lines[1].get_color() == 'blue'


def test_unknown_style_raises(profile):
    with pytest.raises(ValueError, match='Unknown style'):
        plot_volume_fraction(profile, style='stack')


def test_negative_fraction_is_visible():
    z = np.linspace(-10, 30, 41)
    dip = np.where((z > -5) & (z < 0), -0.1, 0.0)
    profile = VolumeFractionProfile(z=z, fractions={'f': dip, 'r': 1 - dip}, labels={'f': 'film', 'r': 'rest'})
    ax = plot_volume_fraction(profile)
    assert ax.get_ylim()[0] < -0.1
