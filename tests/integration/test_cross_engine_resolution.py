# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-engine consistency checks for resolution function width conventions.

The same model + resolution function should produce broadly the same
reflectivity on refnx and refl1d.  A width-convention error at one engine
boundary shows up as a systematic disagreement between them.

.. warning::

   These are **smoke tests, not the regression tests for issue #367.**  They
   compare the engines against each other, so they are blind to any error
   applied consistently to both, and -- measured, not assumed -- they are only
   sensitive enough to catch *one* of the two bugs #367 fixed:

   * ``LinearSpline``: the pre-fix refl1d code **over**-smeared by 2.355x,
     which moves the curve enough to be caught here (measured separation ~2x).
   * ``Pointwise``: the pre-fix refnx code **under**-smeared by 2.355x from an
     already-small width.  That barely moves the curve -- measured separation
     ~1.1x, against a baseline engine disagreement of the same size -- so **no
     tolerance can catch it here.**  The Pointwise test below is a consistency
     check only.

   The actual, exact regression tests for both conventions live in
   ``tests/calculators/test_resolution_conventions.py``, which intercepts the
   widths handed to each backend and asserts them to floating-point precision.
   Fix that file first if these ever conflict.

.. note::

   Reflectivity spans several decades and the engines' different resolution
   algorithms (refnx: pointwise convolution; refl1d: oversampling) disagree
   most at fringe minima, where R is tiny and *relative* differences explode.
   The comparison is therefore made on ``log10(R)``, and the tolerances are
   measured values with roughly 1.5x headroom rather than round numbers.

See GitHub issue #367 for background.
"""

import numpy as np
import pytest

from easyreflectometry.calculators.refl1d.wrapper import Refl1dWrapper
from easyreflectometry.calculators.refnx.wrapper import RefnxWrapper
from easyreflectometry.model import Model
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.model.resolution_functions import LinearSpline
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.model.resolution_functions import Pointwise
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.geomspace(0.005, 0.3, 100)


def _build_simple_model(wrapper):
    """Build an ambient | 100 A film | substrate model on *wrapper*.

    The ambient layer is not optional decoration: both engines treat the first
    layer as the semi-infinite superphase and ignore its thickness.  Without it
    the "film" becomes the ambient, leaving a bare interface with no Kiessig
    fringes -- and resolution smearing acts almost entirely on fringes, so the
    model would be insensitive to the very thing under test.
    """
    ambient = Layer(Material(0.0, 0.0, 'Ambient'), 0.0, 0.0, 'AmbientLayer')
    film = Layer(Material(3.45, 0.0, 'Film'), 100.0, 3.0, 'FilmLayer')
    substrate = Layer(Material(2.07, 0.0, 'Substrate'), 0.0, 3.0, 'SubstrateLayer')
    model = Model(Sample(Multilayer([ambient, film, substrate], 'Item')), 1.0, 0.0, name='MyModel')
    wrapper.register_model(model)
    # Returned: the wrapper holds its models weakly.
    return model


def _both_engines(resolution_function):
    """Return (refnx_reflectivity, refl1d_reflectivity) for one resolution."""
    reflectivities = []
    for wrapper in (RefnxWrapper(), Refl1dWrapper()):
        model = _build_simple_model(wrapper)
        wrapper.set_resolution_function(resolution_function)
        reflectivities.append(wrapper.calculate(Q, model.unique_name))
    return tuple(reflectivities)


def _assert_log_close(refnx_r, refl1d_r, atol):
    """Assert the engines agree to `atol` decades of R at every q."""
    deviation = np.abs(np.log10(refnx_r) - np.log10(refl1d_r))
    assert deviation.max() <= atol, (
        f'engines disagree by {deviation.max():.4f} decades '
        f'(factor {10 ** deviation.max():.2f}) at q={Q[np.argmax(deviation)]:.4f}, tolerance {atol}'
    )


@pytest.mark.fast
@pytest.mark.parametrize(('resolution_pct', 'atol'), [(1.0, 0.07), (5.0, 0.23), (10.0, 0.29)])
def test_percentage_fwhm_consistent_across_engines(resolution_pct, atol):
    """PercentageFwhm gives consistent results across engines.

    Measured disagreement grows with the width (0.041 / 0.149 / 0.191 decades
    at 1% / 5% / 10% dQ/Q), so the tolerance is parametrized with it rather
    than set to one blanket value.  This combination was correct both before
    and after issue #367; the test guards against regression.
    """
    refnx_r, refl1d_r = _both_engines(PercentageFwhm(resolution_pct))
    _assert_log_close(refnx_r, refl1d_r, atol=atol)


@pytest.mark.fast
def test_linear_spline_consistent_across_engines():
    """LinearSpline gives consistent results across engines.

    This one does earn its keep: pre-fix, refl1d read the FWHM knots as sigma
    and over-smeared by 2.355x.  Measured max |dlog10(R)|: 0.085 with the fix,
    0.182 without it, so atol=0.13 separates them with ~1.5x headroom either
    way.
    """
    q_knots = np.linspace(0.001, 0.5, 10)
    fwhm_knots = 0.02 * q_knots + 0.001

    refnx_r, refl1d_r = _both_engines(LinearSpline(q_knots, fwhm_knots))
    _assert_log_close(refnx_r, refl1d_r, atol=0.13)


@pytest.mark.fast
def test_pointwise_consistent_across_engines():
    """Pointwise (sigma from sQz) is consistent across engines.

    Consistency check only.  Pre-fix, refnx under-smeared these widths by
    2.355x, but measured max |dlog10(R)| is 0.092 pre-fix versus 0.085 with
    the fix -- indistinguishable, because under-smearing an already-small
    width barely moves the curve.  Do not add a tolerance here expecting it to
    catch that bug; ``tests/calculators/test_resolution_conventions.py`` is
    what actually pins it.

    The sQz values mirror the LinearSpline knots, so the applied smearing --
    and hence the measured agreement -- matches that test.
    """
    qz = np.linspace(0.001, 0.5, 50)
    r = np.ones_like(qz)  # only kept for serialization round-trips
    sigma = (0.02 * qz + 0.001) / SIGMA_TO_FWHM
    sqz = sigma**2

    refnx_r, refl1d_r = _both_engines(Pointwise([qz, r, sqz]))
    _assert_log_close(refnx_r, refl1d_r, atol=0.13)
