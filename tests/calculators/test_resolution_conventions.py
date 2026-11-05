# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute checks on the width convention each wrapper hands to its backend.

``ResolutionFunction.smearing()`` returns sigma for every resolution type; each
wrapper is then responsible for converting to what its backend expects:

* refnx  -- ``x_err`` is the **FWHM** at each q (a scalar ``x_err`` is instead a
  constant dQ/Q FWHM *percentage*).
* refl1d -- ``QProbe.dQ`` is **sigma**.

The cross-engine tests in ``tests/integration/test_cross_engine_resolution.py``
only pin the engines against each other, so they cannot catch an error applied
consistently to both.  These tests intercept the value at each engine boundary
and assert the exact numbers, which pins the convention absolutely.  In
particular this is the only absolute check on the refl1d resolution path.

See GitHub issue #367 for background.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from refl1d import names
from refnx import reflect

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

Q = np.linspace(0.01, 0.3, 20)

Q_KNOTS = np.linspace(0.001, 0.5, 10)
FWHM_KNOTS = 0.02 * Q_KNOTS + 0.001

QZ = np.linspace(0.001, 0.5, 50)
SIGMA_POINTS = 0.01 * QZ + 0.0005
SQZ = SIGMA_POINTS**2


def _build_refnx():
    """A film on a substrate on a refnx wrapper; the model is returned too, the wrapper holds it weakly."""
    wrapper = RefnxWrapper()
    film = Layer(Material(3.45, 0.0, 'Film'), 100.0, 3.0, 'FilmLayer')
    substrate = Layer(Material(2.07, 0.0, 'Substrate'), 0.0, 0.0, 'SubstrateLayer')
    model = Model(Sample(Multilayer([film, substrate], 'Item')), 1.0, 0.0, name='MyModel')
    wrapper.register_model(model)
    return wrapper, model


def _build_refl1d():
    wrapper = Refl1dWrapper()
    wrapper.reset_storage()
    wrapper.create_material('Substrate')
    wrapper.update_material('Substrate', rho=2.07, irho=0.0)
    wrapper.create_material('Film')
    wrapper.update_material('Film', rho=3.45, irho=0.0)
    wrapper.create_layer('SubstrateLayer')
    wrapper.assign_material_to_layer('Substrate', 'SubstrateLayer')
    wrapper.create_layer('FilmLayer')
    wrapper.assign_material_to_layer('Film', 'FilmLayer')
    wrapper.update_layer('FilmLayer', thickness=100.0, interface=3.0)
    wrapper.create_item('Item')
    wrapper.add_layer_to_item('FilmLayer', 'Item')
    wrapper.add_layer_to_item('SubstrateLayer', 'Item')
    wrapper.create_model('MyModel')
    wrapper.add_item('Item', 'MyModel')
    wrapper.update_model('MyModel', bkg=0.0)
    return wrapper


def _capture_refnx_x_err(monkeypatch, resolution_function):
    """Run RefnxWrapper.calculate and return the x_err handed to refnx."""
    captured = {}
    real_call = reflect.ReflectModel.__call__

    def spy(self, x, p=None, x_err=None):
        captured['x_err'] = x_err
        return real_call(self, x, p=p, x_err=x_err)

    monkeypatch.setattr(reflect.ReflectModel, '__call__', spy)

    wrapper, model = _build_refnx()
    wrapper.set_resolution_function(resolution_function)
    wrapper.calculate(Q, model.unique_name)
    return captured['x_err']


def _capture_refl1d_dq(monkeypatch, resolution_function):
    """Run Refl1dWrapper.calculate and return the dQ handed to refl1d's QProbe."""
    captured = {}
    real_qprobe = names.QProbe

    def spy(**kwargs):
        captured['dQ'] = np.asarray(kwargs['dQ'], dtype=float)
        return real_qprobe(**kwargs)

    monkeypatch.setattr(names, 'QProbe', spy)

    wrapper = _build_refl1d()
    wrapper.set_resolution_function(resolution_function)
    wrapper.calculate(Q, 'MyModel')
    return captured['dQ']


# ----- the constant itself -----


@pytest.mark.fast
def test_sigma_to_fwhm_is_the_gaussian_ratio():
    """Pin SIGMA_TO_FWHM against a literal.

    Every other test in this module imports SIGMA_TO_FWHM -- the same constant
    the production code uses -- so a wrong value would cancel out on both sides
    of the assertion and stay invisible.  This is the one place the constant is
    checked against an external fact: the FWHM/sigma ratio of a Gaussian,
    2*sqrt(2*ln2).
    """
    assert SIGMA_TO_FWHM == pytest.approx(2.3548200450309493)


# ----- refnx expects FWHM -----


@pytest.mark.fast
def test_refnx_receives_fwhm_for_linear_spline(monkeypatch):
    x_err = _capture_refnx_x_err(monkeypatch, LinearSpline(Q_KNOTS, FWHM_KNOTS))

    expected_fwhm = np.interp(Q, Q_KNOTS, FWHM_KNOTS)
    assert_allclose(x_err, expected_fwhm)


@pytest.mark.fast
def test_refnx_receives_fwhm_for_pointwise(monkeypatch):
    x_err = _capture_refnx_x_err(monkeypatch, Pointwise([QZ, np.ones_like(QZ), SQZ]))

    expected_sigma = np.interp(Q, QZ, np.sqrt(SQZ))
    assert_allclose(x_err, expected_sigma * SIGMA_TO_FWHM)


@pytest.mark.fast
def test_refnx_receives_scalar_percentage_for_percentage_fwhm(monkeypatch):
    x_err = _capture_refnx_x_err(monkeypatch, PercentageFwhm(5.0))

    # refnx reads a scalar x_err as a constant dQ/Q FWHM percentage, so the
    # percentage is passed through verbatim -- not converted to a width.
    assert np.isscalar(x_err) or np.ndim(x_err) == 0
    assert_allclose(x_err, 5.0)


# ----- refl1d expects sigma -----


@pytest.mark.fast
def test_refl1d_receives_sigma_for_linear_spline(monkeypatch):
    dq = _capture_refl1d_dq(monkeypatch, LinearSpline(Q_KNOTS, FWHM_KNOTS))

    expected_fwhm = np.interp(Q, Q_KNOTS, FWHM_KNOTS)
    assert_allclose(dq, expected_fwhm / SIGMA_TO_FWHM)


@pytest.mark.fast
def test_refl1d_receives_sigma_for_pointwise(monkeypatch):
    dq = _capture_refl1d_dq(monkeypatch, Pointwise([QZ, np.ones_like(QZ), SQZ]))

    expected_sigma = np.interp(Q, QZ, np.sqrt(SQZ))
    assert_allclose(dq, expected_sigma)


@pytest.mark.fast
def test_refl1d_receives_sigma_for_percentage_fwhm(monkeypatch):
    dq = _capture_refl1d_dq(monkeypatch, PercentageFwhm(5.0))

    expected_sigma = (5.0 / 100.0) * Q / SIGMA_TO_FWHM
    assert_allclose(dq, expected_sigma)


# ----- the two backends must receive widths that differ by exactly SIGMA_TO_FWHM -----


@pytest.mark.fast
def test_engines_receive_widths_differing_by_sigma_to_fwhm(monkeypatch):
    """The whole point of issue #367, stated directly.

    Catches a common-mode error that the cross-engine reflectivity comparison
    cannot see: whatever the widths are, refnx's must be exactly SIGMA_TO_FWHM
    times refl1d's.
    """
    x_err = _capture_refnx_x_err(monkeypatch, LinearSpline(Q_KNOTS, FWHM_KNOTS))
    dq = _capture_refl1d_dq(monkeypatch, LinearSpline(Q_KNOTS, FWHM_KNOTS))

    assert_allclose(x_err, dq * SIGMA_TO_FWHM)
