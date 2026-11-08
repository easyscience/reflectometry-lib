# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Both engines against the reflectivity/analysis validation suite.

Independent truth, kept apart from the migration baselines: these numbers
were not produced by this library (see ``tests/_static/PROVENANCE.md``).
"""

from pathlib import Path

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

TEST4 = np.loadtxt(Path(__file__).parents[1] / '_static' / 'validation_test4.dat')
ENGINES = ['refnx', 'refl1d']


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _model(engine, layers, resolution_function) -> Model:
    """One assembly per (sld, isld, thickness, roughness) row, fronting first."""
    sample = Sample(*[
        Multilayer(Layer(Material(sld, isld), thickness, roughness)) for sld, isld, thickness, roughness in layers
    ])
    model = Model(sample, 1.0, 0.0, resolution_function)
    factory = CalculatorFactory()
    factory.switch(engine)
    model.interface = factory
    return model


def _test0_model(engine, resolution_function) -> Model:
    return _model(
        engine,
        [(2.070, 0.0, 0.0, 0.0), (3.450, 0.1, 100.0, 3.0), (5.000, 0.01, 200.0, 1.0), (6.000, 0.0, 0.0, 5.0)],
        resolution_function,
    )


def _reflectivity(model, q):
    return model.interface.fit_func(q, model.unique_name)


@pytest.mark.parametrize('engine', ENGINES)
def test_unpolarised_test0(engine):
    model = _test0_model(engine, PercentageFwhm(0))
    q = np.array([5.0e-03, 3.7175e-02, 5.45e-02, 1.00535e-01, 2.95565e-01])
    expected = [
        9.665000503913141472e-01,
        3.486325360684768590e-04,
        8.540420179439664689e-05,
        5.815959818366009312e-06,
        4.999742968030015832e-08,
    ]
    assert_almost_equal(_reflectivity(model, q), expected)


@pytest.mark.parametrize('engine', ENGINES)
def test_unpolarised_test2_bare_interface(engine):
    model = _model(engine, [(0.0, 0.0, 0.0, 0.0), (6.360, 0.0, 0.0, 3.0)], PercentageFwhm(0))
    q = np.array([5.0e-03, 7.5645e-02, 1.43305e-01, 2.36835e-01, 5.9205e-01])
    expected = [
        1.000000000000000222e00,
        1.964576414578978456e-04,
        1.280698699505669096e-05,
        1.234290141526865827e-06,
        2.222536631965092181e-09,
    ]
    assert_almost_equal(_reflectivity(model, q), expected)


@pytest.mark.parametrize('engine', ENGINES)
def test_unpolarised_test4_constant_resolution(engine):
    model = _test0_model(engine, PercentageFwhm(5))
    assert_allclose(_reflectivity(model, TEST4[:, 0]), TEST4[:, 1], rtol=0.03)


@pytest.mark.parametrize('engine', ENGINES)
def test_unpolarised_test4_pointwise_resolution(engine):
    # The file carries sigma; the spline takes FWHM.
    model = _test0_model(engine, LinearSpline(TEST4[:, 0], SIGMA_TO_FWHM * TEST4[:, 3]))
    assert_allclose(_reflectivity(model, TEST4[:, 0]), TEST4[:, 1], rtol=0.03)
