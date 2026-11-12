# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Refl1d calculator.
"""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.calculators import CalculatorBase
from easyreflectometry.calculators.refl1d.calculator import Refl1d
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.001, 0.3, 10)
# Within 0.5 % of refnx for the same models (tests/calculators/refnx/test_refnx_calculator.py),
# which has no background; recorded once refl1d oversampling was applied.
FILM = [1.0000e00, 2.1701e-03, 1.1431e-04, 1.9336e-05, 4.9502e-06, 1.5447e-06, 5.4663e-07, 2.2701e-07, 1.2687e-07, 1.0188e-07]
REPEATED = [
    1.0000e00,
    1.7980e-05,
    1.2273e-04,
    2.4062e-06,
    6.7234e-06,
    8.3050e-07,
    1.1546e-06,
    4.1352e-07,
    3.5132e-07,
    2.5349e-07,
]


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _layers():
    ambient = Layer(Material(0.0, 0.0, 'Material1'), 0.0, 0.0, 'Layer1')
    film = Layer(Material(2.0, 0.0, 'Material2'), 10.0, 1.0, 'Layer2')
    substrate = Layer(Material(4.0, 0.0, 'Material3'), 0.0, 1.0, 'Layer3')
    return ambient, film, substrate


def _repeated_model() -> Model:
    ambient, film, substrate = _layers()
    sample = Sample(
        Multilayer(ambient, 'Item1'), RepeatingMultilayer([film, ambient], 10, 'Item2'), Multilayer(substrate, 'Item3')
    )
    return Model(sample, 1.0, 1e-7)


def _served(calculator: Refl1d, model: Model) -> Model:
    calculator.register_model(model)
    return model


def test_init():
    calculator = Refl1d()
    assert calculator.name == 'refl1d'
    assert Refl1d in CalculatorBase._calculators
    assert calculator.models == {}
    assert calculator.supports_magnetism


def test_reflectity_profile():
    calculator = Refl1d()
    model = _served(calculator, Model(Sample(Multilayer(list(_layers()), 'Item')), 1.0, 1e-7))
    assert_almost_equal(calculator.reflectity_profile(Q, model.unique_name), FILM, decimal=4)


def test_reflectity_profile_of_a_repeated_assembly():
    calculator = Refl1d()
    model = _served(calculator, _repeated_model())
    assert_almost_equal(calculator.reflectity_profile(Q, model.unique_name), REPEATED, decimal=4)


def test_magnetism_on_without_a_moment_gives_the_plain_reflectivity():
    calculator = Refl1d()
    calculator.include_magnetism = True
    model = _served(calculator, Model(Sample(Multilayer(list(_layers()), 'Item')), 1.0, 1e-7))
    assert_almost_equal(calculator.reflectity_profile(Q, model.unique_name), FILM, decimal=4)
    assert calculator.include_magnetism


def test_polarized_reflectivity_profiles():
    calculator = Refl1d()
    film = Layer(Material(4.0, 0.0, 'Material2'), 100.0, 0.0, 'Layer2', magnetism=LayerMagnetism(rho_m=2.0, theta_m=45.0))
    sample = Sample(
        Multilayer(
            [Layer(Material(0.0, 0.0, 'Material1'), 0.0, 0.0, 'Layer1'), film, Layer(Material(2.047, 0.0, 'Material3'))],
            'Item',
        )
    )
    model = _served(calculator, Model(sample, 1.0, 0.0))
    calculator.include_magnetism = True
    q = np.linspace(0.005, 0.3, 50)

    channels = calculator.polarized_reflectivity_profiles(q, model.unique_name)

    assert list(channels) == ['pp', 'pm', 'mp', 'mm']
    for reflectivity in channels.values():
        assert len(reflectivity) == len(q)
    # reflectity_profile follows the selected channel
    for key in ['pp', 'pm', 'mp', 'mm']:
        calculator.polarization_channel = key
        assert calculator.polarization_channel.value == key
        assert_almost_equal(calculator.reflectity_profile(q, model.unique_name), channels[key])


def test_sld_profile():
    calculator = Refl1d()
    model = _served(calculator, Model(Sample(Multilayer(list(_layers()), 'Item')), 1.0, 1e-7))
    assert_almost_equal(calculator.sld_profile(model.unique_name)[1][0], 0)
    assert_almost_equal(calculator.sld_profile(model.unique_name)[1][-1], 4)
