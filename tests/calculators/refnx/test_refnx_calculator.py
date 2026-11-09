# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Refnx calculator.
"""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.calculators import CalculatorBase
from easyreflectometry.calculators.refnx.calculator import Refnx
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.001, 0.3, 10)


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


def test_init():
    calculator = Refnx()
    assert calculator.name == 'refnx'
    assert Refnx in CalculatorBase._calculators
    assert calculator.models == {}


def test_reflectity_profile():
    model = Model(Sample(Multilayer(list(_layers()), 'Item')), 1.0, 0.0)
    calculator = Refnx()
    calculator.register_model(model)
    expected = [
        9.99956517e-01,
        2.16286891e-03,
        1.14086254e-04,
        1.93031759e-05,
        4.94188894e-06,
        1.54191953e-06,
        5.45592112e-07,
        2.26619392e-07,
        1.26726993e-07,
        1.01842852e-07,
    ]
    assert_almost_equal(calculator.reflectity_profile(Q, model.unique_name), expected)


def test_reflectity_profile_of_a_repeated_assembly():
    ambient, film, substrate = _layers()
    sample = Sample(
        Multilayer(ambient, 'Item1'), RepeatingMultilayer([film, ambient], 10, 'Item2'), Multilayer(substrate, 'Item3')
    )
    model = Model(sample, 1.0, 0.0)
    calculator = Refnx()
    calculator.register_model(model)
    expected = [
        9.9995652e-01,
        1.7096697e-05,
        1.2253047e-04,
        2.4026928e-06,
        6.7117546e-06,
        8.3209877e-07,
        1.1512901e-06,
        4.1468151e-07,
        3.4981523e-07,
        2.5424356e-07,
    ]
    assert_almost_equal(calculator.reflectity_profile(Q, model.unique_name), expected)


def test_sld_profile():
    model = Model(Sample(Multilayer(list(_layers()), 'Item')), 1.0, 0.0)
    calculator = Refnx()
    calculator.register_model(model)
    assert_almost_equal(calculator.sld_profile(model.unique_name)[1][0], 0)
    assert_almost_equal(calculator.sld_profile(model.unique_name)[1][-1], 4)


def test_magnetism_is_refused():
    with pytest.raises(NotImplementedError):
        Refnx().include_magnetism = True
