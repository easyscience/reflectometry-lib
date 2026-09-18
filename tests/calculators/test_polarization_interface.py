# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Model/factory-level tests for polarization channel selection.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


def _magnetic_model() -> Model:
    vacuum = Material(sld=0, isld=0, name='Vacuum')
    material = Material(sld=4.0, isld=0, name='Sld 4')
    si = Material(sld=2.047, isld=0, name='Si')
    superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
    layer = Layer(material=material, thickness=100, roughness=0, name='Sld 4 Layer')
    subphase = Layer(material=si, thickness=0, roughness=0, name='Si Subphase')
    sample = Sample(Multilayer(superphase), Multilayer(layer), Multilayer(subphase), name='Sample')
    model = Model(sample=sample, scale=1, background=0, name='Magnetic Model')
    model.resolution_function = PercentageFwhm(0)
    return model


Q = np.linspace(0.005, 0.3, 50)


def test_polarized_reflectivity_profiles_through_factory():
    model = _magnetic_model()
    interface = CalculatorFactory()
    interface.switch('refl1d')
    model.interface = interface
    calculator = model.interface()
    calculator.include_magnetism = True
    layer_name = list(calculator._wrapper.storage['layer'].keys())[1]
    calculator._wrapper.update_layer(layer_name, magnetism_rhoM=2, magnetism_thetaM=45)

    # Through the factory (model.interface), not only the calculator (model.interface())
    channels = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)

    assert list(channels.keys()) == ['pp', 'pm', 'mp', 'mm']
    for reflectivity in channels.values():
        assert len(reflectivity) == len(Q)

    # reflectity_profile (the fitting path) follows the selected channel
    calculator.polarization_channel = 'mm'
    assert_allclose(model.interface().reflectity_profile(Q, model.unique_name), channels['mm'], rtol=1e-10)


def test_switch_resets_polarization_state():
    model = _magnetic_model()
    interface = CalculatorFactory()
    interface.switch('refl1d')
    model.interface = interface
    calculator = model.interface()
    calculator.include_magnetism = True
    calculator.polarization_channel = 'mm'

    # switch() constructs a fresh calculator instance: channel and magnetism reset
    interface.switch('refl1d')

    calculator = model.interface()
    assert calculator.polarization_channel is PolarizationChannel.PP
    assert calculator.include_magnetism is False


def test_magnetic_sld_profile_through_factory():
    model = _magnetic_model()
    interface = CalculatorFactory()
    interface.switch('refl1d')
    model.interface = interface
    calculator = model.interface()
    calculator.include_magnetism = True
    layer_name = list(calculator._wrapper.storage['layer'].keys())[1]
    calculator._wrapper.update_layer(layer_name, magnetism_rhoM=2, magnetism_thetaM=45)

    z, sld, sld_magnetic, theta_magnetic = interface.magnetic_sld_profile(model.unique_name)

    assert len(z) == len(sld) == len(sld_magnetic) == len(theta_magnetic)
    # inside the 100 angstrom magnetic layer (zero roughness, so plateaus are exact)
    inside = (z > 25) & (z < 75)
    assert_allclose(sld[inside], 4.0)
    assert_allclose(sld_magnetic[inside], 2.0)
    assert_allclose(theta_magnetic[inside], 45.0)


def test_magnetic_sld_profile_requires_magnetism():
    model = _magnetic_model()
    interface = CalculatorFactory()
    interface.switch('refl1d')
    model.interface = interface

    with pytest.raises(ValueError):
        interface.magnetic_sld_profile(model.unique_name)


def test_refnx_magnetic_sld_profile_raises():
    model = _magnetic_model()
    interface = CalculatorFactory()
    interface.switch('refnx')
    model.interface = interface

    with pytest.raises(NotImplementedError):
        interface.magnetic_sld_profile(model.unique_name)


def test_refnx_include_magnetism_raises():
    interface = CalculatorFactory()
    interface.switch('refnx')

    with pytest.raises(NotImplementedError):
        interface().include_magnetism = True
    assert interface().include_magnetism is False
