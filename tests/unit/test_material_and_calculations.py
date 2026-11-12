# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for MaterialMixture derived sld/isld, special calculations,
and default parameter limits."""

import numpy as np
import pytest
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.limits import apply_default_limits
from easyreflectometry.sample import Material
from easyreflectometry.sample.elements.materials.material_mixture import MaterialMixture
from easyreflectometry.special.calculations import molecular_weight
from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.special.calculations import weighted_average


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


class TestMaterialMixtureSld:
    def test_sld_and_isld_are_floats_from_weighted_average(self):
        material_a = Material(name='A', sld=2.0, isld=0.5)
        material_b = Material(name='B', sld=6.0, isld=1.5)
        mixture = MaterialMixture(material_a=material_a, material_b=material_b, fraction=0.25)

        assert isinstance(mixture.sld, float)
        assert isinstance(mixture.isld, float)
        assert mixture.sld == pytest.approx(weighted_average(2.0, 6.0, 0.25))
        assert mixture.isld == pytest.approx(weighted_average(0.5, 1.5, 0.25))

    def test_sld_and_isld_follow_fraction_changes(self):
        material_a = Material(name='A', sld=2.0, isld=0.0)
        material_b = Material(name='B', sld=6.0, isld=1.0)
        mixture = MaterialMixture(material_a=material_a, material_b=material_b, fraction=0.25)

        mixture.fraction = 0.75

        assert mixture.sld == pytest.approx(5.0)
        assert mixture.isld == pytest.approx(0.75)


class TestNeutronScatteringLength:
    def test_element_without_absorption_has_zero_imaginary_part(self):
        result = neutron_scattering_length('Si')
        assert result.real == pytest.approx(4.1507e-05, rel=1e-3)
        assert result.imag == 0.0

    def test_element_with_absorption_has_negative_imaginary_part(self):
        # Boron has a non-zero imaginary bound coherent scattering length (b_c_i)
        result = neutron_scattering_length('B')
        assert result.real == pytest.approx(5.3e-05, rel=1e-3)
        assert result.imag == pytest.approx(-2.1e-06, rel=1e-3)

    def test_formula_scales_with_stoichiometry(self):
        single = neutron_scattering_length('B')
        double = neutron_scattering_length('B2')
        assert double.real == pytest.approx(2 * single.real)
        assert double.imag == pytest.approx(2 * single.imag)


class TestMolecularWeight:
    def test_molecular_weight_of_water(self):
        assert molecular_weight('H2O') == pytest.approx(18.015, rel=1e-3)


class TestApplyDefaultLimits:
    def test_percentage_limits_set_for_infinite_bounds(self):
        param = Parameter(10.0, display_name='thickness', min=-np.inf, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == pytest.approx(5.0)
        assert param.max == pytest.approx(20.0)

    def test_percentage_limits_leave_finite_bounds_untouched(self):
        param = Parameter(10.0, display_name='roughness', min=2.0, max=30.0)
        apply_default_limits(param, 'roughness')
        assert param.min == 2.0
        assert param.max == 30.0

    def test_percentage_limits_skip_zero_value(self):
        param = Parameter(0.0, display_name='thickness', min=-np.inf, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert np.isinf(param.min)
        assert np.isinf(param.max)

    def test_sld_gets_fixed_limits(self):
        param = Parameter(4.0, display_name='sld', min=-np.inf, max=np.inf)
        apply_default_limits(param, 'sld')
        assert param.min == -1.0
        assert param.max == 10.0
