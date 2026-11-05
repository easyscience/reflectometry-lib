# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Layer class.
"""

import unittest

import numpy as np
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators.factory import CalculatorFactory
from easyreflectometry.calculators.refnx.wrapper import _slab
from easyreflectometry.sample.elements.layers.layer import DEFAULTS
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.materials.material import Material
from easyreflectometry.utils import get_as_parameter


class TestLayer(unittest.TestCase):
    def test_no_arguments(self):
        p = Layer()
        assert_equal(p.name, 'EasyLayer')
        assert_equal(p.interface, None)
        assert_equal(p.material.name, 'EasyMaterial')
        assert_equal(p.thickness.display_name, 'thickness')
        assert_equal(str(p.thickness.unit), 'Å')
        assert_equal(p.thickness.value, 10.0)
        assert_equal(p.thickness.min, 0.0)
        assert_equal(p.thickness.max, np.inf)
        assert_equal(p.thickness.fixed, True)
        assert_equal(p.roughness.display_name, 'roughness')
        assert_equal(str(p.roughness.unit), 'Å')
        assert_equal(p.roughness.value, 3.3)
        assert_equal(p.roughness.min, 0.0)
        assert_equal(p.roughness.max, np.inf)
        assert_equal(p.roughness.fixed, True)

    def test_shuffled_arguments(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(thickness=5.0, material=m, roughness=2.0, name='thinBoron')
        assert_equal(p.name, 'thinBoron')
        assert_equal(p.interface, None)
        assert_equal(p.material.name, 'Boron')
        assert_equal(p.thickness.display_name, 'thickness')
        assert_equal(str(p.thickness.unit), 'Å')
        assert_equal(p.thickness.value, 5.0)
        assert_equal(p.thickness.min, 0.0)
        assert_equal(p.thickness.max, np.inf)
        assert_equal(p.thickness.fixed, True)
        assert_equal(p.roughness.display_name, 'roughness')
        assert_equal(str(p.roughness.unit), 'Å')
        assert_equal(p.roughness.value, 2.0)
        assert_equal(p.roughness.min, 0.0)
        assert_equal(p.roughness.max, np.inf)
        assert_equal(p.roughness.fixed, True)

    def test_only_roughness_key(self):
        p = Layer(roughness=10.0)
        assert_equal(p.roughness.display_name, 'roughness')
        assert_equal(str(p.roughness.unit), 'Å')
        assert_equal(p.roughness.value, 10.0)
        assert_equal(p.roughness.min, 0.0)
        assert_equal(p.roughness.max, np.inf)
        assert_equal(p.roughness.fixed, True)

    def test_only_roughness_key_paramter(self):
        roughness = get_as_parameter('roughness', 10, DEFAULTS)
        roughness.min = -10.0
        p = Layer(roughness=roughness)
        assert_equal(p.roughness.value, 10.0)
        assert_equal(p.roughness.min, -10.0)

    def test_only_thickness_key(self):
        p = Layer(thickness=10.0)
        assert_equal(p.thickness.display_name, 'thickness')
        assert_equal(str(p.thickness.unit), 'Å')
        assert_equal(p.thickness.value, 10.0)
        assert_equal(p.thickness.min, 0.0)
        assert_equal(p.thickness.max, np.inf)
        assert_equal(p.thickness.fixed, True)

    def test_only_thickness_key_paramter(self):
        thickness = get_as_parameter('thickness', 10, DEFAULTS)
        thickness.min = -10.0
        p = Layer(thickness=thickness)
        assert_equal(p.thickness.value, 10.0)
        assert_equal(p.thickness.min, -10.0)

    def test_assign_material(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        k = Material(2.074, 0.0, 'Silicon')
        assert_almost_equal(p.material.sld.value, 6.908)
        assert_almost_equal(p.material.isld.value, -0.278)
        p.assign_material(k)
        assert_almost_equal(p.material.sld.value, 2.074)
        assert_almost_equal(p.material.isld.value, 0.0)

    def test_assign_material_with_interface_refnx(self):
        interface = CalculatorFactory()
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        k = Material(2.074, 0.0, 'Silicon', interface=interface)
        assert_almost_equal(_slab(p).sld.real.value, 6.908)
        assert_almost_equal(_slab(p).sld.imag.value, -0.278)
        p.assign_material(k)
        assert_almost_equal(_slab(p).sld.real.value, 2.074)
        assert_almost_equal(_slab(p).sld.imag.value, 0.0)

    def test_dict_repr(self):
        p = Layer()
        assert p._dict_repr == {
            'EasyLayer': {
                'material': {'EasyMaterial': {'isld': '0.000e-6 1/Å^2', 'sld': '4.186e-6 1/Å^2'}},
                'roughness': '3.300 Å',
                'thickness': '10.000 Å',
            }
        }

    def test_repr(self):
        p = Layer()
        assert (
            p.__repr__()
            == 'EasyLayer:\n  material:\n    EasyMaterial:\n      sld: 4.186e-6 1/Å^2\n      isld: 0.000e-6 1/Å^2\n  thickness: 10.000 Å\n  roughness: 3.300 Å\n'  # noqa: E501
        )  # noqa: E501

    def test_dict_round_trip(self):
        p = Layer()
        p_dict = p.as_dict()
        global_object.registry.clear()

        q = Layer.from_dict(p_dict)
        assert sorted(p.as_dict()) == sorted(q.as_dict())
