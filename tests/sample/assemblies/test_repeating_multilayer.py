# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for RepeatingMultiLayer module
"""

import unittest

from easyscience import global_object
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.sample.assemblies.repeating_multilayer import RepeatingMultilayer
from easyreflectometry.sample.collections.layer_collection import LayerCollection
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.materials.material import Material


class TestRepeatingMultilayer(unittest.TestCase):
    def test_default(self):
        p = RepeatingMultilayer()
        assert_equal(p.name, 'EasyRepeatingMultilayer')
        assert_equal(p._type, 'Repeating Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(len(p.layers), 1)
        assert_equal(p.repetitions.display_name, 'repetitions')
        assert_equal(str(p.repetitions.unit), 'dimensionless')
        assert_equal(p.repetitions.value, 1.0)
        assert_equal(p.repetitions.min, 1)
        assert_equal(p.repetitions.max, 9999)
        assert_equal(p.repetitions.fixed, True)
        assert_equal(p.layers.name, 'EasyLayerCollection')

    def test_default_empty(self):
        p = RepeatingMultilayer(populate_if_none=False)
        assert_equal(p.name, 'EasyRepeatingMultilayer')
        assert_equal(p._type, 'Repeating Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(p.repetitions.display_name, 'repetitions')
        assert_equal(str(p.repetitions.unit), 'dimensionless')
        assert_equal(p.repetitions.value, 1.0)
        assert_equal(p.repetitions.min, 1)
        assert_equal(p.repetitions.max, 9999)
        assert_equal(p.repetitions.fixed, True)
        assert_equal(p.layers.name, 'EasyLayerCollection')

    def test_from_pars(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        layers = LayerCollection(p, q, name='twoLayer')
        o = RepeatingMultilayer(layers, 2.0, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o._type, 'Repeating Multi-layer')
        assert_equal(o.interface, None)
        assert_equal(o.repetitions.display_name, 'repetitions')
        assert_equal(str(o.repetitions.unit), 'dimensionless')
        assert_equal(o.repetitions.value, 2.0)
        assert_equal(o.repetitions.min, 1)
        assert_equal(o.repetitions.max, 9999)
        assert_equal(o.repetitions.fixed, True)
        assert_equal(o.layers.name, 'twoLayer')

    def test_from_pars_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.repetitions.display_name, 'repetitions')
        assert_equal(str(o.repetitions.unit), 'dimensionless')
        assert_equal(o.repetitions.value, 2.0)
        assert_equal(o.repetitions.min, 1)
        assert_equal(o.repetitions.max, 9999)
        assert_equal(o.repetitions.fixed, True)
        assert_equal(o.layers.name, 'thinBoron')

    def test_from_pars_layer_list(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 15.0, 2.0, 'layerPotassium')
        o = RepeatingMultilayer([p, q], 10, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'thinBoron/layerPotassium')
        assert_equal(o.repetitions.value, 10.0)
        assert_equal(o.repetitions.min, 1)
        assert_equal(o.repetitions.max, 9999)

    def test_add_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')

    def test_add_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[1].thick.value, 50.0)

    def test_duplicate_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        o.duplicate_layer(1)
        assert_equal(len(o.layers), 3)
        assert_equal(o.layers[1].name, 'thickPotassium')
        assert_equal(o.layers[2].name, 'thickPotassium duplicate')

    def test_duplicate_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[1].thick.value, 50.0)
        o.duplicate_layer(1)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 3)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[2].thick.value, 50.0)
        assert_raises(
            AssertionError,
            assert_equal,
            o.interface()._wrapper.storage['item'][o.unique_name].components[1].name,
            o.interface()._wrapper.storage['item'][o.unique_name].components[2].name,
        )

    def test_remove_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = RepeatingMultilayer(p, 2.0, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')
        o.remove_layer(1)
        assert_equal(len(o.layers), 1)
        assert_equal(o.layers[0].name, 'thinBoron')

    def test_remove_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = RepeatingMultilayer(p, repetitions=2.0, name='twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')
        o.remove_layer(1)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        assert_equal(o.layers[0].name, 'thinBoron')

    def test_repr(self):
        p = RepeatingMultilayer(populate_if_none=True)
        assert (
            p.__repr__()
            == 'EasyRepeatingMultilayer:\n  EasyLayerCollection:\n  - EasyLayer:\n      material:\n        EasyMaterial:\n          sld: 4.186e-6 1/Å^2\n          isld: 0.000e-6 1/Å^2\n      thickness: 10.000 Å\n      roughness: 3.300 Å\n  repetitions: 1.0\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        p = RepeatingMultilayer(populate_if_none=True)
        p_dict = p.as_dict()
        global_object.map._clear()

        q = RepeatingMultilayer.from_dict(p_dict)
        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_conformal_flags_survive_dict_round_trip(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        layers = [Layer(m, 5.0, 2.0, 'thinBoron'), Layer(k, 50.0, 1.0, 'thickPotassium')]
        p = RepeatingMultilayer(layers, 8, conformal_roughness=True)
        p_dict = p.as_dict()
        assert p_dict['conformal_roughness'] is True
        global_object.map._clear()

        q = RepeatingMultilayer.from_dict(p_dict)
        assert q.conformal_roughness is True
        assert q.layers[1].roughness.independent is False
        assert q.repetitions.value == 8.0
