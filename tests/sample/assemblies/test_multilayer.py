# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for MultiLayer class module
"""

import unittest

from easyscience import global_object
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from easyreflectometry.calculators.factory import CalculatorFactory
from easyreflectometry.sample.assemblies.multilayer import Multilayer
from easyreflectometry.sample.collections.layer_collection import LayerCollection
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.materials.material import Material


class TestMultilayer(unittest.TestCase):
    def test_default(self):
        p = Multilayer()
        assert_equal(p.name, 'EasyMultilayer')
        assert_equal(p._type, 'Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(len(p.layers), 1)
        assert_equal(p.layers.name, 'EasyLayerCollection')

    def test_default_empty(self):
        p = Multilayer(populate_if_none=False)
        assert_equal(p.name, 'EasyMultilayer')
        assert_equal(p._type, 'Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(len(p.layers), 0)

    def test_from_pars(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        layers = LayerCollection(p, q, name='twoLayer')
        o = Multilayer(layers, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o._type, 'Multi-layer')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'twoLayer')

    def test_from_pars_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        o = Multilayer(p, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'thinBoron')

    def test_from_pars_layer_list(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 15.0, 2.0, 'layerPotassium')
        o = Multilayer([p, q], 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'thinBoron/layerPotassium')

    def test_add_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = Multilayer(p, 'twoLayerItem')
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
        o = Multilayer(p, 'twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[1].thick.value, 50.0)

    def test_duplicate_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = Multilayer(p, 'twoLayerItem')
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
        o = Multilayer(p, 'twoLayerItem', interface=interface)
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
        o = Multilayer(p, 'twoLayerItem')
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
        o = Multilayer(p, name='twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')
        o.remove_layer(1)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        assert_equal(o.layers[0].name, 'thinBoron')

    def test_repr(self):
        p = Multilayer()
        assert (
            p.__repr__()
            == 'EasyMultilayer:\n  EasyLayerCollection:\n  - EasyLayer:\n      material:\n        EasyMaterial:\n          sld: 4.186e-6 1/Å^2\n          isld: 0.000e-6 1/Å^2\n      thickness: 10.000 Å\n      roughness: 3.300 Å\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        p = Multilayer()
        p_dict = p.as_dict()
        global_object.map._clear()

        q = Multilayer.from_dict(p_dict)
        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def _two_layers(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        return [Layer(m, 5.0, 2.0, 'thinBoron'), Layer(k, 50.0, 1.0, 'thickPotassium')]

    def test_conformal_kwargs_apply_the_ties(self):
        o = Multilayer(self._two_layers(), conformal_roughness=True, conformal_thickness=True)
        assert o.conformal_roughness is True
        assert o.conformal_thickness is True
        assert o.layers[1].roughness.independent is False
        assert o.layers[1].thickness.independent is False

    def test_conformal_flags_survive_dict_round_trip(self):
        # The ties are raw parameter dependencies, which nothing serializes;
        # the assembly persists the flags and rebuilds the ties on load.
        o = Multilayer(self._two_layers(), conformal_roughness=True)
        o_dict = o.as_dict()
        assert o_dict['conformal_roughness'] is True
        assert o_dict['conformal_thickness'] is False
        global_object.map._clear()

        q = Multilayer.from_dict(o_dict)
        assert q.conformal_roughness is True
        assert q.layers[1].roughness.independent is False
        q.layers[0].roughness.value = 7.0
        assert q.layers[1].roughness.value == 7.0
        assert q.conformal_thickness is False
        assert q.layers[1].thickness.independent is True

    def test_conformal_toggled_after_construction_is_serialized(self):
        # Serialization reads the property (graph truth), not the constructor argument.
        o = Multilayer(self._two_layers())
        o.conformal_thickness = True
        assert o.as_dict()['conformal_thickness'] is True
        o.conformal_thickness = False
        assert o.as_dict()['conformal_thickness'] is False
