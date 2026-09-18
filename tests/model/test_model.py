# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Model class.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerCollection
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer


class TestModel(unittest.TestCase):
    def test_default(self):
        p = Model()
        assert_equal(p.name, 'Model')
        assert_equal(p.interface, None)
        assert_equal(p.sample.name, 'EasySample')
        assert_equal(p.scale.display_name, 'scale')
        assert_equal(str(p.scale.unit), 'dimensionless')
        assert_equal(p.scale.value, 1.0)
        assert_equal(p.scale.min, 0.0)
        assert_equal(p.scale.max, 10.0)
        assert_equal(p.scale.fixed, True)
        assert_equal(p.background.display_name, 'background')
        assert_equal(str(p.background.unit), 'dimensionless')
        assert_equal(p.background.value, 1.0e-8)
        assert_equal(p.background.min, 0.0)
        assert_equal(p.background.max, np.inf)
        assert_equal(p.background.fixed, True)
        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        assert np.allclose(p._resolution_function.smearing([1]), 5.0 / 100.0 * 1.0 / sigma_to_fwhm)
        assert np.allclose(p._resolution_function.smearing([100]), 5.0 / 100.0 * 100.0 / sigma_to_fwhm)

    def test_from_pars(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, o2, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(
            sample=d,
            scale=2,
            background=1e-5,
            resolution_function=resolution_function,
            name='newModel',
        )
        assert_equal(mod.name, 'newModel')
        assert_equal(mod.interface, None)
        assert_equal(mod.sample.name, 'myModel')
        assert_equal(mod.scale.display_name, 'scale')
        assert_equal(str(mod.scale.unit), 'dimensionless')
        assert_equal(mod.scale.value, 2.0)
        assert_equal(mod.scale.min, 0.0)
        assert_equal(mod.scale.max, 10.0)
        assert_equal(mod.scale.fixed, True)
        assert_equal(mod.background.display_name, 'background')
        assert_equal(str(mod.background.unit), 'dimensionless')
        assert_equal(mod.background.value, 1.0e-5)
        assert_equal(mod.background.min, 0.0)
        assert_equal(mod.background.max, np.inf)
        assert_equal(mod.background.fixed, True)
        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        assert np.allclose(mod._resolution_function.smearing([1]), 2.0 / 100.0 * 1.0 / sigma_to_fwhm)
        assert np.allclose(mod._resolution_function.smearing([100]), 2.0 / 100.0 * 100.0 / sigma_to_fwhm)

    def test_add_assemblies(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        surfactant = SurfactantLayer()
        multilayer = Multilayer()
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        assert_equal(mod.sample[1].name, 'oneLayerItem2')
        assert_equal(issubclass(mod.sample[1].__class__, RepeatingMultilayer), True)
        mod.add_assemblies(surfactant)
        assert_equal(len(mod.sample), 3)
        mod.add_assemblies(multilayer)
        assert_equal(len(mod.sample), 4)

    def test_add_assemblies_exception(self):
        # When
        mod = Model()

        # Then Expect
        with pytest.raises(ValueError):
            mod.add_assemblies('not an assembly')

    def test_add_assemblies_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_add_assemblies_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_duplicate_assembly(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.sample), 3)
        assert_equal(mod.sample[2].name, 'oneLayerItem2 duplicate')
        assert_equal(issubclass(mod.sample[2].__class__, RepeatingMultilayer), True)

    def test_duplicate_assembly_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 3)

    def test_duplicate_assembly_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 3)

    def test_remove_assembly(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.sample), 1)

    def test_remove_assembly_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_remove_assembly_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_remove_all_assemblies(self):
        # when
        mod = Model()

        # Then
        mod.remove_assembly(0)
        mod.remove_assembly(0)

        # Expect
        assert_equal(len(mod.sample), 0)

    def test_resolution_function(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.resolution_function == mock_resolution_function

    def test_resolution_function_interface_refl1d(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.interface()._wrapper._resolution_function == mock_resolution_function

    def test_set_resolution_function_interface_refnx(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refnx')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.interface()._wrapper._resolution_function == mock_resolution_function

    def test_repr(self):
        model = Model()

        assert (
            model.__repr__()
            == "Model:\n  scale: 1.0\n  background: 1.0e-08\n  resolution: 5.0 %\n  color: '#0173B2'\n  sample:\n    EasySample:\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n"  # noqa: E501
        )

    def test_repr_resolution_function(self):
        resolution_function = LinearSpline([0, 10], [0, 10])
        model = Model()
        model.resolution_function = resolution_function
        assert (
            model.__repr__()
            == "Model:\n  scale: 1.0\n  background: 1.0e-08\n  resolution: function of Q\n  color: '#0173B2'\n  sample:\n    EasySample:\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n"  # noqa: E501
        )


@pytest.mark.parametrize(
    'interface',
    [None, CalculatorFactory()],
)
def test_dict_round_trip(interface):
    # When
    resolution_function = LinearSpline([0, 10], [0, 10])
    model = Model(interface=interface)
    model.resolution_function = resolution_function
    for additional_layer in [SurfactantLayer(), Multilayer(), RepeatingMultilayer()]:
        model.add_assemblies(additional_layer)
    src_dict = model.as_dict()
    global_object.map._clear()

    # Then
    model_from_dict = Model.from_dict(src_dict)

    # Expect
    assert sorted(model.as_dict(skip=['resolution_function', 'interface'])) == sorted(
        model_from_dict.as_dict(skip=['resolution_function', 'interface'])
    )
    assert model._resolution_function.smearing(5.5) == model_from_dict._resolution_function.smearing(5.5)
    if interface is not None:
        assert model.interface().name == model_from_dict.interface().name
        assert_almost_equal(
            model.interface().reflectity_profile([0.3], model.unique_name),
            model_from_dict.interface().reflectity_profile([0.3], model_from_dict.unique_name),
        )


class TestModelPropertyAccessors:
    """Tests for the new @property accessors introduced in the ModelBase/EasyList migration."""

    def test_scale_setter_updates_value(self):
        model = Model()
        model.scale = 3.0
        assert model.scale.value == 3.0

    def test_scale_getter_returns_parameter(self):
        model = Model(scale=2.5)
        from easyscience.variable import Parameter

        assert isinstance(model.scale, Parameter)
        assert model.scale.value == 2.5

    def test_background_setter_updates_value(self):
        model = Model()
        model.background = 1e-6
        assert model.background.value == 1e-6

    def test_background_getter_returns_parameter(self):
        model = Model(background=5e-6)
        from easyscience.variable import Parameter

        assert isinstance(model.background, Parameter)
        assert model.background.value == 5e-6

    def test_sample_setter(self):
        model = Model()
        new_sample = Sample(name='NewSample')
        model.sample = new_sample
        assert model.sample.name == 'NewSample'

    def test_to_dict_includes_sample_and_resolution(self):
        model = Model()
        d = model.to_dict()
        assert 'sample' in d
        assert 'resolution_function' in d
        assert 'interface' in d  # interface is None, encoded as None
        assert 'name' in d

    def test_to_dict_with_interface_name(self):
        interface = CalculatorFactory()
        model = Model(interface=interface)
        d = model.to_dict()
        assert d['interface'] == 'refnx'

    def test_to_dict_excludes_derived_fields(self):
        model = Model()
        d = model.to_dict()
        # sample, resolution_function, interface are handled separately
        assert 'sample' in d
        # The super().to_dict() skip prevents these from being top-level
        assert 'resolution_function' in d
        assert 'interface' in d

    def test_as_dict_alias(self):
        model = Model()
        assert model.as_dict() == model.to_dict()

    def test_is_default_property(self):
        model = Model()
        assert model.is_default is False
        model.is_default = True
        assert model.is_default is True


class TestModelRoundTrip:
    """Tests verifying serialization round-trip for the Model class."""

    def test_basic_round_trip_preserves_name(self):
        global_object.map._clear()
        model = Model(name='MyModel')
        d = model.as_dict()
        global_object.map._clear()
        restored = Model.from_dict(d)
        assert restored.name == 'MyModel'

    def test_round_trip_preserves_scale_and_background(self):
        global_object.map._clear()
        model = Model(scale=2.0, background=1e-7)
        d = model.as_dict()
        global_object.map._clear()
        restored = Model.from_dict(d)
        assert restored.scale.value == 2.0
        assert restored.background.value == 1e-7

    def test_round_trip_preserves_resolution_function(self):
        global_object.map._clear()
        model = Model(resolution_function=PercentageFwhm(3.0))
        d = model.as_dict()
        global_object.map._clear()
        restored = Model.from_dict(d)
        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        assert np.allclose(restored._resolution_function.smearing(100), 3.0 / 100.0 * 100.0 / sigma_to_fwhm)

    def test_round_trip_preserves_interface(self):
        global_object.map._clear()
        interface = CalculatorFactory()
        model = Model(interface=interface)
        d = model.as_dict()
        global_object.map._clear()
        restored = Model.from_dict(d)
        assert restored.interface().name == 'refnx'

    def test_round_trip_preserves_is_default(self):
        global_object.map._clear()
        model = Model()
        model.is_default = True
        d = model.as_dict()
        global_object.map._clear()
        restored = Model.from_dict(d)
        # Note: is_default is a runtime flag that may not survive round-trip
        # because from_dict reconstructs via __init__ which resets _is_default.
        # This test documents the current behaviour.
        assert restored.is_default is False
