# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for GradientLayer class module
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal

import easyreflectometry.sample.assemblies.gradient_layer
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.sample.assemblies.gradient_layer import GradientLayer
from easyreflectometry.sample.assemblies.gradient_layer import _linear_gradient
from easyreflectometry.sample.assemblies.gradient_layer import _prepare_gradient_layers
from easyreflectometry.sample.elements.materials.material import Material


class TestGradientLayer:
    @pytest.fixture
    def gradient_layer(self) -> GradientLayer:
        global_object.map._clear()

        self.front = Material(10.0, -10.0, 'Material_1')
        self.back = Material(0.0, 0.0, 'Material_2')

        return GradientLayer(
            front_material=self.front,
            back_material=self.back,
            thickness=1.0,
            roughness=2.0,
            discretisation_elements=10,
            name='Test',
            interface=None,
        )

    def test_init(self, gradient_layer: GradientLayer) -> None:
        # When Then Expect
        assert len(gradient_layer.layers) == 10
        assert gradient_layer.name, 'Test'
        assert gradient_layer._type, 'Gradient-layer'
        assert gradient_layer.interface is None
        assert gradient_layer.thickness == 1.0
        assert gradient_layer.back_layer.thickness.value == 0.1

        # The discretisation spans both endpoints inclusively: the front layer
        # equals the front material and the back layer equals the back material.
        assert gradient_layer.front_layer.material.sld.value == 10.0
        assert gradient_layer.back_layer.material.sld.value == 0.0
        assert_almost_equal(gradient_layer.layers[5].material.sld.value, 10.0 + (5 / 9) * (0.0 - 10.0))
        assert gradient_layer.front_layer.material.isld.value == -10.0
        assert gradient_layer.back_layer.material.isld.value == 0.0
        assert_almost_equal(gradient_layer.layers[5].material.isld.value, -10.0 + (5 / 9) * (0.0 - -10.0))

    def test_default(self) -> None:
        # When Then
        result = GradientLayer(name='default-layer')

        # Expect
        assert result.name == 'default-layer'
        assert result._type, 'Gradient-layer'
        assert result.interface is None
        assert len(result.layers) == 10

    def test_from_pars(self) -> None:
        # When
        front = Material(6.908, -0.278, 'Boron')
        back = Material(0.487, 0.000, 'Potassium')

        # Then
        result = GradientLayer(
            front_material=front,
            back_material=back,
            thickness=10.0,
            roughness=1.0,
            discretisation_elements=5,
            name='gradientItem',
        )

        # Expect
        assert result.name, 'gradientItem'
        assert result._type, 'Gradient-layer'
        assert result.interface is None
        assert len(result.layers) == 5

    def test_repr(self, gradient_layer: GradientLayer) -> None:
        # When Then Expect
        expected_str = "thickness: 1.0\ndiscretisation_elements: 10\nback_layer:\n  '9':\n    material:\n      EasyMaterial:\n        sld: 0.000e-6 1/Å^2\n        isld: 0.000e-6 1/Å^2\n    thickness: 0.100 Å\n    roughness: 2.000 Å\nfront_layer:\n  '0':\n    material:\n      EasyMaterial:\n        sld: 10.000e-6 1/Å^2\n        isld: -10.000e-6 1/Å^2\n    thickness: 0.100 Å\n    roughness: 2.000 Å\n"  # noqa: E501
        assert gradient_layer.__repr__() == expected_str

    def test_dict_round_trip(self) -> None:
        # When
        p = GradientLayer()
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = GradientLayer.from_dict(p_dict)

        assert sorted(p.as_dict()) == sorted(q.as_dict())
        assert len(p.layers) == len(q.layers)
        # Just one layer of the generated layers is checked
        assert p.layers[5].__repr__() == q.layers[5].__repr__()

    def test_thickness_setter(self) -> None:
        # When
        gradient_layer = GradientLayer()
        gradient_layer.thickness = 10.0

        # Then
        assert gradient_layer.thickness == 10.0
        assert gradient_layer.front_layer.thickness.value == 1.0
        assert gradient_layer.back_layer.thickness.value == 1.0

    def test_thickness_getter(self, gradient_layer: GradientLayer) -> None:
        # When
        gradient_layer.layers = [MagicMock(), MagicMock()]
        gradient_layer.front_layer.thickness.value = 10.0

        # Then
        # discretisation_elements * discrete_layer_thickness
        assert gradient_layer.thickness == 100.0

    def test_roughness_setter(self, gradient_layer: GradientLayer) -> None:
        # When
        gradient_layer.roughness = 10.0

        # Then
        assert gradient_layer.roughness == 10.0
        assert gradient_layer.front_layer.roughness.value == 10.0
        assert gradient_layer.back_layer.roughness.value == 10.0

    def test_roughness_getter(self, gradient_layer: GradientLayer) -> None:
        # When
        gradient_layer.layers = [MagicMock(), MagicMock()]
        gradient_layer.front_layer.roughness.value = 10.0

        # Then
        assert gradient_layer.roughness == 10.0

    # ----- issue #373 regression coverage -----

    @pytest.mark.parametrize('elements', [0, 1])
    def test_too_few_discretisation_elements_raises(self, elements: int) -> None:
        # When Then Expect: the message matches the `< 2` guard.
        with pytest.raises(ValueError, match='at least 2'):
            GradientLayer(discretisation_elements=elements)

    def test_minimum_discretisation_elements_allowed(self) -> None:
        # When: exactly two elements is the minimum (front + back).
        front = Material(2.0, 0.0, 'Front')
        back = Material(6.0, 0.0, 'Back')
        result = GradientLayer(front_material=front, back_material=back, discretisation_elements=2, name='minimal')

        # Then Expect: both endpoints present, nothing dropped.
        assert len(result.layers) == 2
        assert result.front_layer.material.sld.value == 2.0
        assert result.back_layer.material.sld.value == 6.0

    def test_back_endpoint_included(self, gradient_layer: GradientLayer) -> None:
        # When Then Expect: the final sublayer equals the back material (no off-by-one).
        assert gradient_layer.back_layer.material.sld.value == self.back.sld.value
        assert gradient_layer.back_layer.material.isld.value == self.back.isld.value

    def test_end_materials_excluded_from_parameter_tree(self, gradient_layer: GradientLayer) -> None:
        # When
        parameters = gradient_layer.get_all_parameters()

        # Expect: the frozen end materials are not offered as fittable parameters ...
        assert not any(p is self.front.sld for p in parameters)
        assert not any(p is self.front.isld for p in parameters)
        assert not any(p is self.back.sld for p in parameters)
        assert not any(p is self.back.isld for p in parameters)
        # ... but the discrete sublayers (which do drive reflectivity) remain.
        for layer in gradient_layer.layers:
            assert any(p is layer.material.sld for p in parameters)

    def test_end_materials_excluded_even_when_unfixed(self, gradient_layer: GradientLayer) -> None:
        # When: a user unfixes the endpoint hoping to fit it.
        self.front.sld.fixed = False
        self.back.sld.fixed = False

        # Then Expect: it still never reaches the fitter's free-parameter list,
        # so the silent no-op cannot happen.
        free = gradient_layer.get_free_parameters()
        assert not any(p is self.front.sld for p in free)
        assert not any(p is self.back.sld for p in free)

    def test_end_materials_are_frozen(self, gradient_layer: GradientLayer) -> None:
        # When: the end material is mutated after construction.
        before = [layer.material.sld.value for layer in gradient_layer.layers]
        self.front.sld.value = 4.0
        after = [layer.material.sld.value for layer in gradient_layer.layers]

        # Then Expect: the sublayers keep their construction-time snapshot.
        assert before == after

    def test_sublayers_drive_reflectivity(self) -> None:
        # When
        global_object.map._clear()
        front = Material(2.0, 0.0, 'Front')
        back = Material(6.0, 0.0, 'Back')
        gradient = GradientLayer(front_material=front, back_material=back, thickness=60.0, discretisation_elements=6, name='G')
        model = Model(interface=CalculatorFactory())
        model.add_assemblies(gradient)
        q = [0.05, 0.1, 0.2]
        before = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Then: mutating a sublayer SLD (a real, in-tree parameter) changes reflectivity ...
        gradient.layers[2].material.sld.value = 3.9
        after_sublayer = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))
        # ... while mutating the frozen end material does not.
        front.sld.value = 5.0
        after_endpoint = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Expect
        assert not np.allclose(before, after_sublayer)
        assert np.allclose(after_sublayer, after_endpoint)


def test_linear_gradient_increasing():
    # When Then
    result = _linear_gradient(front_value=1.5, back_value=2.5, discretisation_elements=11)

    # Expect: exactly `discretisation_elements` values, both endpoints included.
    assert len(result) == 11
    assert_almost_equal([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5], result)


def test_linear_gradient_decreasing():
    # When Then
    result = _linear_gradient(front_value=2.5, back_value=1.5, discretisation_elements=11)

    # Expect
    assert len(result) == 11
    assert_almost_equal([2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5], result)


def test_linear_gradient_includes_both_endpoints():
    # When Then: with N values the last element is the back value (no off-by-one).
    result = _linear_gradient(front_value=10.0, back_value=0.0, discretisation_elements=10)

    # Expect
    assert len(result) == 10
    assert result[0] == 10.0
    assert result[-1] == 0.0


def test_linear_gradient_same():
    # When Then
    result = _linear_gradient(front_value=2.5, back_value=2.5, discretisation_elements=10)

    # Expect
    assert len(result) == 10
    assert_almost_equal([2.5] * 10, result)


def test_prepare_gradient_layers(monkeypatch):
    # When
    mock_material_1 = MagicMock()
    mock_material_2 = MagicMock()
    mock_Layer = MagicMock()
    mock_LayerCollection = MagicMock()
    mock_Material = MagicMock(return_value='Material_from_mock')
    mock_linear_gradient = MagicMock(return_value=[1.0, 2.0, 3.0])
    monkeypatch.setattr(
        easyreflectometry.sample.assemblies.gradient_layer,
        '_linear_gradient',
        mock_linear_gradient,
    )
    monkeypatch.setattr(easyreflectometry.sample.assemblies.gradient_layer, 'Layer', mock_Layer)
    monkeypatch.setattr(easyreflectometry.sample.assemblies.gradient_layer, 'Material', mock_Material)
    monkeypatch.setattr(easyreflectometry.sample.assemblies.gradient_layer, 'LayerCollection', mock_LayerCollection)

    # Then
    _prepare_gradient_layers(mock_material_1, mock_material_2, 3, None)

    # When
    assert mock_Material.call_count == 3
    assert mock_Material.call_args_list[0][0] == (1.0, 1.0)
    assert mock_Material.call_args_list[1][0] == (2.0, 2.0)
    assert mock_Material.call_args_list[2][0] == (3.0, 3.0)
    assert mock_Layer.call_count == 3
    assert mock_Layer.call_args_list[0][1]['material'] == 'Material_from_mock'
    assert mock_Layer.call_args_list[0][1]['thickness'] == 0.0
    assert mock_Layer.call_args_list[0][1]['name'] == '0'
    assert mock_Layer.call_args_list[0][1]['interface'] is None
