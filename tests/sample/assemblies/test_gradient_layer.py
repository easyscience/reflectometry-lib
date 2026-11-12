# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for GradientLayer class module
"""

from unittest.mock import MagicMock

import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal

import easyreflectometry.sample.assemblies.gradient_layer
from easyreflectometry.sample.assemblies.gradient_layer import GradientLayer
from easyreflectometry.sample.assemblies.gradient_layer import _linear_gradient
from easyreflectometry.sample.assemblies.gradient_layer import _prepare_gradient_layers
from easyreflectometry.sample.elements.materials.material import Material


class TestGradientLayer:
    @pytest.fixture
    def gradient_layer(self) -> GradientLayer:
        global_object.registry.clear()

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

        assert gradient_layer.front_layer.material.sld.value == 10.0
        assert gradient_layer.layers[5].material.sld.value == 5.0
        assert gradient_layer.back_layer.material.sld.value == 1.0
        assert gradient_layer.front_layer.material.isld.value == -10.0
        assert gradient_layer.layers[5].material.isld.value == -5.0
        assert gradient_layer.back_layer.material.isld.value == -1.0

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
        expected_str = "thickness: 1.0\ndiscretisation_elements: 10\nback_layer:\n  '9':\n    material:\n      EasyMaterial:\n        sld: 1.000e-6 1/Å^2\n        isld: -1.000e-6 1/Å^2\n    thickness: 0.100 Å\n    roughness: 2.000 Å\nfront_layer:\n  '0':\n    material:\n      EasyMaterial:\n        sld: 10.000e-6 1/Å^2\n        isld: -10.000e-6 1/Å^2\n    thickness: 0.100 Å\n    roughness: 2.000 Å\n"  # noqa: E501
        assert gradient_layer.__repr__() == expected_str

    def test_dict_round_trip(self) -> None:
        # When
        p = GradientLayer()
        p_dict = p.as_dict()
        global_object.registry.clear()

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


def test_linear_gradient_increasing():
    # When Then
    result = _linear_gradient(front_value=1.5, back_value=2.5, discretisation_elements=10)

    # Expect
    assert_almost_equal([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5], result)


def test_linear_gradient_decreasing():
    # When Then
    result = _linear_gradient(front_value=2.5, back_value=1.5, discretisation_elements=10)

    # Expect
    assert_almost_equal([2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5], result)


def test_linear_gradient_same():
    # When Then
    result = _linear_gradient(front_value=2.5, back_value=2.5, discretisation_elements=10)

    # Expect
    assert_almost_equal([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5], result)


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
