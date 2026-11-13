# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for BaseAssembly class module
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from easyscience import global_object

from easyreflectometry.sample.assemblies.base_assembly import BaseAssembly


class TestBaseAssembly:
    @pytest.fixture
    def base_assembly(self) -> BaseAssembly:
        global_object.registry.clear()

        self.mock_layer_0 = MagicMock()
        self.mock_layer_1 = MagicMock()
        self.mock_layers = [self.mock_layer_0, self.mock_layer_1]
        self.mock_interface = MagicMock()
        BaseAssembly.__abstractmethods__ = set()
        return BaseAssembly(
            name='name',
            type='type',
            interface=self.mock_interface,
            layers=self.mock_layers,
        )

    def test_init(self, base_assembly: BaseAssembly) -> None:
        # When Then Expect
        assert base_assembly.name == 'name'
        assert base_assembly._type == 'type'
        assert base_assembly.interface == self.mock_interface
        assert base_assembly.layers == self.mock_layers
        assert base_assembly._roughness_constraints_setup is False
        assert base_assembly._thickness_constraints_setup is False

    def test_setup_thickness_constraints(self, base_assembly: BaseAssembly, monkeypatch: Any) -> None:
        # When
        # self.mock_layer_0.thickness = MagicMock()
        # self.mock_layer_1.thickness = MagicMock()

        # Then
        base_assembly._setup_thickness_constraints()

        assert base_assembly._thickness_constraints_setup is True

    def test_enable_thickness_constraints(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly._thickness_constraints_setup = True

        # Then
        base_assembly._enable_thickness_constraints()

        # Expect
        assert self.mock_layer_0.thickness.value == self.mock_layer_0.thickness.value

    def test_enable_thickness_constraints_exception(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly._thickness_constraints_setup = False

        # Then
        with pytest.raises(Exception):
            base_assembly._enable_thickness_constraints()

    def test_disable_thickness_constraints(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly._thickness_constraints_setup = True

        # Then
        base_assembly._disable_thickness_constraints()

        # Expect
        assert self.mock_layer_1.thickness.make_independent.called is True

    def test_setup_roughness_constraints(self, base_assembly: BaseAssembly, monkeypatch: Any) -> None:
        # Then
        base_assembly._setup_roughness_constraints()

        # Expect
        assert base_assembly._roughness_constraints_setup is True

    def test_enable_roughness_constraints(self, base_assembly):
        # When
        base_assembly._roughness_constraints_setup = True

        # Then
        base_assembly._enable_roughness_constraints()

        # Expect
        assert self.mock_layer_0.roughness.value == self.mock_layer_0.roughness.value

    def test_enable_roughness_constraints_exception(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly._roughness_constraints_setup = False

        # Then
        with pytest.raises(Exception):
            base_assembly._enable_roughness_constraints(True)

    def test_disable_roughness_constraints(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly._roughness_constraints_setup = True

        # Then
        base_assembly._disable_roughness_constraints()

        # Expect
        assert self.mock_layer_1.roughness.make_independent.called is True

    def test_front_layer(self, base_assembly: BaseAssembly) -> None:
        # When Then Expect
        assert base_assembly.front_layer == self.mock_layer_0

    def test_front_layer_none(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly.layers = []

        # Then
        result = base_assembly.front_layer

        # Expect
        assert result is None

    def test_set_front_layer(self, base_assembly: BaseAssembly) -> None:
        # When Then
        base_assembly.front_layer = self.mock_layer_1

        # Expect
        assert base_assembly.layers == [self.mock_layer_1, self.mock_layer_1]

    def test_set_front_layer_empty(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly.layers = []

        # Then
        base_assembly.front_layer = self.mock_layer_1

        # Expect
        assert base_assembly.layers == [self.mock_layer_1]

    def test_back_layer(self, base_assembly: BaseAssembly) -> None:
        # When Then Expect
        assert base_assembly.back_layer == self.mock_layer_1

    def test_back_layer_none(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly.layers = []

        # Then
        result = base_assembly.back_layer

        # Expect
        assert result is None

    def test_set_back_layer(self, base_assembly: BaseAssembly) -> None:
        # When Then
        base_assembly.back_layer = self.mock_layer_0

        # Expect
        assert base_assembly.layers == [self.mock_layer_0, self.mock_layer_0]

    def test_set_back_layer_exception(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly.layers = []

        # Then
        with pytest.raises(Exception):
            base_assembly.back_layer = self.mock_layer_1

    def test_set_back_layer_with_front(self, base_assembly: BaseAssembly) -> None:
        # When
        base_assembly.layers = [self.mock_layer_0]

        # Then
        base_assembly.back_layer = self.mock_layer_1

        # Expect
        assert base_assembly.layers == [self.mock_layer_0, self.mock_layer_1]

    def test_layers_setter(self) -> None:
        """The layers property setter should replace the layer list."""
        global_object.registry.clear()
        BaseAssembly.__abstractmethods__ = set()
        assembly = BaseAssembly(
            name='test',
            type='type',
            interface=MagicMock(),
            layers=[MagicMock(), MagicMock()],
        )
        new_layers = [MagicMock(), MagicMock(), MagicMock()]
        assembly.layers = new_layers
        assert assembly.layers == new_layers
        assert len(assembly.layers) == 3
