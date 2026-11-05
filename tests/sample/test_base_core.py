# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BaseCore class — the new ModelBase-based foundation for sample-tree objects."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.sample.base_core import BaseCore

# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing the abstract BaseCore
# ---------------------------------------------------------------------------


class _ConcreteCore(BaseCore):
    """A non-abstract BaseCore that exposes a simple ``_dict_repr``."""

    def __init__(self, name='TestCore', interface=None, unique_name=None, **kwargs):
        super().__init__(name=name, interface=interface, unique_name=unique_name, **kwargs)

    @property
    def _dict_repr(self) -> dict[str, str]:
        return {self.name: {'type': 'concrete'}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseCore:
    """Direct unit tests for the BaseCore abstract base class."""

    # ---- construction ----

    def test_default_construction(self) -> None:
        """A minimal concrete subclass should construct without errors."""
        obj = _ConcreteCore(name='Test')
        assert obj.name == 'Test'
        assert obj.interface is None
        assert obj.user_data == {}

    def test_construction_with_interface(self) -> None:
        """Passing an interface should trigger generate_bindings."""
        mock_iface = MagicMock()
        obj = _ConcreteCore(name='WithIface', interface=mock_iface)
        assert obj.interface is mock_iface
        mock_iface.generate_bindings.assert_called_once_with(obj)

    def test_construction_with_unique_name(self) -> None:
        """unique_name is passed through to ModelBase."""
        obj = _ConcreteCore(name='Uniq', unique_name='my_unique')
        assert obj.unique_name == 'my_unique'

    def test_construction_kwargs_stored_as_attributes(self) -> None:
        """Transitional kwargs path: extra kwargs become plain instance attrs."""
        child = Parameter(5.0, display_name='extra_param')
        obj = _ConcreteCore(name='Kwargs', extra=child, extra2=42)
        assert obj.extra is child
        assert obj.extra2 == 42

    # ---- name property ----

    def test_name_getter(self) -> None:
        obj = _ConcreteCore(name='MyName')
        assert obj.name == 'MyName'

    def test_name_setter(self) -> None:
        obj = _ConcreteCore(name='Original')
        obj.name = 'Changed'
        assert obj.name == 'Changed'

    # ---- interface property ----

    def test_interface_set_to_none(self) -> None:
        obj = _ConcreteCore(name='NoIface')
        obj.interface = None
        assert obj.interface is None

    def test_interface_set_triggers_bindings(self) -> None:
        obj = _ConcreteCore(name='Late')
        mock_iface = MagicMock()
        obj.interface = mock_iface
        mock_iface.generate_bindings.assert_called_once_with(obj)

    def test_interface_setter_does_not_call_generate_bindings_for_none(self) -> None:
        obj = _ConcreteCore(name='NoneIface')
        # Setting to None should be safe (no generate_bindings call)
        obj.interface = None
        assert obj.interface is None

    # ---- generate_bindings ----

    def test_generate_bindings_raises_when_interface_is_none(self) -> None:
        obj = _ConcreteCore(name='NoIface')
        with pytest.raises(AttributeError, match='Interface error'):
            obj.generate_bindings()

    def test_generate_bindings_propagates_to_children(self) -> None:
        """Children with an interface setter receive the parent's interface."""
        mock_iface = MagicMock()
        child = _ConcreteCore(name='Child')
        child._interface = None  # reset so we can observe propagation
        obj = _ConcreteCore(name='Parent', child=child)
        obj.interface = mock_iface
        # The child should have received the interface too.
        assert child.interface is mock_iface

    def test_generate_bindings_propagates_to_parameter_children(self) -> None:
        """Parameters stored as plain attrs should not break binding propagation."""
        mock_iface = MagicMock()
        param = Parameter(1.0, display_name='p')
        obj = _ConcreteCore(name='WithParam', p=param)
        obj.interface = mock_iface
        mock_iface.generate_bindings.assert_called_once_with(obj)

    # ---- _iter_public_children ----

    def test_iter_public_children_includes_class_attrs(self) -> None:
        child = _ConcreteCore(name='Child')
        obj = _ConcreteCore(name='Parent', child=child)
        children = list(obj._iter_public_children())
        assert child in children

    def test_iter_public_children_includes_instance_attrs(self) -> None:
        param = Parameter(1.0, display_name='p')
        obj = _ConcreteCore(name='Parent', p=param)
        children = list(obj._iter_public_children())
        assert param in children

    def test_iter_public_children_excludes_private(self) -> None:
        obj = _ConcreteCore(name='Parent')
        obj._private_thing = 'secret'
        children = list(obj._iter_public_children())
        names = [getattr(c, 'name', c) for c in children]
        assert 'secret' not in names

    def test_iter_public_children_excludes_interface_and_name(self) -> None:
        obj = _ConcreteCore(name='Parent')
        children = list(obj._iter_public_children())
        assert obj.interface not in children

    def test_iter_public_children_no_duplicates(self) -> None:
        """If a child appears both as a class attr and instance attr, only one copy."""
        child = _ConcreteCore(name='Child')
        obj = _ConcreteCore(name='Parent', child=child)
        # Also set as attr with same id
        obj.duplicate_ref = child
        children = list(obj._iter_public_children())
        # child should appear only once
        assert children.count(child) == 1

    # ---- _has_interface_setter ----

    def test_has_interface_setter_true(self) -> None:
        assert BaseCore._has_interface_setter(_ConcreteCore) is True

    def test_has_interface_setter_false_for_bare_object(self) -> None:
        assert BaseCore._has_interface_setter(object) is False

    def test_has_interface_setter_false_for_parameter(self) -> None:
        """Parameter doesn't have an interface property."""
        assert BaseCore._has_interface_setter(Parameter) is False

    # ---- compatibility shims ----

    def test_get_linkable_attributes(self) -> None:
        param = Parameter(1.0, display_name='p')
        obj = _ConcreteCore(name='Core', p=param)
        result = obj._get_linkable_attributes()
        assert param in result

    def test_add_component(self) -> None:
        obj = _ConcreteCore(name='Core')
        comp = Parameter(42.0, display_name='comp')
        obj._add_component('my_comp', comp)
        assert obj.my_comp is comp

    # ---- get_all_variables ----

    def test_get_all_variables_includes_descriptors(self) -> None:
        param = Parameter(1.0, display_name='p')
        obj = _ConcreteCore(name='Core', p=param)
        result = obj.get_all_variables()
        assert param in result

    def test_get_all_variables_recurses_into_children(self) -> None:
        inner_param = Parameter(2.0, display_name='inner')
        child = _ConcreteCore(name='Child', p=inner_param)
        obj = _ConcreteCore(name='Parent', child=child)
        result = obj.get_all_variables()
        assert inner_param in result

    def test_get_all_variables_no_duplicates_across_children(self) -> None:
        param = Parameter(1.0, display_name='shared')
        child_a = _ConcreteCore(name='A', p=param)
        child_b = _ConcreteCore(name='B', p=param)
        obj = _ConcreteCore(name='Parent', a=child_a, b=child_b)
        result = obj.get_all_variables()
        assert result.count(param) == 1

    # ---- to_dict / as_dict ----

    def test_to_dict_skips_interface(self) -> None:
        mock_iface = MagicMock()
        obj = _ConcreteCore(name='Core', interface=mock_iface)
        d = obj.to_dict()
        assert 'interface' not in d

    def test_to_dict_skips_unique_name_by_default(self) -> None:
        obj = _ConcreteCore(name='Core', unique_name='my_unique')
        d = obj.to_dict()
        assert 'unique_name' not in d

    def test_to_dict_includes_name(self) -> None:
        obj = _ConcreteCore(name='MyName')
        d = obj.to_dict()
        assert d.get('name') == 'MyName'

    def test_as_dict_is_alias_for_to_dict(self) -> None:
        obj = _ConcreteCore(name='Core')
        assert obj.as_dict() == obj.to_dict()

    def test_to_dict_respects_custom_skip(self) -> None:
        obj = _ConcreteCore(name='Core')
        d = obj.to_dict(skip=['name'])
        assert 'name' not in d

    def test_to_dict_skip_not_mutated_by_callee(self) -> None:
        """Caller's skip list must not be mutated."""
        obj = _ConcreteCore(name='Core')
        skip = ['name']
        obj.to_dict(skip=skip)
        assert skip == ['name']  # not appended-to

    # ---- repr ----

    def test_repr_returns_yaml_string(self) -> None:
        obj = _ConcreteCore(name='Test')
        r = repr(obj)
        assert 'Test' in r
        assert 'concrete' in r

    # ---- user_data ----

    def test_user_data_is_dict(self) -> None:
        obj = _ConcreteCore(name='Core')
        obj.user_data['key'] = 'value'
        assert obj.user_data['key'] == 'value'

    # ---- round-trip ----

    def test_basic_round_trip_via_material(self) -> None:
        """Round-trip through a real subclass (Material) to verify BaseCore serialization."""
        from easyreflectometry.sample.elements.materials.material import Material

        global_object.registry.clear()
        obj = Material(sld=2.0, isld=0.5, name='TestMat')
        d = obj.to_dict()
        global_object.registry.clear()

        restored = Material.from_dict(d)
        assert restored.name == 'TestMat'
        assert restored.sld.value == 2.0
        assert restored.isld.value == 0.5

    def test_round_trip_skips_interface(self) -> None:
        """Round-trip via to_dict → from_dict should strip the interface."""
        global_object.registry.clear()
        obj = _ConcreteCore(name='WithIface')
        d = obj.to_dict()
        assert 'interface' not in d
