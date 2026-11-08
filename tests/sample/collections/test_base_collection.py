# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import pytest

from easyreflectometry.sample.collections.base_collection import BaseCollection
from easyreflectometry.sample.elements.layers.layer import Layer


class TestBaseCollection:
    def test_constructor(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem_1, elem_2)

        # Expect
        p._interface = mock_interface
        len(p) == 2

    def test_names(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem_1, elem_2)

        # Expect
        assert p.names == ['layer_1', 'layer_2']

    def test_dict_repr(self):
        # When
        elem = Layer(name='layer')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem)

        # Expect
        assert p._dict_repr == {
            'name': [
                {
                    'layer': {
                        'material': {'EasyMaterial': {'isld': '0.000e-6 1/Å^2', 'sld': '4.186e-6 1/Å^2'}},
                        'roughness': '3.300 Å',
                        'thickness': '10.000 Å',
                    }
                }
            ]
        }

    def test_as_dict(self):
        # When
        elem = Layer(name='layer')
        mock_interface = MagicMock()

        # Then
        p = BaseCollection('name', mock_interface, elem)

        # Expect
        assert p.as_dict()['name'] == 'name'
        assert len(p.as_dict()['data']) == 1
        assert p.as_dict()['data'][0]['name'] == 'layer'

    def test_move_up(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_up(3)

        # Expect
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_4'
        assert p[3].name == 'layer_3'

    def test_move_up_to_top_and_further(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_up(3)
        p.move_up(2)
        p.move_up(1)
        p.move_up(0)

        # Then
        assert p[0].name == 'layer_4'
        assert p[1].name == 'layer_1'
        assert p[2].name == 'layer_2'
        assert p[3].name == 'layer_3'

    def test_move_down(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.move_down(2)

        # Expect
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_4'
        assert p[3].name == 'layer_3'

    def test_move_down_to_bottom_and_further(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))
        p.append(Layer(name='layer_5'))

        # Then
        p.move_down(3)
        p.move_down(4)

        # Then
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_2'
        assert p[2].name == 'layer_3'
        assert p[3].name == 'layer_5'
        assert p[4].name == 'layer_4'

    def test_remove(self):
        # When
        elem_1 = Layer(name='layer_1')
        elem_2 = Layer(name='layer_2')
        elem_3 = Layer(name='layer_3')
        mock_interface = MagicMock()

        p = BaseCollection('name', mock_interface, elem_1, elem_2, elem_3)
        p.append(Layer(name='layer_4'))

        # Then
        p.remove_at(1)

        # Then
        assert len(p) == 3
        assert p[0].name == 'layer_1'
        assert p[1].name == 'layer_3'
        assert p[2].name == 'layer_4'

    # ---- new BaseCollection (EasyList-based) specific tests ----

    def test_name_getter_and_setter(self):
        """name property should be readable and writable."""
        p = BaseCollection('original', MagicMock())
        assert p.name == 'original'
        p.name = 'changed'
        assert p.name == 'changed'

    def test_data_property(self):
        """data property should return a read-only copy of the internal list."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        data = p.data
        assert len(data) == 1
        assert data[0].name == 'layer'
        # Mutating the returned copy must not affect the collection
        data.append(Layer(name='extra'))
        assert len(p) == 1

    def test_interface_propagates_to_existing_items(self):
        """Setting interface after construction should propagate to all items."""
        mock_iface = MagicMock()
        elem = Layer(name='layer')
        # Pass interface=None explicitly and items as positional args
        p = BaseCollection('name', None, elem)
        assert p.interface is None
        p.interface = mock_iface
        # The interface setter propagates to items then calls generate_bindings on the mock
        assert elem.interface is mock_iface
        mock_iface.generate_bindings.assert_called()

    def test_interface_propagates_to_inserted_items(self):
        """Items inserted after interface is set should receive the interface."""
        mock_iface = MagicMock()
        p = BaseCollection('name', mock_iface)
        elem = Layer(name='new_layer')
        p.append(elem)
        assert elem.interface is mock_iface

    def test_get_all_variables(self):
        """get_all_variables should collect parameters from all items."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        variables = p.get_all_variables()
        # A Layer has thickness, roughness, and the material's sld/isld
        names = {v.display_name for v in variables}
        assert 'thickness' in names
        assert 'roughness' in names

    def test_get_all_parameters(self):
        """get_all_parameters should filter to only Parameter instances."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        params = p.get_all_parameters()
        for param in params:
            assert param.__class__.__name__ == 'Parameter'

    def test_get_free_parameters(self):
        """get_free_parameters should return only independent, non-fixed parameters."""
        elem = Layer(name='layer')
        # By default thickness/roughness are fixed
        p = BaseCollection('name', MagicMock(), elem)
        free = p.get_free_parameters()
        # By default all params are fixed, so empty
        assert len(free) == 0
        # Unfix one
        elem.thickness.fixed = False
        free = p.get_free_parameters()
        assert len(free) == 1
        assert free[0].display_name == 'thickness'

    def test_to_dict_includes_data_and_name(self):
        """to_dict should serialize data items and collection metadata."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        d = p.to_dict()
        assert d['name'] == 'name'
        assert len(d['data']) == 1
        assert d['data'][0]['name'] == 'layer'

    def test_to_dict_skips_interface(self):
        """to_dict should exclude the interface field."""
        mock_iface = MagicMock()
        p = BaseCollection('name', mock_iface)
        d = p.to_dict()
        assert 'interface' not in d

    def test_to_dict_skips_unique_name_by_default(self):
        """to_dict should drop unique_name (matching legacy behaviour)."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        d = p.to_dict()
        assert 'unique_name' not in d

    def test_as_dict_is_alias_for_to_dict(self):
        """as_dict should delegate to to_dict."""
        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        assert p.as_dict() == p.to_dict()

    def test_deepcopy_round_trips(self):
        """__deepcopy__ should produce an equivalent collection via from_dict."""
        import copy

        elem = Layer(name='layer')
        # Use a concrete subclass (LayerCollection) that properly supports deepcopy
        from easyreflectometry.sample.collections.layer_collection import LayerCollection

        p = LayerCollection(elem, name='test_layers')
        p_copy = copy.deepcopy(p)
        assert len(p_copy) == len(p)
        assert p_copy[0].name == p[0].name

    def test_repr_handles_exception_gracefully(self):
        """__repr__ should not crash even with items lacking _dict_repr."""
        mock_item = MagicMock()
        # Deliberately make _dict_repr raise
        del mock_item._dict_repr
        p = BaseCollection('name', interface=None)
        # Manually insert the mock item bypassing normal insert
        p._data.append(mock_item)
        # Should not raise
        result = repr(p)
        assert isinstance(result, str)

    def test_insert_rejects_non_integer_index(self):
        """insert should raise TypeError for non-integer indices."""
        p = BaseCollection('name', interface=None)
        with pytest.raises(TypeError, match='Index must be an integer'):
            p.insert('not_an_int', Layer(name='x'))

    def test_duplicate_insert_is_warned(self):
        """Inserting an already-present item should warn and skip."""
        import warnings

        elem = Layer(name='layer')
        p = BaseCollection('name', MagicMock(), elem)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.append(elem)
            assert len(w) == 1
            assert 'already in collection' in str(w[0].message)
        # Length unchanged
        assert len(p) == 1

    def test_get_key_uses_name(self):
        """_get_key should use the item's name property."""
        elem = Layer(name='mylayer')
        p = BaseCollection('name', MagicMock(), elem)
        assert p._get_key(elem) == 'mylayer'

    def test_has_interface_setter(self):
        """_has_interface_setter should correctly detect interface-writable types."""
        assert BaseCollection._has_interface_setter(Layer) is True
        assert BaseCollection._has_interface_setter(int) is False
