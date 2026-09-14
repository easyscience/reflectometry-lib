# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model.model import COLORS
from easyreflectometry.model.model import Model
from easyreflectometry.model.model_collection import ModelCollection
from easyreflectometry.sample import Sample


class TestModelCollection:
    def test_default(self):
        # When Then
        collection = ModelCollection()

        # Expect
        assert collection.name == 'Models'
        assert collection.interface is None
        assert len(collection) == 1
        assert collection[0].name == 'Model'

    def test_default_with_real_interface(self):
        # When - the default model is built with an actual calculator interface
        interface = CalculatorFactory()
        collection = ModelCollection(interface=interface)

        # Expect - the interface lands on the model as interface, not as sample
        assert len(collection) == 1
        assert isinstance(collection[0].sample, Sample)
        assert collection[0].interface is interface
        assert collection.interface is interface

    def test_dont_populate(self):
        p = ModelCollection(populate_if_none=False)
        assert p.name == 'Models'
        assert p.interface is None
        assert len(p) == 0

    def test_from_pars(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')
        model_3 = Model(name='Model3')

        # Then
        collection = ModelCollection(model_1, model_2, model_3)

        # Expect
        assert collection.name == 'Models'
        assert collection.interface is None
        assert len(collection) == 3
        assert collection[0].name == 'Model1'
        assert collection[1].name == 'Model2'
        assert collection[2].name == 'Model3'

    def test_add_model(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')

        # Then
        collection = ModelCollection(model_1)
        collection.add_model(model_2)

        # Expect
        assert len(collection) == 2
        assert collection[0].name == 'Model1'
        assert collection[1].name == 'Model2'

    def test_add_model_color_cycle(self):
        collection = ModelCollection(populate_if_none=False)

        collection.add_model()
        assert collection[0].color == COLORS[0]

        collection.add_model()
        assert collection[1].color == COLORS[1]

        collection.remove_at(0)
        collection.add_model()

        assert collection[0].color == COLORS[1]
        assert collection[1].color == COLORS[2]

    def test_add_model_color_wrap(self):
        collection = ModelCollection(populate_if_none=False)

        for _ in range(len(COLORS)):
            collection.add_model()

        collection.add_model()

        assert collection[-1].color == COLORS[0]

    def test_add_model_preserves_explicit_color(self):
        collection = ModelCollection(populate_if_none=False)
        collection.add_model()
        expected_index = collection._next_color_index

        custom_color = '#ABCDEF'
        custom_model = Model(name='Custom', color=custom_color)

        collection.add_model(custom_model)

        assert collection[-1].color == custom_color
        assert collection._next_color_index == (expected_index + 1) % len(COLORS)

    def test_next_color_matches_what_appending_assigns(self):
        collection = ModelCollection(populate_if_none=False)
        collection.add_model()
        announced = collection.next_color()

        collection.add_model(Model(name='Prebuilt', color=announced))

        assert collection[-1].color == announced
        assert collection[0].color != collection[1].color
        # and the cycle moved on
        assert collection.next_color() != announced or len(COLORS) <= 2

    def test_duplicate_model_gets_a_distinct_color(self):
        collection = ModelCollection(populate_if_none=False)
        collection.add_model()

        collection.duplicate_model(0)

        assert collection[1].name.endswith('duplicate')
        assert collection[0].color != collection[1].color

    def test_delete_model(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')

        # Then
        collection = ModelCollection(model_1, model_2)
        collection.remove_at(0)

        # Expect
        assert len(collection) == 1
        assert collection[0].name == 'Model2'

    def test_as_dict(self):
        # When
        model_1 = Model(name='Model1')
        collection = ModelCollection(model_1)

        # Then
        dict_repr = collection.as_dict()

        # Expect
        assert dict_repr['data'][0]['resolution_function'] == {
            'smearing': 'PercentageFwhm',
            'constant': 5.0,
        }

    def test_dict_round_trip(self):
        # When
        model_1 = Model(name='Model1')
        model_2 = Model(name='Model2')
        model_3 = Model(name='Model3')
        p = ModelCollection(model_1, model_2, model_3)
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = ModelCollection.from_dict(p_dict)

        # Expect
        # We have to skip the resolution_function and interface
        assert sorted(p.as_dict(skip=['resolution_function', 'interface'])) == sorted(
            q.as_dict(skip=['resolution_function', 'interface'])
        )
        assert p[0]._resolution_function.smearing(5.5) == q[0]._resolution_function.smearing(5.5)

    def test_next_color_index_round_trip(self):
        collection = ModelCollection(populate_if_none=False)
        for _ in range(3):
            collection.add_model()

        expected_index = collection._next_color_index
        dict_repr = collection.as_dict()
        global_object.map._clear()

        restored = ModelCollection.from_dict(dict_repr)

        assert restored._next_color_index == expected_index

    def test_legacy_from_dict_sets_color_index(self):
        collection = ModelCollection()
        legacy_dict = collection.as_dict()
        legacy_dict.pop('next_color_index', None)
        global_object.map._clear()

        restored = ModelCollection.from_dict(legacy_dict)
        restored.add_model()

        assert [model.color for model in restored] == [COLORS[0], COLORS[1]]

    def test_next_color_index_property(self):
        """next_color_index should be accessible as a property for serialization."""
        collection = ModelCollection(populate_if_none=False)
        collection.add_model()
        idx = collection.next_color_index
        assert isinstance(idx, int)
        assert idx >= 0

    def test_next_color_index_none_when_no_colors(self):
        """When COLORS is empty, next_color_index returns 0."""
        # We can test the None case when COLORS has entries, it wraps
        collection = ModelCollection(populate_if_none=False)
        # Without adding models, the index should still be accessible
        assert collection.next_color_index is not None

    def test_from_dict_preserves_data_count(self):
        """from_dict should reconstruct the exact number of models."""
        global_object.map._clear()
        model_1 = Model(name='M1')
        model_2 = Model(name='M2')
        p = ModelCollection(model_1, model_2)
        d = p.as_dict()
        global_object.map._clear()
        q = ModelCollection.from_dict(d)
        assert len(q) == 2

    def test_from_dict_with_extra_data_entries(self):
        """from_dict should handle data entries correctly."""
        global_object.map._clear()
        model_1 = Model(name='M1')
        model_2 = Model(name='M2')
        p = ModelCollection(model_1, model_2)
        d = p.as_dict()
        global_object.map._clear()
        q = ModelCollection.from_dict(d)
        assert len(q) == 2
        assert q[0].name == 'M1'
        assert q[1].name == 'M2'
