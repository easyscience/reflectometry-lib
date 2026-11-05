# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""What a collection writes besides its items (`_arg_spec`), and what it loads back.

easyscience 3 would encode every keyword-only constructor argument; the
collections pin their fields instead, so the `populate_if_none` control flag
is never written and a collection saved empty loads empty.
"""

import pytest
from easyscience import global_object

from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerCollection
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialCollection
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _reload(collection):
    data = collection.as_dict()
    return data, type(collection).from_dict(data)


@pytest.mark.parametrize(
    'collection',
    [
        lambda: LayerCollection(Layer(), name='Custom layers'),
        lambda: MaterialCollection(Material(), name='Custom materials'),
        lambda: Sample(Multilayer(), name='Custom sample'),
        lambda: ModelCollection(Model(), name='Custom models'),
    ],
)
def test_a_custom_name_round_trips(collection):
    data, loaded = _reload(collection())
    assert 'populate_if_none' not in data
    assert loaded.name == data['name']
    assert loaded.name.startswith('Custom')


def test_an_emptied_model_collection_loads_empty():
    models = ModelCollection()  # populate_if_none=True by default: one model
    models.pop(0)
    data, loaded = _reload(models)
    assert 'populate_if_none' not in data
    assert len(loaded) == 0


def test_an_empty_sample_loads_empty():
    data, loaded = _reload(Sample(populate_if_none=False))
    assert 'populate_if_none' not in data
    assert len(loaded) == 0


def test_the_next_colour_index_round_trips():
    models = ModelCollection(Model(), next_color_index=3)
    data, loaded = _reload(models)
    assert data['next_color_index'] == 3
    assert loaded.next_color_index == 3
