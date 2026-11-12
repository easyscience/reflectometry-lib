# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Which models a calculator serves, as the project changes.

Every test keeps a reference to the model it removes or replaces, so what is
checked is membership, not garbage collection.
"""

import gc
import json

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_array_equal

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 30)


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _served(project: Project) -> set[str]:
    return set(project._calculator().models)


def _names(project: Project) -> set[str]:
    return {model.unique_name for model in project.models}


def _project(n_models: int = 2) -> Project:
    project = Project()
    project.models = ModelCollection(*[_film_model(50.0 * (index + 1)) for index in range(n_models)])
    return project


def test_every_model_of_a_project_is_served():
    project = _project(2)
    assert _served(project) == _names(project)


def test_removal_by_index_stops_serving_the_model():
    project = _project(2)
    removed = project.models[1]
    project.remove_model_at_index(1)
    assert _served(project) == _names(project)
    assert removed.unique_name not in _served(project)


def test_a_moved_model_is_still_served():
    project = _project(3)
    project.move_model(0, 2)
    assert _served(project) == _names(project)


def test_replacing_the_models_serves_only_the_new_ones():
    project = _project(2)
    old = list(project.models)
    project.models = ModelCollection(Model())
    assert _served(project) == _names(project)
    assert not {model.unique_name for model in old} & _served(project)
    # A single model again: evaluating without a model id is unambiguous.
    assert len(project._calculator().reflectity_profile(Q, None)) == len(Q)


def test_default_model_replaces_what_is_served():
    project = _project(2)
    project.default_model()
    assert _served(project) == _names(project)


def test_a_reloaded_project_serves_and_evaluates_its_models():
    project = _project(2)
    expected = [model.interface.fit_func(Q, model.unique_name) for model in project.models]
    data = json.loads(json.dumps(project.as_dict()))
    global_object.registry.clear()
    loaded = Project()
    loaded.from_dict(data)
    # Loading also replaces the materials palette, after the models: that must not unregister them.
    assert _served(loaded) == _names(loaded)
    for model, curve in zip(loaded.models, expected):
        assert_array_equal(model.interface.fit_func(Q, model.unique_name), curve)


def test_a_calculator_switch_serves_the_same_models():
    project = _project(2)
    project.calculator = 'refl1d'
    project.calculator = 'refnx'
    assert _served(project) == _names(project)


def test_a_bare_switch_serves_nothing_until_the_models_are_bound_again():
    model = Model()
    factory = CalculatorFactory()
    model.interface = factory
    factory.switch('refnx')
    assert factory().models == {}
    with pytest.raises(ValueError, match='No model is attached'):
        factory().reflectity_profile(Q, None)
    model.generate_bindings()
    assert list(factory().models) == [model.unique_name]


def test_repeated_create_and_destroy_leaves_nothing_behind():
    factory = CalculatorFactory()
    for _ in range(5):
        model = Model(resolution_function=PercentageFwhm(2.0))
        model.interface = factory
        del model
        gc.collect()
    assert factory().models == {}
    assert factory()._wrapper._models == {}


def test_one_factory_serves_several_models_with_their_own_structure():
    factory = CalculatorFactory()
    thin, thick = _film_model(50.0), _film_model(200.0)
    thin.interface = factory
    thick.interface = factory
    assert set(factory().models) == {thin.unique_name, thick.unique_name}
    assert not np.array_equal(factory.fit_func(Q, thin.unique_name), factory.fit_func(Q, thick.unique_name))


def test_structural_edits_need_no_calls_into_the_calculator():
    factory = CalculatorFactory()
    model = _film_model(100.0)
    model.interface = factory
    before = factory.fit_func(Q, model.unique_name)
    model.add_assemblies(Multilayer(Layer(Material(6.0, 0.0, 'Extra'), 40.0, 2.0, 'ExtraLayer')))
    assert not np.array_equal(factory.fit_func(Q, model.unique_name), before)


def test_a_round_tripped_model_evaluates_as_before():
    factory = CalculatorFactory()
    model = _film_model(100.0)
    model.interface = factory
    expected = factory.fit_func(Q, model.unique_name)
    model_dict = model.as_dict()
    global_object.registry.clear()
    restored = Model.from_dict(model_dict)
    assert restored.interface().name == 'refnx'
    assert list(restored.interface().models) == [restored.unique_name]
    assert_array_equal(restored.interface.fit_func(Q, restored.unique_name), expected)


def _film_model(thickness: float) -> Model:
    sample = Sample(
        Multilayer([
            Layer(Material(0.0, 0.0, 'Ambient'), 0.0, 0.0, 'AmbientLayer'),
            Layer(Material(3.45, 0.0, 'Film'), thickness, 3.0, 'FilmLayer'),
            Layer(Material(2.07, 0.0, 'Substrate'), 0.0, 3.0, 'SubstrateLayer'),
        ])
    )
    return Model(sample, 1.0, 0.0, PercentageFwhm(1.0))
