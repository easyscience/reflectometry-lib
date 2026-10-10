# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Removing or moving a model accounts for everything that refers to it."""

import json

import numpy as np
import pytest
from easyscience import global_object

from easyreflectometry.constraints import constrain
from easyreflectometry.constraints import constrain_equal
from easyreflectometry.data import DataSet1D
from easyreflectometry.inequality_constraints import InequalitySpec
from easyreflectometry.model import Model
from easyreflectometry.project import Project
from easyreflectometry.sample import Sample


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _project(models: int = 3) -> Project:
    """Default model plus duplicates; model 1 shares model 0's film assembly."""
    project = Project()
    project.default_model()
    for _ in range(models - 1):
        project.models.duplicate_model(0)
    reference = project.models[0]
    shared = Model(sample=Sample(reference.sample[0], reference.sample[1], project.models[1].sample[2]), name='shares')
    shared.interface = project._calculator
    project.models.pop(1)
    project.models.insert(1, shared)
    return project


def _film(model):
    return model.sample[1].layers[0]


def _dataset(model, name='d') -> DataSet1D:
    q = np.linspace(0.01, 0.2, 5)
    return DataSet1D(name=name, x=q, y=np.exp(-q), ye=np.full(5, 1e-4), model=model, auto_background=False)


def _round_trip(project: Project) -> Project:
    data = json.loads(json.dumps(project.as_dict()))
    global_object.map._clear()
    loaded = Project()
    loaded.from_dict(data)
    return loaded


class TestPlan:
    def test_lists_experiments_dependents_and_inequalities_without_changing_anything(self):
        project = _project()
        reference, follower = project.models[0], project.models[2]
        constrain_equal(follower.scale, reference.scale)
        project.experiments = {0: _dataset(follower), 1: _dataset(reference)}
        spec = InequalitySpec('a', '<', '2', lhs_paths={'a': 'models/0/scale'})
        project.add_inequality_constraint(spec, validate=False)

        plan = project.plan_model_removal(0)

        assert plan.experiments == [1]
        assert plan.dependents == [follower.scale]
        assert plan.inequality_constraints == [spec]
        assert not follower.scale.independent
        assert len(project.models) == 3

    def test_a_parameter_shared_with_a_survivor_is_not_removed(self):
        project = _project()
        constrain(_film(project.models[2]).thickness, '2 * t', t=_film(project.models[0]).thickness)
        plan = project.plan_model_removal(0)
        # The film is shared with model 1, so model 2's tie survives.
        assert plan.dependents == []


class TestRemove:
    def test_followers_become_independent_and_the_project_saves_and_loads(self):
        project = _project()
        reference, follower = project.models[0], project.models[2]
        reference.scale.value = 0.7
        constrain_equal(follower.scale, reference.scale)

        project.remove_model_at_index(0)

        assert follower.scale.independent
        assert follower.scale.value == pytest.approx(0.7)
        loaded = _round_trip(project)
        assert loaded.models[1].scale.value == pytest.approx(0.7)

    def test_a_tie_to_a_shared_parameter_survives_and_round_trips(self):
        project = _project()
        constrain(_film(project.models[2]).thickness, '2 * t', t=_film(project.models[0]).thickness)
        project.remove_model_at_index(0)
        loaded = _round_trip(project)
        _film(loaded.models[0]).thickness.value = 60.0
        assert _film(loaded.models[1]).thickness.value == pytest.approx(120.0)

    def test_experiments_are_rebound_on_request(self):
        project = _project()
        survivor = project.models[2]
        project.experiments = {0: _dataset(project.models[0])}
        project.remove_model_at_index(0, experiments=2)
        assert project.experiments[0].model is survivor

    def test_inequalities_follow_their_parameters_or_go(self):
        project = _project()
        on_removed = InequalitySpec('a', '<', '2', lhs_paths={'a': 'models/0/scale'})
        on_survivor = InequalitySpec('a', '<', '2', lhs_paths={'a': 'models/2/scale'})
        on_shared = InequalitySpec('a', '>', '1', lhs_paths={'a': 'models/0/sample/1/layers/0/thickness'})
        for spec in (on_removed, on_survivor, on_shared):
            project.add_inequality_constraint(spec, validate=False)
        survivor_scale = project.models[2].scale

        project.remove_model_at_index(0)

        assert project.inequality_constraints == [on_survivor, on_shared]
        assert project.resolve_parameter_path(on_survivor.lhs_paths['a']) is survivor_scale
        assert on_shared.lhs_paths['a'] == 'models/0/sample/1/layers/0/thickness'


def test_moving_a_model_repaths_inequalities():
    project = _project()
    spec = InequalitySpec('a', '<', '2', lhs_paths={'a': 'models/2/scale'})
    project.add_inequality_constraint(spec, validate=False)
    scale = project.models[2].scale
    project.move_model(2, 0)
    assert spec.lhs_paths['a'] == 'models/0/scale'
    assert project.resolve_parameter_path(spec.lhs_paths['a']) is scale


def test_moving_a_model_keeps_the_current_one():
    project = _project()
    current = project.models[0]
    project.move_model(0, 2)
    assert project.models[project.current_model_index] is current


def test_restoring_links_from_a_bad_file_is_reported():
    project = _project()
    data = json.loads(json.dumps(project.as_dict()))
    data['contrasts'] = [[0, 9]]
    data['links'] = [{'follower': 9, 'reference': 0, 'pairs': []}]
    global_object.map._clear()
    loaded = Project()
    loaded.from_dict(data)
    assert len(loaded.load_report) == 2
    assert loaded.links == [] and loaded.contrast_reference(0) is None
