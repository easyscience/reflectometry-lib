# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Objects shared between layers and models keep being shared after save/load."""

import json

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.constraints import constrain
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialSolvated
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.01, 0.25, 30)


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _reload(project: Project, calculator: str | None = None) -> Project:
    """Save through JSON and load into a fresh project, as a file round trip does."""
    data = json.loads(json.dumps(project.as_dict()))
    global_object.map._clear()
    loaded = Project()
    loaded.from_dict(data)
    if calculator is not None:
        loaded.calculator = calculator
    return loaded


def _contrasts() -> Project:
    """Two contrasts: one hydrated film assembly shared by both models, different bulk media."""
    project = Project()
    si = Material(sld=2.07, isld=0, name='Si')
    d2o = Material(sld=6.36, isld=0, name='D2O')
    h2o = Material(sld=-0.56, isld=0, name='H2O')
    film = Material(sld=3.0, isld=0, name='Film')
    hydrated = MaterialSolvated(material=film, solvent=d2o, solvent_fraction=0.2, name='Hydrated')
    inner = MaterialSolvated(material=film, solvent=d2o, solvent_fraction=0.4, name='Inner')
    structure = Multilayer(
        [
            Layer(material=hydrated, thickness=40, roughness=3, name='Outer'),
            Layer(material=inner, thickness=20, roughness=3, name='Inner'),
        ],
        conformal_roughness=True,
        name='Film',
    )

    def model(name, bulk, scale):
        return Model(
            sample=Sample(
                Multilayer(Layer(material=si, thickness=0, roughness=0, name='Si')),
                structure,
                Multilayer(Layer(material=bulk, thickness=0, roughness=3, name=bulk.name)),
            ),
            scale=scale,
            name=name,
        )

    project.models = ModelCollection(model('D2O contrast', d2o, 1.0), model('H2O contrast', h2o, 0.9))
    return project


class TestSharedObjects:
    def test_shared_assembly_and_materials_are_one_object_after_reload(self):
        loaded = _reload(_contrasts())
        first, second = loaded.models
        assert first.sample[1] is second.sample[1]
        outer, inner = first.sample[1].layers
        assert outer.material.solvent is inner.material.solvent is first.sample[2].layers[0].material
        assert outer.material.material is inner.material.material
        assert first.sample[0].layers[0].material is second.sample[0].layers[0].material

    def test_internal_dependencies_target_the_shared_objects(self):
        loaded = _reload(_contrasts())
        outer, inner = loaded.models[0].sample[1].layers
        assert not inner.roughness.independent
        assert list(inner.roughness._dependency_map.values()) == [outer.roughness]
        mixture = outer.material
        mixture.solvent.sld.value = -0.56
        assert mixture._sld.value == pytest.approx(0.8 * 3.0 + 0.2 * -0.56)

    def test_an_edit_reaches_every_owner(self):
        loaded = _reload(_contrasts())
        loaded.models[0].sample[1].layers[0].thickness.value = 55.0
        assert loaded.models[1].sample[1].layers[0].thickness.value == 55.0

    def test_parameters_and_free_set_are_unchanged(self):
        project = _contrasts()
        project.models[0].sample[1].layers[0].thickness.fixed = False
        before = sorted(parameter.name for parameter in project.parameters)
        loaded = _reload(project)
        assert sorted(parameter.name for parameter in loaded.parameters) == before
        free = [parameter for parameter in loaded.parameters if parameter.independent and not parameter.fixed]
        assert free == [loaded.models[0].sample[1].layers[0].thickness]

    @pytest.mark.parametrize('calculator', ['refnx', 'refl1d'])
    def test_curves_are_unchanged(self, calculator):
        project = _contrasts()
        project.calculator = calculator
        before = [project.model_data_for_model_at_index(index, q_range=Q).y for index in range(2)]
        loaded = _reload(project, calculator)
        after = [loaded.model_data_for_model_at_index(index, q_range=Q).y for index in range(2)]
        assert not np.allclose(before[0], before[1])
        for expected, actual in zip(before, after):
            assert_allclose(actual, expected, rtol=1e-10)

    def test_a_user_constraint_on_a_shared_parameter_round_trips(self):
        project = _contrasts()
        structure = project.models[0].sample[1]
        constrain(structure.layers[1].thickness, '0.5 * t', t=structure.layers[0].thickness)
        loaded = _reload(project)
        outer, inner = loaded.models[1].sample[1].layers
        outer.thickness.value = 60.0
        assert inner.thickness.value == pytest.approx(30.0)


def test_an_object_in_two_sibling_slots_stays_one_object():
    from easyreflectometry.sample import MaterialMixture

    project = Project()
    pure = Material(sld=2.0, isld=0, name='Pure')
    mixture = MaterialMixture(pure, pure, 0.3, name='Same twice')
    project.models = ModelCollection(Model(sample=Sample(Multilayer(Layer(material=mixture)))))
    loaded = _reload(project)
    loaded_mixture = loaded.models[0].sample[0].layers[0].material
    assert loaded_mixture.material_a is loaded_mixture.material_b
    assert loaded.load_report == []


class TestPalette:
    def test_no_duplicates_after_reload(self):
        loaded = _reload(_contrasts())
        assert [material.name for material in loaded._materials] == ['Si', 'Hydrated', 'Inner', 'D2O', 'H2O']

    def test_order_and_unused_materials_round_trip(self):
        project = Project()
        project.default_model()
        unused = Material(sld=1.0, isld=0, name='Unused')
        project.add_material(unused)
        project._materials.move_up(1)
        names = [material.name for material in project._materials]
        assert names[:2] == ['D2O', 'Air']
        loaded = _reload(project)
        assert [material.name for material in loaded._materials] == names
        # The palette holds the models' own objects, not copies.
        assert loaded._materials[1] is loaded.models[0].sample[0].layers[0].material

    def test_distinct_materials_with_one_name_stay_distinct(self):
        project = Project()
        first = Material(sld=1.0, isld=0, name='Same')
        second = Material(sld=1.0, isld=0, name='Same')
        project.models = ModelCollection(
            Model(sample=Sample(Multilayer(Layer(material=first)), Multilayer(Layer(material=second))))
        )
        loaded = _reload(project)
        sample = loaded.models[0].sample
        assert sample[0].layers[0].material is not sample[1].layers[0].material


class TestFileFormat:
    def test_a_project_without_shared_model_objects_keeps_the_old_format(self):
        project = Project()
        project.models = ModelCollection(Model())
        assert project.as_dict()['file_format'] == Project.FILE_FORMAT

    def test_sharing_raises_the_format(self):
        assert _contrasts().as_dict()['file_format'] == Project.FILE_FORMAT_SHARED

    def test_a_missing_shared_object_is_reported_not_raised(self):
        data = json.loads(json.dumps(_contrasts().as_dict()))
        data['models']['data'][1]['sample']['data'][1]['@ref'] = 'no-such-id'
        data['fit_settings'] = {'minimizer': 'NoSuchEngine'}
        global_object.map._clear()
        loaded = Project()
        loaded.from_dict(data)
        assert any('missing from the file' in line for line in loaded.load_report)
        assert any('NoSuchEngine' in line for line in loaded.load_report)


class TestExperimentModels:
    def _project(self) -> Project:
        project = Project()
        project.default_model()
        project.models.duplicate_model(0)
        project.models[1].name = project.models[0].name
        return project

    def _dataset(self, model) -> DataSet1D:
        return DataSet1D(name='data', x=Q, y=np.exp(-Q), ye=np.full(Q.size, 1e-4), model=model, auto_background=False)

    def test_pairing_survives_identical_model_names(self):
        project = self._project()
        project.experiments = {0: self._dataset(project.models[0]), 1: self._dataset(project.models[1])}
        project._with_experiments = True
        loaded = _reload(project)
        assert loaded.experiments[1].model is loaded.models[1]
        assert loaded.load_report == []

    def test_a_file_without_indices_reports_an_ambiguous_name(self):
        project = self._project()
        project.experiments = {0: self._dataset(project.models[1])}
        project._with_experiments = True
        data = json.loads(json.dumps(project.as_dict()))
        del data['experiments_model_indices']
        global_object.map._clear()
        loaded = Project()
        loaded.from_dict(data)
        assert loaded.experiments[0].model is loaded.models[0]
        assert any('several models' in line for line in loaded.load_report)

    def test_experiments_without_a_model_load_unbound(self):
        project = Project()
        project.default_model()
        plain = self._dataset(None)
        channel = self._dataset(None)
        project.experiments = {0: plain, 1: PolarizedDataSet(name='p', channels={'pp': channel}, model=None)}
        project._with_experiments = True
        loaded = _reload(project)
        assert loaded.experiments[0].model is None
        assert loaded.experiments[1].model is None
        assert len(loaded.load_report) == 2
