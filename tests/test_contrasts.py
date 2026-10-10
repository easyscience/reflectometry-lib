# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Contrasts: deriving, linking, detaching, fitting a subset and recording the run."""

import json
import warnings

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.constraints import constrain_equal
from easyreflectometry.contrasts import LayoutMismatch
from easyreflectometry.contrasts import ReplaceFormula
from easyreflectometry.contrasts import ReplaceMaterial
from easyreflectometry.contrasts import UnsupportedSubstitution
from easyreflectometry.contrasts import derive_contrast
from easyreflectometry.contrasts import substitution_candidates
from easyreflectometry.data import DataSet1D
from easyreflectometry.fitting import FitScopeError
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialDensity
from easyreflectometry.sample import MaterialSolvated
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer

Q = np.linspace(0.01, 0.25, 40)


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _reload(project: Project) -> Project:
    data = json.loads(json.dumps(project.as_dict()))
    global_object.map._clear()
    loaded = Project()
    loaded.from_dict(data)
    return loaded


def _hydrated_project() -> tuple[Project, Material, Material]:
    """Si | oxide | hydrated film (D2O) | D2O bulk; the solvent object is used three times."""
    project = Project()
    si = Material(sld=2.07, isld=0, name='Si')
    d2o = Material(sld=6.36, isld=0, name='D2O')
    h2o = Material(sld=-0.56, isld=0, name='H2O')
    film = Material(sld=3.0, isld=0, name='Film')
    outer = MaterialSolvated(material=film, solvent=d2o, solvent_fraction=0.3, name='Outer')
    inner = MaterialSolvated(material=film, solvent=d2o, solvent_fraction=0.1, name='Inner')
    model = Model(
        sample=Sample(
            Multilayer(Layer(material=si, thickness=0, roughness=0, name='Si'), name='Substrate'),
            Multilayer(Layer(material=Material(sld=3.47, isld=0, name='SiO2'), thickness=12, roughness=3), name='Oxide'),
            Multilayer(
                [
                    Layer(material=inner, thickness=20, roughness=3, name='Inner'),
                    Layer(material=outer, thickness=40, roughness=3, name='Outer'),
                ],
                name='Film',
                conformal_roughness=True,
            ),
            Multilayer(Layer(material=d2o, thickness=0, roughness=3, name='Bulk'), name='Bulk'),
        ),
        name='D2O',
        resolution_function=PercentageFwhm(5),
    )
    project.models = ModelCollection(model)
    return project, d2o, h2o


class TestDeriveSolventContrast:
    def test_shares_what_the_solvent_does_not_touch_and_ties_the_rest(self):
        project, d2o, h2o = _hydrated_project()
        reference = project.models[0]
        contrast = derive_contrast(reference, name='H2O', substitutions=[ReplaceMaterial(d2o, h2o)])

        assert contrast.sample[0] is reference.sample[0]
        assert contrast.sample[1] is reference.sample[1]
        film, reference_film = contrast.sample[2], reference.sample[2]
        assert film is not reference_film
        for layer, reference_layer in zip(film.layers, reference_film.layers):
            assert layer.material.solvent is h2o
            assert layer.material.material is reference_layer.material.material  # the dry film material is shared
            assert list(layer.thickness.dependency_map.values()) == [reference_layer.thickness]
            assert list(layer.material.solvent_fraction.dependency_map.values()) == [reference_layer.material.solvent_fraction]
        assert contrast.sample[3].layers[0].material is h2o
        assert reference_film.layers[0].material.solvent is d2o  # the reference is unchanged

    def test_bulk_and_mixed_slds_both_change_and_follow_the_shared_dry_material(self):
        project, d2o, h2o = _hydrated_project()
        reference = project.models[0]
        contrast = derive_contrast(reference, name='H2O', substitutions=[ReplaceMaterial(d2o, h2o)])
        outer, reference_outer = contrast.sample[2].layers[1].material, reference.sample[2].layers[1].material
        assert outer._sld.value == pytest.approx(0.7 * 3.0 + 0.3 * -0.56)
        reference_outer.material.sld.value = 4.0
        assert outer._sld.value == pytest.approx(0.7 * 4.0 + 0.3 * -0.56)
        assert reference_outer._sld.value == pytest.approx(0.7 * 4.0 + 0.3 * 6.36)

    def test_nuisance_parameters_are_copied_not_tied(self):
        project, d2o, h2o = _hydrated_project()
        reference = project.models[0]
        reference.scale.value, reference.scale.fixed = 0.9, False
        contrast = derive_contrast(reference, name='H2O', substitutions=[ReplaceMaterial(d2o, h2o)])
        assert contrast.scale.value == pytest.approx(0.9) and contrast.scale.independent and not contrast.scale.fixed
        assert contrast.resolution_function is not reference.resolution_function

    def test_unknown_targets_are_rejected_before_anything_is_built(self):
        project, _, h2o = _hydrated_project()
        with pytest.raises(UnsupportedSubstitution):
            derive_contrast(project.models[0], name='x', substitutions=[ReplaceMaterial(Material(name='Elsewhere'), h2o)])
        silicon = project.models[0].sample[0].layers[0]
        with pytest.raises(UnsupportedSubstitution):
            derive_contrast(project.models[0], name='x', substitutions=[ReplaceFormula(silicon, 'C')])
        surfactant = SurfactantLayer()
        project.models[0].add_assemblies(surfactant)
        for formula in ('', 'C10H((', 'Xx2'):
            with pytest.raises(UnsupportedSubstitution):
                derive_contrast(project.models[0], name='x', substitutions=[ReplaceFormula(surfactant.tail_layer, formula)])


class TestDeriveIsotopicContrast:
    def test_a_new_formula_changes_only_the_copy(self):
        surfactant = SurfactantLayer()
        reference = Model(
            sample=Sample(
                Multilayer(Layer(material=Material(sld=0, isld=0, name='Air'), thickness=0, roughness=0)),
                surfactant,
                Multilayer(Layer(material=Material(sld=6.36, isld=0, name='D2O'), thickness=0, roughness=3)),
            )
        )
        contrast = derive_contrast(reference, name='h-tail', substitutions=[ReplaceFormula(surfactant.tail_layer, 'C34H70')])
        tail = contrast.sample[1].tail_layer
        assert surfactant.tail_layer.molecular_formula == 'C32D64'
        assert tail.molecular_formula == 'C34H70'
        assert tail.material.material.sld.value != pytest.approx(surfactant.tail_layer.material.material.sld.value)
        assert list(tail.area_per_molecule_parameter.dependency_map.values()) == [
            surfactant.tail_layer.area_per_molecule_parameter
        ]
        surfactant.tail_layer.thickness.value = 18.0
        assert tail.thickness.value == pytest.approx(18.0)

    def test_a_density_material_keeps_its_density_tied(self):
        oxide = MaterialDensity(chemical_structure='SiO2', density=2.2, name='SiO2')
        reference = Model(sample=Sample(Multilayer(Layer(material=oxide, thickness=10, roughness=2))))
        contrast = derive_contrast(reference, name='d', substitutions=[ReplaceFormula(oxide, 'SiO2D')])
        copy = contrast.sample[0].layers[0].material
        assert copy.chemical_structure == 'SiO2D'
        assert list(copy.density.dependency_map.values()) == [oxide.density]
        assert copy.scattering_length_real.independent


def test_candidates_are_the_substitutable_objects_in_sample_order():
    project, d2o, _ = _hydrated_project()
    surfactant = SurfactantLayer()
    project.models[0].add_assemblies(surfactant)
    materials, formulas = substitution_candidates(project.models[0])
    names = [material.name for material in materials]
    assert names[:5] == ['Si', 'SiO2', 'Inner', 'Film', 'D2O']
    assert d2o in materials and materials.count(d2o) == 1
    # the molecule materials a surfactant layer builds for itself are not offered, its solvents are
    assert surfactant.tail_layer.material.material not in materials
    assert surfactant.head_layer.solvent in materials
    assert formulas == [surfactant.tail_layer, surfactant.head_layer]


class TestAddContrast:
    def test_joins_the_project_and_round_trips(self):
        project, d2o, h2o = _hydrated_project()
        index = project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        assert index == 1
        assert h2o in project._materials
        assert project.models[1].color != project.models[0].color
        assert project.contrast_reference(1) == 0

        loaded = _reload(project)
        reference, contrast = loaded.models
        assert loaded.contrast_reference(1) == 0
        assert contrast.sample[1] is reference.sample[1]
        outer = contrast.sample[2].layers[1]
        reference.sample[2].layers[1].thickness.value = 55.0
        assert outer.thickness.value == pytest.approx(55.0)
        assert outer.material.material is reference.sample[2].layers[1].material.material

    def test_curves_differ_and_survive_reload(self):
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        before = [project.model_data_for_model_at_index(index, q_range=Q).y for index in range(2)]
        assert not np.allclose(*before)
        loaded = _reload(project)
        for index, expected in enumerate(before):
            assert_allclose(loaded.model_data_for_model_at_index(index, q_range=Q).y, expected, rtol=1e-10)


def _two_layouts() -> Project:
    """The default model and a duplicate (every object copied), plus an oxide layer each."""
    project = Project()
    project.default_model()
    project.models.duplicate_model(0)
    return project


class TestLinks:
    def test_plan_ties_geometry_keeps_nuisance_and_leaves_chemistry_undecided(self):
        project = _two_layouts()
        plan = project.plan_link(1, 0)
        actions = {row.path: row.action for row in plan.rows}
        assert actions['scale'] == actions['background'] == 'skip'
        assert actions['sample/1/layers/0/thickness'] == 'tie'
        assert actions['sample/0/layers/0/thickness'] == 'skip'  # superphase thickness: not used
        assert actions['sample/2/layers/0/material/sld'] == 'undecided'
        with pytest.raises(ValueError, match='Decide'):
            project.apply_link(plan)

    def test_apply_ties_and_unlink_restores(self):
        project = _two_layouts()
        follower, reference = project.models[1], project.models[0]
        follower.sample[1].layers[0].thickness.max = 400.0
        follower.sample[1].layers[0].thickness.value = 120.0
        record = project.apply_link(project.plan_link(1, 0, materials='skip'))
        reference.sample[1].layers[0].thickness.value = 90.0
        assert follower.sample[1].layers[0].thickness.value == pytest.approx(90.0)
        assert follower.sample[2].layers[0].material.sld.independent

        loaded = _reload(project)
        assert len(loaded.links) == 1
        assert loaded.unlink(loaded.links[0]) == [1]
        thickness = loaded.models[1].sample[1].layers[0].thickness
        assert thickness.independent
        assert thickness.value == pytest.approx(120.0) and thickness.max == pytest.approx(400.0)
        assert record not in loaded.links

    def test_existing_ties_are_recognised_and_others_reported(self):
        project = _two_layouts()
        follower, reference = project.models[1], project.models[0]
        constrain_equal(follower.sample[1].layers[0].thickness, reference.sample[1].layers[0].thickness)
        constrain_equal(follower.sample[1].layers[0].roughness, reference.sample[2].layers[0].roughness)
        rows = {row.path: row for row in project.plan_link(1, 0, materials='skip').rows}
        assert rows['sample/1/layers/0/thickness'].action == 'already_tied'
        assert rows['sample/1/layers/0/roughness'].action == 'conflict'

    def test_a_parameter_shared_with_a_third_model_is_not_tied_silently(self):
        project = _two_layouts()
        third = Model(sample=Sample(project.models[1].sample[1]), name='third')
        project.models.append(third)
        row = next(row for row in project.plan_link(1, 0, materials='skip').rows if row.path == 'sample/1/layers/0/thickness')
        assert (row.action, row.shared_with) == ('skip', [2])

    def test_a_cycle_is_refused(self):
        project = _two_layouts()
        constrain_equal(project.models[0].sample[1].layers[0].thickness, project.models[1].sample[1].layers[0].thickness)
        row = next(row for row in project.plan_link(1, 0, materials='skip').rows if row.path == 'sample/1/layers/0/thickness')
        assert row.action == 'conflict'

    def test_models_sharing_their_materials_differently_still_correspond(self):
        # The default model's layers use the palette's own materials; a contrast with the
        # substrate replaced uses another object there. Layer by layer they still match.
        project = Project()
        project.default_model()
        silicon = project.models[0].sample[2].layers[0].material
        project.models[0].sample[1].layers[0].material = silicon  # one material in two layers
        replacement = Material(sld=1.0, isld=0, name='Other')
        project.add_material(replacement)
        project.add_contrast(0, 'other substrate', [ReplaceMaterial(silicon, replacement)])
        plan = project.plan_link(1, 0, materials='skip')
        rows = [row for row in plan.rows if row.path.endswith('layers/0/material/sld')]
        assert [row.path for row in rows] == [
            'sample/0/layers/0/material/sld',
            'sample/1/layers/0/material/sld',
            'sample/2/layers/0/material/sld',
        ]

    def test_mismatched_layouts_are_refused(self):
        project = _two_layouts()
        project.models[1].add_assemblies(Multilayer())
        with pytest.raises(LayoutMismatch):
            project.plan_link(1, 0)

    def test_a_late_failure_ties_nothing(self, monkeypatch):
        project = _two_layouts()
        plan = project.plan_link(1, 0, materials='skip')
        tie_rows = [row for row in plan.rows if row.action == 'tie']
        tie_rows[-1].reference = 'not a parameter'
        with pytest.raises(Exception):
            project.apply_link(plan)
        assert all(row.follower.independent for row in tie_rows)
        assert project.links == []

    def test_records_of_replaced_models_are_not_saved(self):
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        project.apply_link(project.plan_link(1, 0, materials='tie'))
        project.models = ModelCollection(Model(name='new'))
        # A wholesale replacement replaces the palette too (otherwise its old, tied materials stay).
        project._replace_collection(project._get_materials_in_models(), project._materials)
        data = project.as_dict()
        assert 'contrasts' not in data and 'links' not in data
        _reload(project)

    def test_removing_the_reference_restores_the_followers(self):
        project = _two_layouts()
        thickness = project.models[1].sample[1].layers[0].thickness
        thickness.value = 120.0
        project.apply_link(project.plan_link(1, 0, materials='skip'))
        project.remove_model_at_index(0)
        assert thickness.independent and thickness.value == pytest.approx(120.0)
        assert project.links == []


class TestDetach:
    def test_detaching_a_tie_frees_one_parameter(self):
        project = _two_layouts()
        project.apply_link(project.plan_link(1, 0, materials='skip'))
        thickness = project.models[1].sample[1].layers[0].thickness
        project.detach(thickness)
        assert thickness.independent
        assert not project.models[1].sample[1].layers[0].roughness.independent

    def test_detaching_from_a_shared_assembly_copies_it_for_this_model(self):
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        reference, contrast = project.models
        oxide = reference.sample[1]
        project.detach(oxide.layers[0].thickness, 1)

        copy = contrast.sample[1]
        assert copy is not oxide and reference.sample[1] is oxide
        assert copy.layers[0].thickness.independent
        assert list(copy.layers[0].roughness.dependency_map.values()) == [oxide.layers[0].roughness]
        assert copy.layers[0].material is oxide.layers[0].material
        before = project.model_data_for_model_at_index(1, q_range=Q).y
        copy.layers[0].thickness.value = 15.0
        assert not np.allclose(project.model_data_for_model_at_index(1, q_range=Q).y, before)

    def test_a_derived_parameter_cannot_be_detached(self):
        project = Project()
        oxide = MaterialDensity(chemical_structure='SiO2', density=2.2, name='SiO2')
        project.models = ModelCollection(Model(sample=Sample(Multilayer(Layer(material=oxide, thickness=10, roughness=2)))))
        with pytest.raises(ValueError, match='derived'):
            project.detach(oxide.sld)
        assert not oxide.sld.independent

    def test_a_model_without_the_parameter_is_refused(self):
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        project.models.append(Model(name='other'))
        with pytest.raises(ValueError, match='not part of'):
            project.detach(project.models[0].sample[1].layers[0].thickness, 2)

    def test_a_shared_material_cannot_be_detached(self):
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        with pytest.raises(ValueError, match='material'):
            project.detach(project.models[0].sample[1].layers[0].material.sld, 1)


def _synthetic(project: Project, index: int, noise: float = 0.0) -> DataSet1D:
    model = project.models[index]
    y = project.model_data_for_model_at_index(index, q_range=Q).y
    rng = np.random.default_rng(index)
    y = y * (1 + noise * rng.standard_normal(Q.size))
    dataset = DataSet1D(name=model.name, x=Q, y=y, ye=(0.02 * y) ** 2, model=model, auto_background=False)
    return dataset


class TestFitScope:
    def test_a_follower_alone_still_varies_the_root_it_follows(self):
        project = _two_layouts()
        root = project.models[0].sample[1].layers[0].thickness
        root.fixed = False
        project.apply_link(project.plan_link(1, 0, materials='skip'))
        follower_data = _synthetic(project, 1)
        project.experiments = {0: _synthetic(project, 0), 1: follower_data}
        project.experiments[0].include_in_fit = False

        prepared = project.prepare_fit()

        assert prepared.added_roots == [root]
        assert any(parameter is root for parameter in prepared.core_fitter.fit_object.get_fit_parameters())
        assert len(prepared.fitted) == 1  # the excluded data is not in the objective
        assert prepared.labels == [(1, follower_data.name, None)]

    def test_an_experiment_without_a_model_is_an_error_unless_skipped(self):
        project = _two_layouts()
        project.experiments = {0: _synthetic(project, 0), 1: _synthetic(project, 1)}
        project.experiments[1].model = None
        with pytest.raises(FitScopeError, match='no model'):
            project.prepare_fit()
        assert project.prepare_fit(skip_invalid=True).skipped == [project.experiments[1].name]

    def test_nothing_included_is_an_error(self):
        project = _two_layouts()
        project.experiments = {0: _synthetic(project, 0)}
        project.experiments[0].include_in_fit = False
        with pytest.raises(FitScopeError):
            project.prepare_fit()

    def test_inclusion_round_trips_and_resets_on_a_new_import(self):
        project = _two_layouts()
        project.experiments = {0: _synthetic(project, 0), 1: _synthetic(project, 1)}
        project._with_experiments = True
        project.experiments[0].include_in_fit = False
        loaded = _reload(project)
        assert [experiment.include_in_fit for experiment in loaded.experiments.values()] == [False, True]
        loaded.remove_experiment(0)
        assert loaded.experiments[0].include_in_fit


class TestRunRecord:
    def _fitted(self) -> Project:
        project, d2o, h2o = _hydrated_project()
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
        project.experiments = {0: _synthetic(project, 0, noise=0.01), 1: _synthetic(project, 1, noise=0.01)}
        prepared = project.prepare_fit()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = prepared.execute()
        project.record_fit(prepared, results)
        return project

    def test_statistics_partition_and_survive_a_model_switch(self):
        project = self._fitted()
        run = project.last_fit
        assert run.status == 'completed'
        assert [entry[:2] for entry in run.inputs] == [(0, 'D2O'), (1, 'H2O')]
        total = sum(entry['objective_chi2'] for entry in run.per_dataset)
        assert total == pytest.approx(run.pooled['objective_chi2'])
        assert sum(entry['share_of_objective'] for entry in run.per_dataset) == pytest.approx(1.0)
        assert run.pooled['objective_dof'] == run.pooled['objective_n_points'] - run.n_free_parameters
        project.current_model_index = 1
        assert project.last_fit is run

    def test_removing_an_experiment_remaps_the_record(self):
        project = self._fitted()
        project.remove_experiment(0)
        assert [entry[0] for entry in project.last_fit.inputs] == [None, 0]

    def test_a_failed_run_has_no_statistics(self):
        project = _two_layouts()
        project.experiments = {0: _synthetic(project, 0)}
        run = project.record_fit(project.prepare_fit(), None, status='failed')
        assert run.pooled == {} and run.per_dataset == ()


@pytest.mark.slow
def test_a_joint_fit_recovers_the_shared_thickness():
    project, d2o, h2o = _hydrated_project()
    project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, h2o)])
    film = project.models[0].sample[2].layers[1]
    project.experiments = {0: _synthetic(project, 0, noise=0.005), 1: _synthetic(project, 1, noise=0.005)}
    film.thickness.value = 32.0
    film.thickness.fixed = False
    for model in project.models:
        model.scale.fixed = False
    prepared = project.prepare_fit()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        project.record_fit(prepared, prepared.execute())
    assert film.thickness.value == pytest.approx(40.0, rel=0.02)
    assert project.models[1].sample[2].layers[1].thickness.value == pytest.approx(film.thickness.value)
