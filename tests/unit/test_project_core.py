# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Project material-index helpers, resolution auto-selection,
ORSO loading, model data generation, and minimizer serialization."""

import os

import numpy as np
import pytest
from easyscience import global_object

import easyreflectometry
from easyreflectometry.data import DataSet1D
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.project import Project

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def project() -> Project:
    return Project()


class TestGetIndexMaterials:
    def test_get_index_air_adds_material_when_missing(self, project: Project):
        assert len(project._materials) == 0
        index = project.get_index_air()
        assert project._materials[index].name == 'Air'
        assert project._materials[index].sld.value == 0.0

    def test_get_index_si_adds_material_when_missing(self, project: Project):
        index = project.get_index_si()
        assert project._materials[index].name == 'Si'
        assert project._materials[index].sld.value == 2.07

    def test_get_index_sio2_adds_material_when_missing(self, project: Project):
        index = project.get_index_sio2()
        assert project._materials[index].name == 'SiO2'
        assert project._materials[index].sld.value == 3.47

    def test_get_index_d2o_adds_material_when_missing(self, project: Project):
        index = project.get_index_d2o()
        assert project._materials[index].name == 'D2O'
        assert project._materials[index].sld.value == 6.36

    def test_get_index_is_idempotent(self, project: Project):
        first = project.get_index_d2o()
        count_after_first = len(project._materials)
        second = project.get_index_d2o()
        assert first == second
        assert len(project._materials) == count_after_first

    def test_indices_point_at_distinct_materials(self, project: Project):
        indices = {
            'Air': project.get_index_air(),
            'D2O': project.get_index_d2o(),
            'Si': project.get_index_si(),
            'SiO2': project.get_index_sio2(),
        }
        assert len(set(indices.values())) == 4
        for name, index in indices.items():
            assert project._materials[index].name == name


class TestApplyResolutionFunction:
    def test_pointwise_used_when_experiment_has_q_variances(self, project: Project):
        project.default_model()
        model = project.models[0]
        experiment = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=[1e-8, 4e-8])

        project._apply_resolution_function(experiment, model)

        assert isinstance(model.resolution_function, Pointwise)
        # sigma at the data points is sqrt(sQz)
        np.testing.assert_allclose(model.resolution_function.smearing(), np.sqrt([1e-8, 4e-8]))

    def test_percentage_fwhm_fallback_when_q_variances_all_zero(self, project: Project):
        project.default_model()
        model = project.models[0]
        experiment = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=[0.0, 0.0])

        project._apply_resolution_function(experiment, model)

        assert isinstance(model.resolution_function, PercentageFwhm)
        assert model.resolution_function.constant == 5.0

    def test_percentage_fwhm_fallback_when_q_variances_absent(self, project: Project):
        project.default_model()
        model = project.models[0]
        experiment = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1])

        project._apply_resolution_function(experiment, model)

        assert isinstance(model.resolution_function, PercentageFwhm)


class TestLoadOrsoFile:
    def test_load_orso_file_creates_model_and_experiment(self, project: Project):
        # example.ort has no sample.model -> default model; deprecated wrapper
        # now goes through the DataSet1D + title + resolution path.
        with pytest.warns((UserWarning, DeprecationWarning)):
            project.load_orso_file(os.path.join(PATH_STATIC, 'example.ort'))

        assert len(project.models) == 1
        assert len(project.experiments) == 1
        assert isinstance(project.experiments[0], DataSet1D)
        assert project.experiments[0].name == 'Example data file from refnx docs'
        assert project.experiments[0].model is project.models[0]
        assert project._with_experiments is True

    def test_load_orso_file_with_model_builds_sample_from_file(self, project: Project):
        # Ni_example.ort carries a sample.model -> the model comes from the file.
        with pytest.warns(DeprecationWarning):
            project.load_orso_file(os.path.join(PATH_STATIC, 'Ni_example.ort'))

        assert len(project.models) == 1
        assert project.models[0].sample.name == 'Ni on Si'
        assert len(project.experiments) == 1
        assert isinstance(project.experiments[0], DataSet1D)
        assert project.experiments[0].model is project.models[0]
        # The measured sQz feeds a Pointwise resolution function.
        assert isinstance(project.models[0].resolution_function, Pointwise)


class TestModelData:
    def test_model_data_for_model_at_index_returns_reflectivity(self, project: Project):
        project.default_model()
        q_range = np.linspace(0.01, 0.1, 5)

        dataset = project.model_data_for_model_at_index(0, q_range=q_range)

        np.testing.assert_array_equal(dataset.x, q_range)
        assert dataset.y.shape == (5,)
        assert np.all(np.isfinite(dataset.y))
        # Reflectivity near total reflection is of order unity and decays with q
        assert dataset.y[0] > dataset.y[-1]


class TestAsDictMinimizer:
    def test_as_dict_uses_selection_when_no_fitter_exists(self, project: Project):
        # No models -> the lazy fitter property stays None -> fall back to selection
        project_dict = project.as_dict()
        assert project._fitter is None
        assert project_dict['fitter_minimizer'] == project._minimizer_selection.name

    def test_as_dict_reads_minimizer_from_fitter_when_models_exist(self, project: Project):
        project.default_model()
        project_dict = project.as_dict()
        assert project._fitter is not None
        assert project_dict['fitter_minimizer'] == project._fitter.easy_science_multi_fitter.minimizer.name
