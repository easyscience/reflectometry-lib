# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import Parameter
from numpy.testing import assert_allclose

import easyreflectometry
from easyreflectometry.data import DataSet1D
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialCollection
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestProject:
    def test_constructor(self):
        # When Then
        project = Project()

        # Expect
        assert project._info == {
            'name': 'DefaultEasyReflectometryProject',
            'short_description': 'Reflectometry, 1D',
            'modified': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        }
        assert project._path_project_parent == Path(os.path.expanduser('~'))
        assert len(project._materials) == 0
        assert len(project._models) == 0
        assert project._calculator.current_interface_name == 'refnx'
        assert project._experiments == {}
        assert project._report is None
        assert project._created is False
        assert project._with_experiments is False
        assert project._current_material_index == 0
        assert project._current_model_index == 0
        assert project._current_assembly_index == 0
        assert project._current_layer_index == 0
        assert project._fitter_model_index is None
        assert project._fitter is None
        assert project._q_min is None
        assert project._q_max is None
        assert project._q_resolution is None

    def test_reset(self):
        # When
        project = Project()
        project._info['name'] = 'Test Project'
        project._materials.append(Material())
        project._models.append(Model())
        project._calculator = 'calculator'
        project._experiments = 'experiments'
        project._report = 'report'
        project._created = True
        project._with_experiments = True
        project._path_project_parent = 'project_path'
        project._fitter = 'fitter'
        project._current_material_index = 10
        project._current_model_index = 10
        project._current_assembly_index = 10
        project._current_layer_index = 10
        project._fitter_model_index = 10
        project._q_min = 'q_min'
        project._q_max = 'q_max'
        project._q_resolution == 'q_resolution'
        # Then
        project.reset()

        # Expect
        assert project._info == {
            'name': 'DefaultEasyReflectometryProject',
            'short_description': 'Reflectometry, 1D',
            'modified': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        }
        assert project._models.unique_name == 'project_models'
        assert len(project._models) == 0
        assert project._materials.unique_name == 'project_materials'
        assert len(project._materials) == 0

        assert project._path_project_parent == Path(os.path.expanduser('~'))
        assert project._calculator.current_interface_name == 'refnx'
        assert project._experiments == {}
        assert project._report is None
        assert project._created is False
        assert project._with_experiments is False
        assert global_object.map.vertices() == ['project_models', 'project_materials']
        assert project._fitter is None
        assert project._current_material_index == 0
        assert project._current_model_index == 0
        assert project._current_assembly_index == 0
        assert project._current_layer_index == 0
        assert project._fitter_model_index is None
        assert project._q_min is None
        assert project._q_max is None
        assert project._q_resolution is None

    def test_models(self):
        # When
        project = Project()
        models = ModelCollection(Model())
        material = Material()
        project._materials.append(material)

        # Then
        project.models = models

        # Expect
        def remove_interface(d):
            if isinstance(d, dict):
                if 'interface' in d:
                    del d['interface']
                for v in d.values():
                    remove_interface(v)
            elif isinstance(d, list):
                for item in d:
                    remove_interface(item)

        project_models_dict = project.models.as_dict()
        models_dict = models.as_dict()
        models_dict['unique_name'] = 'project_models'
        remove_interface(project_models_dict)
        remove_interface(models_dict)
        # Since as_dict may not include unique_name,
        # remove it for comparison
        for d in [project_models_dict, models_dict]:
            if 'unique_name' in d:
                del d['unique_name']
        assert project_models_dict == models_dict

        assert len(project._materials) == 3
        assert project._materials[0] == material
        assert project._materials[1] == models[0].sample[0].layers[0].material
        assert project._materials[2] == models[0].sample[1].layers[0].material
        assert project.models[0].interface == project._calculator

    def test_default_model(self):
        # When
        global_object.map._clear()
        project = Project()

        # Then
        project.default_model()

        # Expect
        assert len(project._models) == 1
        assert project._models[0].unique_name == 'Model_0'
        assert len(project._models.data[0].sample) == 3
        assert len(project._materials) == 3

    def test_sld_data_for_model_at_index(self):
        # When
        project = Project()
        project.default_model()

        # Then
        sample_data = project.sld_data_for_model_at_index(0)

        # Expect
        assert len(sample_data.x) == 500
        assert_allclose(
            np.array([4.6119497e-08, 6.3189932e00, 6.3350000e00, 2.0740000e00]),
            np.array([
                sample_data.y[0],
                sample_data.y[100],
                sample_data.y[300],
                sample_data.y[499],
            ]),
        )

    def test_sample_data_for_model_at_index(self):
        # When
        project = Project()
        project.default_model()

        # Then
        sample_data = project.sample_data_for_model_at_index(0, np.array([0.01, 0.05, 0.1, 0.5]))

        # Expect
        assert len(sample_data.y) == 4
        assert_allclose(
            np.array([1.00000001e00, 1.74684509e-03, 1.66360864e-04, 1.73359103e-08]),
            sample_data.y,
        )

    def test_model_data_for_model_at_index(self):
        # When
        project = Project()
        project.default_model()

        # Then
        model_data = project.model_data_for_model_at_index(0, np.array([0.01, 0.05, 0.1, 0.5]))

        # Expect
        assert len(model_data.y) == 4
        assert_allclose(
            np.array([
                0.9738701849233727,
                0.0017678986451491123,
                0.00016581714423990004,
                3.3290653551465554e-08,
            ]),
            model_data.y,
        )

    def test_minimizer(self):
        # When
        project = Project()

        # Then Expect
        assert project.minimizer == AvailableMinimizers.LMFit_leastsq

    def test_set_minimizer(self):
        # When
        project = Project()
        project._fitter = MagicMock()
        project._fitter.easy_science_multi_fitter = MagicMock()
        project._fitter.easy_science_multi_fitter.switch_minimizer = MagicMock()

        # Then
        project.minimizer = 'minimizer'

        # Expect
        project._fitter.easy_science_multi_fitter.switch_minimizer.assert_called_once_with('minimizer')

    def test_fitter_none(self):
        # When
        project = Project()

        # Then Expect
        assert project.fitter is None

    def test_fitter_model(self):
        # When
        project = Project()
        project.default_model()

        # Then Expect
        assert isinstance(project.fitter, MultiFitter)

    def test_fitter_same_model_index(self):
        # When
        project = Project()
        project.default_model()
        fitter_0 = project.fitter
        project._models.append(Model())

        # Then
        fitter_1 = project.fitter

        # Expect
        assert fitter_0 is fitter_1

    def test_fitter_new_model_index(self):
        # When
        project = Project()
        project.default_model()
        fitter_0 = project.fitter
        model = Model()
        project._models.append(model)
        project._models[1].interface = project._models[0].interface
        project._current_model_index = 1

        # Then
        fitter_1 = project.fitter

        # Expect
        assert fitter_0 is not fitter_1

    def test_switch_calculator_rebuilds_model_bindings(self):
        # When
        project = Project()
        project.default_model()

        # Then
        project.calculator = 'refl1d'
        reflectivity = project.model_data_for_model_at_index(0, np.array([0.01, 0.05, 0.1, 0.5]))

        # Expect
        assert project.calculator == 'refl1d'
        assert len(reflectivity.y) == 4
        assert np.all(np.isfinite(reflectivity.y))

    def test_experiments(self):
        # When
        project = Project()

        # Then
        project.experiments = 'experiments'

        # Expect
        assert project.experiments == 'experiments'

    def test_path_json(self, tmp_path):
        # When
        project = Project()
        project.set_path_project_parent(tmp_path)

        # Then Expect
        assert project.path_json == Path(tmp_path) / 'DefaultEasyReflectometryProject' / 'project.json'

    def test_add_material(self):
        # When
        project = Project()
        material = Material()

        # Then
        project.add_material(material)

        # Expect
        assert len(project._materials) == 1
        assert project._materials[0] == material

    def test_remove_material(self):
        # When
        project = Project()
        material = Material()
        project.add_material(material)

        # Then
        project.remove_material(0)

        # Expect
        assert len(project._materials) == 0

    def test_remove_material_in_model(self):
        # When
        project = Project()
        model = Model()
        models = ModelCollection(model)
        project.models = models

        # Then
        project.remove_material(0)

        # Expect
        assert len(project._materials) == 2

    def test_default_info(self):
        # When
        project = Project()

        # Then
        info = project._default_info()

        # Expect
        assert info == {
            'name': 'DefaultEasyReflectometryProject',
            'short_description': 'Reflectometry, 1D',
            'modified': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        }

    def test_as_dict(self):
        # When
        project = Project()

        # Then
        project_dict = project.as_dict()

        # Expect
        keys = list(project_dict.keys())
        keys.sort()
        assert keys == [
            'calculator',
            'file_format',
            'fitter_minimizer',
            'info',
            'models',
            'with_experiments',
        ]
        assert project_dict['file_format'] == Project.FILE_FORMAT
        assert project_dict['info'] == {
            'name': 'DefaultEasyReflectometryProject',
            'short_description': 'Reflectometry, 1D',
            'modified': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        }
        assert project_dict['calculator'] == 'refnx'
        assert project_dict['models']['data'] == []
        assert project_dict['with_experiments'] is False

    def test_as_dict_models(self):
        # When
        project = Project()
        models = ModelCollection(Model())
        project.models = models

        # Then
        project_dict = project.as_dict()

        # Expect
        def remove_interface(d):
            if isinstance(d, dict):
                if 'interface' in d:
                    del d['interface']
                for v in d.values():
                    remove_interface(v)
            elif isinstance(d, list):
                for item in d:
                    remove_interface(item)

        models_dict = models.as_dict()
        models_dict['unique_name'] = 'project_models_to_prevent_collisions_on_load'
        remove_interface(models_dict)
        remove_interface(project_dict['models'])
        assert project_dict['models'] == models_dict

    def test_from_dict_missing_file_format_raises(self):
        """Loading a dict without file_format should raise ValueError."""
        project = Project()
        bad_dict = {'info': {}, 'with_experiments': False, 'models': {'data': []}}
        with pytest.raises(ValueError, match='predates file_format=2'):
            project.from_dict(bad_dict)

    def test_from_dict_wrong_file_format_raises(self):
        """Loading a dict with an unsupported file_format should raise ValueError."""
        project = Project()
        bad_dict = {
            'file_format': 99,
            'info': {},
            'with_experiments': False,
            'models': {'data': []},
        }
        with pytest.raises(ValueError, match='Unsupported project file_format'):
            project.from_dict(bad_dict)

    def test_from_dict_correct_file_format_succeeds(self):
        """Loading a dict with the correct file_format should work."""
        global_object.map._clear()
        # Build a valid project dict with at least one model
        src_project = Project()
        src_project._info['name'] = 'Test'
        src_project._info['short_description'] = 'Desc'
        src_project._info['modified'] = '01.01.2025 00:00'
        src_project.default_model()  # ensures at least one model exists
        src_project._with_experiments = False
        good_dict = src_project.as_dict()
        global_object.map._clear()

        project = Project()
        project.from_dict(good_dict)
        assert project._info['name'] == 'Test'
        assert project._with_experiments is False
        assert len(project._models) >= 1

    def test_as_dict_materials_not_in_model(self):
        # When
        project = Project()
        models = ModelCollection(Model())
        project.models = models
        material = Material()
        project.add_material(material)

        # Then
        project_dict = project.as_dict(include_materials_not_in_model=True)

        # Expect
        assert project_dict['materials_not_in_model']['data'][0] == material.as_dict(skip=['interface'])

    def test_as_dict_minimizer(self):
        # When
        project = Project()
        project._fitter = MagicMock()
        project._fitter.easy_science_multi_fitter = MagicMock()
        project._fitter.easy_science_multi_fitter.minimizer = AvailableMinimizers.LMFit

        # Then
        project_dict = project.as_dict()

        # Expect
        assert project_dict['fitter_minimizer'] == 'LMFit'

    def test_replace_collection(self):
        # When
        project = Project()
        material = Material()
        project._materials.append(material)
        new_material = Material()

        # Then
        project._replace_collection(MaterialCollection(new_material), project._materials)

        # Expect
        assert project._materials[0] == new_material
        assert project._materials.unique_name == 'project_materials'

    def test_get_materials_in_models(self):
        # When
        models = ModelCollection(Model())
        project = Project()
        project.models = models
        material = Material(6.908, -0.278, 'Boron')
        project.add_material(material)

        # Then
        materials = project._get_materials_in_models()

        # Expect
        assert len(materials) == 2
        assert materials[0] == models[0].sample[0].layers[0].material
        assert materials[1] == models[0].sample[1].layers[0].material

    def test_dict_round_trip(self):
        # When
        global_object.map._clear()
        project = Project()
        models = ModelCollection(Model())
        project.models = models
        material = Material(6.908, -0.278, 'Boron')
        project.add_material(material)
        minimizer = AvailableMinimizers.LMFit
        project.minimizer = minimizer
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        project_dict = project.as_dict(include_materials_not_in_model=True)
        project_materials_dict = project._materials.as_dict()

        del material
        global_object.map._clear()

        # Then
        new_project = Project()
        new_project.from_dict(project_dict)
        new_project_dict = new_project.as_dict(include_materials_not_in_model=True)
        new_project_materials_dict = new_project._materials.as_dict()

        # Expect
        keys = list(project_dict.keys())
        for key in keys:
            assert project_dict[key] == new_project_dict[key]
        assert project_materials_dict == new_project_materials_dict

    def test_dict_round_trip_experiment_without_xe(self):
        # When - an experiment whose x-uncertainty was explicitly cleared
        global_object.map._clear()
        project = Project()
        project.models = ModelCollection(Model(name='First'), Model(name='Second'))
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath, 1)
        project.experiments[1].xe = None
        project_dict = project.as_dict()

        # Expect - name and model are recorded regardless of xe
        assert project_dict['experiments_names'][1] == 'Example data file from refnx docs'
        assert project_dict['experiments_models'][1] == project.models[1].name
        assert len(project_dict['experiments'][1]) == 3

        # Then - the project loads back with the experiment attached to the same model
        global_object.map._clear()
        new_project = Project()
        new_project.from_dict(project_dict)

        assert list(new_project.experiments.keys()) == [1]
        assert new_project.experiments[1].name == 'Example data file from refnx docs'
        assert new_project.experiments[1].model == new_project.models[1]
        assert_allclose(new_project.experiments[1].x, project.experiments[1].x)
        assert_allclose(new_project.experiments[1].y, project.experiments[1].y)
        assert_allclose(new_project.experiments[1].ye, project.experiments[1].ye)

    def test_save_as_json(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.default_model()
        project._info['name'] = 'Test Project'

        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)

        # Then
        project.save_as_json()

        # Expect
        assert project.path_json.exists()

    def test_save_as_json_overwrite(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        file_info = project.path_json.stat()

        # Then
        project._info['short_description'] = 'short_description'
        project.default_model()
        project.save_as_json(overwrite=True)

        # Expect
        assert str(file_info) != str(project.path_json.stat())

    def test_save_as_json_dont_overwrite(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        file_info = project.path_json.stat()

        # Then
        project._info['short_description'] = 'short_description'
        project.default_model()
        with pytest.raises(FileExistsError):
            project.save_as_json()

        # Expect
        assert str(file_info) == str(project.path_json.stat())

    def test_save_as_json_serialization_failure_keeps_previous_file(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        previous_content = project.path_json.read_text()

        # Then
        project.as_dict = MagicMock(side_effect=ValueError('unreachable constraint parameter'))
        with pytest.raises(ValueError):
            project.save_as_json(overwrite=True)

        # Expect: the previous save survives and no temporary file is left behind
        assert project.path_json.exists()
        assert project.path_json.read_text() == previous_content
        assert list(project.path_json.parent.glob('*.tmp')) == []

    def test_save_as_json_write_failure_keeps_previous_file(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        previous_content = project.path_json.read_text()

        # Then: the destination cannot be replaced, e.g. a locked file on Windows
        with patch('easyreflectometry.project.os.replace', side_effect=PermissionError('locked')):
            with pytest.raises(PermissionError):
                project.save_as_json(overwrite=True)

        # Expect
        assert project.path_json.read_text() == previous_content
        assert list(project.path_json.parent.glob('*.tmp')) == []

    def test_save_as_json_reports_unserializable_content_as_value_error(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        previous_content = project.path_json.read_text()

        # Then: json.dumps raises TypeError for a value it cannot encode
        project.as_dict = MagicMock(return_value={'info': object()})
        with pytest.raises(ValueError, match='cannot be serialized'):
            project.save_as_json(overwrite=True)

        # Expect
        assert project.path_json.read_text() == previous_content
        assert list(project.path_json.parent.glob('*.tmp')) == []

    @pytest.mark.skipif(sys.platform == 'win32', reason='POSIX file modes')
    def test_save_as_json_keeps_the_file_mode_of_an_existing_project_file(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()
        # Group-writable on purpose: distinguishable from the umask default a new file would get.
        os.chmod(project.path_json, 0o664)  # noqa: S103

        # Then
        project._info['short_description'] = 'short_description'
        project.save_as_json(overwrite=True)

        # Expect: the temporary file's 0600 must not leak onto the project file
        assert project.path_json.stat().st_mode & 0o777 == 0o664

    @pytest.mark.skipif(sys.platform == 'win32', reason='POSIX file modes')
    def test_save_as_json_new_file_is_not_owner_only(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        previous_umask = os.umask(0o022)
        try:
            project.save_as_json()
        finally:
            os.umask(previous_umask)

        # Expect
        assert project.path_json.stat().st_mode & 0o777 == 0o644

    def test_save_as_json_overwrite_replaces_content(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.save_as_json()

        # Then
        project._info['short_description'] = 'short_description'
        project.save_as_json(overwrite=True)

        # Expect
        assert 'short_description' in project.path_json.read_text()
        assert list(project.path_json.parent.glob('*.tmp')) == []

    def test_load_from_json(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.default_model()
        project._info['name'] = 'name'
        project._info['short_description'] = 'short_description'
        project._info['samples'] = 'samples'
        project._info['experiments'] = 'experiments'

        project.save_as_json()
        project_dict = project.as_dict()

        global_object.map._clear()
        new_project = Project()

        # Then
        new_project.load_from_json(tmp_path / 'name' / 'project.json')
        # Do it twice to ensure that potential
        # global objects don't collide
        new_project.load_from_json(tmp_path / 'name' / 'project.json')

        # Expect
        assert len(new_project._models) == 1
        assert new_project._info['name'] == 'name'
        assert new_project._info['short_description'] == 'short_description'
        assert new_project._info['samples'] == 'samples'
        assert new_project._info['experiments'] == 'experiments'
        assert project_dict == new_project.as_dict()
        assert new_project._path_project_parent == tmp_path
        assert new_project.created is True

    def test_load_from_json_missing_file_raises_and_keeps_the_project(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        project._info['name'] = 'untouched'

        # Then
        with pytest.raises(FileNotFoundError):
            project.load_from_json(tmp_path / 'nowhere' / 'project.json')

        # Expect
        assert project._info['name'] == 'untouched'
        assert len(project._models) == 1
        assert project.created is False

    def test_create(self, tmp_path):
        # When
        project = Project()
        project.set_path_project_parent(tmp_path)
        project._info['modified'] = 'modified'
        project._info['name'] = 'TestProject'

        # Then
        project.create()

        # Expect
        assert project.path == tmp_path / 'TestProject'
        assert project.path.exists()
        assert (project.path / 'experiments').exists()
        assert project.created is True
        assert project._info == {
            'name': 'TestProject',
            'short_description': 'Reflectometry, 1D',
            'modified': datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        }

    def test_create_existing_directory_raises(self, tmp_path):
        # When
        project = Project()
        project.set_path_project_parent(tmp_path)
        project._info['name'] = 'TestProject'
        project._info['modified'] = 'modified'
        (tmp_path / 'TestProject').mkdir()

        # Then
        with pytest.raises(FileExistsError):
            project.create()

        # Expect: nothing was made and the project does not claim to exist on disk
        assert project.created is False
        assert not (project.path / 'experiments').exists()
        assert project._info['modified'] == 'modified'

    def test_load_experiment(self):
        # When
        global_object.map._clear()
        project = Project()
        model_5 = Model()
        project.models = ModelCollection(Model(), Model(), Model(), Model(), Model(), model_5)
        fpath = os.path.join(PATH_STATIC, 'example.ort')

        # Then
        project.load_experiment_for_model_at_index(fpath, 5)

        # Expect
        assert list(project.experiments.keys()) == [5]
        assert isinstance(project.experiments[5], DataSet1D)
        assert project.experiments[5].name == 'Example data file from refnx docs'
        assert project.experiments[5].model == model_5
        # example.ort carries an sQz column, so the measured resolution is used
        assert isinstance(project.models[5].resolution_function, Pointwise)
        assert isinstance(project.models[4].resolution_function, PercentageFwhm)

    def test_load_experiment_sets_resolution_function_pointwise_when_xe_present(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.models = ModelCollection(Model())

        # Create a simple 4-column data file (x, y, e, xe)
        fpath = tmp_path / 'four_col.txt'
        fpath.write_text('# test data\n0.01 1e-5 1e-6 1e-4\n0.02 2e-5 1e-6 1e-4\n')

        # Then
        project.load_experiment_for_model_at_index(str(fpath))

        # Expect Pointwise because xe (q-resolution) is present
        resolution_function = project.models[0].resolution_function
        assert isinstance(resolution_function, Pointwise)
        # The 4th column is sQz (sigma); smearing() must return it unchanged
        assert_allclose(resolution_function.smearing([0.01, 0.02]), [1e-4, 1e-4])

    def test_load_experiment_keeps_percentage_fwhm_when_no_xe(self, tmp_path):
        # When
        global_object.map._clear()
        project = Project()
        project.models = ModelCollection(Model())

        # Create a simple 3-column data file (x, y, e)
        fpath = tmp_path / 'three_col.txt'
        fpath.write_text('# test data\n0.01 1e-5 1e-6\n0.02 2e-5 1e-6\n')

        # Then
        project.load_experiment_for_model_at_index(str(fpath))

        # No q-resolution data available, so the 5% FWHM default is kept
        assert isinstance(project.models[0].resolution_function, PercentageFwhm)

    def test_apply_resolution_function_prefers_pointwise_falls_back_to_percentage(self):
        # When
        global_object.map._clear()
        project = Project()
        model = Model()
        with_xe = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=[1e-8, 4e-8])
        zero_xe = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=[0.0, 0.0])
        no_xe = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1])

        # Then Expect
        project._apply_resolution_function(with_xe, model)
        assert isinstance(model.resolution_function, Pointwise)
        # xe holds sQz variances; smearing() must return sigma = sqrt(xe)
        assert_allclose(model.resolution_function.smearing([0.01, 0.02]), [1e-4, 2e-4])

        project._apply_resolution_function(zero_xe, model)
        assert isinstance(model.resolution_function, PercentageFwhm)

        project._apply_resolution_function(no_xe, model)
        assert isinstance(model.resolution_function, PercentageFwhm)

    def test_load_all_experiments_from_file_sets_pointwise_when_sqz_present(self):
        # When
        global_object.map._clear()
        project = Project()
        project.models = ModelCollection(Model())
        fpath = os.path.join(PATH_STATIC, 'test_example2.ort')

        # Then
        n_loaded = project.load_all_experiments_from_file(fpath)

        # Expect
        assert n_loaded == 2
        assert list(project.experiments.keys()) == [0, 1]
        assert isinstance(project.models[0].resolution_function, Pointwise)

    def test_experimental_data_at_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.models = ModelCollection(Model())
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)

        # Then
        data = project.experimental_data_for_model_at_index()

        # Expect
        assert data.name == 'Example data file from refnx docs'
        assert data.is_experiment
        assert isinstance(data, DataSet1D)
        assert len(data.x) == 408
        assert len(data.xe) == 408
        assert len(data.y) == 408
        assert len(data.ye) == 408

    def test_q(self):
        # When
        global_object.map._clear()
        project = Project()

        # Then
        q = project.q_min, project.q_max, project.q_resolution

        # Expect
        assert q == (0.001, 0.3, 500)

    def test_set_q(self):
        # When
        project = Project()

        # Then
        project.q_min = 1
        project.q_max = 2
        project.q_resolution = 3

        # Expect
        q = project.q_min, project.q_max, project.q_resolution
        assert q == (1, 2, 3)

    def test_calculator(self):
        # When
        project = Project()

        # Then Expect
        assert project.calculator == 'refnx'

    def test_set_calculator(self):
        # When
        project = Project()

        # Then
        project.calculator = 'refl1d'

        # Expect
        assert project._calculator.current_interface_name == 'refl1d'

    def test_parameters(self):
        # When
        project = Project()
        project.default_model()

        # Then
        parameters = project.parameters

        # Expect: 14 layer/material/model parameters + the model's derived total thickness
        assert len(parameters) == 15
        assert isinstance(parameters[0], Parameter)
        assert any(parameter is project.models[0].total_thickness for parameter in parameters)

    def test_parameters_enabled_flags(self):
        global_object.map._clear()
        project = Project()
        project.default_model()

        sample = project.models[0].sample
        superphase = sample.superphase
        subphase = sample.subphase
        middle_layer = sample[1].front_layer

        assert superphase.thickness.enabled is False
        assert superphase.roughness.enabled is False
        assert subphase.thickness.enabled is False
        assert getattr(subphase.roughness, 'enabled', True) is True
        assert getattr(middle_layer.thickness, 'enabled', True) is True
        assert getattr(middle_layer.roughness, 'enabled', True) is True

    def test_parameters_read_does_not_overwrite_enabled_flag(self):
        global_object.map._clear()
        project = Project()
        project.default_model()

        parameter = project.models[0].sample[0].layers[0].thickness
        parameter.enabled = True

        _ = project.parameters

        assert parameter.enabled is True

    def test_current_experiment_index_getter_and_setter(self):
        global_object.map._clear()
        project = Project()
        # Default value should be 0
        assert project.current_experiment_index == 0

        # Add two experiments to allow setting index 1
        project._experiments[0] = DataSet1D(name='exp0', x=[], y=[], ye=[], xe=[], model=None)
        project._experiments[1] = DataSet1D(name='exp1', x=[], y=[], ye=[], xe=[], model=None)

        # Set to 1 (valid)
        project.current_experiment_index = 1
        assert project.current_experiment_index == 1

        # Set to 0 (valid)
        project.current_experiment_index = 0
        assert project.current_experiment_index == 0

    def test_current_experiment_index_setter_out_of_range(self):
        global_object.map._clear()
        project = Project()
        # Add one experiment
        project._experiments[0] = DataSet1D(name='exp0', x=[], y=[], ye=[], xe=[], model=None)

        # Negative index should raise
        with pytest.raises(ValueError):
            project.current_experiment_index = -1

        # Index >= len(_experiments) should raise
        with pytest.raises(ValueError):
            project.current_experiment_index = 1

    def test_get_materials_from_model(self):
        # When
        global_object.map._clear()
        project = Project()
        material_1 = Material(sld=2.07, isld=0.0, name='Material 1')
        material_2 = Material(sld=3.47, isld=0.0, name='Material 2')
        material_3 = Material(sld=6.36, isld=0.0, name='Material 3')

        layer_1 = Layer(material=material_1, thickness=10, roughness=0, name='Layer 1')
        layer_2 = Layer(material=material_2, thickness=20, roughness=1, name='Layer 2')
        layer_3 = Layer(material=material_3, thickness=0, roughness=2, name='Layer 3')

        sample = Sample(Multilayer([layer_1, layer_2]), Multilayer([layer_3]))
        model = Model(sample=sample)

        # Then
        materials = project._get_materials_from_model(model)

        # Expect
        assert len(materials) == 3
        assert materials[0] == material_1
        assert materials[1] == material_2
        assert materials[2] == material_3

    def test_get_materials_from_model_duplicate_materials(self):
        # When
        global_object.map._clear()
        project = Project()
        # Use the same material in multiple layers
        shared_material = Material(sld=2.07, isld=0.0, name='Shared Material')
        material_2 = Material(sld=3.47, isld=0.0, name='Material 2')

        layer_1 = Layer(material=shared_material, thickness=10, roughness=0, name='Layer 1')
        layer_2 = Layer(material=material_2, thickness=20, roughness=1, name='Layer 2')
        layer_3 = Layer(material=shared_material, thickness=30, roughness=2, name='Layer 3')

        sample = Sample(Multilayer([layer_1, layer_2, layer_3]))
        model = Model(sample=sample)

        # Then
        materials = project._get_materials_from_model(model)

        # Expect - should only include unique materials
        assert len(materials) == 2
        assert materials[0] == shared_material
        assert materials[1] == material_2

    def test_add_sample_from_orso(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()

        initial_model_count = len(project._models)
        initial_material_count = len(project._materials)

        material_1 = Material(sld=4.0, isld=0.0, name='New Material 1')
        material_2 = Material(sld=5.0, isld=0.0, name='New Material 2')
        layer_1 = Layer(material=material_1, thickness=50, roughness=1, name='New Layer 1')
        layer_2 = Layer(material=material_2, thickness=100, roughness=2, name='New Layer 2')
        new_sample = Sample(Multilayer([layer_1, layer_2]))

        # Then
        project.add_sample_from_orso(new_sample)

        # Expect
        assert len(project._models) == initial_model_count + 1
        assert project._models[-1].sample == new_sample
        # The interface should be set by add_sample_from_orso
        assert project._models[-1].interface == project._calculator
        assert len(project._materials) == initial_material_count + 2
        assert material_1 in project._materials
        assert material_2 in project._materials
        assert project.current_model_index == len(project._models) - 1

    def test_add_sample_from_orso_multiple_additions(self):
        # When
        global_object.map._clear()
        project = Project()

        material_1 = Material(sld=2.0, isld=0.0, name='Material A')
        layer_1 = Layer(material=material_1, thickness=10, roughness=0, name='Layer A')
        sample_1 = Sample(Multilayer([layer_1]))

        material_2 = Material(sld=3.0, isld=0.0, name='Material B')
        layer_2 = Layer(material=material_2, thickness=20, roughness=1, name='Layer B')
        sample_2 = Sample(Multilayer([layer_2]))

        # Then
        project.add_sample_from_orso(sample_1)
        project.add_sample_from_orso(sample_2)

        # Expect
        assert len(project._models) == 2
        assert project._models[0].sample == sample_1
        assert project._models[1].sample == sample_2
        assert len(project._materials) == 2
        assert material_1 in project._materials
        assert material_2 in project._materials
        assert project.current_model_index == 1
        # Each loaded sample gets its own colour from the collection's cycle;
        # without this, every ORSO-loaded model rendered in the first palette
        # colour and their curves were indistinguishable.
        assert project._models[0].color != project._models[1].color

    def test_add_sample_from_orso_with_shared_materials(self):
        # When
        global_object.map._clear()
        project = Project()

        # Create first sample with a material
        shared_material = Material(sld=2.0, isld=0.0, name='Shared Material')
        layer_1 = Layer(material=shared_material, thickness=10, roughness=0, name='Layer 1')
        sample_1 = Sample(Multilayer([layer_1]))
        project.add_sample_from_orso(sample_1)

        initial_material_count = len(project._materials)

        # Create second sample using the same material
        layer_2 = Layer(material=shared_material, thickness=20, roughness=1, name='Layer 2')
        sample_2 = Sample(Multilayer([layer_2]))

        # Then
        project.add_sample_from_orso(sample_2)

        # Expect - shared material should not be duplicated
        assert len(project._models) == 2
        # The shared material instance is already in the collection,
        # so count should stay the same
        assert len(project._materials) == initial_material_count

    def test_replace_models_from_orso(self):
        """Test that replace_models_from_orso replaces all existing
        models with a single new model."""
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()

        # Add some models to start with
        material_1 = Material(sld=2.0, isld=0.0, name='Material 1')
        layer_1 = Layer(material=material_1, thickness=10, roughness=0, name='Layer 1')
        sample_1 = Sample(Multilayer([layer_1]))
        project.add_sample_from_orso(sample_1)

        material_2 = Material(sld=3.0, isld=0.0, name='Material 2')
        layer_2 = Layer(material=material_2, thickness=20, roughness=1, name='Layer 2')
        sample_2 = Sample(Multilayer([layer_2]))
        project.add_sample_from_orso(sample_2)

        # Verify we have multiple models
        assert len(project._models) > 1
        len(project._models)

        # Create a new sample to replace all existing models
        new_material = Material(sld=5.0, isld=0.5, name='New Material')
        new_layer = Layer(material=new_material, thickness=50, roughness=2, name='New Layer')
        new_sample = Sample(Multilayer([new_layer]))

        # Then - replace all models with the new sample
        project.replace_models_from_orso(new_sample)

        # Expect - only one model should remain
        assert len(project._models) == 1
        assert project._models[0].sample == new_sample
        # The interface should be set
        assert project._models[0].interface == project._calculator
        # Only the new material should be in the materials collection
        assert len(project._materials) == 1
        assert new_material in project._materials
        # Old materials should not be in the collection
        assert material_1 not in project._materials
        assert material_2 not in project._materials
        # Current model index should be reset to 0
        assert project.current_model_index == 0

    def test_is_default_model_true(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()

        # Then Expect
        assert project.is_default_model(0) is True

    def test_is_default_model_false_non_default_sample(self):
        # When
        global_object.map._clear()
        project = Project()
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project.models = ModelCollection(model)

        # Then Expect
        assert project.is_default_model(0) is False

    def test_is_default_model_index_out_of_range(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()

        # Then Expect
        assert project.is_default_model(-1) is False
        assert project.is_default_model(1) is False
        assert project.is_default_model(100) is False

    def test_is_default_model_multiple_models(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a custom model
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)

        # Then Expect
        assert project.is_default_model(0) is True
        assert project.is_default_model(1) is False

    def test_remove_model_at_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a second model
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)
        assert len(project._models) == 2

        # Then
        project.remove_model_at_index(0)

        # Expect
        assert len(project._models) == 1
        assert project._models[0].sample[0].name == 'Custom Assembly'

    def test_remove_model_at_index_adjusts_current_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a second model
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)
        project._current_model_index = 1
        project._current_assembly_index = 1
        project._current_layer_index = 1

        # Then
        project.remove_model_at_index(0)

        # Expect - current_model_index should be adjusted
        assert project._current_model_index == 0
        assert project._current_assembly_index == 0
        assert project._current_layer_index == 0

    def test_remove_model_at_index_resets_indices_when_at_end(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a second model
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)
        project._current_model_index = 1

        # Then - remove the model at the current index
        project.remove_model_at_index(1)

        # Expect - current_model_index should be clamped to valid range
        assert project._current_model_index == 0
        assert project._current_assembly_index == 0
        assert project._current_layer_index == 0

    def test_remove_model_at_index_removes_experiment_at_same_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a second model
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)
        # Add experiment linked to model 0
        experiment = DataSet1D(
            name='exp0',
            x=[0.01, 0.02],
            y=[1.0, 0.5],
            ye=[0.1, 0.1],
            xe=[0.001, 0.001],
            model=project._models[0],
        )
        project._experiments[0] = experiment

        # Then
        project.remove_model_at_index(0)

        # Expect - experiment mapped to the removed model index is removed
        assert 0 not in project._experiments

    def test_remove_model_at_index_reindexes_experiments_above_removed_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()

        # Add two more models (total = 3)
        material_1 = Material(sld=4.0, isld=0.0, name='Custom Material 1')
        layer_1 = Layer(material=material_1, thickness=50, roughness=1, name='Custom Layer 1')
        model_1 = Model(sample=Sample(Multilayer([layer_1], name='Custom Assembly 1')))
        project._models.append(model_1)

        material_2 = Material(sld=5.0, isld=0.0, name='Custom Material 2')
        layer_2 = Layer(material=material_2, thickness=60, roughness=2, name='Custom Layer 2')
        model_2 = Model(sample=Sample(Multilayer([layer_2], name='Custom Assembly 2')))
        project._models.append(model_2)

        # Add experiments for all model indices 0, 1, 2
        project._experiments[0] = DataSet1D(name='exp0', x=[0.01], y=[1.0], ye=[0.1], xe=[0.001], model=project._models[0])
        project._experiments[1] = DataSet1D(name='exp1', x=[0.02], y=[0.9], ye=[0.1], xe=[0.001], model=project._models[1])
        project._experiments[2] = DataSet1D(name='exp2', x=[0.03], y=[0.8], ye=[0.1], xe=[0.001], model=project._models[2])

        # Then - remove middle model
        project.remove_model_at_index(1)

        # Expect - middle experiment removed and upper one shifted down
        assert set(project._experiments.keys()) == {0, 1}
        assert project._experiments[0].name == 'exp0'
        assert project._experiments[1].name == 'exp2'

    def test_remove_model_at_index_raises_for_last_model(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        assert len(project._models) == 1

        # Then Expect
        with pytest.raises(ValueError, match='Cannot remove the last model'):
            project.remove_model_at_index(0)

    def test_remove_model_at_index_raises_for_invalid_index(self):
        # When
        global_object.map._clear()
        project = Project()
        project.default_model()
        # Add a second model so we have 2
        material = Material(sld=4.0, isld=0.0, name='Custom Material')
        layer = Layer(material=material, thickness=50, roughness=1, name='Custom Layer')
        sample = Sample(Multilayer([layer], name='Custom Assembly'))
        model = Model(sample=sample)
        project._models.append(model)

        # Then Expect - negative index
        with pytest.raises(IndexError, match='out of range'):
            project.remove_model_at_index(-1)

        # Then Expect - index >= len
        with pytest.raises(IndexError, match='out of range'):
            project.remove_model_at_index(2)
