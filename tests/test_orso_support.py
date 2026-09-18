# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ORSO support update (ORSO_UPDATE_TASK.md):

reader hardening (banner discriminator, FWHM->sigma, nan policy, 1/nm Q),
model-language preservation (units, repeats, density materials),
single-file polarized import, the .ort/.orb exporter, and .orb reading.
"""

import os

import numpy as np
import pytest
from easyscience import global_object
from orsopy.fileio import model_language
from orsopy.fileio.base import Column
from orsopy.fileio.base import ComplexValue
from orsopy.fileio.base import ErrorColumn
from orsopy.fileio.base import Value
from orsopy.fileio.orso import Orso
from orsopy.fileio.orso import OrsoDataset
from orsopy.fileio.orso import save_orso

import easyreflectometry
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.orso_utils import SIGMA_TO_FWHM
from easyreflectometry.orso_utils import _load_orso_any
from easyreflectometry.orso_utils import is_orso_file
from easyreflectometry.orso_utils import load_orso_model
from easyreflectometry.orso_utils import sample_to_orso_model
from easyreflectometry.orso_utils import save_orso_experiment
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample.elements.materials.material_density import MaterialDensity

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')

# The Qz/sQz grids the generated fixtures were built from (see task 7a).
FIXTURE_QZ = np.linspace(0.01, 0.3, 12)
FIXTURE_SQZ = FIXTURE_QZ * 0.02


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def project() -> Project:
    return Project()


class TestBannerDiscriminator:
    def test_ort_file_is_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'Ni_example.ort')) is True

    def test_orb_file_is_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'example.orb')) is True

    def test_txt_file_is_not_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'test_example1.txt')) is False

    def test_bannered_but_corrupt_file_raises(self, tmp_path):
        # A file carrying the ORSO banner that fails to parse must raise, not
        # silently fall back to plain-text loading (which drops the header).
        bad = tmp_path / 'bad.ort'
        bad.write_text(
            '# # ORSO reflectivity data file | 1.1 standard | YAML encoding | https://www.reflectometry.org/\n'
            '# data_source: {[[not yaml\n'
            '1.0 2.0 3.0 4.0\n'
        )
        with pytest.raises(ValueError, match='Error loading ORSO file'):
            load(str(bad))

    def test_non_bannered_file_falls_back_to_txt(self, tmp_path):
        plain = tmp_path / 'plain.dat'
        plain.write_text('0.01 1.0 0.1 0.001\n0.02 0.5 0.05 0.002\n')
        data_group = load(str(plain))
        assert 'R_plain' in data_group['data']

    def test_count_datasets_raises_for_corrupt_orso(self, project, tmp_path):
        bad = tmp_path / 'bad.ort'
        bad.write_text('# # ORSO reflectivity data file | 1.1 standard | YAML encoding\n# data_source: {[[\n1 2 3 4\n')
        with pytest.raises(ValueError):
            project.count_datasets_in_file(str(bad))

    def test_count_datasets_multi(self, project):
        assert project.count_datasets_in_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort')) == 2


class TestNanPolicy:
    def test_all_nan_sqz_leaves_xe_empty(self):
        data_group = load(os.path.join(PATH_STATIC, 'nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        assert coords.variances is None

    def test_all_nan_sqz_falls_back_to_percentage_fwhm(self, project):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'nan_sqz.ort'))
        assert isinstance(project.models[0].resolution_function, PercentageFwhm)

    def test_partial_nan_sqz_is_interpolated(self):
        with pytest.warns(UserWarning, match='sQz values are nan'):
            data_group = load(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        assert coords.variances is not None
        assert np.all(np.isfinite(coords.variances))

    def test_partial_nan_sqz_builds_finite_pointwise(self, project):
        project.default_model()
        with pytest.warns(UserWarning, match='sQz values are nan'):
            project.load_new_experiment(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        resolution_function = project.models[0].resolution_function
        assert isinstance(resolution_function, Pointwise)
        assert np.all(np.isfinite(resolution_function.smearing(FIXTURE_QZ)))


class TestValueIsFwhm:
    def test_fwhm_sqz_is_converted_to_sigma(self):
        data_group = load(os.path.join(PATH_STATIC, 'fwhm_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        # The fixture stores sigma * SIGMA_TO_FWHM declared as FWHM; loading
        # must convert back so stored variances are sigma squared.
        np.testing.assert_allclose(np.sqrt(coords.variances), FIXTURE_SQZ)

    def test_sigma_file_is_not_scaled(self):
        data_group = load(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        valid = np.sqrt(coords.variances)[2:-1]
        np.testing.assert_allclose(valid, FIXTURE_SQZ[2:-1])


class TestQUnitConversion:
    def test_1_per_nm_qz_is_converted_to_1_per_angstrom(self):
        data_group = load(os.path.join(PATH_STATIC, 'nm_units.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        np.testing.assert_allclose(coords.values, FIXTURE_QZ)
        np.testing.assert_allclose(np.sqrt(coords.variances), FIXTURE_SQZ)
        assert str(coords.unit) == '1/Å'


class TestModelLanguagePreservation:
    def test_nm_default_length_unit_is_honoured(self):
        # nm_units.ort declares thicknesses as bare magnitudes with no
        # length_unit -> the model-language default (nm) applies.
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        film = [layer for assembly in sample for layer in assembly.layers if layer.name == 'film'][0]
        assert film.thickness.value == pytest.approx(100.0)  # 10 nm
        assert film.roughness.value == pytest.approx(5.0)  # 0.5 nm

    def test_default_roughness_from_globals(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        subphase_layer = sample[-1].layers[0]
        # Si declares roughness 0.3 (nm default) -> 3 A
        assert subphase_layer.roughness.value == pytest.approx(3.0)

    def test_repeated_substack_becomes_repeating_multilayer(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        repeating = [assembly for assembly in sample if isinstance(assembly, RepeatingMultilayer)]
        assert len(repeating) == 1
        assert repeating[0].repetitions.value == 3
        assert [layer.name for layer in repeating[0].layers] == ['A', 'B']
        assert repeating[0].layers[0].thickness.value == pytest.approx(20.0)  # 2 nm

    def test_density_material_stays_density_defined(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'Ni_example.ort')))
        m1 = sample[1].layers[0]
        assert isinstance(m1.material, MaterialDensity)
        assert m1.material.chemical_structure == 'Ni'
        assert m1.material.density.value == pytest.approx(8.9)
        # and the derived SLD is still sensible (Ni ~ 9.4e-6 A^-2)
        assert m1.material.sld.value == pytest.approx(9.4, abs=0.1)


class TestPolarizedSingleFile:
    def test_load_polarized_experiment_from_file(self, project):
        project.default_model()
        index = project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        experiment = project.experiments[index]
        assert isinstance(experiment, PolarizedDataSet)
        assert [channel.value for channel in experiment.available_channels] == ['pp', 'mm']
        assert experiment.name == 'Polarized fixture'
        assert experiment.model is project.models[0]
        np.testing.assert_allclose(experiment['pp'].x, FIXTURE_QZ)

    def test_unclassifiable_dataset_raises(self, project):
        project.default_model()
        # nan_sqz.ort carries no spin-channel polarization -> not coerced.
        with pytest.raises(ValueError, match='declares no (mappable spin channel|polarization)'):
            project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'nan_sqz.ort'))

    def test_multidataset_file_rejected_by_per_channel_loader(self, project):
        project.default_model()
        with pytest.raises(ValueError, match='load_polarized_experiment_from_file'):
            project.load_polarized_experiment({'pp': os.path.join(PATH_STATIC, 'polarized_2ch.ort')})

    def test_duplicate_channel_raises(self, project, tmp_path):
        # Build a file where two datasets declare the same channel.
        datasets = _load_orso_any(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        from orsopy.fileio.data_source import Polarization
        from orsopy.fileio.orso import save_orso

        for orso_dataset in datasets:
            orso_dataset.info.data_source.measurement.instrument_settings.polarization = Polarization('pp')
        duplicate_file = tmp_path / 'dup.ort'
        save_orso(datasets, str(duplicate_file))

        project.default_model()
        with pytest.raises(ValueError, match="Duplicate spin channel 'pp'"):
            project.load_polarized_experiment_from_file(str(duplicate_file))


class TestExporter:
    def test_ort_data_roundtrip_writes_sigma(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        experiment = project.experiments[0]
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)

        back = _load_orso_any(str(out))
        assert len(back) == 1
        data = back[0].data
        np.testing.assert_allclose(data[:, 0], experiment.x)
        np.testing.assert_allclose(data[:, 1], experiment.y)
        # DataSet1D stores variances; the file must carry sigma.
        np.testing.assert_allclose(data[:, 2], np.sqrt(experiment.ye))
        np.testing.assert_allclose(data[:, 3], np.sqrt(experiment.xe))
        # sigma convention declared in the columns
        assert back[0].info.columns[2].value_is == 'sigma'
        assert back[0].info.columns[3].value_is == 'sigma'

    def test_export_reuses_preserved_header(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)
        info = _load_orso_any(str(out))[0].info
        # data_source/reduction provenance from the original file, not synthesized
        assert info.data_source.experiment.title == 'Metal films'
        assert info.data_source.owner.name == 'Joe Bloggs'

    def test_export_writes_model_language(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)
        model = _load_orso_any(str(out))[0].info.data_source.sample.model
        assert model is not None
        assert model.globals.length_unit == 'angstrom'
        # the exported model resolves back into layers
        assert len(model.resolve_to_layers()) >= 2

    def test_absent_errors_written_as_nan(self, tmp_path):
        dataset = DataSet1D(x=np.array([0.01, 0.02]), y=np.array([1.0, 0.5]))
        out = tmp_path / 'nan.ort'
        save_orso_experiment(dataset, str(out))
        data = _load_orso_any(str(out))[0].data
        assert np.all(np.isnan(data[:, 2]))
        assert np.all(np.isnan(data[:, 3]))

    def test_polarized_export_single_multidataset_file(self, project, tmp_path):
        project.default_model()
        index = project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        out = tmp_path / 'pol.ort'
        project.save_experiment_as_orso(str(out), index)

        back = _load_orso_any(str(out))
        assert [d.info.data_set for d in back] == ['pp', 'mm']
        polarizations = [str(d.info.data_source.measurement.instrument_settings.polarization.value) for d in back]
        assert polarizations == ['pp', 'mm']

    def test_repeating_multilayer_roundtrips_via_model_language(self):
        air = Layer(material=Material(sld=0.0, isld=0.0, name='air'), thickness=0, roughness=0, name='air')
        layer_a = Layer(material=Material(sld=4.0, isld=0.0, name='A'), thickness=20, roughness=3, name='A')
        layer_b = Layer(material=Material(sld=2.0, isld=0.0, name='B'), thickness=10, roughness=3, name='B')
        si = Layer(material=Material(sld=2.07, isld=0.0, name='Si'), thickness=0, roughness=3, name='Si')
        sample = Sample(
            Multilayer(air, name='Superphase'),
            RepeatingMultilayer([layer_a, layer_b], repetitions=5, name='rep'),
            Multilayer(si, name='Subphase'),
            name='test',
        )
        orso_model = sample_to_orso_model(sample)
        assert '5 ( A | B )' in orso_model.stack
        assert len(orso_model.resolve_to_layers()) == 12  # 1 + 5*2 + 1

    def test_model_as_orso_returns_model_language_dict(self, project):
        project.default_model()
        orso_dict = project.models[0].as_orso()
        assert 'stack' in orso_dict
        assert 'layers' in orso_dict
        assert orso_dict['globals']['length_unit'] == 'angstrom'


class TestOrbSupport:
    def test_load_orb_file(self):
        data_group = load(os.path.join(PATH_STATIC, 'example.orb'))
        assert 'R_0' in data_group['data']
        coords = data_group['coords'][list(data_group['coords'])[0]]
        np.testing.assert_allclose(coords.values, FIXTURE_QZ)

    def test_orb_write_and_read_roundtrip(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        experiment = project.experiments[0]
        out = tmp_path / 'out.orb'
        project.save_experiment_as_orso(str(out), 0)
        back = _load_orso_any(str(out))
        np.testing.assert_allclose(back[0].data[:, 0], experiment.x)
        np.testing.assert_allclose(back[0].data[:, 2], np.sqrt(experiment.ye))

    def test_load_orb_as_dataset(self):
        dataset = load_as_dataset(os.path.join(PATH_STATIC, 'example.orb'))
        assert isinstance(dataset, DataSet1D)
        np.testing.assert_allclose(dataset.x, FIXTURE_QZ)
        assert dataset.orso_header is not None


class TestFwhmSigmaContract:
    def test_pointwise_serialization_stays_variances(self):
        # The saved-project contract: sQz_data_points round-trip as variances.
        qz = np.array([0.01, 0.02])
        reflectivity = np.array([1.0, 0.5])
        variances = np.array([1e-8, 2e-8])
        pointwise = Pointwise([qz, reflectivity, variances])
        as_dict = pointwise.as_dict()
        np.testing.assert_allclose(as_dict['sQz_data_points'], variances)
        from easyreflectometry.model.resolution_functions import ResolutionFunction

        restored = ResolutionFunction.from_dict(as_dict)
        np.testing.assert_allclose(restored.smearing(qz), np.sqrt(variances))

    def test_sigma_to_fwhm_constant(self):
        assert SIGMA_TO_FWHM == pytest.approx(2.3548, abs=1e-4)


# ---------------------------------------------------------------------------
# Synthetic .ort fixtures
#
# The static files in tests/_static cover the happy paths. The cases below are
# one-off malformed or unusual files, so they are built here rather than
# checked in: the header that provokes the behaviour stays next to the
# assertion about it.
# ---------------------------------------------------------------------------

STANDARD_COLUMNS = [
    Column(name='Qz', unit='1/angstrom', physical_quantity='wavevector transfer'),
    Column(name='R', physical_quantity='reflectivity'),
    ErrorColumn(error_of='R', error_type='uncertainty', value_is='sigma'),
    ErrorColumn(error_of='Qz', error_type='resolution', value_is='sigma'),
]

SYNTHETIC_QZ = np.linspace(0.01, 0.3, 5)
SYNTHETIC_R = np.exp(-SYNTHETIC_QZ * 10.0)


def _write_ort(path, sample_model=None, columns=None, data=None, sample_name='synthetic fixture'):
    """Write a single-dataset .ort file with the given model and columns.

    Parameters
    ----------
    path :
        Destination path.
    sample_model : optional
        An orsopy ``SampleModel`` for ``data_source.sample.model``. By default, None.
    columns : optional
        Column descriptors. By default, the standard Qz, R, sR, sQz four.
    data : optional
        The data array. By default, a four-column exponential decay.
    sample_name : str, optional
        ``data_source.sample.name``. By default, 'synthetic fixture'.

    Returns
    -------
    str
        The path written.
    """
    info = Orso.empty()
    info.data_set = 0
    info.data_source.sample.name = sample_name
    info.data_source.sample.model = sample_model
    info.columns = columns if columns is not None else list(STANDARD_COLUMNS)
    if data is None:
        data = np.column_stack([SYNTHETIC_QZ, SYNTHETIC_R, SYNTHETIC_R * 0.05, SYNTHETIC_QZ * 0.02])
    save_orso([OrsoDataset(info=info, data=data)], str(path))
    return str(path)


def _orso_layer(thickness=0.0, roughness=0.0, sld=0.0, **material_kwargs):
    """A model-language layer, SLD-defined unless material keywords are given."""
    if material_kwargs:
        material = model_language.Material(**material_kwargs)
    else:
        material = model_language.Material(sld=ComplexValue(real=sld))
    return model_language.Layer(thickness=thickness, roughness=roughness, material=material)


def _orso_sample_model(stack, layers=None, **kwargs):
    """A model-language sample model, with an air/Si frame available by default."""
    frame = {'air': _orso_layer(), 'Si': _orso_layer(sld=2.07e-6)}
    frame.update(layers or {})
    return model_language.SampleModel(stack=stack, layers=frame, **kwargs)


def _sample_from(path):
    return load_orso_model(_load_orso_any(path))


class TestColumnValidation:
    def test_missing_error_columns_warn_and_leave_errors_absent(self, tmp_path):
        # A two-column file is incomplete: warn, and treat the error columns as
        # absent rather than inventing zeros (which would become fit weights).
        path = _write_ort(
            tmp_path / 'two_columns.ort',
            columns=STANDARD_COLUMNS[:2],
            data=np.column_stack([SYNTHETIC_QZ, SYNTHETIC_R]),
        )
        with pytest.warns(UserWarning, match='declares only 2 columns'):
            data_group = load(path)
        assert data_group['data']['R_0'].variances is None
        assert data_group['coords']['Qz_0'].variances is None

    def test_misordered_columns_warn(self, tmp_path):
        # Data is read by position, so a file declaring R before Qz is read
        # transposed. The warning is the only signal the user gets.
        columns = [STANDARD_COLUMNS[1], STANDARD_COLUMNS[0]] + STANDARD_COLUMNS[2:]
        path = _write_ort(
            tmp_path / 'misordered.ort',
            columns=columns,
            data=np.column_stack([SYNTHETIC_R, SYNTHETIC_QZ, SYNTHETIC_R * 0.05, SYNTHETIC_QZ * 0.02]),
        )
        with pytest.warns(UserWarning, match="column 0 is 'R'"):
            load(path)

    def test_unknown_q_unit_warns_and_values_are_used_as_is(self, tmp_path):
        columns = [Column(name='Qz', unit='1/m', physical_quantity='wavevector transfer')] + STANDARD_COLUMNS[1:]
        path = _write_ort(tmp_path / 'q_unit.ort', columns=columns)
        with pytest.warns(UserWarning, match="declares Qz unit '1/m'"):
            data_group = load(path)
        # Left unconverted: guessing a scale would be worse than not converting.
        np.testing.assert_allclose(data_group['coords']['Qz_0'].values, SYNTHETIC_QZ)


class TestSrNanPolicy:
    def test_all_nan_sr_leaves_variances_unset(self, tmp_path):
        # All-nan is the spec's "uncertainty unknown": no variances are stored,
        # and no warning is warranted.
        data = np.column_stack([SYNTHETIC_QZ, SYNTHETIC_R, np.full_like(SYNTHETIC_QZ, np.nan), SYNTHETIC_QZ * 0.02])
        data_group = load(_write_ort(tmp_path / 'sr_all_nan.ort', data=data))
        assert data_group['data']['R_0'].variances is None
        # the sQz column is unaffected by the sR policy
        assert data_group['coords']['Qz_0'].variances is not None

    def test_partial_nan_sr_warns_and_is_not_interpolated(self, tmp_path):
        # Unlike sQz, partial-nan sR is kept as-is: interpolating measurement
        # uncertainties would fabricate fit weights.
        sr = SYNTHETIC_R * 0.05
        sr[1] = np.nan
        data = np.column_stack([SYNTHETIC_QZ, SYNTHETIC_R, sr, SYNTHETIC_QZ * 0.02])
        with pytest.warns(UserWarning, match='1 of 5 sR values are nan'):
            data_group = load(_write_ort(tmp_path / 'sr_partial_nan.ort', data=data))
        variances = data_group['data']['R_0'].variances
        assert np.isnan(variances[1])
        assert np.all(np.isfinite(np.delete(variances, 1)))


class TestStackShapes:
    def test_multiple_plain_layer_groups_are_disambiguated(self, tmp_path):
        # Two runs of plain layers separated by a sub-stack: the single-group
        # name 'Loaded layer' would collide, so both runs get an index.
        layers = {
            'f1': _orso_layer(thickness=Value(5.0, 'nm'), sld=3e-6),
            'f2': _orso_layer(thickness=Value(6.0, 'nm'), sld=1e-6),
            'A': _orso_layer(thickness=Value(2.0, 'nm'), sld=4e-6),
            'B': _orso_layer(thickness=Value(1.0, 'nm'), sld=2e-6),
        }
        model = _orso_sample_model('air | f1 | 2 ( A | B ) | f2 | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'two_groups.ort', model))
        assert [assembly.name for assembly in sample] == [
            'Superphase',
            'Loaded layer 0',
            'Multilayer',
            'Loaded layer 1',
            'Subphase',
        ]

    def test_single_repetition_substack_is_a_plain_multilayer(self, tmp_path):
        layers = {
            'A': _orso_layer(thickness=Value(2.0, 'nm'), sld=4e-6),
            'B': _orso_layer(thickness=Value(1.0, 'nm'), sld=2e-6),
        }
        model = _orso_sample_model('air | 1 ( A | B ) | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'one_rep.ort', model))
        middle = sample[1]
        assert isinstance(middle, Multilayer)
        assert not isinstance(middle, RepeatingMultilayer)
        assert [layer.name for layer in middle.layers] == ['A', 'B']

    def test_substack_as_first_stack_item_is_flattened(self, tmp_path):
        # The ambient entry must be a plain layer; a sub-stack there is
        # pathological, so its layers are used directly.
        layers = {
            'A': _orso_layer(thickness=Value(2.0, 'nm'), sld=4e-6),
            'B': _orso_layer(thickness=Value(1.0, 'nm'), sld=2e-6),
        }
        model = _orso_sample_model('2 ( A | B ) | air | Si', layers)
        with pytest.warns(UserWarning, match='First ORSO stack item is a sub-stack'):
            sample = _sample_from(_write_ort(tmp_path / 'first_sub.ort', model))
        assert sample[0].name == 'Superphase'
        assert sample[0].layers[0].name == 'A'

    def test_substack_as_last_stack_item_is_flattened(self, tmp_path):
        layers = {
            'A': _orso_layer(thickness=Value(2.0, 'nm'), sld=4e-6),
            'B': _orso_layer(thickness=Value(1.0, 'nm'), sld=2e-6),
        }
        model = _orso_sample_model('air | Si | 2 ( A | B )', layers)
        with pytest.warns(UserWarning, match='Last ORSO stack item is a sub-stack'):
            sample = _sample_from(_write_ort(tmp_path / 'last_sub.ort', model))
        assert sample[-1].name == 'Subphase'
        assert sample[-1].layers[0].name == 'B'

    def test_unresolvable_stack_raises(self, tmp_path):
        model = _orso_sample_model('')
        with pytest.raises(ValueError, match='Could not resolve ORSO layers'):
            _sample_from(_write_ort(tmp_path / 'empty_stack.ort', model))

    def test_single_layer_stack_raises(self, tmp_path):
        model = _orso_sample_model('air')
        with pytest.raises(ValueError, match='at least 2 layers'):
            _sample_from(_write_ort(tmp_path / 'one_layer.ort', model))


class TestMaterialAndUnitFallbacks:
    def test_layer_without_name_or_formula_falls_back_to_material(self, tmp_path):
        # Layers declared inline in a sub-stack sequence carry no
        # `original_name`, and an SLD-only material has no formula either, so
        # neither the layer nor the material name may end up None.
        sub_stack = model_language.SubStack(
            repetitions=2,
            sequence=[
                _orso_layer(thickness=Value(2.0, 'nm'), sld=4e-6),
                _orso_layer(thickness=Value(1.0, 'nm'), sld=2e-6),
            ],
        )
        model = _orso_sample_model('air | multi | Si', sub_stacks={'multi': sub_stack})
        sample = _sample_from(_write_ort(tmp_path / 'unnamed.ort', model))
        repeating = sample[1]
        assert isinstance(repeating, RepeatingMultilayer)
        assert [layer.name for layer in repeating.layers] == ['material', 'material']
        assert all(layer.material.name == 'material' for layer in repeating.layers)

    def test_unknown_length_unit_warns_and_value_used_as_angstrom(self, tmp_path):
        layers = {'film': _orso_layer(thickness=Value(12.0, 'furlong'), sld=3e-6)}
        model = _orso_sample_model('air | film | Si', layers)
        with pytest.warns(UserWarning, match="Unknown ORSO length unit 'furlong'"):
            sample = _sample_from(_write_ort(tmp_path / 'length_unit.ort', model))
        assert sample[1].layers[0].thickness.value == pytest.approx(12.0)

    def test_absent_thickness_is_zero_and_roughness_takes_the_global_default(self, tmp_path):
        # An omitted thickness has no model-language default, so it reads as 0;
        # an omitted roughness picks up `globals.roughness` (0.3 nm) instead.
        layers = {'film': model_language.Layer(material=model_language.Material(sld=ComplexValue(real=3e-6)))}
        model = _orso_sample_model('air | film | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'no_thickness.ort', model))
        film = sample[1].layers[0]
        assert film.thickness.value == 0.0
        assert film.roughness.value == pytest.approx(3.0)

    def test_null_length_magnitude_is_zero(self, tmp_path):
        # A Value carrying an explicit null magnitude must not become nan.
        layers = {
            'film': model_language.Layer(
                thickness=Value(None, 'nm'),
                roughness=Value(None, 'nm'),
                material=model_language.Material(sld=ComplexValue(real=3e-6)),
            )
        }
        model = _orso_sample_model('air | film | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'null_magnitude.ort', model))
        film = sample[1].layers[0]
        assert film.thickness.value == 0.0
        assert film.roughness.value == 0.0

    def test_sld_in_inverse_nm_squared_is_converted(self, tmp_path):
        # 4e-4 1/nm^2 == 4e-6 1/angstrom^2 == 4.0 in the internal 10^-6 units.
        layers = {
            'film': model_language.Layer(
                thickness=Value(10.0, 'nm'),
                material=model_language.Material(sld=ComplexValue(real=4.0e-4, unit='1/nm^2')),
            )
        }
        model = _orso_sample_model('air | film | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'sld_nm.ort', model))
        assert sample[1].layers[0].material.sld.value == pytest.approx(4.0)

    def test_unknown_sld_unit_warns(self, tmp_path):
        layers = {
            'film': model_language.Layer(
                thickness=Value(10.0, 'nm'),
                material=model_language.Material(sld=ComplexValue(real=4.0e-6, unit='1/barn')),
            )
        }
        model = _orso_sample_model('air | film | Si', layers)
        with pytest.warns(UserWarning, match="Unknown ORSO SLD unit '1/barn'"):
            sample = _sample_from(_write_ort(tmp_path / 'sld_unit.ort', model))
        assert sample[1].layers[0].material.sld.value == pytest.approx(4.0)

    def test_large_sld_warns_about_units(self, tmp_path):
        # 3.47 in a field specified as 1/angstrom^2 almost certainly means the
        # writer stored 10^-6 A^-2; multiplied by 1e6 it becomes 3.47e6.
        layers = {
            'film': model_language.Layer(
                thickness=Value(10.0, 'nm'),
                material=model_language.Material(sld=ComplexValue(real=3.47)),
            )
        }
        model = _orso_sample_model('air | film | Si', layers)
        with pytest.warns(UserWarning, match='seems large for'):
            _sample_from(_write_ort(tmp_path / 'sld_large.ort', model))

    def test_mass_density_in_kg_per_m3_is_converted(self, tmp_path):
        layers = {'Ni': _orso_layer(thickness=Value(10.0, 'nm'), formula='Ni', mass_density=Value(8900.0, 'kg/m^3'))}
        model = _orso_sample_model('air | Ni | Si', layers)
        sample = _sample_from(_write_ort(tmp_path / 'density_kg.ort', model))
        material = sample[1].layers[0].material
        assert isinstance(material, MaterialDensity)
        assert material.density.value == pytest.approx(8.9)

    def test_unknown_mass_density_unit_warns(self, tmp_path):
        layers = {'Ni': _orso_layer(thickness=Value(10.0, 'nm'), formula='Ni', mass_density=Value(8.9, 'stone/gallon'))}
        model = _orso_sample_model('air | Ni | Si', layers)
        with pytest.warns(UserWarning, match="Unknown ORSO mass density unit 'stone/gallon'"):
            sample = _sample_from(_write_ort(tmp_path / 'density_unit.ort', model))
        assert sample[1].layers[0].material.density.value == pytest.approx(8.9)


class TestBannerDiscriminatorEdgeCases:
    def test_missing_file_is_not_orso(self, tmp_path):
        assert is_orso_file(str(tmp_path / 'does_not_exist.ort')) is False

    def test_directory_is_not_orso(self, tmp_path):
        assert is_orso_file(str(tmp_path)) is False

    def test_empty_file_is_not_orso(self, tmp_path):
        empty = tmp_path / 'empty.ort'
        empty.write_bytes(b'')
        assert is_orso_file(str(empty)) is False


class TestExporterFallbacks:
    def test_duplicate_layer_names_get_unique_stack_keys(self):
        # Layers legitimately share a name; the stack keys must not collide, or
        # one definition would silently overwrite the other. Three of them, so
        # the counter has to advance past the first free suffix.
        air = Layer(material=Material(sld=0.0, isld=0.0, name='air'), thickness=0, roughness=0, name='air')
        films = [
            Layer(material=Material(sld=sld, isld=0.0, name='film'), thickness=thickness, roughness=3, name='film')
            for sld, thickness in ((4.0, 20), (2.0, 40), (1.0, 60))
        ]
        si = Layer(material=Material(sld=2.07, isld=0.0, name='Si'), thickness=0, roughness=3, name='Si')
        sample = Sample(
            Multilayer(air, name='Superphase'),
            Multilayer(films, name='Loaded layer'),
            Multilayer(si, name='Subphase'),
            name='duplicates',
        )
        orso_model = sample_to_orso_model(sample)
        assert {'film', 'film_2', 'film_3'} <= set(orso_model.layers)
        resolved = orso_model.resolve_to_layers()
        assert len(resolved) == 5
        # every definition kept its own thickness
        assert [layer.thickness.magnitude for layer in resolved[1:4]] == pytest.approx([20.0, 40.0, 60.0])

    def test_density_material_exports_formula_and_mass_density(self):
        air = Layer(material=Material(sld=0.0, isld=0.0, name='air'), thickness=0, roughness=0, name='air')
        nickel = Layer(
            material=MaterialDensity(chemical_structure='Ni', density=8.9, name='Ni'),
            thickness=100,
            roughness=3,
            name='Ni',
        )
        si = Layer(material=Material(sld=2.07, isld=0.0, name='Si'), thickness=0, roughness=3, name='Si')
        sample = Sample(
            Multilayer(air, name='Superphase'),
            Multilayer(nickel, name='Loaded layer'),
            Multilayer(si, name='Subphase'),
            name='density',
        )
        material = sample_to_orso_model(sample).layers['Ni'].material
        # written as formula + density, not flattened to a numeric SLD
        assert material.formula == 'Ni'
        assert material.mass_density.magnitude == pytest.approx(8.9)
        assert material.mass_density.unit == 'g/cm^3'
        assert material.sld is None

    def test_sample_name_falls_back_to_the_model_sample_name(self, project, tmp_path):
        # A dataset with no preserved header gets a synthesized one, whose
        # sample name is empty until the model supplies it.
        project.default_model()
        project.models[0].sample.name = 'my sample'
        dataset = DataSet1D(x=np.array([0.01, 0.02]), y=np.array([1.0, 0.5]))
        out = tmp_path / 'named.ort'
        save_orso_experiment(dataset, str(out), model=project.models[0])
        assert _load_orso_any(str(out))[0].info.data_source.sample.name == 'my sample'

    def test_polarized_export_synthesizes_absent_instrument_settings(self, project, tmp_path):
        # Some files omit instrument_settings entirely; the exporter must create
        # one to hang the channel polarization off, rather than raising.
        project.default_model()
        index = project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        experiment = project.experiments[index]
        for channel_dataset in experiment.channels.values():
            channel_dataset.orso_header['data_source']['measurement']['instrument_settings'] = None

        out = tmp_path / 'no_settings.ort'
        save_orso_experiment(experiment, str(out))
        back = _load_orso_any(str(out))
        polarizations = [str(d.info.data_source.measurement.instrument_settings.polarization.value) for d in back]
        assert polarizations == ['pp', 'mm']
