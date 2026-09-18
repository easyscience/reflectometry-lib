# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from types import SimpleNamespace

import pytest
from orsopy.fileio import orso

import easyreflectometry
from easyreflectometry.orso_utils import LoadOrso
from easyreflectometry.orso_utils import _get_sld_values
from easyreflectometry.orso_utils import load_data_from_orso_file
from easyreflectometry.orso_utils import load_orso_data
from easyreflectometry.orso_utils import load_orso_model

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.fixture
def orso_data():
    """Load the test ORSO data from Ni_example.ort."""
    return orso.load_orso(os.path.join(PATH_STATIC, 'Ni_example.ort'))


def test_load_orso_model(orso_data):
    """Test loading a model from ORSO data."""
    sample = load_orso_model(orso_data)
    assert sample is not None
    assert sample.name == 'Ni on Si'  # Based on the file

    # Verify sample structure: Superphase, Loaded layer, Subphase
    # Stack in file: air | m1 | SiO2 | Si
    assert len(sample) == 3

    # Check Superphase (first layer from stack: air)
    superphase = sample[0]
    assert superphase.name == 'Superphase'
    assert len(superphase.layers) == 1
    assert superphase.layers[0].material.name == 'air'
    assert superphase.layers[0].thickness.value == 0.0
    assert superphase.layers[0].roughness.value == 0.0
    assert superphase.layers[0].thickness.fixed is True
    assert superphase.layers[0].roughness.fixed is True

    # Check Loaded layer (middle layers: m1, SiO2)
    loaded_layer = sample[1]
    assert loaded_layer.name == 'Loaded layer'
    assert len(loaded_layer.layers) == 2
    assert loaded_layer.layers[0].material.name == 'm1'  # Uses original_name, not formula
    assert loaded_layer.layers[0].thickness.value == 1000.0  # From layer definition
    assert loaded_layer.layers[1].material.name == 'SiO2'
    assert loaded_layer.layers[1].thickness.value == 10.0  # From layer definition

    # Check Subphase (last layer from stack: Si)
    subphase = sample[2]
    assert subphase.name == 'Subphase'
    assert len(subphase.layers) == 1
    assert subphase.layers[0].material.name == 'Si'
    assert subphase.layers[0].thickness.value == 0.0
    assert subphase.layers[0].thickness.fixed is True
    # Subphase roughness should be enabled (not fixed)
    assert subphase.layers[0].roughness.fixed is False


def test_load_orso_data(orso_data):
    """Test loading data from ORSO data."""
    data = load_orso_data(orso_data)
    assert data is not None
    # Check structure, e.g., has R_0 in data
    assert 'R_0' in data['data']


def test_LoadOrso(orso_data):
    """Test the LoadOrso function."""
    sample, data = LoadOrso(orso_data)
    assert sample is not None
    assert data is not None
    # Similar checks as above


def test_load_data_from_orso_file():
    """Test loading data from ORSO file."""
    data = load_data_from_orso_file(os.path.join(PATH_STATIC, 'Ni_example.ort'))
    assert data is not None
    # Check it's a sc.DataGroup
    import scipp as sc

    assert isinstance(data, sc.DataGroup)


def test_orso_sld_unit_conversion(orso_data):
    """Test that SLD values from ORSO are correctly converted
       from A^-2 to 10^-6 A^-2.

    ORSO stores SLD in absolute units (A^-2), e.g., 3.47e-06.
    The internal representation uses 10^-6 A^-2,
    so the value should be 3.47.
    """
    sample = load_orso_model(orso_data)

    # Check SiO2 layer (second layer in Loaded layer assembly)
    # ORSO file has: sld: {real: 3.4700000000000002e-06, imag: 0.0}
    # Expected internal value: 3.47
    loaded_layer = sample[1]
    sio2_layer = loaded_layer.layers[1]
    assert sio2_layer.material.name == 'SiO2'
    assert abs(sio2_layer.material.sld.value - 3.47) < 1e-6, (
        f'Expected SLD ~3.47 (10^-6 A^-2), got {sio2_layer.material.sld.value}'
    )

    # Check Si subphase layer
    # ORSO file has: sld: {real: 2.0699999999999997e-06, imag: 0.0}
    # Expected internal value: 2.07
    subphase = sample[2]
    si_layer = subphase.layers[0]
    assert si_layer.material.name == 'Si'
    assert abs(si_layer.material.sld.value - 2.07) < 1e-6, f'Expected SLD ~2.07 (10^-6 A^-2), got {si_layer.material.sld.value}'

    # Check air superphase layer
    # ORSO file has: sld: {real: 0.0, imag: 0.0}
    # Expected internal value: 0.0
    superphase = sample[0]
    air_layer = superphase.layers[0]
    assert air_layer.material.name == 'air'
    assert abs(air_layer.material.sld.value - 0.0) < 1e-6, f'Expected SLD 0.0 (10^-6 A^-2), got {air_layer.material.sld.value}'


def test_LoadOrso_returns_two_items(orso_data):
    """LoadOrso should return exactly two values: (sample, data)."""
    result = LoadOrso(orso_data)
    assert isinstance(result, tuple)
    assert len(result) == 2
    sample, data = result
    assert sample is not None
    assert data is not None


def test_LoadOrso_with_invalid_file(tmp_path):
    """LoadOrso should raise for a corrupt / non-ORSO file."""
    bad_file = tmp_path / 'bad.ort'
    bad_file.write_text('this is not valid ORSO data')
    with pytest.raises((ValueError, Exception)):
        LoadOrso(str(bad_file))


def test_LoadOrso_with_nonexistent_file():
    """LoadOrso should raise for a path that does not exist."""
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        LoadOrso('/nonexistent/path/to/file.ort')


def test_get_sld_values_defaults_to_zero_when_sld_and_density_missing():
    """_get_sld_values should return (0.0, 0.0) when both
    sld and mass_density are None."""
    material = SimpleNamespace(sld=None, mass_density=None)
    m_sld, m_isld = _get_sld_values(material, 'Unknown')
    assert m_sld == 0.0
    assert m_isld == 0.0


def test_load_orso_model_returns_none_and_warns_when_no_sample_model():
    """load_orso_model should return None and emit a warning
    when the ORSO file has no sample model."""
    orso_data = orso.load_orso(os.path.join(PATH_STATIC, 'test_example1.ort'))
    # Verify the file indeed has no model
    assert orso_data[0].info.data_source.sample.model is None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = load_orso_model(orso_data)

    assert result is None
    assert len(w) == 1
    assert 'does not contain a sample model definition' in str(w[0].message)


def test_load_orso_model_resolves_named_materials(tmp_path):
    """Named materials from the file's `materials` section must be resolved.

    Regression: `load_orso_model` used to rebuild the ORSO ``SampleModel``
    from `stack` and `layers` alone, dropping the `materials` definitions —
    every named material then read as SLD 0. Also covers SLDs stored as
    orsopy ``Value`` objects and layers whose material has neither a name
    nor a formula (must not produce ``None`` names).
    """
    import datetime

    import numpy as np
    from orsopy import fileio
    from orsopy.fileio import model_language

    sample_model = model_language.SampleModel(
        stack='ambient | film | substrate',
        layers={
            'ambient': model_language.Layer(
                thickness=fileio.Value(0.0, 'angstrom'), roughness=fileio.Value(0.0, 'angstrom'), material='vac'
            ),
            'film': model_language.Layer(
                thickness=fileio.Value(35.0, 'angstrom'), roughness=fileio.Value(3.0, 'angstrom'), material='MatA'
            ),
            'substrate': model_language.Layer(
                thickness=fileio.Value(0.0, 'angstrom'), roughness=fileio.Value(2.0, 'angstrom'), material='Si'
            ),
        },
        materials={
            'vac': model_language.Material(sld=fileio.Value(0.0, '1/angstrom^2')),
            'MatA': model_language.Material(sld=fileio.Value(3.0e-6, '1/angstrom^2')),
            'Si': model_language.Material(sld=fileio.Value(2.07e-6, '1/angstrom^2')),
        },
        globals=model_language.ModelParameters(length_unit='angstrom'),
    )
    header = fileio.Orso(
        data_source=fileio.DataSource(
            owner=fileio.Person(name='test', affiliation='test'),
            experiment=fileio.Experiment(
                title='t', instrument='sim', start_date=datetime.datetime(2026, 1, 1), probe='neutron'
            ),
            sample=fileio.Sample(name='named materials', model=sample_model),
            measurement=fileio.Measurement(
                instrument_settings=fileio.InstrumentSettings(
                    incident_angle=fileio.Value(1.0, 'deg'), wavelength=fileio.Value(6.0, 'angstrom')
                ),
                data_files=[],
            ),
        ),
        reduction=fileio.Reduction(software=fileio.Software(name='test')),
        columns=[fileio.Column('Qz', '1/angstrom'), fileio.Column('R')],
        data_set=0,
    )
    q = np.linspace(0.01, 0.1, 5)
    path = tmp_path / 'named_materials.ort'
    fileio.save_orso([fileio.OrsoDataset(header, np.array([q, np.ones_like(q)]).T)], str(path))

    from orsopy.fileio import orso as orso_io

    sample = load_orso_model(orso_io.load_orso(str(path)))

    assert sample is not None
    film = sample[1].layers[0]
    assert film.name == 'film'
    assert film.thickness.value == pytest.approx(35.0)
    assert film.roughness.value == pytest.approx(3.0)
    assert film.material.sld.value == pytest.approx(3.0)  # 1e-6/A^2, resolved from `materials`
    assert sample[2].layers[0].material.sld.value == pytest.approx(2.07)
    # No layer or material name may come back as None
    for assembly in sample:
        for layer in assembly.layers:
            assert layer.name is not None
            assert layer.material.name is not None
