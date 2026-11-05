# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Magnetic moments and number densities in ORSO model-language import and export."""

import warnings

import pytest
from easyscience import global_object
from orsopy.fileio import model_language
from orsopy.fileio.base import ComplexValue
from orsopy.fileio.base import Value

from easyreflectometry.model import Model
from easyreflectometry.orso_utils import sample_to_orso_model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialDensity
from easyreflectometry.sample import MaterialMagnetic
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.special.calculations import MAGNETIC_SCATTERING_LENGTH
from easyreflectometry.special.calculations import number_density_from_mass_density

from .test_orso_support import _orso_layer
from .test_orso_support import _orso_sample_model
from .test_orso_support import _sample_from
from .test_orso_support import _write_ort

FE_RHO_M = 5.0420761  # 1e-6 / angstrom^2: 7.874 g/cm^3, 2.2 muB per atom


@pytest.fixture(autouse=True)
def _clear_map():
    global_object.registry.clear()


def _film_from(tmp_path, **material_kwargs):
    """Import 'air | film | Si' with the given film material; return the film layer."""
    film = model_language.Layer(thickness=Value(10.0, 'nm'), material=model_language.Material(**material_kwargs))
    layers = {'film': film}
    sample = _sample_from(_write_ort(tmp_path / 'film.ort', _orso_sample_model('air | film | Si', layers)))
    return sample[1].layers[0]


def _sample(film: Layer) -> Sample:
    air = Layer(material=Material(sld=0.0, isld=0.0, name='air'), thickness=0, roughness=0, name='air')
    si = Layer(material=Material(sld=2.07, isld=0.0, name='Si'), thickness=0, roughness=3, name='Si')
    return Sample(Multilayer(air), Multilayer(film), Multilayer(si), name='magnetic')


def _round_trip(tmp_path, film: Layer) -> Layer:
    sample = _sample_from(_write_ort(tmp_path / 'round_trip.ort', sample_to_orso_model(_sample(film))))
    return sample[1].layers[0]


class TestImport:
    def test_formula_density_and_moment_give_a_coupled_magnetic_layer(self, tmp_path):
        film = _film_from(tmp_path, formula='Fe', mass_density=Value(7.874, 'g/cm^3'), magnetic_moment=2.2)

        assert isinstance(film.material, MaterialMagnetic)
        assert film.rho_m_coupled
        assert film.magnetism.theta_m.value == 270.0
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_number_density_is_converted_with_the_formula(self, tmp_path):
        film = _film_from(tmp_path, formula='Si', number_density=Value(49.9611, '1/nm^3'))

        assert isinstance(film.material, MaterialDensity)
        assert film.material.density.value == pytest.approx(2.33, rel=1e-5)
        assert film.magnetism is None

    def test_number_density_in_inverse_cubic_angstrom_with_a_moment(self, tmp_path):
        film = _film_from(tmp_path, formula='Fe', number_density=Value(0.0849106, '1/angstrom^3'), magnetic_moment=2.2)

        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M, rel=1e-5)

    def test_moment_is_per_formula_unit(self, tmp_path):
        film = _film_from(tmp_path, formula='Fe3O4', mass_density=Value(5.17, 'g/cm^3'), magnetic_moment=4.1)

        number_density = number_density_from_mass_density(5.17, 231.533)
        expected = number_density * 4.1 * MAGNETIC_SCATTERING_LENGTH * 1e6
        assert film.magnetism.rho_m.value == pytest.approx(expected, rel=1e-4)

    def test_explicit_sld_wins_over_the_formula(self, tmp_path):
        film = _film_from(
            tmp_path,
            formula='Fe',
            mass_density=Value(7.874, 'g/cm^3'),
            sld=ComplexValue(real=7.0e-6),
            magnetic_moment=2.2,
        )

        assert not film.material.sld_coupled
        assert film.material.sld.value == pytest.approx(7.0)
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_sld_and_number_density_without_formula_give_a_direct_magnetic_sld(self, tmp_path):
        film = _film_from(
            tmp_path,
            sld=ComplexValue(real=7.0e-6),
            number_density=Value(0.0849106, '1/angstrom^3'),
            magnetic_moment=2.2,
        )

        assert type(film.material) is Material
        assert not film.rho_m_coupled
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M, rel=1e-5)

    def test_moment_without_density_is_skipped(self, tmp_path):
        with pytest.warns(UserWarning, match='Magnetic moment of "film" not imported'):
            film = _film_from(tmp_path, sld=ComplexValue(real=7.0e-6), magnetic_moment=2.2)
        assert film.magnetism is None

    def test_unknown_moment_unit_is_never_guessed(self, tmp_path):
        with pytest.warns(UserWarning, match="Unknown ORSO magnetic moment unit 'A\\*m\\^2'"):
            film = _film_from(tmp_path, formula='Fe', mass_density=Value(7.874, 'g/cm^3'), magnetic_moment=Value(2.2, 'A*m^2'))
        assert type(film.material) is MaterialDensity
        assert film.magnetism is None

    def test_disagreeing_densities_keep_the_mass_density(self, tmp_path):
        with pytest.warns(UserWarning, match='disagree; the mass density is used'):
            film = _film_from(tmp_path, formula='Si', mass_density=Value(2.33, 'g/cm^3'), number_density=Value(10.0, '1/nm^3'))
        assert film.material.density.value == pytest.approx(2.33)

    def test_relative_density_is_reported(self, tmp_path):
        with pytest.warns(UserWarning, match='relative_density of "film" is not applied'):
            _film_from(tmp_path, formula='Si', mass_density=Value(2.33, 'g/cm^3'), relative_density=0.5)

    def test_repeated_magnetic_layers_are_reported(self, tmp_path):
        layers = {'Fe': _orso_layer(thickness=Value(2.0, 'nm'), formula='Fe', mass_density=7.874, magnetic_moment=2.2)}
        model = _orso_sample_model('air | 3 ( Fe | Si ) | Si', layers)
        with pytest.warns(UserWarning, match='repeats magnetic layers'):
            _sample_from(_write_ort(tmp_path / 'repeated.ort', model))


class TestExport:
    def test_coupled_layer_writes_the_material_moment(self, tmp_path):
        film = Layer(material=MaterialMagnetic(name='Fe'), thickness=100, name='Fe', rho_m_coupled=True)

        loaded = _round_trip(tmp_path, film)

        assert isinstance(loaded.material, MaterialMagnetic)
        assert loaded.rho_m_coupled
        assert loaded.material.magnetic_moment.value == pytest.approx(2.2)
        assert loaded.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_uncoupled_layer_writes_the_matching_moment(self, tmp_path):
        film = Layer(
            material=MaterialDensity(chemical_structure='Fe', density=7.874, name='Fe'),
            thickness=100,
            name='Fe',
            magnetism=LayerMagnetism(rho_m=3.0),
        )

        loaded = _round_trip(tmp_path, film)

        assert loaded.magnetism.rho_m.value == pytest.approx(3.0)

    def test_magnetic_sld_on_an_sld_material_is_reported(self, tmp_path):
        film = Layer(material=Material(sld=8.0, name='film'), thickness=100, name='film', magnetism=LayerMagnetism(rho_m=3.0))

        with pytest.warns(UserWarning, match='Magnetic SLD of layer "film" not written'):
            loaded = _round_trip(tmp_path, film)
        assert loaded.magnetism is None

    def test_direction_is_reported(self, tmp_path):
        film = Layer(material=MaterialMagnetic(name='Fe'), thickness=100, name='Fe', rho_m_coupled=True)
        film.magnetism.theta_m = 300.0

        with pytest.warns(UserWarning, match='Moment direction of layer "Fe" not written'):
            loaded = _round_trip(tmp_path, film)
        assert loaded.magnetism.theta_m.value == 270.0

    def test_magnetic_material_on_a_non_magnetic_layer_writes_no_moment(self):
        film = Layer(material=MaterialMagnetic(name='Fe'), thickness=100, name='Fe')

        assert sample_to_orso_model(_sample(film)).layers['Fe'].material.magnetic_moment is None

    def test_manual_nuclear_sld_survives(self, tmp_path):
        material = MaterialDensity(chemical_structure='Ni', density=8.9, name='Ni')
        material.sld_coupled = False
        material.sld.value = 7.0
        film = Layer(material=material, thickness=100, name='Ni')

        loaded = _round_trip(tmp_path, film)

        assert loaded.material.sld.value == pytest.approx(7.0)

    def test_aligned_coupled_layer_exports_without_warnings(self, tmp_path):
        film = Layer(material=MaterialMagnetic(name='Fe'), thickness=100, name='Fe', rho_m_coupled=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            sample_to_orso_model(_sample(film))

    def test_model_field_is_reported(self):
        model = Model(guide_field_angle=0.0)
        with pytest.warns(UserWarning, match='guide field angle and applied field are not written'):
            model.as_orso()


class TestSignedAndLargeMoments:
    def test_negative_direct_rho_m_round_trips(self, tmp_path):
        film = Layer(
            material=MaterialDensity(chemical_structure='Fe', density=7.874, name='Fe'),
            thickness=100,
            name='Fe',
            magnetism=LayerMagnetism(rho_m=-3.0),
        )

        loaded = _round_trip(tmp_path, film)

        assert loaded.magnetism.rho_m.value == pytest.approx(-3.0)
        assert loaded.material.magnetic_moment.value < 0

    def test_widened_signed_material_moment_round_trips(self, tmp_path):
        iron = MaterialMagnetic(name='Fe')
        iron.magnetic_moment.min = -10.0
        iron.magnetic_moment.value = -1.1
        film = Layer(material=iron, thickness=100, name='Fe', rho_m_coupled=True)

        loaded = _round_trip(tmp_path, film)

        assert loaded.rho_m_coupled
        assert loaded.magnetism.rho_m.value == pytest.approx(-FE_RHO_M / 2)

    def test_moment_above_ten_per_formula_unit(self, tmp_path):
        film = _film_from(tmp_path, formula='Fe10', mass_density=Value(7.874, 'g/cm^3'), magnetic_moment=22.0)

        assert film.material.magnetic_moment.value == pytest.approx(22.0)
        # Ten iron atoms of 2.2 muB each: the same magnetisation as Fe at 2.2.
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_zero_moment(self, tmp_path):
        film = _film_from(tmp_path, formula='Fe', mass_density=Value(7.874, 'g/cm^3'), magnetic_moment=0.0)
        assert film.magnetism.rho_m.value == pytest.approx(0.0)


class TestProjectRefusesMagneticSamplesItCannotCalculate:
    @staticmethod
    def _project_and_sample(tmp_path, calculator):
        from easyreflectometry.project import Project

        project = Project()
        project.calculator = calculator
        project.default_model()
        film = Layer(material=MaterialMagnetic(name='Fe'), thickness=100, name='Fe', rho_m_coupled=True)
        path = _write_ort(tmp_path / 'magnetic.ort', sample_to_orso_model(_sample(film)))
        return project, _sample_from(path)

    @pytest.mark.parametrize('method', ['add_sample_from_orso', 'replace_models_from_orso'])
    def test_refnx_refuses_before_changing_the_project(self, tmp_path, method):
        project, sample = self._project_and_sample(tmp_path, 'refnx')
        models = list(project.models)
        materials = list(project._materials)

        with pytest.raises(NotImplementedError, match='switch to refl1d'):
            getattr(project, method)(sample)

        assert list(project.models) == models
        assert list(project._materials) == materials
        assert project.current_model_index == 0

    @pytest.mark.parametrize('method', ['add_sample_from_orso', 'replace_models_from_orso'])
    def test_refl1d_loads_it(self, tmp_path, method):
        project, sample = self._project_and_sample(tmp_path, 'refl1d')

        getattr(project, method)(sample)

        film = project.models[project.current_model_index].sample[1].layers[0]
        assert film.rho_m_coupled
        assert project.model_data_for_model_at_index(project.current_model_index).y.size > 0
