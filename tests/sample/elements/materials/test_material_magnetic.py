# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import scipy.constants as constants
from easyscience import global_object

from easyreflectometry.contrasts import ReplaceFormula
from easyreflectometry.contrasts import derive_contrast
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialDensity
from easyreflectometry.sample import MaterialMagnetic
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.special.calculations import MAGNETIC_SCATTERING_LENGTH
from easyreflectometry.special.calculations import mass_density_from_number_density
from easyreflectometry.special.calculations import number_density_from_mass_density

FE_RHO_M = 5.0420761  # 1e-6 / angstrom^2: 7.874 g/cm^3, 2.2 muB per atom


@pytest.fixture(autouse=True)
def _clear_map():
    global_object.registry.clear()


class TestConstants:
    def test_magnetic_scattering_length_from_codata(self):
        mu_n = abs(constants.physical_constants['neutron mag. mom.'][0])
        mu_b = constants.physical_constants['Bohr magneton'][0]
        expected = constants.m_n * mu_n * constants.mu_0 * mu_b / (2 * np.pi * constants.hbar**2) * 1e10
        assert MAGNETIC_SCATTERING_LENGTH == pytest.approx(expected, rel=1e-8)

    def test_number_density_conversions(self):
        number_density = number_density_from_mass_density(2.33, 28.0855)
        assert number_density == pytest.approx(0.0499603, rel=1e-5)
        assert mass_density_from_number_density(number_density, 28.0855) == pytest.approx(2.33)


class TestMaterialMagnetic:
    def test_iron_defaults(self):
        iron = MaterialMagnetic()

        assert iron.chemical_structure == 'Fe'
        assert iron.magnetic_moment.value == 2.2
        assert iron.number_density == pytest.approx(0.0849106, rel=1e-5)
        assert iron.rho_m.value == pytest.approx(FE_RHO_M)
        assert not iron.rho_m.independent
        assert iron.magnetic_moment.min == 0.0

    def test_number_density_is_read_only_and_follows_density(self):
        material = MaterialDensity(chemical_structure='Si', density=2.33)
        assert material.number_density == pytest.approx(0.0499611, rel=1e-5)
        material.density = 4.66
        assert material.number_density == pytest.approx(0.0999222, rel=1e-5)
        with pytest.raises(AttributeError):
            material.number_density = 1.0

    @pytest.mark.parametrize(
        ('attribute', 'value', 'factor'),
        [('magnetic_moment', 1.1, 0.5), ('density', 7.874 * 2, 2.0)],
    )
    def test_rho_m_follows_its_inputs(self, attribute, value, factor):
        iron = MaterialMagnetic()
        setattr(iron, attribute, value)
        assert iron.rho_m.value == pytest.approx(FE_RHO_M * factor)

    def test_moment_is_per_formula_unit(self):
        # Fe2 at the same mass density has half the formula units, each with
        # the moment given: a moment per formula unit, not per atom.
        iron = MaterialMagnetic()
        iron.chemical_structure = 'Fe2'
        assert iron.rho_m.value == pytest.approx(FE_RHO_M / 2)
        iron.magnetic_moment = 4.4
        assert iron.rho_m.value == pytest.approx(FE_RHO_M)

    def test_nuclear_decoupling_leaves_rho_m_derived(self):
        iron = MaterialMagnetic()
        iron.sld_coupled = False
        iron.density = 7.874 / 2
        assert iron.rho_m.value == pytest.approx(FE_RHO_M / 2)
        assert iron.sld.value == pytest.approx(8.024, abs=1e-3)

    @pytest.mark.parametrize('sld_coupled', [True, False])
    def test_round_trip_rewires_rho_m(self, sld_coupled):
        iron = MaterialMagnetic(magnetic_moment=1.1)
        iron.sld_coupled = sld_coupled
        if not sld_coupled:
            iron.sld.value = 7.0

        restored = MaterialMagnetic.from_dict(iron.as_dict())
        restored.magnetic_moment = 2.2
        restored.density = 7.874

        assert restored.rho_m.value == pytest.approx(FE_RHO_M)
        assert restored.sld_coupled is sld_coupled
        if not sld_coupled:
            assert restored.sld.value == pytest.approx(7.0)

    def test_isotopic_substitution_keeps_density_and_moment(self):
        iron = MaterialMagnetic(name='Fe')
        film = Layer(material=iron, thickness=100, name='film', rho_m_coupled=True)
        air = Layer(material=Material(sld=0.0, name='air'), thickness=0, name='air')
        reference = Model(Sample(Multilayer(air), Multilayer(film), Multilayer(air)))

        contrast = derive_contrast(reference, name='57Fe', substitutions=[ReplaceFormula(iron, 'Fe[57]')])
        substituted = contrast.sample[1].layers[0].material

        assert substituted.density.value == pytest.approx(7.874)
        assert substituted.magnetic_moment.value == pytest.approx(2.2)
        # N changes with the molecular weight, and rho_m with it.
        ratio = iron.molecular_weight.value / substituted.molecular_weight.value
        assert substituted.rho_m.value == pytest.approx(FE_RHO_M * ratio)


class TestMomentBounds:
    def test_default_bounds(self):
        moment = MaterialMagnetic().magnetic_moment
        assert (moment.min, moment.max) == (0.0, 10.0)

    @pytest.mark.parametrize(('value', 'bounds'), [(-1.3, (-1.3, 10.0)), (22.0, (0.0, 22.0)), (0.0, (0.0, 10.0))])
    def test_an_explicit_moment_widens_the_bounds(self, value, bounds):
        moment = MaterialMagnetic(chemical_structure='Fe10', magnetic_moment=value).magnetic_moment
        assert moment.value == value
        assert (moment.min, moment.max) == bounds

    def test_a_parameter_keeps_its_own_bounds(self):
        from easyscience.variable import Parameter

        given = Parameter(display_name='magnetic_moment', value=1.0, min=0.5, max=2.0)
        assert MaterialMagnetic(magnetic_moment=given).magnetic_moment is given
