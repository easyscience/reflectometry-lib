# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the volume fraction (occupancy) profile."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from scipy.special import erf

from easyreflectometry.sample import Bilayer
from easyreflectometry.sample import GradientLayer
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerAreaPerMolecule
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import MaterialSolvated
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import VolumeFractionProfile
from easyreflectometry.sample import VolumeFractionWarning
from easyreflectometry.sample import volume_fraction_profile
from easyreflectometry.sample.volume_fraction import DEFAULT_POINTS
from easyreflectometry.sample.volume_fraction import _resolve_slabs
from easyreflectometry.sample.volume_fraction import _smear


def _clear():
    global_object.map._clear()


def _sample(*film_layers, front=None, back=None):
    """Air | film layers | D2O, each phase in its own Multilayer."""
    if front is None:
        front = Layer(Material(0.0, 0.0, 'Air'), 0, 0, name='Air')
    if back is None:
        back = Layer(Material(6.36, 0.0, 'D2O'), 0, 3, name='D2O bulk')
    assemblies = [Multilayer([front])]
    if film_layers:
        assemblies.append(Multilayer(list(film_layers)))
    assemblies.append(Multilayer([back]))
    return Sample(*assemblies)


def _by_label(profile_or_slabs, labels=None):
    """Map label -> value for a profile or for one slab's composition."""
    if isinstance(profile_or_slabs, VolumeFractionProfile):
        return {profile_or_slabs.labels[key]: values for key, values in profile_or_slabs.fractions.items()}
    return {labels[key]: value for key, value in profile_or_slabs.items()}


class TestComposition:
    def setup_method(self):
        _clear()

    def test_pure_material_layer(self):
        sample = _sample(Layer(Material(2.0, 0.0, 'Si'), 50, 2, name='film'))
        slabs, labels = _resolve_slabs(sample)
        assert [slab.name for slab in slabs] == ['Air', 'film', 'D2O bulk']
        assert _by_label(slabs[1].composition, labels) == {'Si': 1.0}
        assert list(labels.values()) == ['Air', 'Si', 'D2O']

    def test_solvated_layer(self):
        si = Material(2.07, 0.0, 'Si')
        d2o = Material(6.36, 0.0, 'D2O')
        sample = _sample(Layer(MaterialSolvated(si, d2o, 0.3), 40, 3, name='film'), back=Layer(d2o, 0, 3, name='bulk'))
        slabs, labels = _resolve_slabs(sample)
        composition = _by_label(slabs[1].composition, labels)
        assert composition == pytest.approx({'Si': 0.7, 'D2O': 0.3})
        # the shared D2O object is one component, present in the film and the subphase
        assert len(labels) == 3

    def test_mixture_layer(self):
        a = Material(1.0, 0.0, 'A')
        b = Material(5.0, 0.0, 'B')
        sample = _sample(Layer(MaterialMixture(a, b, 0.25), 20, 1, name='film'))
        slabs, labels = _resolve_slabs(sample)
        assert _by_label(slabs[1].composition, labels) == pytest.approx({'A': 0.75, 'B': 0.25})

    def test_layer_area_per_molecule(self):
        # Algebraic compatibility with the layer SLD (1-f) b/(t A) + f rho_solvent, which is the
        # current material model; this is not a validated molecular-volume convention.
        d2o = Material(6.36, 0.0, 'D2O')
        layer = LayerAreaPerMolecule('C10H18NO8P', 10.0, d2o, 0.3, 48.2, 3.0, name='heads')
        sample = _sample(layer, back=Layer(d2o, 0, 3, name='bulk'))
        slabs, labels = _resolve_slabs(sample)
        assert _by_label(slabs[1].composition, labels) == pytest.approx({'C10H18NO8P': 0.7, 'D2O': 0.3})

    def test_same_name_different_sld_stay_separate(self):
        sample = _sample(
            Layer(Material(2.0, 0.0, 'poly'), 20, 1, name='a'),
            Layer(Material(4.0, 0.0, 'poly'), 20, 1, name='b'),
        )
        _, labels = _resolve_slabs(sample)
        assert list(labels.values()) == ['Air', 'poly', 'poly (2)', 'D2O']

    def test_equivalent_materials_merge_by_default(self):
        sample = _sample(
            Layer(Material(6.36, 0.0, 'D2O'), 20, 1, name='a'),
            Layer(Material(6.36, 0.0, 'D2O'), 20, 1, name='b'),
        )
        _, labels = _resolve_slabs(sample)
        assert list(labels.values()) == ['Air', 'D2O']
        _, labels = _resolve_slabs(sample, merge_equivalent=False)
        assert list(labels.values()) == ['Air', 'D2O', 'D2O (2)', 'D2O (3)']

    def test_bilayer_heads_tails_and_solvents_merge(self):
        si = Layer(Material(2.07, 0.0, 'Si'), 0, 0, name='Si')
        bilayer = Bilayer(constrain_heads=True)
        back = Layer(Material(6.36, 0.0, 'D2O'), 0, 3, name='bulk')
        sample = Sample(Multilayer([si]), bilayer, Multilayer([back]))
        profile = volume_fraction_profile(sample)
        assert list(profile.labels.values()) == ['Si', 'C10H18NO8P', 'D2O', 'C32D64']
        # the subphase and the four generated solvent objects are one trace that ends at 1
        assert _by_label(profile)['D2O'][-1] == pytest.approx(1.0)

    def test_heads_with_different_thickness_stay_separate(self):
        d2o = Material(6.36, 0.0, 'D2O')
        front = LayerAreaPerMolecule('C10H18NO8P', 10.0, d2o, 0.3, 48.2, 3.0, name='front')
        back = LayerAreaPerMolecule('C10H18NO8P', 12.0, d2o, 0.3, 48.2, 3.0, name='back')
        _, labels = _resolve_slabs(_sample(front, back, back=Layer(d2o, 0, 3, name='bulk')))
        assert list(labels.values()) == ['Air', 'C10H18NO8P', 'D2O', 'C10H18NO8P (2)']

    def test_repeating_multilayer_is_expanded(self):
        layers = [Layer(Material(2.0, 0.0, 'A'), 10, 1, name='a'), Layer(Material(4.0, 0.0, 'B'), 20, 2, name='b')]
        sample = Sample(
            Multilayer([Layer(Material(0.0, 0.0, 'Air'), 0, 0, name='Air')]),
            RepeatingMultilayer(layers, repetitions=3),
            Multilayer([Layer(Material(6.36, 0.0, 'D2O'), 0, 3, name='bulk')]),
        )
        slabs, _ = _resolve_slabs(sample)
        assert [slab.thickness for slab in slabs] == [0, 10, 20, 10, 20, 10, 20, 0]
        assert [slab.roughness for slab in slabs] == [0, 1, 2, 1, 2, 1, 2, 3]

    def test_gradient_layer_is_a_linear_mixing_gradient(self):
        front = Material(0.0, 0.0, 'front')
        back = Material(10.0, 0.0, 'back')
        gradient = GradientLayer(front, back, thickness=10, roughness=0, discretisation_elements=5)
        sample = Sample(
            Multilayer([Layer(Material(0.0, 0.0, 'Air'), 0, 0, name='Air')]),
            gradient,
            Multilayer([Layer(Material(6.36, 0.0, 'D2O'), 0, 3, name='bulk')]),
        )
        slabs, labels = _resolve_slabs(sample)
        for index, slab in enumerate(slabs[1:-1]):
            assert _by_label(slab.composition, labels) == pytest.approx({'front': 1 - index / 5, 'back': index / 5})

    def test_edited_gradient_layer_is_refused(self):
        gradient = GradientLayer(Material(0.0, 0.0, 'front'), Material(10.0, 0.0, 'back'), 10, 0, 5)
        sample = Sample(
            Multilayer([Layer(Material(0.0, 0.0, 'Air'), 0, 0, name='Air')]),
            gradient,
            Multilayer([Layer(Material(6.36, 0.0, 'D2O'), 0, 3, name='bulk')]),
        )
        gradient.layers[2].material.sld.value = 99.0
        with pytest.raises(ValueError, match='edited after construction'):
            _resolve_slabs(sample)

    def test_bounding_slabs_have_zero_thickness(self):
        # Parameters forbid negative roughness, so only the thickness convention is observable here.
        front = Layer(Material(0.0, 0.0, 'Air'), 30, 0, name='Air')
        back = Layer(Material(6.36, 0.0, 'D2O'), 30, 3, name='bulk')
        slabs, _ = _resolve_slabs(_sample(Layer(Material(2.0, 0.0, 'Si'), 50, 2, name='film'), front=front, back=back))
        assert [slab.thickness for slab in slabs] == [0, 50, 0]
        assert [slab.roughness for slab in slabs] == [0, 2, 3]

    def test_two_slab_sample_and_empty_sample(self):
        profile = volume_fraction_profile(_sample())
        assert_allclose(profile.total, 1.0)
        assert list(profile.labels.values()) == ['Air', 'D2O']
        with pytest.raises(ValueError, match='fewer than two layers'):
            volume_fraction_profile(Sample(populate_if_none=False))


class TestProfileMaths:
    def setup_method(self):
        _clear()

    def test_two_medium_analytic_erf(self):
        back = Layer(Material(6.36, 0.0, 'D2O'), 0, 2.0, name='bulk')
        profile = volume_fraction_profile(_sample(back=back))
        phi = _by_label(profile)
        assert_allclose(phi['D2O'], 0.5 * (1 + erf(profile.z / (2.0 * np.sqrt(2)))), atol=1e-15)
        assert_allclose(phi['Air'], 1 - phi['D2O'], atol=1e-15)

    def test_zero_roughness_is_a_step_with_one_on_the_interface(self):
        back = Layer(Material(6.36, 0.0, 'D2O'), 0, 0.0, name='bulk')
        z = np.array([-1e-9, 0.0, 1e-9])
        profile = volume_fraction_profile(_sample(back=back), z=z)
        assert_allclose(_by_label(profile)['D2O'], [0.0, 1.0, 1.0])

    @pytest.mark.parametrize('case', ['solvated', 'bilayer', 'repeating', 'gradient'])
    def test_closure(self, case):
        air = Layer(Material(0.0, 0.0, 'Air'), 0, 0, name='Air')
        d2o = Material(6.36, 0.0, 'D2O')
        bulk = Layer(d2o, 0, 3, name='bulk')
        if case == 'solvated':
            film = Multilayer([Layer(MaterialSolvated(Material(2.0, 0.0, 'Si'), d2o, 0.3), 40, 3, name='film')])
        elif case == 'bilayer':
            film = Bilayer()
        elif case == 'repeating':
            film = RepeatingMultilayer(
                [Layer(Material(2.0, 0.0, 'A'), 10, 1, name='a'), Layer(Material(4.0, 0.0, 'B'), 20, 2, name='b')],
                repetitions=4,
            )
        else:
            film = GradientLayer(Material(0.0, 0.0, 'f'), Material(10.0, 0.0, 'b'), 10, 1, 5)
        profile = volume_fraction_profile(Sample(Multilayer([air]), film, Multilayer([bulk])))
        assert_allclose(profile.total, 1.0, atol=1e-12)

    def test_isolated_interface_is_at_the_midpoint(self):
        si = Material(2.07, 0.0, 'Si')
        d2o = Material(6.36, 0.0, 'D2O')
        film = Layer(MaterialSolvated(si, d2o, 0.3), 200, 2.0, name='film')
        profile = volume_fraction_profile(_sample(film, back=Layer(d2o, 0, 2.0, name='bulk')), z=np.array([0.0, 200.0]))
        phi = _by_label(profile)
        assert_allclose(phi['Si'], [0.35, 0.35])
        assert_allclose(phi['D2O'], [0.15, 0.65])
        assert_allclose(phi['Air'], [0.5, 0.0], atol=1e-12)

    def test_unequal_roughness_gives_a_negative_fraction_that_is_reported_not_clipped(self):
        film = Layer(Material(3.0, 0.0, 'film'), 10, 1.0, name='film')
        back = Layer(Material(6.0, 0.0, 'sub'), 0, 10.0, name='sub')
        with pytest.warns(VolumeFractionWarning, match="'film' reaches -0.0"):
            profile = volume_fraction_profile(_sample(film, back=back))
        phi_film = _by_label(profile)['film']
        index = np.argmin(phi_film)
        assert phi_film[index] == pytest.approx(-0.0994, abs=5e-4)
        assert profile.z[index] == pytest.approx(-2.5, abs=0.2)
        assert_allclose(profile.total, 1.0, atol=1e-12)

    def test_material_amount_is_conserved(self):
        d2o = Material(6.36, 0.0, 'D2O')
        film_a = Layer(MaterialSolvated(Material(2.0, 0.0, 'poly'), d2o, 0.4), 30, 3.0, name='a')
        film_b = Layer(MaterialSolvated(Material(2.0, 0.0, 'poly'), d2o, 0.1), 50, 5.0, name='b')
        back = Layer(d2o, 0, 4.0, name='bulk')
        z = np.linspace(-60, 140, 40001)
        profile = volume_fraction_profile(_sample(film_a, film_b, back=back), z=z)
        integral = np.trapezoid(_by_label(profile)['poly'], z)
        assert integral == pytest.approx(30 * 0.6 + 50 * 0.9, rel=1e-6)

    def test_matches_refnx_create_occupancy(self):
        refnx_reflect = pytest.importorskip('refnx.reflect')
        air, si, poly, d2o = (refnx_reflect.SLD(v) for v in (0.0, 2.07, 4.0, 6.36))
        structure = air(0, 0) | si(40, 3, vfsolv=0.3) | poly(60, 5, vfsolv=0.1) | d2o(0, 8)
        z_ref, vfp_ref = refnx_reflect.create_occupancy(structure, solvent_slab=-1)

        era_d2o = Material(6.36, 0.0, 'D2O')
        sample = _sample(
            Layer(MaterialSolvated(Material(2.07, 0.0, 'Si'), era_d2o, 0.3), 40, 3, name='si'),
            Layer(MaterialSolvated(Material(4.0, 0.0, 'poly'), era_d2o, 0.1), 60, 5, name='poly'),
            back=Layer(era_d2o, 0, 8, name='bulk'),
        )
        profile = volume_fraction_profile(sample)
        assert np.array_equal(profile.z, z_ref)
        phi = _by_label(profile)
        for label, row in zip(['Air', 'Si', 'poly', 'D2O'], vfp_ref):
            assert_allclose(phi[label], row, atol=1e-12)

    def test_caller_grid_is_used_as_given(self):
        sample = _sample(Layer(Material(2.0, 0.0, 'Si'), 50, 2, name='film'))
        z = np.array([-10, 0, 25, 50, 60])
        profile = volume_fraction_profile(sample, z=z)
        assert profile.z.dtype == float
        assert_allclose(profile.z, z)
        assert len(volume_fraction_profile(sample).z) == DEFAULT_POINTS
        assert len(volume_fraction_profile(sample, max_delta_z=0.01).z) > DEFAULT_POINTS

    def test_invalid_grid_arguments(self):
        sample = _sample(Layer(Material(2.0, 0.0, 'Si'), 50, 2, name='film'))
        with pytest.raises(ValueError, match='not both'):
            volume_fraction_profile(sample, z=np.array([0.0]), max_delta_z=1.0)
        with pytest.raises(ValueError, match='finite'):
            volume_fraction_profile(sample, z=np.array([0.0, np.nan]))
        with pytest.raises(ValueError, match='one-dimensional'):
            volume_fraction_profile(sample, z=np.zeros((2, 2)))
        with pytest.raises(ValueError, match='positive'):
            volume_fraction_profile(sample, max_delta_z=-1.0)

    def test_fractions_follow_live_parameters(self):
        si = Material(2.07, 0.0, 'Si')
        d2o = Material(6.36, 0.0, 'D2O')
        solvated = MaterialSolvated(si, d2o, 0.3)
        sample = _sample(Layer(solvated, 200, 1, name='film'), back=Layer(d2o, 0, 1, name='bulk'))
        z = np.array([100.0])
        assert _by_label(volume_fraction_profile(sample, z=z))['Si'][0] == pytest.approx(0.7)
        solvated.solvent_fraction = 0.5
        assert _by_label(volume_fraction_profile(sample, z=z))['Si'][0] == pytest.approx(0.5)


class TestGrouping:
    def setup_method(self):
        _clear()
        self.profile = volume_fraction_profile(
            _sample(
                Layer(Material(2.0, 0.0, 'poly'), 20, 1, name='a'),
                Layer(Material(4.0, 0.0, 'poly'), 20, 1, name='b'),
            )
        )

    def test_grouped_sums_and_keeps_the_rest_in_order(self):
        grouped = self.profile.grouped({'polymer': ['poly', 'poly (2)']})
        assert list(grouped.labels.values()) == ['polymer', 'Air', 'D2O']
        phi = _by_label(self.profile)
        assert_allclose(grouped.fractions['polymer'], phi['poly'] + phi['poly (2)'])
        assert_allclose(grouped.total, 1.0, atol=1e-12)
        # members can also be given by key, and the original is untouched
        key = self.profile.components[1]
        assert_allclose(self.profile.grouped({'p': [key]}).fractions['p'], phi['poly'])
        assert list(self.profile.labels.values()) == ['Air', 'poly', 'poly (2)', 'D2O']

    def test_invalid_groups(self):
        with pytest.raises(ValueError, match='Unknown component'):
            self.profile.grouped({'x': ['nothing']})
        with pytest.raises(ValueError, match='must not overlap'):
            self.profile.grouped({'x': ['poly'], 'y': ['poly']})
        with pytest.raises(ValueError, match='must not overlap'):
            self.profile.grouped({'x': ['poly', 'poly']})
        with pytest.raises(ValueError, match='no members'):
            self.profile.grouped({'x': []})
        with pytest.raises(ValueError, match='collides'):
            self.profile.grouped({'Air': ['poly']})


def test_smear_is_a_sum_of_steps():
    z = np.linspace(-10, 30, 5)
    values = np.array([0.0, 1.0, 0.25])
    profile = _smear(z, np.array([0.0, 20.0]), np.array([0.0, 0.0]), values)
    assert_allclose(profile, [0.0, 1.0, 1.0, 0.25, 0.25])
