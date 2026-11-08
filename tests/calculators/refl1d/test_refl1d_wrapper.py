# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The refl1d wrapper builds its sample from the model on every evaluation.

Agreement with independent reference data is in
``tests/calculators/test_validation_suite.py``, agreement with the old storage
wrapper in ``tests/calculators/test_calculator_baselines.py``; the magnetism
flag as the model changes in ``tests/calculators/test_magnetism_reconciliation.py``.
"""

from unittest.mock import patch

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from refl1d import names
from refl1d.sample.layers import Repeat

from easyreflectometry.calculators.polarization import POLARIZATION_CHANNEL_TO_INDEX
from easyreflectometry.calculators.polarization import PolarizationChannel
from easyreflectometry.calculators.refl1d.wrapper import OVERSAMPLING_FACTOR
from easyreflectometry.calculators.refl1d.wrapper import RESOLUTION_PADDING
from easyreflectometry.calculators.refl1d.wrapper import Refl1dWrapper
from easyreflectometry.calculators.refl1d.wrapper import _build_sample
from easyreflectometry.calculators.refl1d.wrapper import _kernel_inputs
from easyreflectometry.calculators.refl1d.wrapper import _oversampled_regions
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 100)


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def film_model(rho: float = 4.0, rho_m: float | None = None, theta_m: float = 270.0) -> Model:
    """Vacuum | 100 A film of *rho*, magnetic when *rho_m* is given | Si."""
    magnetism = None if rho_m is None else LayerMagnetism(rho_m=rho_m, theta_m=theta_m)
    sample = Sample(
        Multilayer(Layer(Material(0.0, 0.0, 'Vacuum'), 0.0, 0.0, 'Superphase'), 'Fronting'),
        Multilayer(Layer(Material(rho, 0.0, 'Film'), 100.0, 0.0, 'Film', magnetism=magnetism), 'Film'),
        Multilayer(Layer(Material(2.047, 0.0, 'Si'), 0.0, 0.0, 'Subphase'), 'Backing'),
    )
    return Model(sample, 1.0, 0.0, name='Model')


def wrapper_for(*models: Model, magnetism: bool = False) -> Refl1dWrapper:
    """The wrapper holds its models weakly: the caller keeps them alive."""
    wrapper = Refl1dWrapper()
    for model in models:
        wrapper.register_model(model)
    wrapper.magnetism = magnetism
    return wrapper


def channels_of(model: Model) -> dict[str, np.ndarray]:
    return wrapper_for(model, magnetism=True).calculate_polarized(Q, model.unique_name)


def unpolarized(rho: float) -> np.ndarray:
    model = film_model(rho)
    return wrapper_for(model).calculate(Q)


class TestSampleBuild:
    def test_refl1d_receives_the_layers_substrate_first(self):
        sample = _build_sample(_kernel_inputs(film_model(), magnetism=False))
        assert [slab.material.rho.value for slab in sample.layers] == [2.047, 4.0, 0.0]
        assert [slab.thickness.value for slab in sample.layers] == [0.0, 100.0, 0.0]

    @pytest.mark.parametrize(('repetitions', 'kinds'), [(1.0, ['Slab'] * 4), (5.0, ['Slab', 'Repeat', 'Slab'])])
    def test_only_repeated_assemblies_become_a_repeat(self, repetitions, kinds):
        film = Layer(Material(2.0, 0.0), 10.0, 1.0)
        spacer = Layer(Material(1.0, 0.0), 20.0, 1.0)
        sample = Sample(
            Multilayer(Layer(Material(0.0, 0.0), 0.0, 0.0)),
            RepeatingMultilayer([film, spacer], repetitions),
            Multilayer(Layer(Material(4.0, 0.0), 0.0, 1.0)),
        )
        built = _build_sample(_kernel_inputs(Model(sample, 1.0, 0.0), magnetism=False))
        assert [type(layer).__name__ for layer in built.layers] == kinds
        if repetitions != 1:
            repeat = built.layers[1]
            assert isinstance(repeat, Repeat)
            assert repeat.repeat.value == repetitions
            # Substrate first inside the repeat too.
            assert [slab.thickness.value for slab in repeat.stack.layers] == [20.0, 10.0]

    def test_magnetism_reaches_refl1d_only_when_it_is_on(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        off = _build_sample(_kernel_inputs(model, magnetism=False))
        on = _build_sample(_kernel_inputs(model, magnetism=True))
        assert all(slab.magnetism is None for slab in off.layers)
        assert [slab.magnetism is None for slab in on.layers] == [True, False, True]
        assert (on.layers[1].magnetism.rhoM.value, on.layers[1].magnetism.thetaM.value) == (2.0, 45.0)

    def test_values_are_read_at_evaluation_time(self):
        model = film_model()
        wrapper = wrapper_for(model)
        before = wrapper.calculate(Q)

        # A change on the model only, with no call into the calculator
        model.sample[1].layers[0].thickness.value = 55.0

        assert not np.array_equal(before, wrapper.calculate(Q))

    def test_structural_edits_need_no_mirroring(self):
        model = film_model()
        model.sample[1].add_layer(Layer(Material(3.0, 0.0, 'Extra'), 20.0, 1.0, 'ExtraLayer'))
        assert len(_build_sample(_kernel_inputs(model, magnetism=False)).layers) == 4

    def test_material_mixture_gives_its_mixed_sld(self):
        mixture = MaterialMixture(Material(2.0, 0.0), Material(6.0, 0.0), fraction=0.25)
        sample = Sample(Multilayer(Layer(Material(0.0, 0.0), 0.0, 0.0)), Multilayer(Layer(mixture, 0.0, 1.0)))
        built = _build_sample(_kernel_inputs(Model(sample, 1.0, 0.0), magnetism=False))
        assert built.layers[0].material.rho.value == pytest.approx(3.0)

    def test_scale_and_background_are_taken_from_the_model(self):
        plain = film_model()
        scaled = film_model()
        scaled.scale.value, scaled.background.value = 2.0, 1e-6
        assert_allclose(wrapper_for(scaled).calculate(Q), 2.0 * wrapper_for(plain).calculate(Q) + 1e-6, rtol=1e-8)

    def test_an_evaluation_leaves_the_model_untouched(self):
        model = film_model(rho_m=2.0, theta_m=45.0)

        def state():
            return [(p.value, p.error, p.min, p.max, p.fixed, p.independent) for p in model.get_all_parameters()]

        before = state()
        wrapper = wrapper_for(model, magnetism=True)
        wrapper.calculate(Q)
        wrapper.calculate_polarized(Q, None)
        wrapper.sld_profile()
        wrapper.magnetic_sld_profile()
        assert state() == before

    def test_sld_profile_ends(self):
        _, sld = wrapper_for(model := film_model()).sld_profile(model.unique_name)
        assert sld[0] == pytest.approx(0.0)
        assert sld[-1] == pytest.approx(2.047)


class TestPolarized:
    def test_calculate_polarized_shape(self):
        channels = channels_of(film_model(rho_m=2.0, theta_m=45.0))
        assert list(channels) == ['pp', 'pm', 'mp', 'mm']
        for reflectivity in channels.values():
            assert isinstance(reflectivity, np.ndarray)
            assert len(reflectivity) == len(Q)
            assert np.all(np.isfinite(reflectivity))

    def test_calculate_follows_selected_channel(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        channels = wrapper.calculate_polarized(Q, None)
        for channel in PolarizationChannel:
            wrapper.polarization_channel = channel
            assert_allclose(wrapper.calculate(Q), channels[channel.value], rtol=1e-10)
            assert_allclose(wrapper.calculate_channel(Q, None, channel), channels[channel.value], rtol=1e-10)

    def test_zero_magnetic_sld_degenerates_to_the_unpolarized_result(self):
        channels = channels_of(film_model(rho_m=0.0))
        reference = unpolarized(4.0)
        assert_allclose(channels['pp'], reference, rtol=1e-5)
        assert_allclose(channels['mm'], reference, rtol=1e-5)
        # Tolerance pinned from the observed numerics of refl1d 1.0.0 (machine noise).
        assert np.max(channels['pm']) < 1e-16
        assert np.max(channels['mp']) < 1e-16

    def test_channel_ordering(self):
        # Pins the pp/mm halves of POLARIZATION_CHANNEL_TO_INDEX with physics, guarding
        # against a pp/mm swap: with the moment collinear with the neutron polarization
        # axis there is no spin flip and the non-spin-flip channels see rho +/- rhoM.
        # refl1d returns cross-sections in probe._xs_names order ['mm','mp','pm','pp'].
        # With the default guide field (Aguide=270), thetaM=270 is the
        # moment-parallel-to-field orientation: spin-up ('pp') sees rho + rhoM,
        # spin-down ('mm') sees rho - rhoM. thetaM=90 (anti-parallel) swaps the two.
        rho, rho_m = 4.0, 2.0
        plus, minus = unpolarized(rho + rho_m), unpolarized(rho - rho_m)

        aligned = channels_of(film_model(rho, rho_m=rho_m, theta_m=270.0))
        assert_allclose(aligned['pp'], plus, rtol=1e-4, atol=1e-9)
        assert_allclose(aligned['mm'], minus, rtol=1e-4, atol=1e-9)
        assert np.max(aligned['pm']) < 1e-16
        assert np.max(aligned['mp']) < 1e-16

        anti_aligned = channels_of(film_model(rho, rho_m=rho_m, theta_m=90.0))
        assert_allclose(anti_aligned['pp'], minus, rtol=1e-4, atol=1e-9)
        assert_allclose(anti_aligned['mm'], plus, rtol=1e-4, atol=1e-9)

    def test_channel_index_map_matches_refl1d_xs_names(self):
        # The wrapper extracts Experiment.reflectivity() results in probe.xs order,
        # which is PolarizedQProbe._xs_names; a reordering on either side fails here.
        xs_names = names.PolarizedQProbe._xs_names
        assert len(xs_names) == 4
        for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items():
            assert xs_names[index] == channel.value

    def test_spin_flip(self):
        # A moment perpendicular to the neutron polarization (thetaM=0) flips spins;
        # a collinear one (thetaM=90) essentially does not.
        aligned = channels_of(film_model(rho_m=2.0, theta_m=90.0))
        perpendicular = channels_of(film_model(rho_m=2.0, theta_m=0.0))
        assert np.max(perpendicular['pm']) > 1e3 * np.max(aligned['pm'])
        assert np.max(perpendicular['pm']) > 1e-6  # absolute sanity floor
        # pm and mp are identical by symmetry for a non-chiral, non-absorptive sample.
        assert_allclose(perpendicular['pm'], perpendicular['mp'], rtol=1e-10)

    def test_scale_and_background_reach_every_cross_section(self):
        plain = channels_of(film_model(rho_m=2.0, theta_m=45.0))
        scaled_model = film_model(rho_m=2.0, theta_m=45.0)
        scaled_model.scale.value, scaled_model.background.value = 2.0, 1e-6
        scaled = channels_of(scaled_model)
        for key in ['pp', 'pm', 'mp', 'mm']:
            assert_allclose(scaled[key], 2.0 * plain[key] + 1e-6, rtol=1e-8)

    def test_the_guide_field_reaches_refl1d(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        before = wrapper.calculate_polarized(Q, None)
        model.guide_field_angle.value = 0.0
        assert not np.allclose(before['pm'], wrapper.calculate_polarized(Q, None)['pm'])

    def test_polarized_reflectivity_requires_magnetism(self):
        model = film_model()
        with pytest.raises(ValueError, match='requires magnetism'):
            wrapper_for(model).calculate_polarized(Q, None)

    def test_magnetic_sld_profile_requires_magnetism(self):
        model = film_model(rho_m=2.0)
        with pytest.raises(ValueError, match='requires magnetism'):
            wrapper_for(model).magnetic_sld_profile()

    def test_malformed_output_is_rejected(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)

        def returning(cross_sections):
            experiment = patch('easyreflectometry.calculators.refl1d.wrapper.names.Experiment')
            mock = experiment.start()
            mock.return_value.reflectivity.return_value = [(Q, r) for r in cross_sections]
            return experiment

        bad = np.ones(len(Q))
        bad[0] = np.nan
        # index 1 is 'mp' and index 3 is 'pp' in refl1d's (mm, mp, pm, pp) order
        cases = [
            ([np.ones(len(Q))] * 3, 'expected 4'),
            ([np.ones(len(Q)), np.ones(len(Q) - 1), np.ones(len(Q)), np.ones(len(Q))], 'malformed mp'),
            ([np.ones(len(Q))] * 3 + [bad], 'malformed pp'),
        ]
        for cross_sections, message in cases:
            experiment = returning(cross_sections)
            try:
                with pytest.raises(RuntimeError, match=message):
                    # A new background each time, so no cached result is served.
                    model.background.value += 1e-9
                    wrapper.calculate_polarized(Q, None)
            finally:
                experiment.stop()


class TestPolarizedCache:
    @pytest.fixture
    def kernel_calls(self, monkeypatch):
        calls = []
        real = names.Experiment.reflectivity

        def counting(self, *args, **kwargs):
            calls.append(1)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(names.Experiment, 'reflectivity', counting)
        return calls

    def test_all_four_channels_come_from_one_kernel_call(self, kernel_calls):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        channels = wrapper.calculate_polarized(Q, None)
        for channel in PolarizationChannel:
            assert_array_equal(wrapper.calculate_channel(Q, None, channel), channels[channel.value])
            wrapper.polarization_channel = channel
            assert_array_equal(wrapper.calculate(Q), channels[channel.value])
        assert len(kernel_calls) == 1

    def test_returned_arrays_are_copies(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        first = wrapper.calculate_polarized(Q, None)
        expected = first['pp'].copy()
        first['pp'][:] = 0.0
        wrapper.calculate(Q)[:] = 0.0
        assert_array_equal(wrapper.calculate_polarized(Q, None)['pp'], expected)

    @pytest.mark.parametrize(
        'edit',
        [
            'fixed_value',
            'dependent_value',
            'layer_added',
            'material_used_elsewhere',
            'repeat_count',
            'moment',
            'guide_field',
            'background',
        ],
    )
    def test_any_kernel_input_edit_recalculates(self, kernel_calls, edit):
        water = MaterialMixture(Material(-0.56, 0.0), Material(6.36, 0.0), fraction=0.5)
        film = Layer(Material(4.0, 0.0), 100.0, 3.0, magnetism=LayerMagnetism(rho_m=2.0, theta_m=45.0))
        repeated = RepeatingMultilayer([Layer(Material(1.0, 0.0), 30.0, 3.0)], 2.0)
        substrate = Layer(water, 0.0, 3.0)
        model = Model(
            Sample(Multilayer(Layer(Material(0.0, 0.0), 0.0, 0.0)), Multilayer(film), repeated, Multilayer(substrate))
        )
        wrapper = wrapper_for(model, magnetism=True)
        before = wrapper.calculate_polarized(Q, None)
        assert film.thickness.fixed

        if edit == 'fixed_value':
            film.thickness.value = 120.0
        elif edit == 'dependent_value':
            water.fraction.value = 0.9
        elif edit == 'layer_added':
            model.sample[1].add_layer(Layer(Material(5.0, 0.0), 20.0, 3.0))
        elif edit == 'material_used_elsewhere':
            film.material = repeated.layers[0].material
        elif edit == 'repeat_count':
            repeated.repetitions.value = 3.0
        elif edit == 'moment':
            film.magnetism.theta_m.value = 0.0
        elif edit == 'guide_field':
            model.guide_field_angle.value = 0.0
        elif edit == 'background':
            model.background.value = 1e-6

        after = wrapper.calculate_polarized(Q, None)
        assert len(kernel_calls) == 2
        assert not all(np.array_equal(before[c], after[c]) for c in before)

    def test_each_resolution_at_the_same_q_keeps_its_own_result(self, kernel_calls):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        narrow = PercentageFwhm(1.0)
        wide = Pointwise([Q, np.ones_like(Q), (0.05 * Q / 2.3548) ** 2])
        r_narrow = wrapper.calculate_polarized(Q, None, resolution_function=narrow)
        r_wide = wrapper.calculate_polarized(Q, None, resolution_function=wide)
        assert not np.allclose(r_narrow['pp'], r_wide['pp'])
        # Both grids stay cached for the same model state.
        assert_array_equal(wrapper.calculate_polarized(Q, None, resolution_function=narrow)['pp'], r_narrow['pp'])
        assert_array_equal(wrapper.calculate_polarized(Q, None, resolution_function=wide)['pp'], r_wide['pp'])
        assert len(kernel_calls) == 2

    def test_the_cache_goes_with_the_model(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        wrapper.calculate_polarized(Q, None)
        entry = wrapper._models[model.unique_name]
        assert entry.cache
        wrapper.discard_model(model.unique_name)
        assert model.unique_name not in wrapper._models


class TestOversampling:
    """refl1d evaluates the kernel on a finer grid than the data before smearing onto it."""

    def test_the_region_spans_the_data_padded_by_the_resolution(self):
        q = np.linspace(1, 10, 10)
        dq = np.linspace(0.01, 0.1, 10)
        (region,) = _oversampled_regions(q, dq)
        assert region.Q_start == pytest.approx(1 - RESOLUTION_PADDING * 0.01)
        assert region.Q_end == pytest.approx(10 + RESOLUTION_PADDING * 0.1)
        assert region.n == OVERSAMPLING_FACTOR * len(q)

    def test_the_region_covers_the_widest_point_at_a_repeated_q(self):
        # A merged dataset: each q twice, a sharp and a broad width, sharp first.
        q = np.repeat([0.01, 0.3], 2)
        dq = np.array([1e-5, 1e-3, 1e-4, 2e-2])
        (region,) = _oversampled_regions(q, dq)
        assert region.Q_start == pytest.approx(0.01 - RESOLUTION_PADDING * 1e-3)
        assert region.Q_end == pytest.approx(0.3 + RESOLUTION_PADDING * 2e-2)

    @pytest.mark.parametrize('magnetism', [False, True], ids=['plain', 'polarized'])
    def test_the_kernel_is_evaluated_on_the_oversampled_grid(self, monkeypatch, magnetism):
        model = film_model(rho_m=2.0 if magnetism else None)
        wrapper = wrapper_for(model, magnetism=magnetism)
        grids = []
        real = names.Experiment.reflectivity

        def recording(self, *args, **kwargs):
            grids.append(np.asarray(self.probe.calc_Q))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(names.Experiment, 'reflectivity', recording)
        wrapper.calculate(Q)
        # The data points plus the even grid (np.unique merges coinciding points).
        assert len(grids[0]) > OVERSAMPLING_FACTOR * len(Q)
        assert set(Q) <= set(grids[0])


class TestMagnetismFlag:
    def test_polarization_channel_normalization(self):
        wrapper = wrapper_for(magnetism=True)
        wrapper.polarization_channel = PolarizationChannel.MM
        assert wrapper.polarization_channel is PolarizationChannel.MM
        wrapper.polarization_channel = 'pm'
        assert wrapper.polarization_channel is PolarizationChannel.PM
        for bad in ['MM', 'xx', None]:
            with pytest.raises(ValueError):
                wrapper.polarization_channel = bad

    def test_polarization_channel_requires_magnetism(self):
        wrapper = Refl1dWrapper()
        with pytest.raises(ValueError):
            wrapper.polarization_channel = 'mm'
        wrapper.polarization_channel = 'pp'
        assert wrapper.polarization_channel is PolarizationChannel.PP

    def test_disabling_magnetism_resets_the_channel_and_keeps_the_moments(self):
        model = film_model(rho_m=2.0, theta_m=45.0)
        wrapper = wrapper_for(model, magnetism=True)
        polarized = wrapper.calculate_polarized(Q, None)
        wrapper.polarization_channel = 'mm'

        wrapper.magnetism = False

        assert wrapper.polarization_channel is PolarizationChannel.PP
        assert_allclose(wrapper.calculate(Q), unpolarized(4.0), rtol=1e-10)
        # The model still holds the moment: switching back gives the same channels.
        wrapper.magnetism = True
        assert wrapper.polarization_channel is PolarizationChannel.PP
        for channel, values in wrapper.calculate_polarized(Q, None).items():
            assert_allclose(values, polarized[channel], rtol=1e-10)

    def test_the_channel_survives_reset_storage(self):
        wrapper = wrapper_for(magnetism=True)
        wrapper.polarization_channel = 'mm'
        wrapper.reset_storage()
        assert wrapper.polarization_channel is PolarizationChannel.MM
        assert wrapper.magnetism is True
