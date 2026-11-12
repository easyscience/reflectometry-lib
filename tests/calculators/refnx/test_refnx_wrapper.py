# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The refnx wrapper builds its structure from the model on every evaluation.

Agreement with independent reference data is in
``tests/calculators/test_validation_suite.py``, agreement with the old storage
wrapper in ``tests/calculators/test_calculator_baselines.py``.
"""

import gc

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from refnx import reflect

from easyreflectometry.calculators.refnx.wrapper import RefnxWrapper
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.001, 0.3, 25)


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def build_model(repetitions: float = 1.0, scale: float = 1.0, background: float = 0.0) -> Model:
    """Ambient | (film, ambient) x repetitions | substrate."""
    ambient_layer = Layer(Material(0.0, 0.0, 'Ambient'), 0.0, 0.0, 'AmbientLayer')
    film_layer = Layer(Material(2.0, 0.0, 'Film'), 10.0, 1.0, 'FilmLayer')
    substrate_layer = Layer(Material(4.0, 0.0, 'Substrate'), 0.0, 1.0, 'SubstrateLayer')
    sample = Sample(
        Multilayer(ambient_layer, 'Fronting'),
        RepeatingMultilayer([film_layer, ambient_layer], repetitions, 'Stack'),
        Multilayer(substrate_layer, 'Backing'),
    )
    return Model(sample, scale, background, name='Model')


def wrapper_for(*models: Model) -> RefnxWrapper:
    wrapper = RefnxWrapper()
    for model in models:
        wrapper.register_model(model)
    return wrapper


class TestReferenceModels:
    def test_three_layer_reflectivity(self):
        # Given, the model the storage-based calculator tests pinned
        model = build_model()
        wrapper = wrapper_for(model)
        wrapper.set_resolution_function(PercentageFwhm(5.0))

        # When/Then
        expected = [
            9.99956517e-01,
            2.16286891e-03,
            1.14086254e-04,
            1.93031759e-05,
            4.94188894e-06,
            1.54191953e-06,
            5.45592112e-07,
            2.26619392e-07,
            1.26726993e-07,
            1.01842852e-07,
        ]
        assert_almost_equal(wrapper.calculate(np.linspace(0.001, 0.3, 10), model.unique_name), expected, decimal=6)

    def test_sld_profile_ends(self):
        model = build_model()
        z, sld = wrapper_for(model).sld_profile(model.unique_name)
        assert_almost_equal(sld[0], 0)
        assert_almost_equal(sld[-1], 4)

    def test_scale_and_background_are_taken_from_the_model(self):
        plain = build_model()
        scaled = build_model(scale=2.0, background=1e-3)
        assert_allclose(wrapper_for(scaled).calculate(Q), 2.0 * wrapper_for(plain).calculate(Q) + 1e-3)

    def test_material_mixture_gives_its_mixed_sld(self):
        mixture = MaterialMixture(Material(2.0, 0.0), Material(6.0, 0.0), fraction=0.25)
        sample = Sample(
            Multilayer(Layer(Material(0.0, 0.0, 'Ambient'), 0.0, 0.0, 'AmbientLayer')),
            Multilayer(Layer(mixture, 0.0, 1.0, 'MixtureLayer')),
        )
        structure = RefnxWrapper._build_structure(Model(sample, 1.0, 0.0))
        assert_almost_equal(structure.components[-1].sld.real.value, 3.0)


class TestStructureTraversal:
    def test_layer_and_assembly_order_is_preserved(self):
        model = build_model()
        names = [component.name for component in RefnxWrapper._build_structure(model).components]
        assert names == [
            model.sample[0].layers[0].unique_name,
            model.sample[1].layers[0].unique_name,
            model.sample[1].layers[1].unique_name,
            model.sample[2].layers[0].unique_name,
        ]

    def test_single_repetition_assemblies_are_flattened(self):
        structure = RefnxWrapper._build_structure(build_model(repetitions=1.0))
        assert all(isinstance(component, reflect.Slab) for component in structure.components)

    def test_repeated_assemblies_become_a_stack(self):
        model = build_model(repetitions=5.0)
        stacks = [c for c in RefnxWrapper._build_structure(model).components if isinstance(c, reflect.Stack)]
        assert len(stacks) == 1
        assert stacks[0].name == model.sample[1].unique_name
        assert stacks[0].repeats.value == 5.0
        assert len(stacks[0].components) == 2

    def test_values_are_read_at_evaluation_time(self):
        model = build_model()
        wrapper = wrapper_for(model)
        before = wrapper.calculate(Q)

        # A change on the model only, with no call into the calculator
        model.sample[1].layers[0].thickness.value = 55.0

        assert not np.array_equal(before, wrapper.calculate(Q))
        assert RefnxWrapper._build_structure(model).components[1].thick.value == 55.0

    def test_structural_edits_need_no_mirroring(self):
        model = build_model()
        model.sample[1].add_layer(Layer(Material(3.0, 0.0, 'Extra'), 20.0, 1.0, 'ExtraLayer'))
        assert len(RefnxWrapper._build_structure(model).components) == 5

    def test_an_evaluation_leaves_the_model_untouched(self):
        model = build_model(repetitions=3.0)
        before = [(p.value, p.error, p.min, p.max, p.fixed, p.independent) for p in model.get_all_parameters()]
        wrapper = wrapper_for(model)
        wrapper.calculate(Q)
        wrapper.sld_profile()
        assert [(p.value, p.error, p.min, p.max, p.fixed, p.independent) for p in model.get_all_parameters()] == before


class TestMembership:
    def test_models_are_registered_by_unique_name(self):
        first, second = build_model(), build_model()
        assert wrapper_for(first, second).models == {first.unique_name: first, second.unique_name: second}

    def test_the_sole_model_is_the_default(self):
        model = build_model()
        wrapper = wrapper_for(model)
        assert_array_equal(wrapper.calculate(Q), wrapper.calculate(Q, model.unique_name))

    def test_several_models_require_a_model_id(self):
        first, second = build_model(), build_model()
        with pytest.raises(ValueError, match='pass `model_id`'):
            wrapper_for(first, second).calculate(Q)

    def test_each_model_is_evaluated_from_its_own_structure(self):
        thin, thick = build_model(), build_model()
        thick.sample[1].layers[0].thickness.value = 100.0
        wrapper = wrapper_for(thin, thick)
        assert not np.array_equal(wrapper.calculate(Q, thin.unique_name), wrapper.calculate(Q, thick.unique_name))
        assert_array_equal(wrapper.calculate(Q, thin.unique_name), wrapper_for(thin).calculate(Q))

    def test_no_model_attached_raises(self):
        with pytest.raises(ValueError, match='No model is attached'):
            RefnxWrapper().calculate(Q)

    def test_unknown_model_id_raises(self):
        model = build_model()
        with pytest.raises(ValueError, match='No model with unique name'):
            wrapper_for(model).calculate(Q, 'not_a_model')

    def test_discard_and_reset(self):
        first, second = build_model(), build_model()
        wrapper = wrapper_for(first, second)
        wrapper.discard_model(first.unique_name)
        assert list(wrapper.models) == [second.unique_name]
        wrapper.reset_storage()
        assert wrapper.models == {}

    def test_registering_again_keeps_the_resolution(self):
        model = build_model()
        wrapper = wrapper_for(model)
        resolution = PercentageFwhm(3.0)
        wrapper.set_resolution_function(resolution, model.unique_name)
        wrapper.register_model(model)
        assert wrapper.resolution_function_for(model.unique_name) is resolution

    def test_a_dead_model_takes_its_entry_and_resolution_with_it(self):
        wrapper = RefnxWrapper()
        for _ in range(3):
            model = build_model()
            wrapper.register_model(model)
            wrapper.set_resolution_function(PercentageFwhm(3.0), model.unique_name)
            name = model.unique_name
            del model
            gc.collect()
            assert wrapper.models == {}
            assert wrapper._models == {}
            assert isinstance(wrapper.resolution_function_for(name), PercentageFwhm)
            assert wrapper.resolution_function_for(name).constant == PercentageFwhm().constant

    def test_a_recycled_name_keeps_the_new_models_entry(self):
        wrapper = RefnxWrapper()
        dead = build_model()
        name = dead.unique_name
        wrapper.register_model(dead)
        global_object.registry.discard(name)
        reborn = Model(name='reborn', unique_name=name)
        wrapper.register_model(reborn)
        del dead
        gc.collect()
        assert wrapper.models == {name: reborn}


class TestResolution:
    def test_precedence_dataset_then_model_then_default(self):
        model = build_model()
        wrapper = wrapper_for(model)
        default, own, dataset = PercentageFwhm(1.0), PercentageFwhm(4.0), PercentageFwhm(8.0)
        wrapper.set_resolution_function(default)
        assert wrapper._resolution_for(model.unique_name, None) is default
        wrapper.set_resolution_function(own, model.unique_name)
        assert wrapper._resolution_for(model.unique_name, None) is own
        assert wrapper._resolution_for(model.unique_name, dataset) is dataset

    def test_percentage_fwhm_is_passed_as_a_scalar(self, monkeypatch):
        assert _captured_x_err(monkeypatch, PercentageFwhm(5.0)) == 5.0

    @pytest.mark.parametrize(
        'resolution_function',
        [
            LinearSpline(np.array([0.0, 0.15, 0.3]), np.array([0.0005, 0.002, 0.006])),
            Pointwise([Q, np.ones_like(Q), (0.01 * Q + 1e-4) ** 2]),
        ],
        ids=['spline', 'pointwise'],
    )
    def test_other_resolutions_are_pointwise_fwhm(self, monkeypatch, resolution_function):
        assert_allclose(_captured_x_err(monkeypatch, resolution_function), resolution_function.smearing(Q) * SIGMA_TO_FWHM)


def _captured_x_err(monkeypatch, resolution_function):
    captured = {}
    real_call = reflect.ReflectModel.__call__

    def spy(self, x, p=None, x_err=None):
        captured['x_err'] = x_err
        assert self.dq_type == 'pointwise'
        return real_call(self, x, p=p, x_err=x_err)

    monkeypatch.setattr(reflect.ReflectModel, '__call__', spy)
    model = build_model()
    wrapper_for(model).calculate(Q, resolution_function=resolution_function)
    return captured['x_err']


class TestMagnetism:
    def test_magnetism_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            RefnxWrapper().magnetism = True

    def test_without_magnetism_only_pp_exists(self):
        model = build_model()
        wrapper = wrapper_for(model)
        assert_array_equal(wrapper.calculate_channel(Q, None, 'pp'), wrapper.calculate(Q))
        with pytest.raises(ValueError, match='requires magnetism'):
            wrapper.calculate_channel(Q, None, 'mm')
