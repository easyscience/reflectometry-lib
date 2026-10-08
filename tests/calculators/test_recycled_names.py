# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""A backend entry belongs to the object it was created for, not to its name.

easyscience gives the ``unique_name`` of a dead object out again, while the
calculator keeps every entry it ever created under that name. A new object
bound to a recycled name must get a blank entry, not the dead object's state
(GitHub issue #426: a non-magnetic layer landing on a dead magnetic layer's
name became magnetic). Re-binding the *same* object must keep its entry, as
the item stacks hold references to it.
"""

import gc

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.01, 0.3, 40)


@pytest.fixture(autouse=True)
def _isolated_global_object():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _film_model(name: str, interface, magnetism: LayerMagnetism | None = None) -> Model:
    superphase = Layer(material=Material(0, 0, name=f'Vacuum {name}'), thickness=0, roughness=0, name=f'Super {name}')
    film = Layer(
        material=Material(4.0, 0, name=f'Film {name}'), thickness=100, roughness=3, magnetism=magnetism, name=f'Film {name}'
    )
    subphase = Layer(material=Material(2.07, 0, name=f'Si {name}'), thickness=0, roughness=3, name=f'Sub {name}')
    sample = Sample(Multilayer(superphase), Multilayer(film), Multilayer(subphase), name=f'Sample {name}')
    return Model(sample=sample, scale=1, background=0, name=name, interface=interface)


def _magnetism() -> LayerMagnetism:
    return LayerMagnetism(rho_m=1.5, theta_m=270.0)


def _released_pp_curve(interface) -> np.ndarray:
    """The pp curve of a model that is released on return -- and, holding reference cycles, not yet collected."""
    model = _film_model('released', interface, magnetism=_magnetism())
    return interface().reflectivity_profile_channel(Q, model.unique_name, 'pp')


class TestIssue426:
    def test_identical_model_after_a_released_one_gives_the_same_curve(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        reference = _released_pp_curve(interface)

        model = _film_model('again', interface, magnetism=_magnetism())

        for channel in ('pp', 'mm'):
            expected = _film_model('fresh', interface, magnetism=_magnetism())
            # `fresh` is created while `model` is alive, so its names are new.
            assert_allclose(
                interface().reflectivity_profile_channel(Q, model.unique_name, channel),
                interface().reflectivity_profile_channel(Q, expected.unique_name, channel),
                rtol=1e-12,
            )
        assert_allclose(interface().reflectivity_profile_channel(Q, model.unique_name, 'pp'), reference, rtol=1e-12)

    def test_non_magnetic_model_after_a_released_magnetic_one_is_not_magnetic(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        _released_pp_curve(interface)

        model = _film_model('plain', interface)

        wrapper = interface()._wrapper
        assert not interface().include_magnetism
        for multilayer in model.sample:
            assert multilayer.layers[0].unique_name not in wrapper._layer_magnetism
            assert wrapper.storage['layer'][multilayer.layers[0].unique_name].magnetism is None


class TestRecycledNameGetsABlankEntry:
    """The rule itself, on names recycled deterministically rather than by collection timing."""

    def test_layer_on_a_dead_magnetic_layers_name_is_blank(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        material = Material(4.0, 0, name='film material')
        dead = Layer(material=material, thickness=100, roughness=3, magnetism=_magnetism(), name='dead', interface=interface)
        name = dead.unique_name
        wrapper = interface()._wrapper
        assert name in wrapper._layer_magnetism
        del dead
        gc.collect()

        reborn = Layer(material=material, thickness=50, roughness=1, name='reborn', unique_name=name, interface=interface)

        assert reborn.unique_name == name
        assert name not in wrapper._layer_magnetism
        assert wrapper.storage['layer'][name].magnetism is None
        assert wrapper.get_layer_value(name, 'thickness') == 50
        assert not interface().include_magnetism

    @pytest.mark.parametrize('engine', ['refnx', 'refl1d'])
    def test_material_on_a_dead_materials_name_is_blank(self, engine):
        interface = CalculatorFactory()
        interface.switch(engine)
        dead = Material(4.0, 0.5, name='dead', interface=interface)
        name = dead.unique_name
        wrapper = interface()._wrapper
        stale = wrapper.storage['material'][name]
        del dead
        gc.collect()

        reborn = Material(1.0, 0, name='reborn', unique_name=name, interface=interface)

        assert reborn.unique_name == name
        assert wrapper.storage['material'][name] is not stale
        assert wrapper.get_material_value(name, 'rho' if engine == 'refl1d' else 'real') == 1.0

    def test_model_on_a_dead_models_name_drops_its_resolution(self):
        from easyreflectometry.model import PercentageFwhm
        from easyreflectometry.model import Pointwise

        interface = CalculatorFactory()
        dead = _film_model('dead', interface)
        dead.resolution_function = Pointwise([Q, np.ones_like(Q), (0.01 * Q) ** 2])
        name = dead.unique_name
        wrapper = interface()._wrapper
        del dead
        gc.collect()

        reborn = Model(name='reborn', unique_name=name, interface=interface)

        assert reborn.unique_name == name
        # The reborn model registered its own (default) resolution, not the dead one's.
        assert wrapper.resolution_function_for(name) is reborn.resolution_function
        assert isinstance(wrapper.resolution_function_for(name), PercentageFwhm)


class TestSameObjectKeepsItsEntry:
    def test_rebinding_a_magnetic_layer_keeps_its_slab_and_magnetism(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        layer = Layer(material=Material(4.0, 0, name='m'), thickness=100, roughness=3, magnetism=_magnetism(), name='layer')
        multilayer = Multilayer(layer, name='stack', interface=interface)
        wrapper = interface()._wrapper
        slab = wrapper.storage['layer'][layer.unique_name]

        layer.interface = interface
        multilayer.interface = interface

        assert wrapper.storage['layer'][layer.unique_name] is slab
        assert slab.magnetism is not None
        assert slab.magnetism.rhoM.value == 1.5
        assert list(wrapper.storage['item'][multilayer.unique_name].stack) == [slab]

    def test_calculator_switch_starts_from_blank_ownership(self):
        interface = CalculatorFactory()
        model = _film_model('model', interface)
        interface.switch('refl1d')
        model.interface = interface
        interface.switch('refnx')
        model.interface = interface
        interface.switch('refl1d')
        model.interface = interface

        # Every object is bound exactly once per fresh calculator; no stale entries.
        wrapper = interface()._wrapper
        assert set(wrapper.storage['layer']) == {ml.layers[0].unique_name for ml in model.sample}
        assert set(wrapper.storage['model']) == {model.unique_name}
