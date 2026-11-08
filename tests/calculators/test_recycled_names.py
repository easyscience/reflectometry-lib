# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""A calculator entry belongs to the model it was created for, not to its name.

easyscience gives the ``unique_name`` of a dead object out again. The
calculator keeps state only per model (see ``StatelessWrapperBase``); a new
model registered under a recycled name must not inherit the dead one's
(GitHub issue #426: a non-magnetic layer landing on a dead magnetic layer's
name became magnetic, back when the calculator kept an entry per layer).
"""

import gc

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators.refl1d.wrapper import _kernel_inputs
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.01, 0.3, 40)


@pytest.fixture(autouse=True)
def _isolated_global_object():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


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

        assert not interface().include_magnetism
        assemblies = _kernel_inputs(model, magnetism=True)[0]
        assert all(layer[-1] is None for _, layers in assemblies for layer in layers)


class TestRecycledNameGetsABlankEntry:
    """The rule itself, on names recycled deterministically rather than by collection timing."""

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
