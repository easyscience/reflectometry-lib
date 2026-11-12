# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""`include_magnetism` as the models change.

The flag is explicit calculator state. The calculator switches it off only
when the models it serves have just stopped being magnetic, and never on.
"""

import numpy as np
import pytest
from easyscience import global_object

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 30)


@pytest.fixture(autouse=True)
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _model(magnetic: bool = True) -> Model:
    """Vacuum | film (magnetic or not) | Si, each in its own assembly."""
    magnetism = LayerMagnetism(rho_m=2.0, theta_m=45.0) if magnetic else None
    return Model(
        Sample(
            Multilayer(Layer(Material(0.0, 0.0), 0.0, 0.0)),
            Multilayer(Layer(Material(4.0, 0.0), 100.0, 3.0, magnetism=magnetism)),
            Multilayer(Layer(Material(2.047, 0.0), 0.0, 3.0)),
        )
    )


def _served(*models: Model) -> CalculatorFactory:
    factory = CalculatorFactory()
    factory.switch('refl1d')
    for model in models:
        model.interface = factory
    return factory


def _evaluate(factory: CalculatorFactory, model: Model) -> None:
    factory.fit_func(Q, model.unique_name)


def _film(model: Model) -> Layer:
    return model.sample[1].layers[0]


def test_clearing_the_last_magnetic_layer_switches_magnetism_off():
    model = _model()
    factory = _served(model)
    factory().polarization_channel = 'mm'

    _film(model).magnetism = None

    assert factory().include_magnetism is False
    assert factory().polarization_channel is PolarizationChannel.PP


def test_clearing_one_while_another_model_stays_magnetic_keeps_it_on():
    first, second = _model(), _model()
    factory = _served(first, second)

    _film(first).magnetism = None
    _evaluate(factory, first)

    assert factory().include_magnetism is True


@pytest.mark.parametrize('removal', ['layer', 'assembly'])
def test_a_structural_removal_switches_magnetism_off_at_the_next_evaluation(removal):
    model = _model()
    factory = _served(model)
    factory().polarization_channel = 'pm'

    if removal == 'layer':
        model.sample[1].remove_layer(0)
    else:
        model.remove_assembly(1)
    _evaluate(factory, model)

    assert factory().include_magnetism is False
    assert factory().polarization_channel is PolarizationChannel.PP


def test_removing_the_magnetic_model_switches_magnetism_off():
    project = Project()
    project.calculator = 'refl1d'
    project.models = ModelCollection(_model(magnetic=False), _model())
    calculator = project._calculator()
    assert calculator.include_magnetism is True

    project.remove_model_at_index(1)

    assert calculator.include_magnetism is False


@pytest.mark.parametrize('first_magnetic', [True, False], ids=['magnetic-first', 'magnetic-last'])
def test_reassigning_the_models_keeps_the_selected_channel(first_magnetic):
    # The setter resets the calculator before binding the models again; that
    # brief moment with no models must not lose the channel (#449 regression).
    models = [_model(), _model(magnetic=False)]
    project = Project()
    project.calculator = 'refl1d'
    project.models = ModelCollection(*(models if first_magnetic else models[::-1]))
    magnetic = models[0]
    calculator = project._calculator()
    calculator.polarization_channel = 'mm'
    mm = project._calculator.fit_func(Q, magnetic.unique_name)

    project.models = ModelCollection(*project.models)

    assert calculator.include_magnetism is True
    assert calculator.polarization_channel is PolarizationChannel.MM
    np.testing.assert_allclose(project._calculator.fit_func(Q, magnetic.unique_name), mm)


def test_replacing_magnetic_models_by_non_magnetic_ones_still_resets_the_channel():
    project = Project()
    project.calculator = 'refl1d'
    project.models = ModelCollection(_model())
    calculator = project._calculator()
    calculator.polarization_channel = 'mm'

    project.models = ModelCollection(_model(magnetic=False))

    assert calculator.include_magnetism is False
    assert calculator.polarization_channel is PolarizationChannel.PP


def test_a_channel_picked_after_a_structural_removal_sees_it():
    model = _model()
    factory = _served(model)
    model.remove_assembly(1)

    # The setter reconciles first: the removal is seen before the channel is checked.
    with pytest.raises(ValueError, match='requires magnetism'):
        factory().polarization_channel = 'mm'
    assert factory().include_magnetism is False


def test_explicit_off_with_magnetic_layers_stays_off():
    model = _model()
    factory = _served(model)

    factory().include_magnetism = False
    for _ in range(3):
        _evaluate(factory, model)

    assert factory().include_magnetism is False


def test_explicit_off_then_on_is_kept():
    model = _model()
    factory = _served(model)

    factory().include_magnetism = False
    _evaluate(factory, model)
    factory().include_magnetism = True
    _evaluate(factory, model)

    assert factory().include_magnetism is True


def test_explicit_on_after_the_moments_were_removed_while_off_is_kept():
    model = _model()
    factory = _served(model)
    factory().include_magnetism = False
    _film(model).magnetism = None

    factory().include_magnetism = True
    _evaluate(factory, model)

    assert factory().include_magnetism is True


def test_on_before_any_magnetic_layer_survives_evaluations():
    model = _model(magnetic=False)
    factory = _served(model)

    factory().include_magnetism = True
    for _ in range(3):
        _evaluate(factory, model)

    assert factory().include_magnetism is True


def test_a_selected_channel_survives_an_unrelated_evaluation():
    model = _model()
    factory = _served(model)
    factory().polarization_channel = 'mp'

    _film(model).thickness.value = 120.0
    _evaluate(factory, model)

    assert factory().polarization_channel is PolarizationChannel.MP


def test_reconciliation_never_switches_magnetism_on():
    model = _model()
    factory = _served(model)
    _film(model).magnetism = None
    assert factory().include_magnetism is False

    # Magnetic again, but only through the tree: nothing switches the flag on.
    model.sample[2].layers[0]._magnetism = LayerMagnetism(rho_m=1.0)
    _evaluate(factory, model)

    assert factory().include_magnetism is False
