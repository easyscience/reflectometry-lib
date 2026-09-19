# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests: bound *dependent* parameters must reach the calculator.

Historically `easyscience.Parameter._update()` recomputed a dependent
parameter's value but never pushed it through `_callback.fset`, so the
calculator (refnx / refl1d) kept the value from binding time. Every
constraint expressed through a bound dependent parameter was then a silent
no-op during simulation and fitting:

- `MaterialMixture`: `fraction` had zero effect on reflectivity and never
  moved in a fit (`_sld` / `_isld` are dependent parameters bound to the
  calculator),
- `GradientLayer` (and other assemblies): the thickness / roughness
  constraints updated only the front layer on the calculator side.

These tests pin the fixed behavior end-to-end against real calculators.
"""

import numpy as np
import pytest
from easyscience import global_object

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import GradientLayer
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

CALCULATORS = ['refnx', 'refl1d']


def _mixture_model(fraction: float) -> tuple[Model, MaterialMixture]:
    """Air | mixture(100 A) | Si substrate, only the mixture is interesting."""
    air = Material(0.0, 0.0, 'Air')
    si = Material(2.07, 0.0, 'Si')
    mixture = MaterialMixture(
        material_a=Material(2.0, 0.0, 'A'),
        material_b=Material(6.0, 0.0, 'B'),
        fraction=fraction,
        name='Mix',
    )
    superphase = Layer(air, 0, 0, 'Air layer')
    mix_layer = Layer(mixture, 100, 3, 'Mix layer')
    subphase = Layer(si, 0, 3, 'Si substrate')
    sample = Sample(
        Multilayer(superphase),
        Multilayer(mix_layer),
        Multilayer(subphase),
        name='Mix sample',
    )
    model = Model(sample, 1, 1e-9, PercentageFwhm(2.0), 'Mix model')
    return model, mixture


class TestMaterialMixtureReachesCalculator:
    @pytest.fixture(autouse=True)
    def _clean_map(self):
        global_object.map._clear()
        yield
        global_object.map._clear()

    @pytest.mark.parametrize('calculator', CALCULATORS)
    def test_fraction_change_updates_calculator_storage(self, calculator):
        # When
        model, mixture = _mixture_model(fraction=0.2)
        interface = CalculatorFactory()
        interface.switch(calculator)
        model.interface = interface

        # Then: binding pushed the initial dependent value to the calculator
        assert mixture.sld._callback.fget() == pytest.approx(mixture.sld.value_no_call_back)

        # When: the independent driver changes
        mixture.fraction = 0.8

        # Expect: python-side dependency math ran ...
        expected_sld = 2.0 * (1 - 0.8) + 6.0 * 0.8
        assert mixture.sld.value_no_call_back == pytest.approx(expected_sld)
        # ... and, crucially, the calculator-side storage was updated too
        # (this was the silent no-op: fset was never called from _update).
        assert mixture.sld._callback.fget() == pytest.approx(expected_sld)
        # ... and reading .value does not clobber the result with a stale one
        assert mixture.sld.value == pytest.approx(expected_sld)

    @pytest.mark.parametrize('calculator', CALCULATORS)
    def test_fraction_change_updates_reflectivity(self, calculator):
        # When
        model, mixture = _mixture_model(fraction=0.2)
        interface = CalculatorFactory()
        interface.switch(calculator)
        model.interface = interface
        q = np.linspace(0.01, 0.3, 50)
        before = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Then
        mixture.fraction = 0.8
        after = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Expect: the fraction drives the computed reflectivity
        assert not np.allclose(before, after)


class TestAssemblyConstraintsReachCalculator:
    @pytest.fixture(autouse=True)
    def _clean_map(self):
        global_object.map._clear()
        yield
        global_object.map._clear()

    def _gradient_model(self) -> tuple[Model, GradientLayer]:
        gradient = GradientLayer(
            front_material=Material(2.0, 0.0, 'Front'),
            back_material=Material(6.0, 0.0, 'Back'),
            thickness=60.0,
            roughness=1.0,
            discretisation_elements=6,
            name='Gradient',
        )
        model = Model(interface=None)
        model.add_assemblies(gradient)
        return model, gradient

    @pytest.mark.parametrize('calculator', CALCULATORS)
    def test_thickness_constraint_updates_all_sublayers(self, calculator):
        # When
        model, gradient = self._gradient_model()
        interface = CalculatorFactory()
        interface.switch(calculator)
        model.interface = interface

        # Then: the assembly setter drives the front layer; the dependent
        # sublayers must follow it into the calculator storage.
        gradient.thickness = 120.0
        per_sublayer = 120.0 / gradient.discretisation_elements

        # Expect
        for sublayer in gradient.layers:
            assert sublayer.thickness.value_no_call_back == pytest.approx(per_sublayer)
            assert sublayer.thickness._callback.fget() == pytest.approx(per_sublayer)

    @pytest.mark.parametrize('calculator', CALCULATORS)
    def test_roughness_constraint_updates_all_sublayers(self, calculator):
        # When
        model, gradient = self._gradient_model()
        interface = CalculatorFactory()
        interface.switch(calculator)
        model.interface = interface

        # Then
        gradient.roughness = 5.0

        # Expect
        for sublayer in gradient.layers:
            assert sublayer.roughness.value_no_call_back == pytest.approx(5.0)
            assert sublayer.roughness._callback.fget() == pytest.approx(5.0)

    @pytest.mark.parametrize('calculator', CALCULATORS)
    def test_thickness_change_updates_reflectivity(self, calculator):
        # When
        model, gradient = self._gradient_model()
        interface = CalculatorFactory()
        interface.switch(calculator)
        model.interface = interface
        q = np.linspace(0.01, 0.3, 50)
        before = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Then
        gradient.thickness = 120.0
        after = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Expect
        assert not np.allclose(before, after)


class TestMaterialMixtureFractionFits:
    @pytest.fixture(autouse=True)
    def _clean_map(self):
        global_object.map._clear()
        yield
        global_object.map._clear()

    def test_fit_moves_fraction_to_truth(self):
        """End-to-end: a fit recovers the fraction that generated the data.

        Before the easyscience fix the minimizer saw zero gradient in
        `fraction` (the calculator never received the updated mixed SLD),
        so the fit left the start value untouched.
        """
        # When: synthesize data at the true fraction
        truth = 0.7
        start = 0.3
        model, mixture = _mixture_model(fraction=truth)
        model.interface = CalculatorFactory()
        q = np.linspace(0.01, 0.3, 100)
        y_true = np.asarray(model.interface().reflectivity_profile(q, model.unique_name))

        # Then: start the fit elsewhere with fraction as the only free parameter
        mixture.fraction = start
        mixture.fraction.fixed = False
        data = DataSet1D(
            name='synthetic',
            x=q,
            y=y_true,
            ye=(0.01 * y_true) ** 2,  # ye stores variances
        )
        fitter = MultiFitter(model)
        result = fitter.fit_single_data_set_1d(data)

        # Expect: the fraction moved and recovered the truth
        fitted = mixture.fraction.value
        assert result.success
        assert fitted != pytest.approx(start), 'fraction did not move during fit'
        assert fitted == pytest.approx(truth, abs=0.05)
