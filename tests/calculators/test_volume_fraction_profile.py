# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The volume fraction profile against the calculators' SLD profiles."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialSolvated
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import volume_fraction_profile
from easyreflectometry.sample.volume_fraction import _resolve_slabs
from easyreflectometry.sample.volume_fraction import _smear


def _solvated_model():
    """Air | solvated Si | solvated polymer (absorbing) | D2O, every component with one SLD."""
    air = Material(0.0, 0.0, 'Air')
    si = Material(2.07, 0.0, 'Si')
    poly = Material(4.0, 0.2, 'poly')
    d2o = Material(6.36, 0.0, 'D2O')
    sample = Sample(
        Multilayer([Layer(air, 0, 0, name='air')]),
        Multilayer([
            Layer(MaterialSolvated(si, d2o, 0.3), 40, 3, name='si'),
            Layer(MaterialSolvated(poly, d2o, 0.1), 60, 5, name='poly'),
        ]),
        Multilayer([Layer(d2o, 0, 8, name='bulk')]),
    )
    materials = {'Air': air, 'Si': si, 'poly': poly, 'D2O': d2o}
    return Model(sample=sample), materials


def _reconstructed_sld(profile, materials):
    labels = {key: label for key, label in profile.labels.items()}
    real = sum(materials[labels[key]].sld.value * phi for key, phi in profile.fractions.items())
    imag = sum(materials[labels[key]].isld.value * phi for key, phi in profile.fractions.items())
    return real, imag


class TestAgainstCalculators:
    def setup_method(self):
        global_object.map._clear()

    def test_reconstructs_refnx_sld_profile_on_the_same_grid(self):
        model, materials = _solvated_model()
        interface = CalculatorFactory()
        interface.switch('refnx')
        model.interface = interface
        z_ref, sld_ref = interface().sld_profile(model.unique_name)

        profile = volume_fraction_profile(model.sample)
        assert np.array_equal(profile.z, z_ref)
        real, imag = _reconstructed_sld(profile, materials)
        assert_allclose(real, sld_ref, atol=1e-9)

        # the imaginary part against an independent smear of the per-slab iSLDs
        slabs, labels = _resolve_slabs(model.sample)
        name_of = {key: label for key, label in labels.items()}
        slab_isld = [sum(materials[name_of[key]].isld.value * f for key, f in slab.composition.items()) for slab in slabs]
        thickness = np.array([slab.thickness for slab in slabs])
        expected_imag = _smear(
            profile.z, np.cumsum(thickness[:-1]), np.array([s.roughness for s in slabs[1:]]), np.array(slab_isld)
        )
        assert_allclose(imag, expected_imag, atol=1e-12)
        assert expected_imag.max() == pytest.approx(0.9 * 0.2, abs=1e-3)

    def test_is_independent_of_the_active_calculator(self):
        model, _ = _solvated_model()
        profiles = {}
        for name in ('refnx', 'refl1d'):
            interface = CalculatorFactory()
            interface.switch(name)
            model.interface = interface
            z, _ = interface().sld_profile(model.unique_name)
            # closure holds on the backend's own grid too
            assert_allclose(volume_fraction_profile(model.sample, z=np.asarray(z)).total, 1.0, atol=1e-12)
            profiles[name] = volume_fraction_profile(model.sample)
        assert np.array_equal(profiles['refnx'].z, profiles['refl1d'].z)
        for key in profiles['refnx'].fractions:
            assert_allclose(profiles['refnx'].fractions[key], profiles['refl1d'].fractions[key])
