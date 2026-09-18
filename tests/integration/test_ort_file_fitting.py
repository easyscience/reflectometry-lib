# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end fit of a real ORT dataset.

Not part of ``tests/test_ort_file.py`` because the ``fit_model`` fixture
runs a genuine Bumps_simplex fit (max_evaluations=3000), which takes ~40s
on its own -- by far the single most expensive item in the unit test run.
Run explicitly with ``pixi run integration-tests`` (or
``pytest tests/integration``), not part of the default ``pytest tests``/
``unit-tests`` run.
"""

import logging
import os

import numpy as np
import pytest
from easyscience.fitting import AvailableMinimizers

import easyreflectometry
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.fixture(scope='module')
def load_data():
    path = os.path.join(PATH_STATIC, 'amor_reduced_iofq.ort')
    logging.info('Loading data from %s', path)
    data = load(path)
    return data


@pytest.fixture(scope='module')
def fit_model(load_data):
    data = load_data
    # Rescale data
    reflectivity = data['data']['R_0'].values
    scale_factor = 1 / np.max(reflectivity)
    data['data']['R_0'].values *= scale_factor
    data['data']['R_0'].variances *= scale_factor**2

    # Create a model for the sample

    si = Material(sld=2.07, isld=0.0, name='Si')
    sio2 = Material(sld=3.47, isld=0.0, name='SiO2')
    d2o = Material(sld=6.33, isld=0.0, name='D2O')
    dlipids = Material(sld=5.0, isld=0.0, name='DLipids')

    superphase = Layer(material=si, thickness=0, roughness=0, name='Si superphase')
    sio2_layer = Layer(material=sio2, thickness=20, roughness=4, name='SiO2 layer')
    dlipids_layer = Layer(material=dlipids, thickness=40, roughness=4, name='DLipids layer')
    subphase = Layer(material=d2o, thickness=0, roughness=5, name='D2O subphase')

    multi_sample = Sample(
        Multilayer(superphase),
        Multilayer(sio2_layer),
        Multilayer(dlipids_layer),
        Multilayer(subphase),
        name='Multilayer Structure',
    )

    multi_layer_model = Model(
        sample=multi_sample,
        scale=1,
        background=0.000001,
        resolution_function=PercentageFwhm(5),
        name='Multilayer Model',
    )

    # Set the fitting parameters

    sio2_layer.roughness.min = 3
    sio2_layer.roughness.max = 12
    sio2_layer.material.sld.min = 3.47
    sio2_layer.material.sld.max = 5
    sio2_layer.thickness.min = 10
    sio2_layer.thickness.max = 30

    subphase.material.sld.min = 6
    dlipids_layer.thickness.min = 30
    dlipids_layer.thickness.max = 60
    dlipids_layer.roughness.min = 3
    dlipids_layer.roughness.max = 10
    dlipids_layer.material.sld.min = 4
    dlipids_layer.material.sld.max = 6
    multi_layer_model.scale.min = 0.8
    multi_layer_model.scale.max = 1.2
    multi_layer_model.background.min = 1e-6
    multi_layer_model.background.max = 1e-3

    sio2_layer.roughness.free = True
    sio2_layer.material.sld.free = True
    sio2_layer.thickness.free = True
    subphase.material.sld.free = True
    dlipids_layer.thickness.free = True
    dlipids_layer.roughness.free = True
    dlipids_layer.material.sld.free = True
    multi_layer_model.scale.free = True
    multi_layer_model.background.free = True

    # Run the model and plot the results

    multi_layer_model.interface = CalculatorFactory()

    fitter1 = MultiFitter(multi_layer_model)
    fitter1.switch_minimizer(AvailableMinimizers.Bumps_simplex)
    fitter1.easy_science_multi_fitter.max_evaluations = 3000

    analysed = fitter1.fit(data)
    return analysed


@pytest.mark.slow
def test_analyze_reduced_data__fit_model_success(fit_model):
    assert fit_model['success'] is True


def test_analyze_reduced_data__fit_model_reasonable(fit_model):
    assert fit_model['reduced_chi'] < 6.0
