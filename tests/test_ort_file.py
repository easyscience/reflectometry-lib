# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os

import numpy as np
import pytest

import easyreflectometry
from easyreflectometry.data import load

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.fixture(scope='module')
def load_data():
    path = os.path.join(PATH_STATIC, 'amor_reduced_iofq.ort')
    logging.info('Loading data from %s', path)
    data = load(path)
    return data


def test_read_reduced_data__check_structure(load_data):
    data_keys = load_data['data'].keys()
    coord_keys = load_data['coords'].keys()
    for key in data_keys:
        if key in coord_keys:
            assert len(load_data['data'][key].values) == len(load_data['coords'][key].values)


def test_validate_physical_data__r_values_non_negative(load_data):
    for key in load_data['data'].keys():
        assert all(load_data['data'][key].values >= 0)


def test_validate_physical_data__r_values_finite(load_data):
    for key in load_data['data'].keys():
        assert all(np.isfinite(load_data['data'][key].values))


@pytest.mark.skip('Currently no warning implemented')
def test_validate_physical_data__r_values_ureal_positive(load_data):
    a = load_data['data']['R_0'].values
    b = 1 + 2 * np.sqrt(load_data['data']['R_0'].variances)
    for val_a, val_b in zip(a, b):
        if val_a > val_b:
            pytest.warns(
                UserWarning,
                reason=f'Reflectivity value {val_a} is unphysically large compared to its uncertainty {val_b}',
            )
    assert all(load_data['data']['R_0'].values <= 1 + 2 * np.sqrt(load_data['data']['R_0'].variances))


def test_validate_physical_data__q_values_non_negative(load_data):
    for key in load_data['coords'].keys():
        assert all(load_data['coords'][key].values >= 0)


def test_validate_physical_data__q_values_ureal_positive(load_data):
    for key in load_data['coords'].keys():
        # Reflectometry data is usually with the range of 0-5,
        # so 10 is a safe upper limit
        assert all(load_data['coords'][key].values < 10)


def test_validate_physical_data__q_values_finite(load_data):
    for key in load_data['coords'].keys():
        assert all(np.isfinite(load_data['coords'][key].values < 10))


@pytest.mark.skip('Currently no meta data to check')
def test_validate_meta_data__required_meta_data() -> None:
    pytest.fail(reason='Currently no meta data to check')
