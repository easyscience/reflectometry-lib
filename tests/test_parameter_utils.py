# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_equal

from easyreflectometry.utils import get_as_parameter

PARAMETER_DETAILS = {
    'test_parameter': {
        'description': 'Test parameter',
        'url': 'https://veryrealwebsite.com',
        'value': 1.0,
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    'test_parameter_10': {
        'description': 'Test parameter 10',
        'url': 'https://veryrealwebsite_10.com',
        'value': 10.0,
        'min': -10,
        'max': 10,
        'fixed': False,
    },
}


def test_get_as_parameter():
    # When
    test_parameter = None

    # Then
    test_parameter = get_as_parameter('test_parameter', test_parameter, PARAMETER_DETAILS)

    # Expected
    assert test_parameter.display_name == 'test_parameter'
    assert_equal(str(test_parameter.unit), 'dimensionless')
    assert_equal(test_parameter.value, 1.0)
    assert_equal(test_parameter.min, 0.0)
    assert_equal(test_parameter.max, np.inf)
    assert_equal(test_parameter.fixed, True)
    assert_equal(test_parameter.description, 'Test parameter')


def test_get_as_parameter_from_float():
    # When
    test_parameter = 2.0

    # Then
    test_parameter = get_as_parameter('test_parameter', float(test_parameter), PARAMETER_DETAILS)

    # Expected
    assert_equal(test_parameter.value, 2.0)


def test_get_as_parameter_from_int():
    # When
    test_parameter = 2.0

    # Then
    test_parameter = get_as_parameter('test_parameter', int(test_parameter), PARAMETER_DETAILS)

    # Expected
    assert_equal(test_parameter.value, 2.0)


def test_get_as_parameter_from_parameter():
    # When
    test_parameter_10 = get_as_parameter('test_parameter_10', None, PARAMETER_DETAILS)

    # Then
    test_parameter = get_as_parameter('test_parameter', test_parameter_10, PARAMETER_DETAILS)

    # Expected
    assert test_parameter is test_parameter_10
    assert_equal(str(test_parameter.unit), 'dimensionless')
    assert_equal(test_parameter.value, 10.0)
    assert_equal(test_parameter.min, -10.0)
    assert_equal(test_parameter.max, 10.0)
    assert_equal(test_parameter.fixed, False)
    assert_equal(test_parameter.description, 'Test parameter 10')


def test_get_as_parameter_not_number():
    # When
    test_parameter = '2.0'

    # Then Expected
    with pytest.raises(ValueError):
        test_parameter = get_as_parameter('test_parameter', test_parameter, PARAMETER_DETAILS)


def test_dict_remains_unchanged():
    expected_parameter_details = {
        'test_parameter': {
            'description': 'Test parameter',
            'url': 'https://veryrealwebsite.com',
            'value': 1.0,
            'min': 0,
            'max': np.inf,
            'fixed': True,
        }
    }
    # When
    test_parameter = 2.0

    # Then
    test_parameter = get_as_parameter('test_parameter', int(test_parameter), PARAMETER_DETAILS)

    # Expected
    assert_equal(expected_parameter_details['test_parameter'], PARAMETER_DETAILS['test_parameter'])
