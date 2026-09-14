# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from easyreflectometry.data.data_store import DataSet1D


class TestDataSet1D:
    def test_constructor_default_values(self):
        # When - Create with minimal arguments
        data = DataSet1D()

        # Then - Check defaults
        assert data.name == 'Series'
        assert_array_equal(data.x, np.array([]))
        assert_array_equal(data.y, np.array([]))
        assert_array_equal(data.ye, np.array([]))
        assert_array_equal(data.xe, np.array([]))
        assert data.x_label == 'x'
        assert data.y_label == 'y'
        assert data.model is None
        assert data._color is None

    def test_constructor_with_values(self):
        # When
        data = DataSet1D(
            x=[1, 2, 3],
            y=[4, 5, 6],
            ye=[7, 8, 9],
            xe=[10, 11, 12],
            x_label='label_x',
            y_label='label_y',
            name='MyDataSet1D',
        )

        # Then
        assert data.name == 'MyDataSet1D'
        assert_almost_equal(data.x, [1, 2, 3])
        assert data.x_label == 'label_x'
        assert_almost_equal(data.xe, [10, 11, 12])
        assert_almost_equal(data.y, [4, 5, 6])
        assert data.y_label == 'label_y'
        assert_almost_equal(data.ye, [7, 8, 9])

    def test_constructor_converts_lists_to_arrays(self):
        # When
        data = DataSet1D(x=[1, 2, 3], y=[4, 5, 6])

        # Then
        assert isinstance(data.x, np.ndarray)
        assert isinstance(data.y, np.ndarray)
        assert isinstance(data.ye, np.ndarray)
        assert isinstance(data.xe, np.ndarray)

    def test_constructor_mismatched_lengths_raises_error(self):
        # When/Then
        with pytest.raises(ValueError, match='x and y must be the same length'):
            DataSet1D(x=[1, 2, 3], y=[4, 5])

    def test_constructor_with_model_sets_background(self):
        # Given
        mock_model = Mock()
        x_data = [1, 2, 3, 4]
        y_data = [1, 2, 0.5, 3]

        # When
        _ = DataSet1D(x=x_data, y=y_data, model=mock_model)

        # Then
        assert mock_model.background == np.min(y_data)

    def test_model_property(self):
        # Given
        mock_model = Mock()
        data = DataSet1D(x=[1, 2, 3], y=[4, 5, 6])

        # When
        data.model = mock_model

        # Then
        assert data.model == mock_model

    def test_model_setter_does_not_update_background(self):
        # Given
        mock_model = Mock()
        mock_model.background = 1e-8  # Original background value
        data = DataSet1D(x=[1, 2, 3, 4], y=[1, 2, 0.5, 3])

        # When
        data.model = mock_model

        # Then - background should NOT be overwritten by model setter
        assert mock_model.background == 1e-8

    def test_is_experiment_property(self):
        # Given
        data_with_model = DataSet1D(model=Mock())
        data_without_model = DataSet1D()

        # When/Then
        assert data_with_model.is_experiment is True
        assert data_without_model.is_experiment is False

    def test_is_simulation_property(self):
        # Given
        data_with_model = DataSet1D(model=Mock())
        data_without_model = DataSet1D()

        # When/Then
        assert data_with_model.is_simulation is False
        assert data_without_model.is_simulation is True

    def test_data_points(self):
        # When
        data = DataSet1D(x=[1, 2, 3], y=[4, 5, 6], ye=[7, 8, 9], xe=[10, 11, 12])

        # Then
        points = list(data.data_points())
        assert points == [(1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12)]

    def test_repr(self):
        # When
        data = DataSet1D(x=[1, 2, 3], y=[4, 5, 6], x_label='Q', y_label='R')

        # Then
        expected = "1D DataStore of 'Q' Vs 'R' with 3 data points"
        assert str(data) == expected

    def test_repr_empty_data(self):
        # When
        data = DataSet1D()

        # Then
        expected = "1D DataStore of 'x' Vs 'y' with 0 data points"
        assert str(data) == expected

    def test_default_error_arrays_when_none(self):
        # When
        data = DataSet1D(x=[1, 2, 3], y=[4, 5, 6])

        # Then
        assert_array_equal(data.ye, np.zeros(3))
        assert_array_equal(data.xe, np.zeros(3))
