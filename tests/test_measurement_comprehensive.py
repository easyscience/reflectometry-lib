# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive tests for measurement and DataSet1D functionality.
Tests for all functions in measurement.py and data_store.py modules.
"""

__author__ = 'tests'

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import easyreflectometry
from easyreflectometry.data.data_store import DataSet1D
from easyreflectometry.data.measurement import _load_txt
from easyreflectometry.data.measurement import dataset_from_datagroup
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset
from easyreflectometry.data.measurement import merge_datagroups
from easyreflectometry.orso_utils import load_data_from_orso_file

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestMeasurementFunctions:
    """Test all measurement loading functions."""

    def test_load_function_with_orso_file(self):
        """Test that load() correctly identifies and loads ORSO files."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        result = load(fpath)

        assert 'data' in result
        assert 'coords' in result
        assert len(result['data']) > 0
        assert len(result['coords']) > 0

    def test_load_function_with_txt_file(self):
        """Test that load() falls back to txt loading for non-ORSO files."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        result = load(fpath)

        assert 'data' in result
        assert 'coords' in result
        assert 'R_test_example1' in result['data']
        assert 'Qz_test_example1' in result['coords']

    def test_load_as_dataset_returns_dataset1d(self):
        """Test that load_as_dataset returns a proper DataSet1D object."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert hasattr(dataset, 'x')
        assert hasattr(dataset, 'y')
        assert hasattr(dataset, 'xe')
        assert hasattr(dataset, 'ye')
        assert len(dataset.x) == len(dataset.y)

    def test_load_as_dataset_extracts_correct_basename(self):
        """Test that load_as_dataset correctly extracts file basename."""
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        dataset = load_as_dataset(fpath)

        # Should work without error and have data
        assert len(dataset.x) > 0
        assert len(dataset.y) > 0

    def test_merge_datagroups_preserves_all_data(self):
        """Test that merge_datagroups combines multiple data groups correctly."""
        fpath1 = os.path.join(PATH_STATIC, 'test_example1.txt')
        fpath2 = os.path.join(PATH_STATIC, 'ref_concat_1.txt')

        group1 = load(fpath1)
        group2 = load(fpath2)

        merged = merge_datagroups(group1, group2)

        # Should have data from both groups
        assert len(merged['data']) >= len(group1['data'])
        assert len(merged['coords']) >= len(group1['coords'])

    def test_merge_datagroups_single_group(self):
        """Test that merge_datagroups works with a single group."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        group = load(fpath)

        merged = merge_datagroups(group)

        # Should be equivalent to original
        assert len(merged['data']) == len(group['data'])
        assert len(merged['coords']) == len(group['coords'])

    def test_load_txt_handles_comma_delimiter(self):
        """Test that _load_txt correctly handles comma-delimited files."""
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        result = _load_txt(fpath)

        assert 'data' in result
        assert 'coords' in result
        # Should successfully parse comma-delimited data
        data_key = list(result['data'].keys())[0]
        assert len(result['data'][data_key].values) > 0

    def test_load_txt_handles_three_columns(self):
        """Test that _load_txt handles files with only 3 columns (no xe)."""
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        result = _load_txt(fpath)

        coords_key = list(result['coords'].keys())[0]
        # xe should be zeros
        assert_array_equal(
            result['coords'][coords_key].variances,
            np.zeros_like(result['coords'][coords_key].values),
        )

    def test_load_txt_with_insufficient_columns(self):
        """Test that _load_txt raises error for files with too few columns."""
        # Create temporary file with only 2 columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('1.0 2.0\n3.0 4.0\n')
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match='File must contain at least 3 columns'):
                _load_txt(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_orso_with_multiple_datasets(self):
        """Test that _load_orso handles files with multiple datasets."""
        fpath = os.path.join(PATH_STATIC, 'test_example2.ort')
        result = load_data_from_orso_file(fpath)

        # Should have multiple data entries
        assert len(result['data']) > 1
        assert 'attrs' in result

    def test_load_orso_preserves_metadata(self):
        """Test that _load_orso preserves ORSO metadata in attrs."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        result = load_data_from_orso_file(fpath)

        assert 'attrs' in result
        # Should have orso_header in attrs
        for data_key in result['data']:
            assert data_key in result['attrs']
            assert 'orso_header' in result['attrs'][data_key]


class TestDataSet1DComprehensive:
    """Comprehensive tests for DataSet1D class."""

    def test_constructor_all_parameters(self):
        """Test DataSet1D constructor with all parameters."""
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        xe = [0.1, 0.1, 0.1, 0.1]
        ye = [1, 2, 3, 4]

        dataset = DataSet1D(
            name='TestData',
            x=x,
            y=y,
            xe=xe,
            ye=ye,
            x_label='Q (Å⁻¹)',
            y_label='Reflectivity',
            model=None,
        )

        assert dataset.name == 'TestData'
        assert_array_equal(dataset.x, np.array(x))
        assert_array_equal(dataset.y, np.array(y))
        assert_array_equal(dataset.xe, np.array(xe))
        assert_array_equal(dataset.ye, np.array(ye))
        assert dataset.x_label == 'Q (Å⁻¹)'
        assert dataset.y_label == 'Reflectivity'

    def test_is_experiment_vs_simulation_properties(self):
        """Test is_experiment and is_simulation properties."""
        # Dataset without model is simulation
        sim_data = DataSet1D(x=[1, 2], y=[3, 4])
        assert sim_data.is_simulation is True
        assert sim_data.is_experiment is False

        # Dataset with model is experiment
        exp_data = DataSet1D(x=[1, 2], y=[3, 4], model=Mock())
        assert exp_data.is_experiment is True
        assert exp_data.is_simulation is False

    def test_data_points_iterator(self):
        """Test the data_points method returns correct tuples."""
        dataset = DataSet1D(x=[1, 2, 3], y=[10, 20, 30], xe=[0.1, 0.2, 0.3], ye=[1, 2, 3])

        points = list(dataset.data_points())
        expected = [(1, 10, 1, 0.1), (2, 20, 2, 0.2), (3, 30, 3, 0.3)]
        assert points == expected

    def test_model_property_setter_does_not_update_background(self):
        """Test that setting model via setter does not overwrite background."""
        dataset = DataSet1D(x=[1, 2, 3, 4], y=[5, 1, 8, 3])
        mock_model = Mock()
        mock_model.background = 1e-8  # Original value

        dataset.model = mock_model

        assert mock_model.background == 1e-8  # background should NOT be changed by setter

    def test_repr_string_representation(self):
        """Test the string representation of DataSet1D."""
        dataset = DataSet1D(x=[1, 2, 3], y=[4, 5, 6], x_label='Momentum Transfer', y_label='Intensity')

        expected = "1D DataStore of 'Momentum Transfer' Vs 'Intensity' with 3 data points"
        assert str(dataset) == expected


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_complete_workflow_orso_file(self):
        """Test complete workflow: load ORSO file -> create dataset."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert len(dataset.x) > 0
        assert len(dataset.x) == len(dataset.y) == len(dataset.ye)

    def test_complete_workflow_txt_file(self):
        """Test complete workflow: load txt file -> create dataset (no model)."""
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert dataset.model is None
        assert dataset.is_simulation is True
        assert dataset.is_experiment is False

    def test_merge_multiple_files_workflow(self):
        """Test workflow for merging multiple data files."""
        # Load multiple files
        fpath1 = os.path.join(PATH_STATIC, 'test_example1.txt')
        fpath2 = os.path.join(PATH_STATIC, 'ref_concat_1.txt')

        group1 = load(fpath1)
        group2 = load(fpath2)

        # Merge data groups
        merged = merge_datagroups(group1, group2)

        # Create datasets from merged data
        # This tests that merged data can be used to create datasets
        assert len(merged['data']) >= 2
        assert len(merged['coords']) >= 2

    def test_error_handling_robustness(self):
        """Test error handling in various edge cases."""
        # Test mismatched array lengths
        with pytest.raises(ValueError, match='x and y must be the same length'):
            DataSet1D(x=[1, 2, 3], y=[4, 5])

        # Test file not found
        with pytest.raises(FileNotFoundError):
            _load_txt('nonexistent_file.txt')

    def test_data_consistency_checks(self):
        """Test that data remains consistent across operations."""
        # Create dataset
        original_x = [1, 2, 3, 4]
        original_y = [10, 20, 30, 40]
        dataset = DataSet1D(x=original_x, y=original_y)

        # Verify the input lists are copied into arrays and remain consistent
        assert_array_equal(dataset.x, np.array(original_x))
        assert_array_equal(dataset.y, np.array(original_y))
        assert_array_equal(dataset.ye, np.zeros(len(original_x)))
        assert_array_equal(dataset.xe, np.zeros(len(original_x)))


if __name__ == '__main__':
    # Run all tests if script is executed directly
    pytest.main([__file__, '-v'])


class TestDatasetFromDatagroup:
    """The DataGroup -> DataSet1D step shared by the loaders that parse once."""

    def test_defaults_to_the_first_dataset(self):
        """Without a data_key the first entry is taken (the documented default)."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        data_group = load(fpath)

        dataset = dataset_from_datagroup(data_group)

        first_data_key = list(data_group['data'].keys())[0]
        first_coords_key = list(data_group['coords'].keys())[0]
        assert_array_equal(dataset.y, data_group['data'][first_data_key].values)
        assert_array_equal(dataset.x, data_group['coords'][first_coords_key].values)

    def test_explicit_data_key_selects_that_dataset(self):
        """A multi-dataset ORSO file is addressed one dataset at a time."""
        fpath = os.path.join(PATH_STATIC, 'polarized_2ch.ort')
        data_group = load(fpath)
        keys = list(data_group['data'].keys())
        assert len(keys) > 1

        second = dataset_from_datagroup(data_group, data_key=keys[1])

        assert_array_equal(second.y, data_group['data'][keys[1]].values)

    def test_orso_header_is_attached_for_orso_files(self):
        """The parsed header rides along so the exporter can reuse provenance."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        dataset = dataset_from_datagroup(load(fpath))

        assert dataset.orso_header is not None
        assert 'data_source' in dataset.orso_header

    def test_orso_header_is_none_for_plain_text(self):
        """A text file carries no header; the attribute is still set, to None."""
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        dataset = dataset_from_datagroup(load(fpath))

        assert dataset.orso_header is None
