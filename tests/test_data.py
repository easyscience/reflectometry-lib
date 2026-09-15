# SPDX-FileCopyrightText: 2022 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import os
import tempfile
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from orsopy.fileio import Header
from orsopy.fileio import load_orso

import easyreflectometry
from easyreflectometry.data import DataSet1D
from easyreflectometry.data.measurement import _load_txt
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset
from easyreflectometry.data.measurement import merge_datagroups
from easyreflectometry.orso_utils import load_data_from_orso_file

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestData(unittest.TestCase):
    def test_load_with_orso(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = load(fpath)
        o_data = load_orso(fpath)
        assert er_data['attrs']['R_spin_up']['orso_header'].value == Header.asdict(o_data[0].info)
        assert_almost_equal(er_data['data']['R_spin_up'].values, o_data[0].data[:, 1])
        assert_almost_equal(er_data['coords']['Qz_spin_up'].values, o_data[0].data[:, 0])
        assert_almost_equal(er_data['data']['R_spin_up'].variances, np.square(o_data[0].data[:, 2]))
        assert_almost_equal(er_data['coords']['Qz_spin_up'].variances, np.square(o_data[0].data[:, 3]))

    def test_load_with_txt(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        er_data = load(fpath)
        n_data = np.loadtxt(fpath)
        data_name = 'R_test_example1'
        coords_name = 'Qz_test_example1'
        assert_almost_equal(er_data['data'][data_name].values, n_data[:, 1])
        assert_almost_equal(er_data['coords'][coords_name].values, n_data[:, 0])
        # Relative tolerance: the resolution variances are ~1e-9, well inside the default
        # absolute tolerance of assert_almost_equal, which would accept zeros here.
        assert_allclose(er_data['data'][data_name].variances, np.square(n_data[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(er_data['coords'][coords_name].variances, np.square(n_data[:, 3]), rtol=1e-12, atol=0)

    def test_load_with_txt_extra_columns(self):
        """Columns beyond the leading Qz, R, sR, sQz are ignored, not an error."""
        fpath = os.path.join(PATH_STATIC, 'ref_five_col.txt')
        er_data = load(fpath)
        n_data = np.loadtxt(fpath)
        data_name = 'R_ref_five_col'
        coords_name = 'Qz_ref_five_col'
        assert_almost_equal(er_data['data'][data_name].values, n_data[:, 1])
        assert_almost_equal(er_data['coords'][coords_name].values, n_data[:, 0])
        # Relative tolerance: the resolution variances are ~1e-9, well inside the default
        # absolute tolerance of assert_almost_equal, which would accept zeros here.
        assert_allclose(er_data['data'][data_name].variances, np.square(n_data[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(er_data['coords'][coords_name].variances, np.square(n_data[:, 3]), rtol=1e-12, atol=0)

    def test_load_with_txt_extra_columns_single_row(self):
        """A single-row file stays two-dimensional, so the leading columns are still indexable."""
        row = np.array([[1.03563296e-02, 3.88100068e00, 4.33909068e00, 5.17816478e-05, 5.0]])
        with tempfile.TemporaryDirectory() as directory:
            fpath = os.path.join(directory, 'single_row.txt')
            np.savetxt(fpath, row)
            er_data = load(fpath)

        assert_allclose(er_data['data']['R_single_row'].values, row[:, 1], rtol=1e-12, atol=0)
        assert_allclose(er_data['coords']['Qz_single_row'].values, row[:, 0], rtol=1e-12, atol=0)
        assert_allclose(er_data['data']['R_single_row'].variances, np.square(row[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(er_data['coords']['Qz_single_row'].variances, np.square(row[:, 3]), rtol=1e-12, atol=0)

    def test_load_with_txt_extra_columns_comma_delimited(self):
        """Extra columns are ignored for a comma-delimited file too, not only whitespace."""
        rows = np.array([
            [1.03563296e-02, 3.88100068e00, 4.33909068e00, 5.17816478e-05, 5.0],
            [1.06717294e-02, 1.16430511e01, 8.89252719e00, 5.33586471e-05, 5.5],
        ])
        with tempfile.TemporaryDirectory() as directory:
            fpath = os.path.join(directory, 'comma_five.txt')
            np.savetxt(fpath, rows, delimiter=',')
            er_data = load(fpath)

        assert_allclose(er_data['data']['R_comma_five'].values, rows[:, 1], rtol=1e-12, atol=0)
        assert_allclose(er_data['coords']['Qz_comma_five'].values, rows[:, 0], rtol=1e-12, atol=0)
        assert_allclose(er_data['data']['R_comma_five'].variances, np.square(rows[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(er_data['coords']['Qz_comma_five'].variances, np.square(rows[:, 3]), rtol=1e-12, atol=0)

    def test_load_with_txt_commas(self):
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        er_data = load(fpath)
        x, y, e = np.loadtxt(fpath, delimiter=',', comments='#', unpack=True)
        data_name = 'R_ref_concat_1'
        coords_name = 'Qz_ref_concat_1'
        assert_almost_equal(er_data['data'][data_name].values, y)
        assert_almost_equal(er_data['coords'][coords_name].values, x)
        assert_almost_equal(er_data['data'][data_name].variances, np.square(e))

    def test_orso1(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = load_data_from_orso_file(fpath)
        o_data = load_orso(fpath)
        assert er_data['attrs']['R_spin_up']['orso_header'].value == Header.asdict(o_data[0].info)
        assert_almost_equal(er_data['data']['R_spin_up'].values, o_data[0].data[:, 1])
        assert_almost_equal(er_data['coords']['Qz_spin_up'].values, o_data[0].data[:, 0])
        assert_almost_equal(er_data['data']['R_spin_up'].variances, np.square(o_data[0].data[:, 2]))
        assert_almost_equal(er_data['coords']['Qz_spin_up'].variances, np.square(o_data[0].data[:, 3]))

    def test_orso2(self):
        fpath = os.path.join(PATH_STATIC, 'test_example2.ort')
        er_data = load_data_from_orso_file(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(list(reversed(o_data))):
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_orso3(self):
        fpath = os.path.join(PATH_STATIC, 'test_example3.ort')
        er_data = load_data_from_orso_file(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(o_data):
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_orso4(self):
        fpath = os.path.join(PATH_STATIC, 'test_example4.ort')
        er_data = load_data_from_orso_file(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(o_data):
            print(list(er_data.keys()))
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_txt(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        er_data = _load_txt(fpath)
        n_data = np.loadtxt(fpath)
        data_name = 'R_test_example1'
        coords_name = 'Qz_test_example1'
        assert_almost_equal(er_data['data'][data_name].values, n_data[:, 1])
        assert_almost_equal(er_data['coords'][coords_name].values, n_data[:, 0])
        # Relative tolerance: the resolution variances are ~1e-9, well inside the default
        # absolute tolerance of assert_almost_equal, which would accept zeros here.
        assert_allclose(er_data['data'][data_name].variances, np.square(n_data[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(er_data['coords'][coords_name].variances, np.square(n_data[:, 3]), rtol=1e-12, atol=0)

    def test_load_as_dataset_orso(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert dataset.name == 'Series'  # Default name
        assert len(dataset.x) > 0
        assert len(dataset.y) > 0
        assert len(dataset.xe) > 0
        assert len(dataset.ye) > 0

        # Compare with direct load
        data_group = load(fpath)
        coords_key = list(data_group['coords'].keys())[0]
        data_key = list(data_group['data'].keys())[0]

        assert_almost_equal(dataset.x, data_group['coords'][coords_key].values)
        assert_almost_equal(dataset.y, data_group['data'][data_key].values)
        assert_almost_equal(dataset.xe, data_group['coords'][coords_key].variances)
        assert_almost_equal(dataset.ye, data_group['data'][data_key].variances)

    def test_load_as_dataset_txt(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert len(dataset.x) > 0
        assert len(dataset.y) > 0

        # Compare with numpy loadtxt
        n_data = np.loadtxt(fpath)
        assert_almost_equal(dataset.x, n_data[:, 0])
        assert_almost_equal(dataset.y, n_data[:, 1])
        assert_almost_equal(dataset.ye, np.square(n_data[:, 2]))
        assert_almost_equal(dataset.xe, np.square(n_data[:, 3]))

    def test_load_as_dataset_txt_comma_delimited(self):
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        dataset = load_as_dataset(fpath)

        assert isinstance(dataset, DataSet1D)
        assert len(dataset.x) > 0
        assert len(dataset.y) > 0

        # Should have zero xe since file only has 3 columns
        assert_almost_equal(dataset.xe, np.zeros_like(dataset.x))

    def test_load_as_dataset_uses_correct_names(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        dataset = load_as_dataset(fpath)
        data_group = load(fpath)

        # Should use first available key if expected key not found
        expected_coords_key = list(data_group['coords'].keys())[0]
        expected_data_key = list(data_group['data'].keys())[0]

        assert_almost_equal(dataset.x, data_group['coords'][expected_coords_key].values)
        assert_almost_equal(dataset.y, data_group['data'][expected_data_key].values)

    def test_merge_datagroups_single_group(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        data_group = load(fpath)

        merged = merge_datagroups(data_group)

        # Should be identical to original
        assert list(merged['data'].keys()) == list(data_group['data'].keys())
        assert list(merged['coords'].keys()) == list(data_group['coords'].keys())

        for key in data_group['data']:
            assert_almost_equal(merged['data'][key].values, data_group['data'][key].values)
        for key in data_group['coords']:
            assert_almost_equal(merged['coords'][key].values, data_group['coords'][key].values)

    def test_merge_datagroups_multiple_groups(self):
        fpath1 = os.path.join(PATH_STATIC, 'test_example1.txt')
        fpath2 = os.path.join(PATH_STATIC, 'ref_concat_1.txt')

        group1 = load(fpath1)
        group2 = load(fpath2)

        merged = merge_datagroups(group1, group2)

        # Should contain keys from both groups
        all_data_keys = set(group1['data'].keys()) | set(group2['data'].keys())
        all_coords_keys = set(group1['coords'].keys()) | set(group2['coords'].keys())

        assert set(merged['data'].keys()) == all_data_keys
        assert set(merged['coords'].keys()) == all_coords_keys

    def test_merge_datagroups_shared_key_concatenates(self):
        """Groups sharing a key are concatenated, in argument order, keeping uncertainties.

        The two groups hold different data and differ in length, so a reversed
        concatenation, a dropped group, or lost variances all fail here; merging a
        file with itself would detect none of them.
        """
        first = np.array([
            [1.0e-02, 9.0e-01, 1.0e-02, 5.0e-05],
            [2.0e-02, 8.0e-01, 2.0e-02, 6.0e-05],
            [3.0e-02, 7.0e-01, 3.0e-02, 7.0e-05],
        ])
        second = np.array([[4.0e-02, 6.0e-01, 4.0e-02, 8.0e-05]])
        # Same basename in two directories, so the two groups share their data and
        # coordinate keys -- which is what sends the merge down the concatenating branch.
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, 'a', 'shared.txt')
            second_path = os.path.join(directory, 'b', 'shared.txt')
            for path, rows in ((first_path, first), (second_path, second)):
                os.makedirs(os.path.dirname(path))
                np.savetxt(path, rows)
            group_one = load(first_path)
            group_two = load(second_path)

        merged = merge_datagroups(group_one, group_two)

        data_name = 'R_shared'
        coords_name = 'Qz_shared'
        expected = np.concatenate([first, second])
        merged_data = merged['data'][data_name]
        merged_coords = merged['coords'][coords_name]

        assert merged_data.dims == group_one['data'][data_name].dims
        assert merged_coords.unit == group_one['coords'][coords_name].unit
        assert len(merged_data.values) == len(first) + len(second)
        assert_allclose(merged_data.values, expected[:, 1], rtol=1e-12, atol=0)
        assert_allclose(merged_coords.values, expected[:, 0], rtol=1e-12, atol=0)
        assert_allclose(merged_data.variances, np.square(expected[:, 2]), rtol=1e-12, atol=0)
        assert_allclose(merged_coords.variances, np.square(expected[:, 3]), rtol=1e-12, atol=0)

    def test_merge_datagroups_with_attrs(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        data_group = load(fpath)

        # Create a second group without attrs
        fpath2 = os.path.join(PATH_STATIC, 'test_example1.txt')
        group2 = load(fpath2)

        merged = merge_datagroups(data_group, group2)

        # Should preserve attrs from first group
        if 'attrs' in data_group:
            assert 'attrs' in merged

    def test_load_txt_three_columns(self):
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        er_data = _load_txt(fpath)

        basename = 'ref_concat_1'
        data_name = f'R_{basename}'
        coords_name = f'Qz_{basename}'

        assert data_name in er_data['data']
        assert coords_name in er_data['coords']

        # xe should be zeros for 3-column file
        assert_almost_equal(
            er_data['coords'][coords_name].variances,
            np.zeros_like(er_data['coords'][coords_name].values),
        )

    def test_load_txt_with_zero_errors(self):
        fpath = os.path.join(PATH_STATIC, 'ref_zero_var.txt')
        er_data = _load_txt(fpath)

        basename = 'ref_zero_var'
        data_name = f'R_{basename}'

        # Should handle zero errors without issues
        assert data_name in er_data['data']
        # Some variances should be zero
        assert np.any(er_data['data'][data_name].variances == 0)

    def test_load_txt_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_txt('nonexistent_file.txt')

    def test_load_txt_insufficient_columns(self):
        # Create a temporary file with insufficient columns
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('1.0 2.0\n')  # Only 2 columns
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match='File must contain at least 3 columns'):
                _load_txt(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_orso_multiple_datasets(self):
        fpath = os.path.join(PATH_STATIC, 'test_example2.ort')
        er_data = load_data_from_orso_file(fpath)

        # Should handle multiple datasets
        assert len(er_data['data']) > 1
        assert len(er_data['coords']) > 1

        # All should have corresponding coords
        for data_key in er_data['data']:
            # Find corresponding coord key
            coord_key_found = False
            for coord_key in er_data['coords']:
                if data_key.replace('R_', '') in coord_key:
                    coord_key_found = True
                    break
            assert coord_key_found, f'No corresponding coord found for {data_key}'

    def test_load_orso_with_attrs(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = load_data_from_orso_file(fpath)

        # Should have attrs with ORSO headers
        assert 'attrs' in er_data
        for data_key in er_data['data']:
            assert data_key in er_data['attrs']
            assert 'orso_header' in er_data['attrs'][data_key]

    def test_load_orso_with_units(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = load_data_from_orso_file(fpath)

        # Coords should have units
        for coord_key in er_data['coords']:
            # Check if unit is properly set (scipp units)
            coord_data = er_data['coords'][coord_key]
            assert hasattr(coord_data, 'unit')

    def test_load_fallback_to_txt(self):
        # Test that load() falls back to _load_txt when load_data_from_orso_file fails
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        result = load(fpath)

        # Should successfully load as txt
        assert 'data' in result
        assert 'coords' in result

        basename = 'test_example1'
        data_name = f'R_{basename}'
        assert data_name in result['data']

    def test_load_as_dataset_basename_extraction(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        _ = load_as_dataset(fpath)

        # Verify that basename is correctly extracted and used
        data_group = load(fpath)
        basename = os.path.splitext(os.path.basename(fpath))[0]
        expected_data_name = f'R_{basename}'
        expected_coords_name = f'Qz_{basename}'

        # Should find the correct keys in the data group
        assert expected_data_name in data_group['data'] or list(data_group['data'].keys())[0]
        assert expected_coords_name in data_group['coords'] or list(data_group['coords'].keys())[0]
