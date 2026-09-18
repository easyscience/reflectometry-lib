# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from easyscience import global_object

import easyreflectometry
from easyreflectometry import Project
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.summary import Summary

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestSummary:
    @pytest.fixture
    def project(self) -> Project:
        global_object.map._clear()
        project = Project()
        project.default_model()
        return project

    def test_constructor(self, project: Project) -> None:
        # When Then
        result = Summary(project)

        # Expect
        assert result._project == project

    def test_compile_html_summary(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary._project_information_section = MagicMock(return_value='project result html')
        summary._sample_section = MagicMock(return_value='sample result html')
        summary._experiments_section = MagicMock(return_value='experiments results html')
        summary._refinement_section = MagicMock(return_value='refinement result html')
        summary._figures_section = MagicMock()

        # Then
        result = summary.compile_html_summary()

        # Expect
        summary._figures_section.assert_not_called()
        assert 'project result html' in result
        assert 'sample result html' in result
        assert 'experiments results html' in result
        assert 'refinement result html' in result
        assert 'figures_section' not in result

    def test_compile_html_summary_with_figures(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary._project_information_section = MagicMock(return_value='project result html')
        summary._sample_section = MagicMock(return_value='sample result html')
        summary._experiments_section = MagicMock(return_value='experiments results html')
        summary._refinement_section = MagicMock(return_value='refinement result html')
        summary._figures_section = MagicMock(return_value='figures result html')

        # Then
        result = summary.compile_html_summary(figures=True)

        # Expect
        assert 'figures result html' in result

    def test_save_html_summary(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        summary.compile_html_summary = MagicMock(return_value='html')
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.html')

        # Then
        summary.save_html_summary(file_path)

        # Expect
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            assert f.read() == 'html'

    def test_save_pdf_summary(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        summary.compile_html_summary = MagicMock(return_value='html')
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.pdf')

        # Then
        summary.save_pdf_summary(file_path)

        # Expect
        assert os.path.exists(file_path)

    def test_project_information_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._project_information_section()

        # Expect
        assert 'DefaultEasyReflectometryProject' in html
        assert 'Reflectometry, 1D' in html

    def test_sample_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._sample_section()

        # Expect
        assert 'Name' in html
        assert 'Value' in html
        assert 'Unit' in html
        assert 'Error' in html

        assert 'sld' in html
        assert 'isld' in html
        assert 'thickness' in html
        assert 'background' in html

    def test_experiments_section(self, project: Project) -> None:
        # When
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        summary = Summary(project)

        # Then
        html = summary._experiments_section()

        # Expect
        assert 'Example data file from refnx docs' in html
        assert 'No. of data points' in html
        assert '408' in html
        assert 'Resolution function' in html
        assert 'Pointwise' in html

    def test_experiments_section_percentage_fhwm(self, project: Project) -> None:
        # When
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        project.models[0].resolution_function = PercentageFwhm(5)
        summary = Summary(project)

        # Then
        html = summary._experiments_section()

        # Expect
        assert 'PercentageFwhm 5%' in html

    def test_experiments_section_polarized(self, project: Project, tmp_path) -> None:
        # When
        # A polarized experiment holds one DataSet1D per spin channel, not the
        # x/y arrays an ordinary experiment has — the section used to raise
        # AttributeError, which killed the app when QML read the summary.
        channel_paths = {}
        for channel, suffix in (('pp', 'uu'), ('mm', 'dd')):
            path = tmp_path / f'sample_{suffix}.txt'
            q = np.linspace(0.01, 0.2, 20)
            reflectivity = np.exp(-q * 30)
            np.savetxt(path, np.column_stack([q, reflectivity, 0.01 * reflectivity]))
            channel_paths[channel] = str(path)
        project.calculator = 'refl1d'
        project.load_polarized_experiment(channel_paths)
        summary = Summary(project)

        # Then
        html = summary._experiments_section()

        # Expect: one row per measured channel
        assert 'Polarized experiment 0 (pp)' in html
        assert 'Polarized experiment 0 (mm)' in html
        assert html.count('No. of data points') == 2
        assert '20' in html

    def test_compile_html_summary_polarized(self, project: Project, tmp_path) -> None:
        # When
        channel_paths = {}
        for channel, suffix in (('pp', 'uu'), ('mm', 'dd')):
            path = tmp_path / f'sample_{suffix}.txt'
            q = np.linspace(0.01, 0.2, 20)
            reflectivity = np.exp(-q * 30)
            np.savetxt(path, np.column_stack([q, reflectivity, 0.01 * reflectivity]))
            channel_paths[channel] = str(path)
        project.calculator = 'refl1d'
        project.load_polarized_experiment(channel_paths)
        summary = Summary(project)

        # Then Expect: the whole report compiles for a polarized experiment
        assert 'Polarized experiment 0 (pp)' in summary.compile_html_summary()

    def test_refinement_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._refinement_section()

        # Expect
        assert 'refnx' in html
        assert 'LMFit_leastsq' in html
        assert 'No. of parameters:' in html
        assert 'No. of fixed parameters:' in html
        assert '14' in html
        assert 'No. of free parameters:' in html
        assert '0' in html
        assert 'No. of constraints' in html

    def test_save_sld_plot(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.jpg')

        # Then
        summary.save_sld_plot(file_path)

        # Expect
        assert os.path.exists(file_path)

    @pytest.mark.skip(reason='Matplotlib issue with headless CI environments')
    def test_save_fit_experiment_plot(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.jpg')
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)

        # Then
        summary.save_fit_experiment_plot(file_path)

        # Expect
        assert os.path.exists(file_path)

    def test_figures_section_static(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary.save_sld_plot = MagicMock()
        summary.save_fit_experiment_plot = MagicMock()

        # Then
        html = summary._figures_section(interactive=False)

        # Expect
        summary.save_sld_plot.assert_called_once()
        summary.save_fit_experiment_plot.assert_called_once()
        assert 'sld_plot' in html
        assert 'fit_experiment_plot' in html

    def test_figures_section_interactive(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary.save_sld_plot = MagicMock()
        summary.save_fit_experiment_plot = MagicMock()

        # Then
        html = summary._figures_section(interactive=True)

        # Expect
        # Interactive figures must not fall back to the static image plots.
        summary.save_sld_plot.assert_not_called()
        summary.save_fit_experiment_plot.assert_not_called()
        # Two interactive plotly charts with the library embedded inline once.
        assert html.count('class="plotly-graph-div"') == 2
        assert 'Plotly.newPlot' in html


class TestSummaryPolarized:
    """Report figures of a polarized (per-channel) experiment."""

    @staticmethod
    def _write_channel_file(directory, name: str) -> str:
        import numpy as np

        path = directory / name
        q = np.linspace(0.01, 0.2, 20)
        reflectivity = np.exp(-q * 30)
        np.savetxt(path, np.column_stack([q, reflectivity, 0.01 * reflectivity]))
        return str(path)

    @pytest.fixture
    def polarized_project(self, tmp_path) -> Project:
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_polarized_experiment({
            'pp': self._write_channel_file(tmp_path, 'sample_uu.txt'),
            'mm': self._write_channel_file(tmp_path, 'sample_dd.txt'),
        })
        return project

    def test_measured_series_one_entry_per_channel(self, polarized_project: Project) -> None:
        # When
        summary = Summary(polarized_project)

        # Then
        series = summary._measured_series()

        # Expect
        assert [entry[0] for entry in series] == ['Experiment (pp)', 'Experiment (mm)']
        assert series[0][2] != series[1][2]  # distinct channel colors
        assert [entry[3].value for entry in series] == ['pp', 'mm']

    @pytest.fixture
    def project(self) -> Project:
        global_object.map._clear()
        project = Project()
        project.default_model()
        return project

    def test_measured_series_unpolarized_and_empty(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then Expect: nothing loaded
        assert summary._measured_series() == []

        # When an ordinary experiment is loaded
        project.load_experiment_for_model_at_index(os.path.join(PATH_STATIC, 'example.ort'))

        # Expect one channel-less entry
        series = summary._measured_series()
        assert len(series) == 1
        assert series[0][0] == 'Experiment' and series[0][3] is None

    def test_model_curve_skips_uncalculable_channel(self, polarized_project: Project) -> None:
        # When: a non-magnetic model cannot produce a spin-flip cross-section
        summary = Summary(polarized_project)

        # Then Expect: no misleading overlay is generated for it
        assert summary._model_curve('pm') is None
        assert summary._model_curve('pp') is not None

    def test_fit_experiment_figure_has_a_trace_per_channel(self, polarized_project: Project) -> None:
        # When
        summary = Summary(polarized_project)

        # Then
        figure = summary._fit_experiment_plotly_figure()

        # Expect: one measured trace per channel plus the calculable model curves
        names = [trace.name for trace in figure.data]
        assert 'Experiment (pp)' in names
        assert 'Experiment (mm)' in names
        assert 'Model (pp)' in names
