# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry import Project
from easyreflectometry.contrasts import ReplaceMaterial
from easyreflectometry.data import DataSet1D
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.sample import Material
from easyreflectometry.summary import Summary
from easyreflectometry.summary.summary import _named_parameters

PATH_STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_static')


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

        # Then - the dataset is fitted with the resolution it was measured with
        assert 'Pointwise' in summary._experiments_section()
        # ...or, without one, with its model's
        project.experiments[0].resolution_function = None
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
        assert 'No. of parameters (all models):' in html
        assert 'No. of fixed parameters (all models):' in html
        assert '16' in html
        assert 'No. of free parameters (all models):' in html
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
            'pm': self._write_channel_file(tmp_path, 'sample_ud.txt'),
            'mm': self._write_channel_file(tmp_path, 'sample_dd.txt'),
        })
        return project

    def test_reflectivity_series_one_entry_per_channel(self, polarized_project: Project) -> None:
        # When
        summary = Summary(polarized_project)

        # Then
        series = summary._reflectivity_series()

        # Expect: one series per measured channel, each in its own channel color, none shifted
        name = polarized_project.experiments[0].name
        assert [entry.label for entry in series] == [f'{name} (pp)', f'{name} (pm)', f'{name} (mm)']
        assert len({entry.color for entry in series}) == 3
        assert all(entry.offset == 0 for entry in series)

    def test_spin_flip_channel_of_a_non_magnetic_model_has_no_curve(self, polarized_project: Project) -> None:
        # When: a non-magnetic model cannot produce a spin-flip cross-section
        series = {entry.label[-3:-1]: entry for entry in Summary(polarized_project)._reflectivity_series()}

        # Then Expect: its data is drawn without a misleading overlay
        assert series['pm'].measured is not None
        assert series['pm'].calculated is None
        assert series['pp'].calculated is not None

    @pytest.fixture
    def project(self) -> Project:
        global_object.map._clear()
        project = Project()
        project.default_model()
        return project

    def test_reflectivity_series_unpolarized_and_empty(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then Expect: with nothing loaded, the model's curve alone
        series = summary._reflectivity_series()
        assert len(series) == 1
        assert series[0].measured is None and series[0].calculated is not None
        assert series[0].label == project.models[0].name

        # When an ordinary experiment is loaded
        project.load_experiment_for_model_at_index(os.path.join(PATH_STATIC, 'example.ort'))

        # Expect one series, the data and its curve, in the model's color
        series = summary._reflectivity_series()
        assert len(series) == 1
        assert series[0].label == project.experiments[0].name
        assert series[0].measured is project.experiments[0]
        assert series[0].color == project.models[0].color
        assert_allclose(series[0].calculated.x, series[0].measured.x)

    def test_fit_experiment_figure_has_a_legend_entry_per_channel(self, polarized_project: Project) -> None:
        # When
        summary = Summary(polarized_project)

        # Then
        figure = summary._fit_experiment_plotly_figure()

        # Expect: data and curve of a channel share a legend entry. A non-magnetic model
        # calculates only pp (as the unpolarized curve), so pm and mm are drawn as data alone.
        name = polarized_project.experiments[0].name
        legend = [trace.name for trace in figure.data if trace.showlegend is not False]
        assert legend == [f'{name} (pp)', f'{name} (pm)', f'{name} (mm)']
        assert [trace.mode for trace in figure.data] == ['markers', 'lines', 'markers', 'markers']


class TestSummaryMultipleExperiments:
    """Report figures of several contrasts, each with its own experiment."""

    Q = np.linspace(0.01, 0.25, 60)

    @pytest.fixture
    def contrasts(self) -> Project:
        """The default model and two contrasts, each with synthetic data from its own model."""
        global_object.map._clear()
        project = Project()
        project.default_model()
        d2o = project._materials[1]
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, Material(sld=-0.56, isld=0, name='H2O'))])
        project.add_contrast(0, 'CMSi', [ReplaceMaterial(d2o, Material(sld=2.07, isld=0, name='CMSi'))])
        experiments = {}
        for index, model in enumerate(project.models):
            y = project.model_data_for_model_at_index(index, q_range=self.Q).y
            experiments[index] = DataSet1D(
                name=f'{model.name} data', x=self.Q, y=y, ye=(0.02 * y) ** 2, model=model, auto_background=False
            )
        project.experiments = experiments
        return project

    def test_one_series_per_experiment_in_its_models_color(self, contrasts: Project) -> None:
        # When
        series = Summary(contrasts)._reflectivity_series()

        # Then Expect: every experiment, shifted a decade apart, each on its own model's curve
        assert [entry.label for entry in series] == ['Model data', 'H2O data', 'CMSi data']
        assert [entry.color for entry in series] == [model.color for model in contrasts.models]
        assert [entry.offset for entry in series] == [0, 1, 2]
        assert [entry.legend for entry in series] == ['Model data', 'H2O data (×10)', 'CMSi data (×10^2)']
        for entry, experiment in zip(series, contrasts.experiments.values()):
            assert entry.measured is experiment
            # The data was made from its own model, so its curve passes through it.
            assert_allclose(entry.calculated.y, experiment.y, rtol=1e-8)

    def test_a_recorded_fit_draws_its_own_experiments(self, contrasts: Project) -> None:
        # When: the last fit used two of the three experiments
        prepared = contrasts.prepare_fit([contrasts.experiments[2], contrasts.experiments[0]])
        contrasts.record_fit(prepared, None, status='failed')

        # Then
        series = Summary(contrasts)._reflectivity_series()

        # Expect: only those, in the fit's order
        assert [entry.label for entry in series] == ['CMSi data', 'Model data']
        assert [entry.offset for entry in series] == [0, 1]

    def test_a_fit_whose_experiments_are_gone_draws_every_experiment(self, contrasts: Project) -> None:
        # When: the only fitted experiment has been removed since
        contrasts.record_fit(contrasts.prepare_fit([contrasts.experiments[1]]), None, status='failed')
        contrasts.remove_experiment(1)

        # Then
        series = Summary(contrasts)._reflectivity_series()

        # Expect: the figure falls back to every experiment still loaded
        assert [entry.label for entry in series] == ['Model data', 'CMSi data']

    def test_without_experiments_every_model_is_drawn(self, contrasts: Project) -> None:
        # When
        contrasts.experiments = {}

        # Then
        series = Summary(contrasts)._reflectivity_series()

        # Expect: one calculated curve per model, shifted the same way
        assert [entry.label for entry in series] == [model.name for model in contrasts.models]
        assert all(entry.measured is None for entry in series)
        assert [entry.offset for entry in series] == [0, 1, 2]

    def test_one_sld_profile_per_model(self, contrasts: Project) -> None:
        # When
        summary = Summary(contrasts)

        # Then
        series = summary._sld_series()
        figure = summary._sld_plotly_figure()

        # Expect
        names = [model.name for model in contrasts.models]
        assert [entry.label for entry in series] == names
        assert [entry.color for entry in series] == [model.color for model in contrasts.models]
        assert [trace.name for trace in figure.data] == names

    def test_interactive_reflectivity_figure_shifts_every_experiment(self, contrasts: Project) -> None:
        # When
        figure = Summary(contrasts)._fit_experiment_plotly_figure()

        # Then Expect: data and curve per experiment, shifted by its legend's factor
        assert len(figure.data) == 6
        legend = [trace.name for trace in figure.data if trace.showlegend is not False]
        assert legend == ['Model data', 'H2O data (×10)', 'CMSi data (×10^2)']
        for offset, experiment in enumerate(contrasts.experiments.values()):
            measured, calculated = figure.data[2 * offset], figure.data[2 * offset + 1]
            assert measured.legendgroup == calculated.legendgroup
            assert_allclose(measured.y, np.asarray(experiment.y) * 10.0**offset)

    def test_static_figures_are_saved(self, contrasts: Project, tmp_path) -> None:
        # When
        summary = Summary(contrasts)

        # Then
        summary.save_fit_experiment_plot(tmp_path / 'reflectivity.png')
        summary.save_sld_plot(tmp_path / 'sld.png')

        # Expect
        assert (tmp_path / 'reflectivity.png').exists()
        assert (tmp_path / 'sld.png').exists()


class TestSummarySharedParameters:
    @pytest.fixture
    def contrast_project(self) -> Project:
        """The default model plus an H2O contrast, which reuses the superphase and subphase assemblies."""
        global_object.map._clear()
        project = Project()
        project.default_model()
        d2o = project._materials[1]
        project.add_contrast(0, 'H2O', [ReplaceMaterial(d2o, Material(sld=-0.56, isld=0, name='H2O'))])
        return project

    def test_shared_parameter_is_listed_once(self, contrast_project: Project) -> None:
        # When
        rows = list(_named_parameters(contrast_project))

        # Then Expect: one row per distinct parameter object, none left out
        ids = [id(parameter) for _, parameter in rows]
        assert len(ids) == len(set(ids))
        every = {id(parameter) for model in contrast_project.models for parameter in model.get_all_parameters()}
        assert set(ids) == every

    def test_shared_parameter_names_every_owner(self, contrast_project: Project) -> None:
        # When
        names = {id(parameter): name for name, parameter in _named_parameters(contrast_project)}
        reference, contrast = contrast_project.models

        # Then Expect: the shared subphase layer is attributed to both models, in model order
        shared = reference.sample[2].layers[0].thickness
        assert shared is contrast.sample[2].layers[0].thickness
        assert contrast_project.parameter_models(shared) == [0, 1]
        assert names[id(shared)] == 'Model+H2O Si Layer thickness'

        # And an unshared parameter keeps its single model's name
        own = reference.sample[1].layers[0].thickness
        assert contrast_project.parameter_models(own) == [0]
        assert names[id(own)] == 'Model D2O Layer thickness'

    def test_owner_names_agree_with_project(self, contrast_project: Project) -> None:
        # When
        rows = list(_named_parameters(contrast_project))
        models = contrast_project.models

        # Then Expect: every row is prefixed by exactly the models the project says own it
        for name, parameter in rows:
            owners = '+'.join(models[index].name for index in contrast_project.parameter_models(parameter))
            assert name.startswith(owners + ' ') or name == owners, name

    def test_sample_section_shows_shared_owners(self, contrast_project: Project) -> None:
        # When
        html = Summary(contrast_project)._sample_section()

        # Then Expect
        assert 'Model+H2O Si Layer thickness' in html
