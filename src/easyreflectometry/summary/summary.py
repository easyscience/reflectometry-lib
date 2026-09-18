# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import logging
from html import escape
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
from easyscience import global_object
from xhtml2pdf import pisa

from easyreflectometry import Project

from .html_templates import HTML_DATA_COLLECTION_TEMPLATE
from .html_templates import HTML_FIGURES_TEMPLATE
from .html_templates import HTML_INTERACTIVE_FIGURES_TEMPLATE
from .html_templates import HTML_PARAMETER_HEADER_TEMPLATE
from .html_templates import HTML_PARAMETER_TEMPLATE
from .html_templates import HTML_PROJECT_INFORMATION_TEMPLATE
from .html_templates import HTML_REFINEMENT_TEMPLATE
from .html_templates import HTML_TEMPLATE


@contextlib.contextmanager
def _silence_pdf_converter():
    """Silence xhtml2pdf's verbose output during PDF conversion.

    xhtml2pdf emits a large amount of ``log.debug`` output (image tags, file
    objects, column widths, parsed schemes, ...) and a few stray ``print``
    statements while rendering. When the host application has configured logging
    at DEBUG level this floods stdout. For the duration of the conversion we
    raise the ``xhtml2pdf`` logger level so its debug records are dropped, and
    redirect stdout to swallow the stray prints. Both are restored afterwards.
    """
    xhtml2pdf_logger = logging.getLogger('xhtml2pdf')
    previous_level = xhtml2pdf_logger.level
    xhtml2pdf_logger.setLevel(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        xhtml2pdf_logger.setLevel(previous_level)


_NAME_MAX_LEN = 20
# Fixed per-channel colors for polarized experiments, shared with the app so a
# channel keeps the same color in the GUI and in the report.
_CHANNEL_COLORS = {'pp': '#0173B2', 'pm': '#029E73', 'mp': '#CC78BC', 'mm': '#DE8F05'}
# Custom href scheme used to pass the full name to QML via TextEdit.hoveredLink.
_TOOLTIP_SCHEME = 'nametooltip'

# URLs for known calculation engines and minimizer packages.
_ENGINE_URLS: dict[str, str] = {
    'refnx': 'https://refnx.readthedocs.io',
    'refl1d': 'https://refl1d.readthedocs.io',
    'lm': 'https://lmfit.github.io/lmfit-py/',
    'bumps': 'https://bumps.readthedocs.io',
    'dfo': 'https://github.com/fitbenchmarking/dfo-ls',
}


def _engine_link(name: str, package: str | None = None) -> str:
    """Return an HTML hyperlink for an engine, including its version.

    Falls back to plain text when no URL is known for the engine.
    """
    url = _ENGINE_URLS.get(name) or _ENGINE_URLS.get(package or '')
    display = escape(name)
    if package:
        try:
            ver = version(package)
        except PackageNotFoundError:
            ver = None
        if ver:
            display = f'{display} (v{ver})'

    if url:
        return f'<a href="{url}">{display}</a>'
    return display


def _format_value(value: float, sig_figs: int) -> str:
    """Format a numeric value for summary display.

    *sig_figs* significant figures; fall back to 1-decimal exponential when
    the formatted string is too long.  Zero is shown as '0.0'.
    """
    if value == 0.0:
        return '0.0'
    s = f'{value:.{sig_figs}g}'
    if len(s) <= sig_figs + 4:
        return s
    return f'{value:.1e}'


def _truncate_name(name: str, max_len: int = _NAME_MAX_LEN) -> str:
    """Return an HTML snippet with a truncated name and tooltip for the full text.

    Browsers use the ``title`` attribute; QML reads the ``href`` via
    ``TextEdit.hoveredLink`` and shows a native ToolTip.
    """
    safe = escape(name)
    if len(name) <= max_len:
        return safe
    short = escape(name[:max_len].rstrip())
    encoded = quote(name, safe='')
    return f'<a style="color:inherit;text-decoration:none;" href="{_TOOLTIP_SCHEME}:{encoded}" title="{safe}">{short}…</a>'


class Summary:
    def __init__(self, project: Project):
        """Init function."""
        self._project = project

    def compile_html_summary(self, figures: bool = False, interactive: bool = True) -> str:
        """Compile html summary.

        Parameters
        ----------
        figures
            Whether to render the figures section.
        interactive
            When ``True`` (the default) the figures are rendered as interactive
            plotly charts suitable for an HTML report. Set to ``False`` to fall
            back to static images, e.g. when the html is handed to the PDF
            converter, which cannot run the embedded JavaScript.
        """
        html = HTML_TEMPLATE

        html = html.replace('project_information_section', self._project_information_section())

        html = html.replace('sample_section', self._sample_section())

        experiments_section = self._experiments_section()
        if experiments_section == '':  # no experiments
            experiments_section = '<td>No experiments</td>'
        html = html.replace('experiments_section', experiments_section)

        html = html.replace('refinement_section', self._refinement_section())

        if figures:
            html = html.replace('figures_section', self._figures_section(interactive=interactive))
        else:
            html = html.replace('figures_section', '')

        return html

    def save_html_summary(self, filename: str) -> None:
        """Save html summary."""
        html = self.compile_html_summary(figures=True, interactive=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)

    def save_pdf_summary(self, filename: str) -> None:
        """Save pdf summary."""
        # The PDF converter (xhtml2pdf) cannot execute the JavaScript that powers
        # the interactive plotly charts, so embed static images instead.
        html = self.compile_html_summary(figures=True, interactive=False)

        with open(filename, 'w+b') as result_file:
            with _silence_pdf_converter():
                pisa_status = pisa.CreatePDF(
                    html,
                    dest=result_file,
                )

            if pisa_status.err:
                print('An error occured when generating PDF summary!')

    def save_sld_plot(self, filename: str) -> None:
        """Save sld plot."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        sld = self._project.sld_data_for_model_at_index(0)
        ax.plot(sld.x, sld.y, color='blue')

        ax.set_xlabel('z (Å)')
        ax.set_ylabel('SLD (Å⁻²)')
        fig.legend(['SLD'])
        fig.savefig(filename, dpi=600)
        plt.close()

    def _measured_series(self) -> list:
        """Measured data of experiment 0 as ``(label, dataset, color, channel)`` entries.

        One entry per measured spin channel for a polarized experiment (each
        with its own channel color and channel key), a single channel-less
        entry for an ordinary experiment, and an empty list when no experiment
        is loaded.
        """
        try:
            experiment = self._project.experimental_data_for_model_at_index(0)
        except IndexError:
            return []

        channels = self._project.experiment_channels_at_index(0)
        if not channels:
            return [('Experiment', experiment, 'red', None)]
        return [
            (f'Experiment ({channel.value})', experiment[channel], _CHANNEL_COLORS[channel.value], channel)
            for channel in channels
        ]

    def _model_curve(self, channel=None):
        """Calculated curve for experiment 0, per spin channel when asked.

        Returns None when the requested channel cannot be calculated (e.g. a
        spin-flip channel on a non-magnetic model) — an incorrect overlay is
        worse than none.
        """
        try:
            return self._project.model_data_for_model_at_index(0, channel=channel)
        except (ValueError, NotImplementedError) as exception:
            logging.getLogger(__name__).warning(
                'No calculated curve for channel %s: %s', getattr(channel, 'value', channel), exception
            )
            return None

    def save_fit_experiment_plot(self, filename: str) -> None:
        """Save fit experiment plot."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        legends = []

        measured_series = self._measured_series()
        # One calculated curve per measured channel; the ordinary single curve
        # when the experiment is unpolarized or nothing is loaded.
        model_channels = [entry[3] for entry in measured_series] or [None]
        for channel in model_channels:
            model = self._model_curve(channel)
            if model is None:
                continue
            color = 'blue' if channel is None else _CHANNEL_COLORS[channel.value]
            ax.plot(model.x, np.log10(model.y), color=color)
            legends.append('Model' if channel is None else f'Model ({channel.value})')

        for label, dataset, color, _channel in measured_series:
            ax.plot(dataset.x, np.log10(dataset.y), color=color)
            legends.append(label)

        ax.set_xlabel('Q (Å⁻¹)')
        ax.set_ylabel('Reflectivity')
        fig.legend(legends)
        fig.savefig(filename, dpi=600)
        plt.close()

    def _project_information_section(self) -> str:
        """Project information section."""
        html_project = HTML_PROJECT_INFORMATION_TEMPLATE

        name = self._project._info['name']
        short_description = self._project._info['short_description']
        html_project = html_project.replace('project_title', f'{name}')
        html_project = html_project.replace('project_description', f'{short_description}')
        html_project = html_project.replace('num_experiments', f'{len(self._project.experiments)}')
        return html_project

    def _sample_section(self) -> str:
        """Sample section."""
        html_parameters = []

        html_parameter = HTML_PARAMETER_HEADER_TEMPLATE
        html_parameter = html_parameter.replace('parameter_name', 'Name')
        html_parameter = html_parameter.replace('parameter_value', 'Value')
        html_parameter = html_parameter.replace('parameter_unit', 'Unit')
        html_parameter = html_parameter.replace('parameter_error', 'Error')
        html_parameters.append(html_parameter)

        # Get parameters directly from the model instead of using project.parameters
        model = self._project._models[self._project.current_model_index]
        parameters = model.get_all_parameters()

        for parameter in parameters:
            path = global_object.map.find_path(model.unique_name, parameter.unique_name)
            if 0 < len(path):
                name = f'{global_object.map.get_item_by_key(path[-2]).name} {global_object.map.get_item_by_key(path[-1]).name}'
            else:
                name = parameter.name
            value = parameter.value
            unit = parameter.unit
            error = parameter.error

            html_parameter = HTML_PARAMETER_TEMPLATE
            html_parameter = html_parameter.replace('parameter_name', f'{name}')
            html_parameter = html_parameter.replace('parameter_value', _format_value(value, 3))
            html_parameter = html_parameter.replace('parameter_unit', f'{unit}')
            # An unfitted parameter has no uncertainty; a literal '0.0' would
            # read as a perfectly determined value, so leave the cell empty.
            error_str = _format_value(error, 2) if error else ''
            html_parameter = html_parameter.replace('parameter_error', error_str)
            html_parameters.append(html_parameter)

        html_parameters_str = '\n'.join(html_parameters)

        return html_parameters_str

    def _experiments_section(self) -> str:
        """Experiments section."""
        html_experiments = []

        for idx, experiment in self._project.experiments.items():
            # A polarized experiment holds one dataset per spin channel; report
            # one row per channel, as the plots already draw one curve each.
            channels = getattr(experiment, 'available_channels', None)
            if channels is None:
                rows = [(experiment.name, experiment)]
            else:
                rows = [(f'{experiment.name} ({channel.value})', experiment[channel]) for channel in channels]
            for row_name, dataset in rows:
                html_experiments.append(self._experiment_row(row_name, dataset, experiment.model))

        html_experiments_str = '\n'.join(html_experiments)

        return html_experiments_str

    def _experiment_row(self, experiment_name, dataset, model) -> str:
        """One row of the experiments table for a single measured dataset."""
        num_data_points = len(dataset.x)
        resolution_function = model.resolution_function.as_dict()['smearing']
        if resolution_function == 'PercentageFwhm':
            precentage = model.resolution_function.as_dict()['constant']
            resolution_function = f'{resolution_function} {precentage}%'
        range_min = min(dataset.y)
        range_max = max(dataset.y)
        range_units = 'Å⁻¹'
        html_experiment = HTML_DATA_COLLECTION_TEMPLATE
        html_experiment = html_experiment.replace('experiment_name', _truncate_name(experiment_name))
        html_experiment = html_experiment.replace('range_min', _format_value(range_min, 2))
        html_experiment = html_experiment.replace('range_max', _format_value(range_max, 2))
        html_experiment = html_experiment.replace('range_units', f'{range_units}')
        html_experiment = html_experiment.replace('num_data_points', f'{num_data_points}')
        html_experiment = html_experiment.replace('resolution_function', f'{resolution_function}')
        return html_experiment

    def _refinement_section(self) -> str:
        """Refinement section."""
        html_refinement = HTML_REFINEMENT_TEMPLATE

        # Get parameters directly from the model
        model = self._project._models[self._project.current_model_index]
        parameters = model.get_all_parameters()

        # Dependent parameters (user constraints, derived values such as the
        # total thickness) are neither free nor fixed: they never enter a fit.
        num_free_params = sum(1 for parameter in parameters if parameter.independent and parameter.free)
        num_fixed_params = sum(1 for parameter in parameters if parameter.independent and not parameter.free)
        num_constraints = sum(1 for parameter in parameters if not parameter.independent)
        num_params = num_free_params + num_fixed_params + num_constraints

        goodness_of_fit = self._compute_goodness_of_fit()

        html_refinement = html_refinement.replace(
            'calculation_engine',
            _engine_link(self._project._calculator.current_interface_name),
        )
        html_refinement = html_refinement.replace(
            'minimization_engine',
            _engine_link(self._project.minimizer.name, self._project.minimizer.package),
        )
        html_refinement = html_refinement.replace('goodness_of_fit', goodness_of_fit)
        html_refinement = html_refinement.replace('num_total_params', f'{num_params}')
        html_refinement = html_refinement.replace('num_free_params', f'{num_free_params}')
        html_refinement = html_refinement.replace('num_fixed_params', f'{num_fixed_params}')
        html_refinement = html_refinement.replace('num_constriants', f'{num_constraints}')
        return html_refinement

    def _compute_goodness_of_fit(self) -> str:
        """Return reduced chi² as a formatted string, or 'N/A' if no fit has been run."""
        last_fit_results = getattr(self._project, '_last_fit_results', None)
        if not last_fit_results:
            return 'N/A'
        try:
            if len(last_fit_results) == 1:
                gof = float(last_fit_results[0].reduced_chi2)
            else:
                total_chi2 = sum(float(r.chi2) for r in last_fit_results)
                total_points = sum(len(r.x) for r in last_fit_results)
                n_pars = last_fit_results[0].n_pars
                dof = total_points - n_pars
                gof = total_chi2 / dof if dof > 0 else 0.0
            return f'{gof:.4g}'
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return 'N/A'

    def _figures_section(self, interactive: bool = True) -> str:
        """Figures section.

        When *interactive* is ``True`` the figures are rendered as interactive
        plotly charts embedded directly in the html. Otherwise static images are
        written to disk and referenced (used for the PDF report).
        """
        if interactive:
            return self._interactive_figures_section()

        html_figures = HTML_FIGURES_TEMPLATE
        path_sld = self._project.path / 'sld_plot.jpg'
        path_fit_experiment = self._project.path / 'fit_experiment_plot.jpg'

        self.save_sld_plot(path_sld)
        self.save_fit_experiment_plot(path_fit_experiment)

        html_figures = html_figures.replace('path_sld_plot', str(path_sld))
        html_figures = html_figures.replace('path_fit_experiment_plot', str(path_fit_experiment))
        return html_figures

    def _interactive_figures_section(self) -> str:
        """Build the interactive (plotly) figures section for the html report."""
        import plotly.io as pio

        fig_sld = self._sld_plotly_figure()
        fig_fit_experiment = self._fit_experiment_plotly_figure()

        # Embed plotly.js inline once (with the first figure) so the saved report
        # is self-contained and renders without an internet connection.
        sld_div = pio.to_html(
            fig_sld,
            include_plotlyjs=True,
            full_html=False,
            default_width='640px',
            default_height='480px',
        )
        fit_experiment_div = pio.to_html(
            fig_fit_experiment,
            include_plotlyjs=False,
            full_html=False,
            default_width='640px',
            default_height='480px',
        )

        html_figures = HTML_INTERACTIVE_FIGURES_TEMPLATE
        html_figures = html_figures.replace('sld_plot_div', sld_div)
        html_figures = html_figures.replace('fit_experiment_plot_div', fit_experiment_div)
        return html_figures

    def _sld_plotly_figure(self):
        """Interactive SLD profile figure."""
        import plotly.graph_objects as go

        sld = self._project.sld_data_for_model_at_index(0)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.asarray(sld.x),
                y=np.asarray(sld.y),
                mode='lines',
                name='SLD',
                line={'color': 'blue'},
            )
        )
        fig.update_layout(
            xaxis_title='z (Å)',
            yaxis_title='SLD (Å⁻²)',
            template='simple_white',
            margin={'l': 70, 'r': 20, 't': 30, 'b': 50},
            legend={'x': 0.99, 'xanchor': 'right', 'y': 0.99, 'yanchor': 'top'},
        )
        return fig

    def _fit_experiment_plotly_figure(self):
        """Interactive reflectivity (model vs. experiment) figure."""
        import plotly.graph_objects as go

        fig = go.Figure()

        measured_series = self._measured_series()
        # One calculated curve per measured channel; the ordinary single curve
        # when the experiment is unpolarized or nothing is loaded.
        model_channels = [entry[3] for entry in measured_series] or [None]
        for channel in model_channels:
            model = self._model_curve(channel)
            if model is None:
                continue
            color = 'blue' if channel is None else _CHANNEL_COLORS[channel.value]
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(model.x),
                    y=np.asarray(model.y),
                    mode='lines',
                    name='Model' if channel is None else f'Model ({channel.value})',
                    line={'color': color},
                )
            )

        for label, dataset, color, _channel in measured_series:
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(dataset.x),
                    y=np.asarray(dataset.y),
                    mode='markers',
                    name=label,
                    marker={'color': color, 'size': 4},
                )
            )

        fig.update_layout(
            xaxis_title='Q (Å⁻¹)',
            yaxis_title='Reflectivity',
            yaxis_type='log',
            template='simple_white',
            margin={'l': 70, 'r': 20, 't': 30, 'b': 50},
            legend={'x': 0.99, 'xanchor': 'right', 'y': 0.99, 'yanchor': 'top'},
        )
        return fig
