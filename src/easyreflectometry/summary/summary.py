# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import logging
from dataclasses import dataclass
from html import escape
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter
from xhtml2pdf import pisa

from easyreflectometry import Project
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.sample import paths

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
# Color of an ordinary experiment that has no model to take one from.
_DEFAULT_COLOR = '#0173B2'
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


def _named_parameters(project: Project):
    """``(display name, parameter)`` for every parameter of every model, each once.

    The display name is the name of the model, of the object holding the parameter,
    and the parameter's own. A parameter shared by several models (a contrast reusing
    an assembly of its reference) is one value fitted for all of them, so it carries
    every owner's name, joined by '+' as in the app's parameter table.
    """
    found: dict[int, tuple[Parameter, str]] = {}  # first encounter, in walk order
    owners: dict[int, list[str]] = {}

    def walk(obj, owner_name: str, seen: set[int]):
        for _, child in paths.children(obj):
            if id(child) in seen:
                continue
            seen.add(id(child))
            if isinstance(child, DescriptorNumber):
                if isinstance(child, Parameter):
                    yield child, owner_name
            else:
                yield from walk(child, getattr(child, 'name', owner_name), seen)

    # Each model is walked in full, shared objects included, to learn every owner.
    for model in project.models:
        for parameter, owner_name in walk(model, model.name, set()):
            found.setdefault(id(parameter), (parameter, owner_name))
            owners.setdefault(id(parameter), []).append(model.name)

    for key, (parameter, owner_name) in found.items():
        model_names = '+'.join(owners[key])
        # A model's own parameter (scale) has the model as its owner: name it once.
        yield ' '.join(dict.fromkeys([model_names, owner_name, parameter.name])), parameter


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


@dataclass(frozen=True)
class _ReflectivitySeries:
    """One entry of the reflectivity figure: a measured dataset, the curve it is
    compared to, or both, drawn in one color and shifted by one scale factor.

    `measured` is None when no experiment is loaded and a model is drawn alone;
    `calculated` is None when the curve cannot be calculated (e.g. a spin-flip
    channel of a non-magnetic model): no overlay is better than a wrong one.
    """

    label: str
    measured: DataSet1D | None
    calculated: DataSet1D | None
    color: str
    offset: int = 0

    @property
    def scale(self) -> float:
        return 10.0**self.offset

    @property
    def legend(self) -> str:
        """The label, with the factor the series is shifted by when it is shifted."""
        if self.offset == 0:
            return self.label
        return f'{self.label} (×10)' if self.offset == 1 else f'{self.label} (×10^{self.offset})'


@dataclass(frozen=True)
class _SldSeries:
    """The SLD profile of one model, in the model's color."""

    label: str
    profile: DataSet1D
    color: str


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
        """Save the SLD figure: one profile per model."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for series in self._sld_series():
            ax.plot(series.profile.x, series.profile.y, color=series.color, label=series.label)

        ax.set_xlabel('z (Å)')
        ax.set_ylabel('SLD (Å⁻²)')
        if ax.has_data():
            ax.legend(loc='best')
        fig.savefig(filename, dpi=600)
        plt.close()

    def save_fit_experiment_plot(self, filename: str) -> None:
        """Save the reflectivity figure; see :meth:`_reflectivity_series` for what it draws."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')

        for series in self._reflectivity_series():
            # One legend entry per series: on the curve when there is one, else on the data.
            if series.measured is not None:
                ax.plot(
                    series.measured.x,
                    np.asarray(series.measured.y) * series.scale,
                    ls='',
                    marker='.',
                    color=series.color,
                    alpha=0.6,
                    label=series.legend if series.calculated is None else None,
                )
            if series.calculated is not None:
                ax.plot(
                    series.calculated.x,
                    np.asarray(series.calculated.y) * series.scale,
                    ls='-',
                    color=series.color,
                    zorder=10,
                    label=series.legend,
                )

        ax.set_xlabel('Q (Å⁻¹)')
        ax.set_ylabel('Reflectivity')
        if ax.has_data():
            ax.legend(loc='best')
        fig.savefig(filename, dpi=600)
        plt.close()

    def _sld_series(self) -> list[_SldSeries]:
        """The SLD profile of every model, each in its model's color."""
        return [
            _SldSeries(model.name, self._project.sld_data_for_model_at_index(index), model.color)
            for index, model in enumerate(self._project.models)
        ]

    def _drawn_datasets(self) -> list[tuple[int, PolarizationChannel | None]]:
        """``(experiment key, channel)`` of every measured dataset the reflectivity figure draws.

        The datasets of the last recorded fit, when there is one and they are still
        loaded, so that the figure describes the fit the report's numbers come from.
        Otherwise every dataset of every experiment: one per spin channel of a
        polarized experiment, one for an ordinary experiment.
        """
        experiments = self._project.experiments
        run = self._project.last_fit
        if run is not None:
            fitted = []
            for key, _name, channel in run.inputs:
                entry = (key, None if channel is None else PolarizationChannel(channel))
                if entry not in fitted and self._is_loaded(*entry):
                    fitted.append(entry)
            if fitted:
                return fitted

        drawn = []
        for key in sorted(experiments):
            channels = self._project.experiment_channels_at_index(key)
            drawn.extend([(key, channel) for channel in channels] if channels else [(key, None)])
        return drawn

    def _is_loaded(self, key, channel: PolarizationChannel | None) -> bool:
        """Whether the project still holds the dataset of `key` and `channel`.

        The experiment at a recorded key may have been replaced since the fit.
        """
        experiment = self._project.experiments.get(key)
        if isinstance(experiment, PolarizedDataSet):
            return channel is not None and channel in experiment.available_channels
        return experiment is not None and channel is None

    def _reflectivity_series(self) -> list[_ReflectivitySeries]:
        """What the reflectivity figure draws, shared by the static and the interactive figure.

        Each measured dataset of :meth:`_drawn_datasets` with the curve the fit compares
        it to: its experiment's model, at its q, with its resolution. An ordinary
        experiment is drawn in its model's color, a polarized one in one color per
        spin channel. With more than one experiment, the n-th is shifted up by 10^n
        so that the curves do not overlap. With no experiment loaded, the calculated
        curve of every model over the project's q range, shifted the same way.
        """
        drawn = self._drawn_datasets()
        if not drawn:
            models = self._project.models
            shift = len(models) > 1
            return [
                _ReflectivitySeries(
                    label=model.name,
                    measured=None,
                    calculated=self._project.model_data_for_model_at_index(index),
                    color=model.color,
                    offset=index if shift else 0,
                )
                for index, model in enumerate(models)
            ]

        keys = list(dict.fromkeys(key for key, _channel in drawn))
        shift = len(keys) > 1
        series = []
        for key, channel in drawn:
            experiment = self._project.experiments[key]
            name = experiment.name or f'Experiment {key}'
            if channel is None:
                label, measured = name, experiment
                color = getattr(experiment.model, 'color', None) or _DEFAULT_COLOR
            else:
                label, measured = f'{name} ({channel.value})', experiment[channel]
                color = _CHANNEL_COLORS[channel.value]
            series.append(
                _ReflectivitySeries(
                    label=label,
                    measured=measured,
                    calculated=self._calculated_curve(key, channel),
                    color=color,
                    offset=keys.index(key) if shift else 0,
                )
            )
        return series

    def _calculated_curve(self, key, channel: PolarizationChannel | None = None) -> DataSet1D | None:
        """The curve the fit compares the dataset of `key` and `channel` to, or None.

        None when it cannot be calculated: a spin-flip channel of a non-magnetic
        model, a calculator without magnetism, or an experiment without a model.
        """
        try:
            return self._project.model_data_for_experiment_at_index(key, channel=channel)
        except (ValueError, NotImplementedError) as exception:
            logging.getLogger(__name__).warning(
                'No calculated curve for experiment %s, channel %s: %s',
                key,
                getattr(channel, 'value', channel),
                exception,
            )
            return None

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

        for name, parameter in _named_parameters(self._project):
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
        # The dataset's own resolution when it has one, else its model's.
        resolution = getattr(dataset, 'resolution_function', None) or getattr(model, 'resolution_function', None)
        resolution_function = 'n/a' if resolution is None else resolution.as_dict()['smearing']
        if resolution_function == 'PercentageFwhm':
            precentage = resolution.as_dict()['constant']
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
        """Refinement section: the last recorded fit, then the project's parameter inventory."""
        html_refinement = HTML_REFINEMENT_TEMPLATE

        # The fit's own facts come from its record, not from the project as it is now.
        run = self._project.last_fit
        if run is None:
            minimizer_name, minimizer_package = self._project.minimizer.name, self._project.minimizer.package
            fit_free_params = 'N/A'
        else:
            member = AvailableMinimizers.__members__.get(run.minimizer)
            minimizer_name, minimizer_package = run.minimizer, member.package if member else None
            fit_free_params = f'{run.n_free_parameters}' if run.status == 'completed' else 'N/A'

        parameters = self._project.parameters

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
        html_refinement = html_refinement.replace('minimization_engine', _engine_link(minimizer_name, minimizer_package))
        html_refinement = html_refinement.replace('goodness_of_fit', goodness_of_fit)
        html_refinement = html_refinement.replace('num_fit_free_params', fit_free_params)
        html_refinement = html_refinement.replace('num_total_params', f'{num_params}')
        html_refinement = html_refinement.replace('num_free_params', f'{num_free_params}')
        html_refinement = html_refinement.replace('num_fixed_params', f'{num_fixed_params}')
        html_refinement = html_refinement.replace('num_constriants', f'{num_constraints}')
        return html_refinement

    def _compute_goodness_of_fit(self) -> str:
        """The pooled reduced chi² of the last recorded fit, or 'N/A'."""
        run = self._project.last_fit
        reduced = run.pooled.get('objective_reduced_chi2') if run is not None else None
        return 'N/A' if reduced is None else f'{reduced:.4g}'

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
        """Interactive SLD figure: one profile per model."""
        import plotly.graph_objects as go

        fig = go.Figure()
        for series in self._sld_series():
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(series.profile.x),
                    y=np.asarray(series.profile.y),
                    mode='lines',
                    name=series.label,
                    line={'color': series.color},
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
        """Interactive reflectivity figure; see :meth:`_reflectivity_series` for what it draws."""
        import plotly.graph_objects as go

        fig = go.Figure()
        for index, series in enumerate(self._reflectivity_series()):
            # The data and its curve toggle together under one legend entry.
            group = f'series-{index}'
            if series.measured is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(series.measured.x),
                        y=np.asarray(series.measured.y) * series.scale,
                        mode='markers',
                        name=series.legend,
                        legendgroup=group,
                        showlegend=series.calculated is None,
                        marker={'color': series.color, 'size': 4},
                    )
                )
            if series.calculated is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(series.calculated.x),
                        y=np.asarray(series.calculated.y) * series.scale,
                        mode='lines',
                        name=series.legend,
                        legendgroup=group,
                        line={'color': series.color},
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
