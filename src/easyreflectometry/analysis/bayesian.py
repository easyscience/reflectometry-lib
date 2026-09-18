# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian posterior analysis for reflectometry fitting results."""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
from easyscience.fitting.minimizers.minimizer_base import MINIMIZER_PARAMETER_PREFIX

try:
    import arviz as _arviz

    _HAS_ARVIZ = True
except ImportError:
    _HAS_ARVIZ = False


def _require_arviz():
    if not _HAS_ARVIZ:
        raise ImportError('The ``arviz`` library is required for trace plots and R-hat. Install it with ``pip install arviz``.')


def _require_plotly():
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'The ``plotly`` library is required for posterior plots. Install it with ``pip install plotly``.'
        ) from exc


def _wrap_pair_label(name: str, max_len: int = 16) -> str:
    """Insert ``<br>`` line breaks so long parameter labels don't overlap.

    Dotted names (e.g. ``layer1.thickness``) break on dots; plain names
    word-wrap on spaces at roughly *max_len* characters per line.
    """
    name = name.strip()
    if not name:
        return name
    if '.' in name:
        parts = [p.strip() for p in name.split('.') if p.strip()]
        if parts:
            return '<br>'.join([*(f'{p}.' for p in parts[:-1]), parts[-1]])
    if len(name) <= max_len:
        return name
    words = name.split()
    if len(words) == 1:
        return name
    lines: list[str] = []
    current = ''
    for word in words:
        if current and len(current) + 1 + len(word) > max_len:
            lines.append(current)
            current = word
        else:
            current = f'{current} {word}' if current else word
    if current:
        lines.append(current)
    return '<br>'.join(lines)


def _to_arviz_data(draws: np.ndarray, param_names: list[str]):
    """Convert posterior draws to an arviz InferenceData object.

    :param draws: Posterior samples, shape ``(n_samples, n_params)`` or
        ``(n_chains, n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: arviz InferenceData object.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim == 2:
        draws = draws[np.newaxis, ...]  # (1, n_samples, n_params)

    # Build a dict of {param_name: (chain, draw) array}
    posterior_dict = {}
    for i, name in enumerate(param_names):
        posterior_dict[name] = draws[:, :, i]

    # arviz < 1.0 takes the posterior variables as a keyword argument; arviz
    # >= 1.0 removed it in favour of a single {group: {var: array}} mapping.
    # The 1.x call cannot go first: 0.x accepts the mapping without error but
    # misreads it as one variable named 'posterior'.
    try:
        return _arviz.from_dict(posterior=posterior_dict)
    except TypeError:
        return _arviz.from_dict({'posterior': posterior_dict})


class PosteriorResults:
    """Container for Bayesian posterior samples with analysis methods.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column of ``draws``).
    :type param_names: list[str]
    :param logp: Log-posterior values, shape ``(n_samples,)``, or ``None``.
    :type logp: np.ndarray | None
    :param sampler_state: Raw sampler state object (e.g. BUMPS ``DreamState``), or ``None``.
    :type sampler_state: Any | None
    """

    def __init__(
        self,
        draws: np.ndarray,
        param_names: list[str],
        logp: np.ndarray | None = None,
        sampler_state: Any | None = None,
    ):
        self.draws = np.asarray(draws)
        self.param_names = list(param_names)
        self.logp = np.asarray(logp) if logp is not None else None
        self.sampler_state = sampler_state

    def __repr__(self) -> str:
        n_samples, n_params = self.draws.shape
        return f'PosteriorResults(n_samples={n_samples}, n_params={n_params}, param_names={self.param_names})'

    def summary(self) -> str:
        """Return a formatted summary table with mean, sd, and equal-tailed 95% credible interval for each parameter.

        :return: Formatted summary table as a string.
        :rtype: str
        """
        return posterior_summary(self.draws, self.param_names)

    def corner(self) -> Any:
        """Return the parameter-correlation corner plot as a Plotly Figure.

        Requires the ``plotly`` library.

        :return: Plotly Figure.
        """
        return plot_corner(self.draws, self.param_names)

    def trace(self, **kwargs) -> None:
        """Plot MCMC trace plot.

        Requires the ``arviz`` library.

        :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``.
        """
        plot_trace(self.draws, self.param_names, **kwargs)

    def distribution(self) -> Any:
        """Return the per-parameter marginal posterior distributions as a Plotly Figure.

        Each panel overlays the posterior histogram, a smooth KDE marginal, the
        95% credible interval, the median, and the best posterior sample (when
        ``logp`` is available). Requires the ``plotly`` library.

        :return: Plotly Figure.
        """
        return plot_distribution(self.draws, self.param_names, logp=self.logp, return_figure=True)

    def credible_interval(self, alpha: float = 0.95) -> dict:
        """Compute equal-tailed credible intervals for each parameter.

        :param alpha: Credible interval width (e.g. 0.95 for 95%).
        :type alpha: float
        :return: Dictionary mapping parameter name to ``(lower, upper)``.
        :rtype: dict
        """
        return credible_intervals(self.draws, self.param_names, alpha=alpha)

    def save(self, path: str) -> None:
        """Persist this posterior trace to disk.

        Convenience wrapper around :func:`save_posterior`.

        :param path: File path prefix (see :func:`save_posterior`).
        :type path: str
        """
        save_posterior(self, path)

    def gelman_rubin(self) -> dict | None:
        """Compute the Gelman-Rubin R-hat convergence diagnostic.

        Requires the ``arviz`` library and posterior draws with at least
        two chains, i.e. shape ``(n_chains, n_draws, n_params)`` with
        ``n_chains >= 2``. R-hat is undefined for a single chain.

        :return: Dictionary mapping parameter name to R-hat value, or ``None``
            if ``arviz`` is not available.
        :rtype: dict | None
        :raises ValueError: If ``self.draws`` does not contain at least two
            chains.
        """
        if not _HAS_ARVIZ:
            warnings.warn(
                'The ``arviz`` library is required for Gelman-Rubin R-hat. Install it with ``pip install arviz``.',
                UserWarning,
            )
            return None
        if self.draws.ndim < 3 or self.draws.shape[0] < 2:
            raise ValueError(
                'Gelman-Rubin R-hat requires posterior draws with at least 2 chains '
                '(shape ``(n_chains, n_draws, n_params)`` with ``n_chains >= 2``).'
            )
        data = _to_arviz_data(self.draws, self.param_names)
        rhat = _arviz.rhat(data)
        return {name: float(rhat[name].values) for name in self.param_names}


def posterior_summary(draws: np.ndarray, param_names: list[str]) -> str:
    """Return a formatted summary table with mean, sd, and the equal-tailed
    2.5%/97.5% posterior quantiles for each parameter.

    The reported interval is the equal-tailed 95% credible interval; it is
    not a highest-density interval (HDI) and the two coincide only for
    symmetric unimodal posteriors.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: Formatted summary table as a string.
    :rtype: str
    """
    draws = np.asarray(draws)
    lines = [f'{"parameter":<30s} {"mean":>10s} {"sd":>10s} {"q2.5%":>10s} {"q97.5%":>10s}']
    for i, name in enumerate(param_names):
        col = draws[:, i]
        lo, hi = np.percentile(col, [2.5, 97.5])
        lines.append(f'{name:<30s} {col.mean():>10.4f} {col.std():>10.4f} {lo:>10.4f} {hi:>10.4f}')
    return '\n'.join(lines)


# Posterior pair-plot palette and styling, mirroring easydiffraction so the
# corner plots match across the two libraries.
_POSTERIOR_PAIR_MIN_SAMPLE_COUNT = 2
_POSTERIOR_PAIR_COVARIANCE_RANK = 2
_POSTERIOR_PAIR_SCATTER_MAX_POINTS = 1500
_POSTERIOR_PAIR_CONTOUR_GRID_SIZE = 80
_POSTERIOR_PAIR_DENSITY_GRID_SIZE = 256

_POSTERIOR_PAIR_MARGINAL_LINE_COLOR = 'rgb(44, 160, 44)'
_POSTERIOR_PAIR_MARGINAL_FILL_COLOR = 'rgba(44, 160, 44, 0.22)'
_POSTERIOR_PAIR_SCATTER_COLOR = 'rgba(140, 140, 140, 0.22)'
_POSTERIOR_HISTOGRAM_FILL_COLOR = 'rgba(120, 120, 120, 0.38)'
_POSTERIOR_HISTOGRAM_LINE_COLOR = 'rgba(120, 120, 120, 0.24)'
_POSTERIOR_INTERVAL_95_FILL_COLOR = 'rgba(214, 39, 40, 0.14)'
_POSTERIOR_MEDIAN_LINE_COLOR = 'rgb(80, 80, 80)'
_POSTERIOR_POINT_ESTIMATE_LINE_COLOR = 'rgb(214, 39, 40)'
_POSTERIOR_CONTOUR_FILL_COLORSCALE = [
    [0.0, 'rgba(224, 233, 255, 0.62)'],
    [0.35, 'rgba(183, 203, 255, 0.70)'],
    [0.60, 'rgba(138, 169, 252, 0.78)'],
    [0.82, 'rgba(96, 131, 242, 0.84)'],
    [1.0, 'rgba(58, 86, 224, 0.90)'],
]
_POSTERIOR_NEGATIVE_CONTOUR_FILL_COLORSCALE = [
    [0.0, 'rgba(255, 224, 224, 0.62)'],
    [0.35, 'rgba(250, 188, 188, 0.70)'],
    [0.60, 'rgba(245, 148, 148, 0.78)'],
    [0.82, 'rgba(237, 104, 104, 0.84)'],
    [1.0, 'rgba(215, 48, 39, 0.90)'],
]
_POSTERIOR_CONTOUR_LINE_COLORSCALE = [
    [0.0, 'rgba(183, 203, 255, 0.94)'],
    [0.35, 'rgba(183, 203, 255, 0.94)'],
    [0.35, 'rgba(138, 169, 252, 0.95)'],
    [0.60, 'rgba(138, 169, 252, 0.95)'],
    [0.60, 'rgba(96, 131, 242, 0.96)'],
    [0.82, 'rgba(96, 131, 242, 0.96)'],
    [0.82, 'rgba(58, 86, 224, 0.98)'],
    [1.0, 'rgba(58, 86, 224, 0.98)'],
]
_POSTERIOR_NEGATIVE_CONTOUR_LINE_COLORSCALE = [
    [0.0, 'rgba(250, 188, 188, 0.94)'],
    [0.35, 'rgba(250, 188, 188, 0.94)'],
    [0.35, 'rgba(245, 148, 148, 0.95)'],
    [0.60, 'rgba(245, 148, 148, 0.95)'],
    [0.60, 'rgba(237, 104, 104, 0.96)'],
    [0.82, 'rgba(237, 104, 104, 0.96)'],
    [0.82, 'rgba(215, 48, 39, 0.98)'],
    [1.0, 'rgba(215, 48, 39, 0.98)'],
]


def _posterior_axis_bounds(values: np.ndarray) -> tuple[float, float] | None:
    """Return padded ``(lower, upper)`` plotting bounds for one posterior axis.

    :param values: Posterior samples for a single parameter.
    :type values: np.ndarray
    :return: Padded bounds, or ``None`` if there are no finite samples.
    :rtype: tuple[float, float] | None
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_range = data_max - data_min
    padding = 0.05 * data_range if data_range > 0 else max(abs(data_min), 1.0) * 0.05
    if padding == 0:
        padding = 1e-6
    return data_min - padding, data_max + padding


def _posterior_density_curve(
    values: np.ndarray,
    grid_size: int = _POSTERIOR_PAIR_DENSITY_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate a 1-D Gaussian-KDE marginal density normalised to unit area.

    :param values: Posterior samples for a single parameter.
    :type values: np.ndarray
    :param grid_size: Number of grid points at which to evaluate the density.
    :type grid_size: int
    :return: ``(grid, density)`` arrays, or ``None`` if a smooth density could
        not be estimated (e.g. ``scipy`` missing or degenerate samples).
    :rtype: tuple[np.ndarray, np.ndarray] | None
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return None

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < _POSTERIOR_PAIR_MIN_SAMPLE_COUNT:
        return None

    bounds = _posterior_axis_bounds(data)
    if bounds is None:
        return None
    grid = np.linspace(bounds[0], bounds[1], num=grid_size)

    if np.allclose(data, data[0]):
        bandwidth = max(abs(data[0]) * 0.01, 1e-6)
        density = np.exp(-0.5 * ((grid - data[0]) / bandwidth) ** 2)
        density /= bandwidth * np.sqrt(2.0 * np.pi)
    else:
        try:
            density = np.asarray(gaussian_kde(data)(grid), dtype=float)
        except (np.linalg.LinAlgError, ValueError):
            return None

    area = np.trapezoid(density, grid)
    if area <= 0:
        return None
    return grid, density / area


def _posterior_density_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid_size: int = _POSTERIOR_PAIR_CONTOUR_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Estimate a 2-D Gaussian-KDE density surface for one pair panel.

    :param x_values: Posterior samples for the x-axis parameter.
    :type x_values: np.ndarray
    :param y_values: Posterior samples for the y-axis parameter.
    :type y_values: np.ndarray
    :param grid_size: Number of grid points per axis.
    :type grid_size: int
    :return: ``(x_grid, y_grid, density)``, or ``None`` if a smooth surface
        could not be estimated (e.g. ``scipy`` missing or degenerate samples).
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return None

    x_data = np.asarray(x_values, dtype=float)
    y_data = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]
    if x_data.size < _POSTERIOR_PAIR_MIN_SAMPLE_COUNT or y_data.size < _POSTERIOR_PAIR_MIN_SAMPLE_COUNT:
        return None
    if np.allclose(x_data, x_data[0]) and np.allclose(y_data, y_data[0]):
        return None

    pair_data = np.vstack([x_data, y_data])
    covariance = np.cov(pair_data)
    if np.linalg.matrix_rank(covariance) < _POSTERIOR_PAIR_COVARIANCE_RANK:
        return None

    x_bounds = _posterior_axis_bounds(x_data)
    y_bounds = _posterior_axis_bounds(y_data)
    if x_bounds is None or y_bounds is None:
        return None
    x_grid = np.linspace(x_bounds[0], x_bounds[1], num=grid_size)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], num=grid_size)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    try:
        kde = gaussian_kde(pair_data)
        density = np.asarray(kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])), dtype=float)
    except (np.linalg.LinAlgError, ValueError):
        return None
    density = density.reshape(mesh_x.shape)
    if not np.any(np.isfinite(density)):
        return None
    return x_grid, y_grid, density


def _posterior_contour_colorscales(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[list, list]:
    """Return sign-aware fill and line contour palettes for one pair panel.

    Negatively correlated parameter pairs use a red palette; everything else
    uses blue.

    :param x_values: Posterior samples for the x-axis parameter.
    :type x_values: np.ndarray
    :param y_values: Posterior samples for the y-axis parameter.
    :type y_values: np.ndarray
    :return: ``(fill_colorscale, line_colorscale)``.
    :rtype: tuple[list, list]
    """
    x_data = np.asarray(x_values, dtype=float)
    y_data = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if np.count_nonzero(mask) >= _POSTERIOR_PAIR_MIN_SAMPLE_COUNT:
        correlation = float(np.corrcoef(x_data[mask], y_data[mask])[0, 1])
        if np.isfinite(correlation) and correlation < 0:
            return _POSTERIOR_NEGATIVE_CONTOUR_FILL_COLORSCALE, _POSTERIOR_NEGATIVE_CONTOUR_LINE_COLORSCALE
    return _POSTERIOR_CONTOUR_FILL_COLORSCALE, _POSTERIOR_CONTOUR_LINE_COLORSCALE


def _add_corner_marginal(
    fig: Any,
    go: Any,
    *,
    values: np.ndarray,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    """Add a diagonal marginal-density panel (smooth KDE, histogram fallback).

    :param fig: The Plotly Figure being built.
    :param go: The ``plotly.graph_objects`` module.
    :param values: Posterior samples for the diagonal parameter.
    :type values: np.ndarray
    :param row: 1-based subplot row.
    :type row: int
    :param col: 1-based subplot column.
    :type col: int
    :param show_legend: Whether this trace should add the legend entry.
    :type show_legend: bool
    """
    curve = _posterior_density_curve(values)
    if curve is not None:
        grid, density = curve
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode='lines',
                line=dict(color=_POSTERIOR_PAIR_MARGINAL_LINE_COLOR, width=1),
                fill='tozeroy',
                fillcolor=_POSTERIOR_PAIR_MARGINAL_FILL_COLOR,
                name='Marginal density',
                legendgroup='marginal',
                showlegend=show_legend,
                hoverinfo='skip',
            ),
            row=row,
            col=col,
        )
        return
    # scipy unavailable or degenerate samples: fall back to a histogram.
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=40,
            histnorm='probability density',
            marker=dict(
                color=_POSTERIOR_PAIR_MARGINAL_FILL_COLOR,
                line=dict(color=_POSTERIOR_PAIR_MARGINAL_LINE_COLOR, width=1),
            ),
            name='Marginal density',
            legendgroup='marginal',
            showlegend=show_legend,
            hoverinfo='skip',
        ),
        row=row,
        col=col,
    )


def _corner_contour_traces(
    go: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[Any, Any] | None:
    """Build filled and line 2-D KDE contour traces for one pair panel.

    :param go: The ``plotly.graph_objects`` module.
    :param x_values: Posterior samples for the x-axis parameter.
    :type x_values: np.ndarray
    :param y_values: Posterior samples for the y-axis parameter.
    :type y_values: np.ndarray
    :return: ``(fill_trace, line_trace)``, or ``None`` if no smooth surface
        could be estimated.
    :rtype: tuple[Any, Any] | None
    """
    surface = _posterior_density_surface(x_values, y_values)
    if surface is None:
        return None
    x_grid, y_grid, density = surface

    fill_colorscale, line_colorscale = _posterior_contour_colorscales(x_values, y_values)
    density_max = float(np.max(density))
    contour_start = density_max * 0.20
    contour_end = density_max * 0.95
    contour_size = density_max * 0.15

    fill_density = np.array(density, copy=True)
    fill_density[fill_density < contour_start] = np.nan
    fill_trace = go.Contour(
        x=x_grid,
        y=y_grid,
        z=fill_density,
        contours=dict(
            coloring='fill',
            showlabels=False,
            showlines=False,
            start=contour_start,
            end=contour_end,
            size=contour_size,
        ),
        colorscale=fill_colorscale,
        zmin=contour_start,
        zmax=contour_end,
        connectgaps=False,
        hoverinfo='skip',
        showscale=False,
        zorder=1,
    )
    line_trace = go.Contour(
        x=x_grid,
        y=y_grid,
        z=density,
        contours=dict(
            coloring='lines',
            showlabels=False,
            start=contour_start,
            end=contour_end,
            size=contour_size,
        ),
        colorscale=line_colorscale,
        zmin=contour_start,
        zmax=contour_end,
        line=dict(width=0.9),
        hoverinfo='skip',
        showscale=False,
        zorder=2,
    )
    return fill_trace, line_trace


def plot_corner(draws: np.ndarray, param_names: list[str]) -> Any:
    """Build a parameter-correlation corner plot as a Plotly Figure.

    Smooth Gaussian-KDE marginal densities on the diagonal, a posterior
    scatter overlay with filled 2-D KDE contours on the lower triangle, and a
    hidden upper triangle.  Contours are coloured blue for positively
    correlated pairs and red for negatively correlated ones.  This mirrors the
    posterior pair plot in ``easydiffraction``.  Requires ``plotly`` (and
    ``scipy`` for the KDE smoothing; without it the diagonal falls back to a
    histogram and the contours are omitted).

    :param draws: Posterior samples, shape ``(n_samples, n_params)`` or
        ``(n_chains, n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: Plotly Figure.
    """
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    draws = np.asarray(draws)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    n_params = len(param_names)

    wrapped_labels = [_wrap_pair_label(name) for name in param_names]

    # Full draws drive the smooth KDE surfaces; the scatter overlay is thinned
    # so large posteriors stay responsive to render and pan.
    n_samples = draws.shape[0]
    if n_samples > _POSTERIOR_PAIR_SCATTER_MAX_POINTS:
        stride = max(1, n_samples // _POSTERIOR_PAIR_SCATTER_MAX_POINTS)
        scatter_draws = draws[::stride]
    else:
        scatter_draws = draws

    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Track which trace types have already been added to the legend.
    legend_shown = {'marginal': False, 'scatter': False, 'contour': False}

    for row in range(n_params):
        for col in range(n_params):
            r, c = row + 1, col + 1
            if col > row:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                continue
            if col == row:
                _add_corner_marginal(
                    fig,
                    go,
                    values=draws[:, row],
                    row=r,
                    col=c,
                    show_legend=not legend_shown['marginal'],
                )
                legend_shown['marginal'] = True
            else:
                fig.add_trace(
                    go.Scatter(
                        x=scatter_draws[:, col],
                        y=scatter_draws[:, row],
                        mode='markers',
                        marker=dict(size=4, color=_POSTERIOR_PAIR_SCATTER_COLOR),
                        name='Posterior samples',
                        legendgroup='scatter',
                        showlegend=not legend_shown['scatter'],
                        hoverinfo='skip',
                        zorder=0,
                    ),
                    row=r,
                    col=c,
                )
                legend_shown['scatter'] = True
                contour_traces = _corner_contour_traces(go, draws[:, col], draws[:, row])
                if contour_traces is not None:
                    fill_trace, line_trace = contour_traces
                    fill_trace.name = 'Posterior contours'
                    fill_trace.legendgroup = 'contour'
                    fill_trace.showlegend = not legend_shown['contour']
                    line_trace.legendgroup = 'contour'
                    line_trace.showlegend = False
                    fig.add_trace(fill_trace, row=r, col=c)
                    fig.add_trace(line_trace, row=r, col=c)
                    legend_shown['contour'] = True

    # Axis labels: outer edges only (bottom row x-axes, leftmost column y-axes,
    # including the top-left diagonal cell so its parameter is identifiable).
    for i, label in enumerate(wrapped_labels):
        fig.update_xaxes(title_text=label, title_font=dict(size=10), row=n_params, col=i + 1)
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=i + 1, col=1)
    # Diagonal y-axes are probability density — hide their tick labels (except the
    # top-left, where ticks would be the only cue about the density scale).
    for i in range(1, n_params):
        fig.update_yaxes(showticklabels=False, row=i + 1, col=i + 1)

    fig.update_layout(
        height=max(450, 180 * n_params),
        width=max(550, 180 * n_params + 140),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.02,
            font=dict(size=11),
            itemsizing='constant',
        ),
        plot_bgcolor='white',
        margin=dict(l=80, r=160, t=30, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside')
    fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside')
    return fig


def plot_trace(draws: np.ndarray, param_names: list[str], return_figure: bool = False, **kwargs) -> Any:
    """Plot MCMC trace plot.

    When *return_figure* is ``True`` a Plotly ``Figure`` is returned instead of
    being displayed inline; the caller is responsible for rendering it.  This
    requires the ``plotly`` package.

    :param draws: Posterior samples, shape ``(n_chains, n_draws, n_params)`` or
        ``(n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param return_figure: Return a Plotly Figure instead of rendering inline.
    :type return_figure: bool
    :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``
        when *return_figure* is ``False``.
    :return: Plotly Figure when *return_figure* is ``True``, otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 2:
        draws = draws[np.newaxis, ...]  # (1, n_draws, n_params)
    # draws shape: (n_chains, n_draws, n_params)

    if return_figure:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            warnings.warn(
                'The ``plotly`` library is required to build the trace figure. Install it with ``pip install plotly``.',
                UserWarning,
                stacklevel=2,
            )
            return None

        n_params = len(param_names)
        n_chains = draws.shape[0]
        fig = make_subplots(
            rows=n_params,
            cols=2,
            column_widths=[0.6, 0.4],
        )
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, name in enumerate(param_names):
            row = i + 1
            show_legend = i == 0
            for c in range(n_chains):
                chain_draws = draws[c, :, i]
                color = colors[c % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        y=chain_draws,
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=f'chain {c}',
                        legendgroup=f'chain {c}',
                        showlegend=show_legend,
                    ),
                    row=row,
                    col=1,
                )
                fig.add_trace(
                    go.Histogram(
                        x=chain_draws,
                        marker_color=color,
                        opacity=0.6,
                        name=f'chain {c}',
                        legendgroup=f'chain {c}',
                        showlegend=show_legend,
                        nbinsx=40,
                    ),
                    row=row,
                    col=2,
                )
            fig.update_yaxes(title_text=name, title_font=dict(size=10), row=row, col=1)
            fig.update_xaxes(title_text=name, title_font=dict(size=10), row=row, col=2)
            fig.update_yaxes(title_text='Count', title_font=dict(size=10), row=row, col=2)
        fig.update_xaxes(title_text='Draw index', title_font=dict(size=10), row=n_params, col=1)
        fig.update_layout(
            height=max(300, 200 * n_params),
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
        )
        return fig

    _require_arviz()
    idata = _to_arviz_data(draws, param_names)
    _arviz.plot_trace(idata, var_names=param_names, **kwargs)
    return None


def _posterior_marginal_y_range(
    values: np.ndarray,
    density_curve: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, float] | None:
    """Return a ``(0, max)`` y-axis range covering histogram and KDE density.

    The range spans the larger of the histogram (probability-density normalised)
    and smooth-KDE peaks so credible-interval bands and reference lines, which
    are drawn as full-height traces, reach the top of the panel.

    :param values: Posterior samples for a single parameter.
    :type values: np.ndarray
    :param density_curve: ``(grid, density)`` from :func:`_posterior_density_curve`, or ``None``.
    :type density_curve: tuple[np.ndarray, np.ndarray] | None
    :return: ``(0.0, padded_max)`` range, or ``None`` if no density is available.
    :rtype: tuple[float, float] | None
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    maxima: list[float] = []
    if data.size:
        hist, _ = np.histogram(data, bins=40, density=True)
        if hist.size:
            maxima.append(float(np.max(hist)))
    if density_curve is not None:
        maxima.append(float(np.max(density_curve[1])))
    if not maxima:
        return None
    y_max = max(maxima)
    if y_max <= 0:
        return None
    return 0.0, y_max * 1.08


def _posterior_interval_band_trace(
    go: Any,
    *,
    x0: float,
    x1: float,
    y_range: tuple[float, float],
    name: str,
    color: str,
    show_legend: bool,
) -> Any:
    """Return a filled rectangle marking a credible interval.

    :param go: The ``plotly.graph_objects`` module.
    :param x0: Lower interval bound.
    :type x0: float
    :param x1: Upper interval bound.
    :type x1: float
    :param y_range: Panel y-axis range the band should span.
    :type y_range: tuple[float, float]
    :param name: Legend/trace name.
    :type name: str
    :param color: Fill colour.
    :type color: str
    :param show_legend: Whether this trace adds the legend entry.
    :type show_legend: bool
    :return: A Plotly Scatter trace.
    """
    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
        mode='lines',
        fill='toself',
        fillcolor=color,
        line=dict(color=color, width=0),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        hoverinfo='skip',
    )


def _posterior_reference_line_trace(
    go: Any,
    *,
    x_value: float,
    y_range: tuple[float, float],
    name: str,
    color: str,
    dash: str,
    show_legend: bool,
) -> Any:
    """Return a vertical reference line for a posterior marginal panel.

    :param go: The ``plotly.graph_objects`` module.
    :param x_value: Parameter value at which to draw the line.
    :type x_value: float
    :param y_range: Panel y-axis range the line should span.
    :type y_range: tuple[float, float]
    :param name: Legend/trace name.
    :type name: str
    :param color: Line colour.
    :type color: str
    :param dash: Plotly dash style (e.g. ``'dash'``, ``'dot'``).
    :type dash: str
    :param show_legend: Whether this trace adds the legend entry.
    :type show_legend: bool
    :return: A Plotly Scatter trace.
    """
    return go.Scatter(
        x=[x_value, x_value],
        y=[y_range[0], y_range[1]],
        mode='lines',
        line=dict(color=color, width=2, dash=dash),
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        hovertemplate=f'{name}: %{{x:.4f}}<extra></extra>',
    )


def plot_distribution(
    draws: np.ndarray,
    param_names: list[str],
    logp: np.ndarray | None = None,
    return_figure: bool = False,
    **kwargs,
) -> Any:
    """Plot marginal posterior distributions for each parameter.

    Each panel overlays, mirroring ``easydiffraction``'s
    ``project.display.posterior.distribution()``:

    * the posterior **histogram** (probability-density normalised),
    * a smooth **Gaussian-KDE marginal** density curve (requires ``scipy``;
      omitted if unavailable or the samples are degenerate),
    * a shaded **95% credible interval** (equal-tailed 2.5/97.5 percentiles),
    * a dashed **median** line, and
    * a dotted **best posterior sample** line when *logp* is supplied.

    When *return_figure* is ``True`` a Plotly ``Figure`` is returned.

    :param draws: Posterior samples, shape ``(n_samples, n_params)`` or
        ``(n_chains, n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param logp: Log-posterior values, shape ``(n_samples,)``. When given, the
        draw with the largest value marks the best posterior sample.
    :type logp: np.ndarray | None
    :param return_figure: Return a Plotly Figure instead of rendering inline.
    :type return_figure: bool
    :param kwargs: Additional keyword arguments (currently unused).
    :return: Plotly Figure when *return_figure* is ``True``, otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    # draws shape: (n_samples, n_params)

    best_index: int | None = None
    if logp is not None:
        logp = np.asarray(logp).reshape(-1)
        if logp.size == draws.shape[0] and np.any(np.isfinite(logp)):
            best_index = int(np.nanargmax(logp))

    if not return_figure:
        return None

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        warnings.warn(
            'The ``plotly`` library is required to build the distribution figure. Install it with ``pip install plotly``.',
            UserWarning,
            stacklevel=2,
        )
        return None

    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols)

    # Each trace type contributes a single shared legend entry.
    legend_shown = {
        'histogram': False,
        'marginal': False,
        'interval': False,
        'median': False,
        'best': False,
    }

    for i, name in enumerate(param_names):
        row = i // n_cols + 1
        col = i % n_cols + 1
        values = draws[:, i]

        density_curve = _posterior_density_curve(values)
        y_range = _posterior_marginal_y_range(values, density_curve)
        lower, upper = (float(v) for v in np.percentile(values, [2.5, 97.5]))

        # Credible-interval band first so it sits behind the density traces.
        if y_range is not None:
            fig.add_trace(
                _posterior_interval_band_trace(
                    go,
                    x0=lower,
                    x1=upper,
                    y_range=y_range,
                    name='95% credible interval',
                    color=_POSTERIOR_INTERVAL_95_FILL_COLOR,
                    show_legend=not legend_shown['interval'],
                ),
                row=row,
                col=col,
            )
            legend_shown['interval'] = True

        fig.add_trace(
            go.Histogram(
                x=values,
                histnorm='probability density',
                marker=dict(
                    color=_POSTERIOR_HISTOGRAM_FILL_COLOR,
                    line=dict(color=_POSTERIOR_HISTOGRAM_LINE_COLOR, width=1),
                ),
                opacity=0.82,
                nbinsx=40,
                name='Posterior histogram',
                legendgroup='histogram',
                showlegend=not legend_shown['histogram'],
                hovertemplate='sample=%{x:.4f}<br>density: %{y:.2f}<extra></extra>',
            ),
            row=row,
            col=col,
        )
        legend_shown['histogram'] = True

        if density_curve is not None:
            grid, density = density_curve
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=density,
                    mode='lines',
                    line=dict(color=_POSTERIOR_PAIR_MARGINAL_LINE_COLOR, width=2),
                    fill='tozeroy',
                    fillcolor=_POSTERIOR_PAIR_MARGINAL_FILL_COLOR,
                    name='Marginal density',
                    legendgroup='marginal',
                    showlegend=not legend_shown['marginal'],
                    hovertemplate=f'{name}: %{{x:.4f}}<br>density: %{{y:.4f}}<extra></extra>',
                ),
                row=row,
                col=col,
            )
            legend_shown['marginal'] = True

        if y_range is not None:
            fig.add_trace(
                _posterior_reference_line_trace(
                    go,
                    x_value=float(np.median(values)),
                    y_range=y_range,
                    name='Median',
                    color=_POSTERIOR_MEDIAN_LINE_COLOR,
                    dash='dash',
                    show_legend=not legend_shown['median'],
                ),
                row=row,
                col=col,
            )
            legend_shown['median'] = True
            if best_index is not None:
                fig.add_trace(
                    _posterior_reference_line_trace(
                        go,
                        x_value=float(values[best_index]),
                        y_range=y_range,
                        name='Best posterior sample',
                        color=_POSTERIOR_POINT_ESTIMATE_LINE_COLOR,
                        dash='dot',
                        show_legend=not legend_shown['best'],
                    ),
                    row=row,
                    col=col,
                )
                legend_shown['best'] = True
            fig.update_yaxes(range=list(y_range), row=row, col=col)

        if density_curve is not None:
            fig.update_xaxes(
                range=[float(density_curve[0][0]), float(density_curve[0][-1])],
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text=name, title_font=dict(size=11), row=row, col=col)
        fig.update_yaxes(title_text='Probability density', title_font=dict(size=11), row=row, col=col)

    fig.update_layout(
        height=max(300, 250 * n_rows),
        barmode='overlay',
        showlegend=True,
        legend=dict(font=dict(size=10)),
        plot_bgcolor='white',
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside')
    fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside')
    return fig


def credible_intervals(
    draws: np.ndarray,
    param_names: list[str],
    alpha: float = 0.95,
) -> dict:
    """Compute equal-tailed credible intervals for each parameter.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param alpha: Credible interval width (e.g. 0.95 for 95%).
    :type alpha: float
    :return: Dictionary mapping parameter name to ``(lower, upper)``.
    :rtype: dict
    """
    draws = np.asarray(draws)
    tail = (1.0 - alpha) / 2.0
    lo_pct = tail * 100
    hi_pct = (1.0 - tail) * 100
    result = {}
    for i, name in enumerate(param_names):
        col = draws[:, i]
        lo, hi = np.percentile(col, [lo_pct, hi_pct])
        result[name] = (float(lo), float(hi))
    return result


def _save_parameter_state(model) -> dict:
    """Save the current values and errors of all free parameters in a model.

    :param model: A reflectometry model with ``get_parameters()``.
    :return: Dictionary mapping ``unique_name`` to ``(value, error)``.
    :rtype: dict
    """
    state = {}
    for param in model.get_parameters():
        # Dependent parameters (constraints, `Model.total_thickness`) are derived
        # from the others: they cannot be written back, and restoring the
        # parameters they follow already restores them.
        if param.independent:
            state[param.unique_name] = (param.value, param.error)
    return state


def _restore_parameter_state(model, state: dict) -> None:
    """Restore parameter values and errors from a saved state.

    :param model: A reflectometry model with ``get_parameters()``.
    :param state: Dictionary mapping ``unique_name`` to ``(value, error)``.
    """
    for param in model.get_parameters():
        if param.unique_name in state:
            param.value = state[param.unique_name][0]
            param.error = state[param.unique_name][1]


def _apply_draw(model, draws: np.ndarray, param_names: list[str], row: int) -> None:
    """Apply a single posterior draw to the model parameters.

    Parameter lookup uses ``unique_name``, matching the BUMPS names after
    removing the minimizer prefix, which avoids collisions when repeated models
    or multi-contrast fits contain similarly named parameters.

    :param model: A reflectometry model with ``get_parameters()``.
    :param draws: Posterior samples array.
    :param param_names: Parameter names matching the columns of ``draws``.
    :param row: Index of the draw to apply.
    """
    param_lookup = {p.unique_name: p for p in model.get_parameters()}
    for j, name in enumerate(param_names):
        if name in param_lookup:
            param_lookup[name].value = float(draws[row, j])


def posterior_predictive_reflectivity(
    draws: np.ndarray,
    param_names: list[str],
    model,
    q_values: np.ndarray,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the posterior predictive reflectivity with credible intervals.

    Parameter values and errors are saved before applying any posterior draw
    and restored in a ``finally`` block, so the model is not left mutated.

    :param draws: Posterior samples, shape ``(n_samples_posterior, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names matching the columns of ``draws``.
    :type param_names: list[str]
    :param model: A reflectometry model with ``interface.fit_func``.
    :param q_values: Q values at which to evaluate reflectivity.
    :type q_values: np.ndarray
    :param n_samples: Number of posterior draws to use (last ``n_samples``).
    :type n_samples: int
    :return: Tuple of ``(median, lower_95, upper_95)`` reflectivity arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    draws = np.asarray(draws)
    q_values = np.asarray(q_values)

    n_total = draws.shape[0]
    n_use = min(n_samples, n_total)
    sample_indices = range(n_total - n_use, n_total)

    saved_state = _save_parameter_state(model)
    try:
        reflectivity_samples = []
        for i in sample_indices:
            _apply_draw(model, draws, param_names, i)
            r_calc = model.interface.fit_func(q_values, model.unique_name)
            reflectivity_samples.append(np.asarray(r_calc))
    finally:
        _restore_parameter_state(model, saved_state)

    reflectivity_samples = np.array(reflectivity_samples)
    median = np.median(reflectivity_samples, axis=0)
    lower = np.percentile(reflectivity_samples, 2.5, axis=0)
    upper = np.percentile(reflectivity_samples, 97.5, axis=0)
    return median, lower, upper


def posterior_predictive_sld_profile(
    draws: np.ndarray,
    param_names: list[str],
    model,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the posterior predictive SLD profile with credible intervals.

    Parameter values and errors are saved before applying any posterior draw
    and restored in a ``finally`` block, so the model is not left mutated.

    :param draws: Posterior samples, shape ``(n_samples_posterior, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names matching the columns of ``draws``.
    :type param_names: list[str]
    :param model: A reflectometry model with ``interface.sld_profile``.
    :param n_samples: Number of posterior draws to use (last ``n_samples``).
    :type n_samples: int
    :return: Tuple of ``(z, median, lower_95, upper_95)`` SLD profile arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    draws = np.asarray(draws)

    n_total = draws.shape[0]
    n_use = min(n_samples, n_total)
    sample_indices = range(n_total - n_use, n_total)

    saved_state = _save_parameter_state(model)
    try:
        sld_samples = []
        z_shared = None
        for i in sample_indices:
            _apply_draw(model, draws, param_names, i)
            z, sld = model.interface.sld_profile(model.unique_name)
            if z_shared is None:
                z_shared = np.asarray(z)
            sld_samples.append(np.asarray(sld))
    finally:
        _restore_parameter_state(model, saved_state)

    sld_samples = np.array(sld_samples)
    median = np.median(sld_samples, axis=0)
    lower = np.percentile(sld_samples, 2.5, axis=0)
    upper = np.percentile(sld_samples, 97.5, axis=0)
    return z_shared, median, lower, upper


# ===================================================================
# Persistence helpers — save / load a posterior trace to / from disk
# ===================================================================

_SIDECAR_SCHEMA_VERSION = 1


def _easyreflectometry_version() -> str:
    """Return the installed easyreflectometry version string."""
    try:
        from importlib.metadata import version as _v

        return _v('easyreflectometry')
    except Exception:
        return 'unknown'


def _data_fingerprint(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    w_list: list[np.ndarray],
) -> str | None:
    """Return a SHA-256 hex digest of concatenated (x|y|weights), or None."""
    try:
        h = hashlib.sha256()
        for arr in list(x_list) + list(y_list) + list(w_list):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()
    except Exception:
        return None


def save_posterior(results: 'PosteriorResults', path: str) -> None:
    """Persist a sampling trace to disk using BUMPS' native state files.

    Writes ``<path>-*.mc`` (BUMPS ``save_state`` output) plus a sidecar
    ``<path>.params.json`` holding parameter names and metadata so that
    :func:`load_posterior` can reconstruct a fully populated
    :class:`PosteriorResults` without re-deriving names from the model.

    Note that ``save_state`` writes **multiple** files (one per DREAM
    component: chain, point, and stats).  The ``path`` argument is a
    prefix; the actual files will be ``<path>-chain.mc``,
    ``<path>-point.mc``, and ``<path>-stats.mc``.

    :param results: The posterior results to persist.  Must have a
        non-``None`` ``sampler_state``.
    :type results: PosteriorResults
    :param path: File path prefix.  BUMPS appends its own suffixes.
    :type path: str
    :raises ValueError: If ``results.sampler_state`` is ``None``.
    :raises TypeError: If ``results.sampler_state`` is not a BUMPS
        ``MCMCDraw`` object.
    """
    from bumps.dream.state import MCMCDraw
    from bumps.dream.state import save_state

    if results.sampler_state is None:
        raise ValueError(
            'This PosteriorResults has no sampler_state, so the chain '
            'cannot be saved or resumed. Re-run sample() and wrap the '
            "returned dict's 'state' value into PosteriorResults."
        )
    if not isinstance(results.sampler_state, MCMCDraw):
        raise TypeError(
            f'sampler_state must be a BUMPS MCMCDraw object, got '
            f'{type(results.sampler_state).__name__}. Only BUMPS DREAM '
            'traces can be persisted with save_posterior.'
        )

    save_state(results.sampler_state, path)

    # Write the sidecar JSON
    sidecar = {
        'schema_version': _SIDECAR_SCHEMA_VERSION,
        'param_names': results.param_names,
        'easyreflectometry_version': _easyreflectometry_version(),
    }
    with open(f'{path}.params.json', 'w') as f:
        json.dump(sidecar, f, indent=2)


def load_posterior(path: str, skip: int = 0) -> 'PosteriorResults':
    """Reload a trace saved by :func:`save_posterior` into a
    :class:`PosteriorResults`.

    The returned object's ``sampler_state`` can be fed back into the core
    ``Sampler`` (via ``Sampler.load_state(...)`` / ``Sampler.extend(...)``)
    to extend the chain.

    :param path: File path prefix used in :func:`save_posterior`.
    :type path: str
    :param skip: Discard the first ``skip`` saved generations on load,
        forwarded to ``bumps.dream.state.load_state(path, skip=skip)``.
        Useful for trimming additional burn-in without re-sampling.
    :type skip: int
    :return: A fully populated :class:`PosteriorResults`.
    :rtype: PosteriorResults
    """
    from bumps.dream.state import load_state

    state = load_state(path, skip=skip)
    _draw = state.draw()
    draws = _draw.points
    logp = _draw.logp  # .logp is on the Draw object, NOT state.logp

    # Restore param_names: prefer the sidecar; fall back to state.labels
    param_names: list[str] | None = None
    try:
        with open(f'{path}.params.json', 'r') as f:
            sidecar = json.load(f)
        if sidecar.get('schema_version') == _SIDECAR_SCHEMA_VERSION:
            param_names = sidecar.get('param_names')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    if param_names is None:
        # Fallback: strip BUMPS 'p' prefix from state.labels
        param_names = [
            lbl[len(MINIMIZER_PARAMETER_PREFIX) :] if lbl.startswith(MINIMIZER_PARAMETER_PREFIX) else lbl
            for lbl in state.labels
        ]

    return PosteriorResults(
        draws=draws,
        param_names=param_names,
        logp=logp,
        sampler_state=state,
    )
