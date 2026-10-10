# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import contextlib
import copy
import dataclasses
import datetime
import functools
import warnings
import weakref
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

import numpy as np
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting import Sampler
from easyscience.fitting.multi_fitter import MultiFitter as EasyScienceMultiFitter
from easyscience.variable import Parameter

from easyreflectometry._bumps_constraints import NON_BUMPS_ERROR
from easyreflectometry._bumps_constraints import applied as _constraints_applied
from easyreflectometry._bumps_constraints import is_applied as _constraints_active
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fit_settings import FitSettings
from easyreflectometry.fit_settings import normalize_objective as _validate_objective
from easyreflectometry.fit_settings import requires_finite_bounds
from easyreflectometry.model import Model
from easyreflectometry.model import Pointwise
from easyreflectometry.model import ResolutionFunction

_EPS = 1e-30


class _ConstrainedEasyScienceMultiFitter(EasyScienceMultiFitter):
    """EasyScience ``MultiFitter`` whose raw ``fit`` honours inequality constraints.

    ``fit`` is a read-only property on the base class (it builds a fresh
    callable per access), so the interception lives in an override rather
    than a monkey-patch. An explicit ``constraints_factory`` keyword is
    consumed here and enforced by this library — the core takes no such
    argument and would swallow it through ``**kwargs``. Without one, a call
    running inside an active :meth:`MultiFitter._constraints` block passes
    through untouched, leaving the outer factory in force.
    """

    #: Weak reference to the owning :class:`MultiFitter` (weak to avoid a
    #: reference cycle); ``None`` disables the interception.
    _constraints_owner = None

    @property
    def fit(self) -> Callable:
        original = EasyScienceMultiFitter.fit.fget(self)
        owner = self._constraints_owner() if self._constraints_owner is not None else None
        if owner is None:
            return original

        @functools.wraps(original)
        def fit_with_constraints(*args, **kwargs):
            explicit = kwargs.pop('constraints_factory', None)
            # An explicit factory wins over an outer block; without one, an
            # outer block has already attached what it resolved.
            if explicit is None and _constraints_active():
                return original(*args, **kwargs)
            with owner._constraints(explicit, fitter=self):
                return original(*args, **kwargs)

        return fit_with_constraints


def _prepare_fit_arrays(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    variances: np.ndarray,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prepare x, y_eff, and weights arrays for fitting based on the objective mode.

    For ``legacy_mask``, zero-variance points are removed from all arrays.
    For ``hybrid``, valid-variance points use standard WLS while zero-variance
    points use Mighell-transformed y and weights.
    For ``mighell``, all points use the Mighell transform.

    Note: ``variances`` here means σ² (the scipp convention), not σ.

    Parameters
    ----------
    x_vals : np.ndarray
        Independent variable values.
    y_vals : np.ndarray
        Observed dependent variable values.
    variances : np.ndarray
        Variance (σ²) of each observed point.
    objective : str
        One of 'legacy_mask', 'hybrid', 'mighell'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        Tuple of (x_out, y_eff, weights, stats) where stats is a dict
        with keys 'valid', 'mighell_substituted', 'masked'.
    """
    n = len(y_vals)
    zero_mask = variances <= 0.0
    n_zero = int(np.sum(zero_mask))
    n_valid = n - n_zero

    if objective == 'legacy_mask':
        valid = ~zero_mask
        x_out = x_vals[valid]
        y_eff = y_vals[valid]
        if n_valid > 0:
            weights = 1.0 / np.sqrt(variances[valid])
        else:
            weights = np.array([])
        stats = {
            'valid': n_valid,
            'mighell_substituted': 0,
            'masked': n_zero,
            'transformed_all_points': False,
        }
        return x_out, y_eff, weights, stats

    # hybrid or mighell
    y_eff = np.copy(y_vals)
    sigma = np.empty(n)

    if objective == 'mighell':
        apply_mighell = np.ones(n, dtype=bool)
    else:
        # hybrid: apply Mighell only to zero-variance points
        apply_mighell = zero_mask

    # Standard WLS for non-Mighell points
    standard = ~apply_mighell
    if np.any(standard):
        sigma[standard] = np.sqrt(variances[standard])

    # Mighell transform for selected points
    if np.any(apply_mighell):
        y_m = y_vals[apply_mighell]
        delta = np.minimum(y_m, 1.0)
        y_eff[apply_mighell] = y_m + delta
        sigma[apply_mighell] = np.sqrt(np.maximum(y_m + 1.0, _EPS))

    weights = 1.0 / sigma
    n_mighell = int(np.sum(apply_mighell))
    stats = {
        'valid': n - n_mighell,
        'mighell_substituted': n_mighell,
        'masked': 0,
        'transformed_all_points': bool(objective == 'mighell'),
    }
    return x_vals, y_eff, weights, stats


def _compute_weighted_chi2(y_obs: np.ndarray, y_calc: np.ndarray, sigma: np.ndarray) -> float:
    """Return weighted chi-square for finite, strictly positive uncertainties."""
    valid = np.isfinite(y_obs) & np.isfinite(y_calc) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(valid):
        return 0.0
    residual = (y_obs[valid] - y_calc[valid]) / sigma[valid]
    return float(np.sum(residual**2))


def _compute_reduced_chi2(chi2: float, n_points: int, n_params: int) -> float | None:
    """Return reduced chi-square or None when degrees of freedom are not positive."""
    dof = int(n_points) - int(n_params)
    if dof <= 0:
        return None
    return float(chi2 / dof)


def _fit_result_reduced_chi(result: FitResults, n_points: int | None = None) -> float:
    """Return reduced chi-square from either supported FitResults attribute name."""
    for attribute in ('reduced_chi', 'reduced_chi2'):
        value = getattr(result, attribute, None)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    if n_points is not None:
        reduced_chi = _compute_reduced_chi2(float(result.chi2), n_points, result.n_pars)
        if reduced_chi is not None:
            return reduced_chi
    raise AttributeError('FitResults object has neither reduced_chi nor reduced_chi2')


def _bind_fit_func(func: Callable, unique_name: str) -> Callable:
    """Bind a model's fit function to its ``unique_name`` for ``EasyScienceMultiFitter``.

    ``EasyScienceMultiFitter`` calls each fit function positionally as
    ``func(x, *extra_args)``; the model's ``interface.fit_func`` expects its
    ``unique_name`` as that extra positional argument, so it has to be closed
    over here rather than passed through the fitter's call signature.
    """

    def wrapped(*args, **kwargs):
        return func(*args, unique_name, **kwargs)

    return wrapped


def _resolution_for_fitted_points(
    resolution_function: ResolutionFunction | None,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    variances: np.ndarray,
    objective: str,
) -> ResolutionFunction | None:
    """The dataset resolution restricted to the points :func:`_prepare_fit_arrays` keeps.

    ``legacy_mask`` drops zero-variance points, so the fit evaluates the model
    on a subset of ``x``. A merged dataset can hold the same q twice with
    different widths; looked up by q alone, the subset would lose which width
    belongs to which point (:meth:`Pointwise.smearing` only keeps per-point
    widths when evaluated on exactly its stored q). The widths are therefore
    evaluated on the full grid and selected together with the points.
    """
    if resolution_function is None or objective != 'legacy_mask':
        return resolution_function
    keep = ~(variances <= 0.0)
    if np.all(keep):
        return resolution_function
    sigma = np.asarray(resolution_function.smearing(x_vals), dtype=float)
    return Pointwise([x_vals[keep], y_vals[keep], np.square(sigma[keep])])


def _emit_array_prep_warnings(stats: dict, y_vals: np.ndarray, label: str, *, action: str = 'fitting', extra: str = '') -> None:
    """Warn about zero-variance handling applied by :func:`_prepare_fit_arrays`.

    Parameters
    ----------
    stats : dict
        The ``stats`` dict returned by :func:`_prepare_fit_arrays`.
    y_vals : np.ndarray
        The original (pre-transform) y values, used for the "all points" count.
    label : str
        Identifies what was fitted/sampled, e.g. ``'reflectivity 1'`` or
        ``'channel pp'``.
    action : str, optional
        Verb describing the operation, e.g. ``'fitting'`` or ``'sampling'``. By default, 'fitting'.
    extra : str, optional
        Extra sentence(s) appended to the Mighell-related warnings (e.g. a
        likelihood-validity caveat for MCMC). By default, ''.
    """
    if stats['masked'] > 0:
        warnings.warn(
            f'Masked {stats["masked"]} data point(s) in {label} due to zero variance during {action}.',
            UserWarning,
        )
    if stats.get('transformed_all_points'):
        warnings.warn(
            f'Applied Mighell transform to all {len(y_vals)} point(s) in {label} during {action}.{extra}',
            UserWarning,
        )
    elif stats['mighell_substituted'] > 0:
        warnings.warn(
            f'Applied Mighell substitution to {stats["mighell_substituted"]} '
            f'zero-variance point(s) in {label} during {action}.{extra}',
            UserWarning,
        )


def _classical_metrics_for(original: dict, model_curve: np.ndarray, result: FitResults, n_points: int | None = None) -> dict:
    """The classical (positive-variance-only) and objective-space chi-square of one dataset.

    Reduced values are not given per dataset: their degrees of freedom belong
    to the pooled fit (see :class:`FitRun` and the ``MultiFitter`` properties).

    Parameters
    ----------
    original : dict
        Dict with keys ``'y'`` and ``'variances'`` holding the un-transformed
        observed values and their variances (σ²).
    model_curve : np.ndarray
        Model evaluated at the original x values.
    result : FitResults
        The minimizer's result for this dataset/channel.
    n_points : int | None, optional
        Number of points actually fitted. If ``None``, derived from ``result.x``.

    Returns
    -------
    dict
        Keys ``'classical_chi2'``, ``'n_classical_points'``, ``'objective_chi2'``
        and ``'n_points'`` (the points fitted).
    """
    sigma_classical = np.sqrt(np.clip(original['variances'], 0.0, None))
    return {
        'classical_chi2': _compute_weighted_chi2(original['y'], model_curve, sigma_classical),
        'n_classical_points': int(np.sum(original['variances'] > 0.0)),
        'objective_chi2': float(result.chi2),
        'n_points': int(np.size(result.x) if n_points is None else n_points),
    }


def _free_roots(models: list) -> list[Parameter]:
    """Free parameters that parameters of `models` follow, directly or through other
    dependencies, but that no model in `models` owns: a tie to a model left out of
    the fit. Leaving them out of the fit would hold them fixed."""
    owned = {id(parameter) for model in models for parameter in model.get_all_parameters()}
    pending = [parameter for model in models for parameter in model.get_all_parameters() if not parameter.independent]
    roots, seen = {}, set()
    while pending:
        parameter = pending.pop()
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        for leader in (parameter.dependency_map or {}).values():
            if not isinstance(leader, Parameter):
                continue
            if not leader.independent:
                pending.append(leader)
            elif not leader.fixed and id(leader) not in owned:
                roots[id(leader)] = leader
    return list(roots.values())


class FitScopeError(ValueError):
    """The experiments asked for cannot be fitted together (none selected, or one has no model)."""


class FitPreconditionError(ValueError):
    """A fit that cannot start: the minimizer needs finite bounds some free parameters lack.

    Attributes
    ----------
    parameters : list
        The offending parameters, each once.
    """

    def __init__(self, message: str, parameters: list):
        super().__init__(message)
        self.parameters = parameters


@dataclass
class _FitInput:
    """One dataset (or spin channel) to fit, before zero-variance handling."""

    x: np.ndarray
    y: np.ndarray
    variances: np.ndarray
    model: Model
    label: str
    channel: Any = None
    #: The dataset's own resolution; None smears with the model's.
    resolution_function: ResolutionFunction | None = None
    #: A stored fit function to use when neither a channel nor a resolution
    #: of its own applies (the plain ``DataGroup`` and single-dataset fits).
    fit_func: Callable | None = None


@dataclass
class PreparedFit:
    """Everything one fit run needs, fixed when it is prepared.

    Built by :meth:`MultiFitter.prepare`, the one place that applies the
    zero-variance objective, decides which resolution each fit function is
    bound to, and configures the core fitter. Settings edited afterwards do not
    reach it.

    Attributes
    ----------
    settings : FitSettings | None
        Snapshot of the settings; None for a fitter without settings.
    objective : str
        The resolved zero-variance objective.
    channels : list
        Spin channel per dataset (None when unpolarized); keys polarized results.
    original : list[dict]
        ``x``, ``y``, ``variances`` as measured, all points; for the classical metrics.
    fitted : list[dict]
        ``x``, ``y``, ``weights`` handed to the engine.
    stats : list[dict]
        Zero-variance handling counts per dataset.
    curve_funcs : list[Callable]
        Full-resolution evaluators, for model curves and classical metrics.
    fit_funcs : list[Callable]
        Evaluators bound to the fitted points' resolution, as executed.
    core_fitter : EasyScienceMultiFitter
        The fitter that is executed.
    fit_kwargs : dict
        Method-specific option keyword arguments for the engine.
    added_roots : list[Parameter]
        Free parameters the fitted models follow but do not own (a tie to a
        model left out of the fit), varied so the tie does not hold them fixed.
    labels : list[tuple]
        ``(experiment key, experiment name, channel)`` per dataset, set by
        :meth:`Project.prepare_fit`; empty otherwise.
    skipped : list[str]
        Experiments left out because they have no model (``skip_invalid``).
    """

    settings: FitSettings | None
    objective: str
    channels: list
    original: list[dict]
    fitted: list[dict]
    stats: list[dict]
    curve_funcs: list[Callable]
    fit_funcs: list[Callable]
    core_fitter: EasyScienceMultiFitter
    fit_kwargs: dict
    added_roots: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    @property
    def x(self) -> list[np.ndarray]:
        return [arrays['x'] for arrays in self.fitted]

    @property
    def y(self) -> list[np.ndarray]:
        return [arrays['y'] for arrays in self.fitted]

    @property
    def weights(self) -> list[np.ndarray]:
        return [arrays['weights'] for arrays in self.fitted]

    def call_kwargs(self, **kwargs) -> dict:
        """The engine keyword arguments of this run merged with per-call ones (which win)."""
        merged = copy.deepcopy(self.fit_kwargs)
        explicit = kwargs.pop('minimizer_kwargs', None)
        if explicit:
            merged['minimizer_kwargs'] = {**merged.get('minimizer_kwargs', {}), **explicit}
        merged.update(kwargs)
        return merged

    def execute(self, **kwargs) -> list[FitResults]:
        """Run the fit synchronously; ``kwargs`` go to the core fitter's ``fit``."""
        return self.core_fitter.fit(x=self.x, y=self.y, weights=self.weights, **self.call_kwargs(**kwargs))

    def finalize(self, results: list[FitResults]) -> list[dict]:
        """Objective and classical metrics of a finished run, one dict per dataset.

        The classical metrics are computed from the measured arrays and the
        full-resolution curves, not from the transformed fitted arrays.
        """
        return [
            _classical_metrics_for(original, curve(original['x']), result, n_points=len(fitted['x']))
            for original, fitted, curve, result in zip(self.original, self.fitted, self.curve_funcs, results)
        ]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


@dataclass(frozen=True)
class FitRun:
    """The record of one completed (or failed, or cancelled) fit, kept by the project.

    It belongs to the run, not to a fitter, so it outlives a change of the
    current model; and it names its datasets, so it can be read after the
    experiments were rearranged.

    Statistics (only for a completed run):

    - ``pooled``: ``objective_chi2`` (the engine's, on the transformed fitted
      points), ``objective_n_points``, ``objective_dof`` (points minus
      :attr:`n_free_parameters`), ``objective_reduced_chi2``; and the same
      four ``classical_*`` values over the measured points with a positive
      variance. A reduced value is None when its dof is not positive.
    - ``per_dataset``, in :attr:`inputs` order: ``objective_chi2``,
      ``objective_n_points``, ``objective_chi2_per_point``, ``classical_chi2``,
      ``classical_n_points``, ``classical_chi2_per_point`` (None without
      points) and ``share_of_objective`` (None when the pooled chi-square is
      zero). The per-dataset objective values add up to the pooled one: the
      engine's per-dataset results are disjoint slices of the pooled residuals.
    """

    status: str
    completed_at: str
    minimizer: str
    objective: str
    #: ``(experiment key, experiment name, channel)`` per fitted dataset; the key
    #: is None for data that is not (or no longer) a project experiment.
    inputs: tuple
    n_free_parameters: int = 0
    pooled: dict = field(default_factory=dict)
    per_dataset: tuple = ()

    @classmethod
    def from_results(
        cls,
        prepared: PreparedFit,
        results: list[FitResults] | None,
        status: str,
        minimizer: str,
        metrics: list[dict] | None = None,
    ) -> 'FitRun':
        """The record of a run; `metrics` is ``prepared.finalize(results)`` when already computed."""
        inputs = tuple(prepared.labels) or tuple((None, f'dataset {index}', None) for index in range(len(prepared.fitted)))
        completed_at = datetime.datetime.now().isoformat(timespec='seconds')
        if status != 'completed' or not results:
            return cls(status, completed_at, minimizer, prepared.objective, inputs)
        n_free = int(results[0].n_pars)
        per_dataset = [
            {
                'objective_chi2': metric['objective_chi2'],
                'objective_n_points': metric['n_points'],
                'objective_chi2_per_point': _ratio(metric['objective_chi2'], metric['n_points']),
                'classical_chi2': metric['classical_chi2'],
                'classical_n_points': metric['n_classical_points'],
                'classical_chi2_per_point': _ratio(metric['classical_chi2'], metric['n_classical_points']),
            }
            for metric in (metrics if metrics is not None else prepared.finalize(results))
        ]
        pooled = {}
        for kind, points in (('objective', 'objective_n_points'), ('classical', 'classical_n_points')):
            chi2 = float(sum(entry[f'{kind}_chi2'] for entry in per_dataset))
            n_points = int(sum(entry[points] for entry in per_dataset))
            pooled[f'{kind}_chi2'] = chi2
            pooled[f'{kind}_n_points'] = n_points
            pooled[f'{kind}_dof'] = n_points - n_free
            pooled[f'{kind}_reduced_chi2'] = _ratio(chi2, n_points - n_free)
        for entry in per_dataset:
            entry['share_of_objective'] = _ratio(entry['objective_chi2'], pooled['objective_chi2'])
        return cls(status, completed_at, minimizer, prepared.objective, inputs, n_free, pooled, tuple(per_dataset))

    def remapped(self, keys: dict[int, int]) -> 'FitRun':
        """This record after experiments were re-keyed (old key -> new key; a removed one -> None)."""
        inputs = tuple((None if key is None else keys.get(key), name, channel) for key, name, channel in self.inputs)
        return dataclasses.replace(self, inputs=inputs)


class MultiFitter:
    def __init__(self, *args: Model, objective: str = 'hybrid'):
        r"""A convenience class for the :py:class:`easyscience.Fitting.Fitting`
        which will populate the :py:class:`sc.DataGroup` appropriately
        after the fitting is performed.

        Parameters
        ----------
        *args : Model
            Reflectometry model(s).
        objective : str, optional
            Zero-variance handling strategy. One of
            ``'hybrid'`` (default, Mighell for zero-variance, WLS otherwise),
            ``'mighell'`` (Mighell transform for all points),
            ``'legacy_mask'`` (drop zero-variance points),
            ``'auto'`` (alias for ``'hybrid'``). By default, 'hybrid'.
        """

        self._fit_func = [_bind_fit_func(m.interface.fit_func, m.unique_name) for m in args]
        self._models = args
        self.easy_science_multi_fitter = self._build_easy_science_fitter(args, self._fit_func)
        self._fit_results: list[FitResults] | None = None
        self._classical_fit_metrics: list[dict] | None = None
        self._objective = _validate_objective(objective)
        self._sampler: Sampler | None = None
        # Set by `for_experiments`: the datasets the fit functions correspond
        # to, and the spin channel each one is evaluated on (None = unpolarized).
        self.fit_datasets: list[DataSet1D] = []
        self.fit_channels: list[Any] = []
        # Optional zero-argument callable returning the ``constraints_factory``
        # for the next fit (or ``None``). ``Project.fitter`` binds it to
        # ``Project.build_constraints_factory`` so inequality constraints
        # registered on the project are applied without passing them
        # explicitly; an explicit ``constraints_factory=`` argument wins.
        self.constraints_factory_provider: Callable[[], Callable | None] | None = None
        # The settings runs are prepared from (a snapshot is taken per run).
        # None: the stored core fitter's own minimizer, tolerance and budget.
        self.settings: FitSettings | None = None

    def _build_easy_science_fitter(self, models, fit_funcs, keep_owner: bool = False) -> EasyScienceMultiFitter:
        """Build the EasyScience fitter, with its raw ``fit`` honouring constraints.

        :meth:`for_experiments` documents that the caller drives
        ``easy_science_multi_fitter.fit(...)`` directly (e.g. a GUI worker
        thread), which would bypass :meth:`_constraints` and silently fit an
        unconstrained problem. The returned fitter routes that path through
        the constraints machinery: :attr:`constraints_factory_provider` is
        resolved at call time (and in the calling thread — the shim's context
        variable is thread-local), and non-BUMPS engines are rejected rather
        than silently dropping the constraints.

        The stored fitter refers to its owner weakly (the owner holds it). A
        prepared run's fitter (``keep_owner``) holds it strongly: the caller may
        keep only the run, and the constraints need the owner at execution.
        """
        fitter = _ConstrainedEasyScienceMultiFitter(models, fit_funcs)
        fitter._constraints_owner = (lambda: self) if keep_owner else weakref.ref(self)
        return fitter

    def _resolve_constraints_factory(self, explicit: Callable | None) -> Callable | None:
        if explicit is not None:
            return explicit
        if self.constraints_factory_provider is not None:
            return self.constraints_factory_provider()
        return None

    @contextlib.contextmanager
    def _constraints(self, explicit: Callable | None, fitter: EasyScienceMultiFitter | None = None):
        """Make any inequality constraints active for the duration of the block.

        The penalties are attached to the BUMPS problem as it is built (see
        :mod:`easyreflectometry._bumps_constraints`); nothing is handed to the
        core as a keyword. Engines that cannot enforce the constraints are
        rejected rather than left to fit an unconstrained problem.

        Parameters
        ----------
        explicit : Callable | None
            Factory passed to the fit call, or None to use
            :attr:`constraints_factory_provider`.
        fitter : EasyScienceMultiFitter | None, optional
            The fitter whose minimizer is about to run, when it is not this
            one's — ``fit_polarized`` builds its own. By default, None.
        """
        factory = self._resolve_constraints_factory(explicit)
        if factory is not None:
            minimizer = (fitter or self.easy_science_multi_fitter).minimizer
            package = getattr(minimizer, 'package', None)
            if package != 'bumps':
                raise ValueError(NON_BUMPS_ERROR.format(package=package))
        with _constraints_applied(factory):
            yield

    @staticmethod
    def _keep_constraints_on_extend(sampler: Sampler, factory: Callable | None) -> None:
        """Re-enter the constraints context around ``sampler.extend()``.

        The factory is current only for the duration of the sampling call that
        built the problem; without this a continued chain would silently sample
        an unpenalised posterior.
        """
        if factory is None:
            return
        original = sampler.extend

        @functools.wraps(original)
        def extend(*args, **kwargs):
            with _constraints_applied(factory):
                return original(*args, **kwargs)

        sampler.extend = extend

    @classmethod
    def for_experiments(
        cls,
        experiments: list[DataSet1D | PolarizedDataSet],
        objective: str = 'hybrid',
        constraints_factory_provider: Callable[[], Callable | None] | None = None,
    ) -> 'MultiFitter':
        """Build a fitter for a mixed list of unpolarized and polarized experiments.

        Every experiment contributes one fit function per dataset: an ordinary
        experiment one, a polarized experiment one per measured spin channel,
        each evaluating that channel's spin cross-section against the single
        model the channels share. Structural parameters are therefore common to
        all channels and the magnetic ones are constrained by all of them at
        once, exactly as in :meth:`fit_polarized` — but here several experiments
        (and several models) can be fitted together, which is what an
        application's "fit everything that is loaded" action needs.

        The resulting fitter is *not* run. :meth:`prepare` (with no inputs)
        prepares a run over :attr:`fit_datasets`, which a GUI can then execute
        from a worker thread. Each fit function is bound to its dataset's own
        resolution. Inequality constraints are resolved at execution through
        :attr:`constraints_factory_provider` — pass ``constraints_factory_provider``
        (e.g. ``project.build_constraints_factory``) or set the attribute
        before fitting; with none set, no inequality constraints are enforced.

        The fitter starts from the core defaults (minimizer, tolerance, budget)
        unless :attr:`settings` is set, as :meth:`Project.prepare_fit` does.

        Parameters
        ----------
        experiments : list[DataSet1D | PolarizedDataSet]
            The loaded experiments, in the order they should be fitted.
        objective : str, optional
            Zero-variance handling strategy, see :meth:`__init__`. By default, 'hybrid'.
        constraints_factory_provider : Callable[[], Callable | None] | None, optional
            Zero-argument callable returning the ``constraints_factory`` for
            the next fit, typically ``project.build_constraints_factory``;
            stored as :attr:`constraints_factory_provider`. By default, None
            (no inequality constraints are applied).

        Returns
        -------
        MultiFitter
            Fitter whose ``easy_science_multi_fitter`` has one fit function per
            entry of ``fit_datasets``, with ``fit_channels`` holding the
            matching :class:`PolarizationChannel` (``None`` when unpolarized).
        """
        if not experiments:
            raise ValueError('At least one experiment is required to build a fitter.')

        models: list[Model] = []
        datasets: list[DataSet1D] = []
        channels: list[Any] = []
        for experiment in experiments:
            model = experiment.model
            if model is None:
                raise ValueError(f"Experiment '{getattr(experiment, 'name', experiment)}' has no model to fit.")
            # `in` would compare models by value; identity is what matters here.
            if not any(model is known for known in models):
                models.append(model)
            experiment_channels = getattr(experiment, 'available_channels', None)
            if experiment_channels is None:
                datasets.append(experiment)
                channels.append(None)
                continue
            for channel in experiment_channels:
                datasets.append(experiment[channel])
                channels.append(channel)

        fitter = cls(*models, objective=objective)
        fitter.fit_datasets = datasets
        fitter.fit_channels = channels
        # Bound with every point's resolution; `prepare` rebinds for a subset.
        fitter._fit_func = [fitter._bind(item, item.resolution_function) for item in fitter._experiment_inputs()]
        fitter.easy_science_multi_fitter = fitter._build_easy_science_fitter(models, fitter._fit_func)
        fitter.constraints_factory_provider = constraints_factory_provider
        return fitter

    def _experiment_inputs(self) -> list[_FitInput]:
        """One input per entry of :attr:`fit_datasets`."""
        inputs = []
        for dataset, channel in zip(self.fit_datasets, self.fit_channels):
            name = getattr(dataset, 'name', None) or 'dataset'
            inputs.append(
                _FitInput(
                    x=np.asarray(dataset.x),
                    y=np.asarray(dataset.y),
                    variances=np.asarray(dataset.ye),
                    model=dataset.model,
                    label=name if channel is None else f'{name} channel {channel.value}',
                    channel=channel,
                    resolution_function=getattr(dataset, 'resolution_function', None),
                )
            )
        return inputs

    def _datagroup_inputs(self, data: sc.DataGroup) -> tuple[list[str], list[_FitInput]]:
        """Inputs for the reflectivity curves of a ``DataGroup``; curve *k* belongs to model *k*."""
        refl_nums = [k[3:] for k in data['coords'].keys() if k.startswith('Qz_')]
        inputs = [
            _FitInput(
                x=data['coords'][f'Qz_{i}'].values,
                y=data['data'][f'R_{i}'].values,
                variances=data['data'][f'R_{i}'].variances,
                model=self._models[index],
                label=f'reflectivity {i}',
                fit_func=self._fit_func[index],
            )
            for index, i in enumerate(refl_nums)
        ]
        return refl_nums, inputs

    @staticmethod
    def _bind(item: _FitInput, resolution_function: ResolutionFunction | None = None) -> Callable:
        """The fit function of one input, bound to ``resolution_function``.

        With neither a channel nor a resolution of its own, an input carrying a
        stored fit function uses it, so a plain fit reuses the stored core fitter.
        """
        if item.fit_func is not None and item.channel is None and resolution_function is None:
            return item.fit_func
        func = item.model.interface.fit_func_for(channel=item.channel, resolution_function=resolution_function)
        return _bind_fit_func(func, item.model.unique_name)

    def _resolve_objective(self, objective: str | None, settings: FitSettings | None) -> str:
        if objective is not None:
            return _validate_objective(objective)
        if settings is not None:
            return _validate_objective(settings.objective)
        return self._objective

    def prepare(
        self,
        inputs: list[_FitInput] | None = None,
        *,
        objective: str | None = None,
        action: str = 'fitting',
        extra_warning: str = '',
    ) -> PreparedFit:
        """Prepare one fit run: zero-variance handling, bindings, configured core fitter.

        This is the only place those rules are applied; :meth:`fit`,
        :meth:`fit_single_data_set_1d`, :meth:`fit_polarized`,
        :meth:`mcmc_sample` and an application's worker all run what it returns.
        Preparing never changes this fitter, so preparing again (with another
        objective, say) starts from the measured data each time.

        Parameters
        ----------
        inputs : list[_FitInput] | None, optional
            Datasets to fit. By default the :attr:`fit_datasets` set by
            :meth:`for_experiments`.
        objective : str | None, optional
            Zero-variance objective; by default the settings' or this fitter's.
        action : str, optional
            Verb for the zero-variance warnings. By default, 'fitting'.
        extra_warning : str, optional
            Sentence appended to the Mighell warnings. By default, ''.

        Returns
        -------
        PreparedFit
            The run, ready to execute.

        Raises
        ------
        ValueError
            If the settings are invalid (however they were edited), or
            ``legacy_mask`` leaves a dataset without points.
        FitPreconditionError
            If a free parameter's bounds are unusable for the minimizer.

        Note
        ----
        The run owns copies of the measured arrays, so editing a dataset
        afterwards changes neither what it fits nor its metrics. Model
        parameters and constraints stay live: they are what the fit changes.
        """
        if inputs is None:
            inputs = self._experiment_inputs()
        if not inputs:
            raise ValueError('Nothing to fit: no datasets were given.')
        settings = copy.deepcopy(self.settings)
        if settings is not None:
            settings.validate()
        obj = self._resolve_objective(objective, settings)

        original, fitted, stats, fit_funcs, curve_funcs, models = [], [], [], [], [], []
        for item in inputs:
            x_vals, y_vals, variances = np.array(item.x), np.array(item.y), np.array(item.variances)
            x_out, y_eff, weights, item_stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)
            _emit_array_prep_warnings(item_stats, y_vals, item.label, action=action, extra=extra_warning)
            if obj == 'legacy_mask' and len(x_out) == 0:
                raise ValueError(f'Cannot fit {item.label}: all points have zero variance.')
            original.append({'x': x_vals, 'y': y_vals, 'variances': variances})
            fitted.append({'x': x_out, 'y': y_eff, 'weights': weights})
            stats.append(item_stats)
            # `legacy_mask` drops points: a per-point resolution must drop the
            # same ones, so the fitted function is bound to the subset while
            # the curve function keeps every point.
            fitted_resolution = _resolution_for_fitted_points(item.resolution_function, x_vals, y_vals, variances, obj)
            fit_funcs.append(self._bind(item, fitted_resolution))
            curve_funcs.append(
                fit_funcs[-1] if fitted_resolution is item.resolution_function else self._bind(item, item.resolution_function)
            )
            if not any(item.model is known for known in models):
                models.append(item.model)

        roots = _free_roots(models)
        core_fitter = self._core_fitter_for(models, fit_funcs, settings, roots)
        self._check_fit_preconditions(core_fitter)
        self._warn_on_bound_starts(core_fitter)
        return PreparedFit(
            settings=settings,
            objective=obj,
            channels=[item.channel for item in inputs],
            original=original,
            fitted=fitted,
            stats=stats,
            curve_funcs=curve_funcs,
            fit_funcs=fit_funcs,
            core_fitter=core_fitter,
            fit_kwargs=settings.engine_kwargs() if settings is not None else {},
            added_roots=roots,
        )

    def _core_fitter_for(self, models: list, fit_funcs: list[Callable], settings: FitSettings | None, roots: list = ()):
        """The core fitter a prepared run executes.

        Without settings, a run over exactly the stored fit functions uses the
        stored core fitter, as direct configuration of it (notebooks, tests)
        expects; any other run gets a fresh one carrying the stored minimizer,
        tolerance and budget. With settings, a fresh one configured from them.
        `roots` (see :func:`_free_roots`) join the fit objects, so the core
        varies them like any free parameter of the models.
        """
        if (
            settings is None
            and not roots
            and len(fit_funcs) == len(self._fit_func)
            and all(a is b for a, b in zip(fit_funcs, self._fit_func))
        ):
            return self.easy_science_multi_fitter
        core_fitter = self._build_easy_science_fitter([*models, *roots], fit_funcs, keep_owner=True)
        if settings is not None:
            settings.configure(core_fitter)
        else:
            stored = self.easy_science_multi_fitter
            core_fitter.switch_minimizer(stored.minimizer.enum)
            core_fitter.tolerance = stored.tolerance
            core_fitter.max_evaluations = stored.max_evaluations
        return core_fitter

    @staticmethod
    def _check_fit_preconditions(core_fitter) -> None:
        """Refuse a run whose minimizer needs finite bounds that some free parameter lacks.

        Checked before any engine call, over the unique free independent
        parameters. Ordered bounds and a value inside them need no check here:
        ``Parameter`` refuses anything else.
        """
        minimizer = getattr(core_fitter, 'minimizer', None)
        enum = getattr(minimizer, 'enum', None)
        if not isinstance(enum, AvailableMinimizers) or not requires_finite_bounds(enum):
            return  # also: a stand-in fitter (tests) has nothing to check
        unbounded, seen = [], set()
        for parameter in core_fitter.fit_object.get_fit_parameters():
            if id(parameter) not in seen and not (np.isfinite(parameter.min) and np.isfinite(parameter.max)):
                unbounded.append(parameter)
            seen.add(id(parameter))
        if unbounded:
            names = ', '.join(f"'{parameter.name}'" for parameter in unbounded)
            raise FitPreconditionError(
                f'{enum.name} requires finite bounds on all free parameters; {names} have none.', unbounded
            )

    @staticmethod
    def _warn_on_bound_starts(core_fitter) -> None:
        """Warn when lmfit's ``leastsq`` would start a free parameter on one of its bounds.

        ``leastsq`` maps bounded parameters through a sine transform whose
        derivative vanishes at the bound, so such a parameter can stay where it
        started and the fit stall. A previous fit that ended on a bound leaves
        exactly this state behind.
        """
        enum = getattr(getattr(core_fitter, 'minimizer', None), 'enum', None)
        if not isinstance(enum, AvailableMinimizers) or (enum.package, enum.method) != ('lm', 'leastsq'):
            return
        on_bound = []
        for parameter in {id(p): p for p in core_fitter.fit_object.get_fit_parameters()}.values():
            span = parameter.max - parameter.min
            tolerance = 1e-6 * span if np.isfinite(span) else 0.0
            if any(
                np.isfinite(bound) and abs(parameter.value - bound) <= tolerance for bound in (parameter.min, parameter.max)
            ):
                on_bound.append(f"'{parameter.name}'")
        if on_bound:
            warnings.warn(
                f'{", ".join(on_bound)} start on a bound. LMFit leastsq may not move such a parameter; '
                'move it inside its range or use LMFit_scipy_least_squares or a BUMPS minimizer.',
                UserWarning,
            )

    def fit(
        self,
        data: sc.DataGroup,
        id: int = 0,
        objective: str | None = None,
        constraints_factory: Callable | None = None,
    ) -> sc.DataGroup:
        """Perform the fitting and populate the DataGroups with the result.

        Parameters
        ----------
        data : sc.DataGroup
            DataGroup to be fitted to and populated.
        id : int, optional
            Unused parameter kept for backward compatibility. By default, 0.
        objective : str | None, optional
            Per-call override for the zero-variance objective.
            If ``None``, uses the instance default set at construction. By default, None.
        constraints_factory : Callable | None, optional
            Inequality constraints to enforce (BUMPS engines only); see
            :mod:`easyreflectometry.inequality_constraints`. Defaults to what
            :attr:`constraints_factory_provider` returns. By default, None.

        Returns
        -------
        sc.DataGroup
            A new DataGroup with fitted model curves, SLD profiles, and fit statistics.
        """
        refl_nums, inputs = self._datagroup_inputs(data)
        prepared = self.prepare(inputs, objective=objective)
        with self._constraints(constraints_factory, fitter=prepared.core_fitter):
            result = prepared.execute()
        self.record_fit_results(result, prepared.finalize(result))
        new_data = data.copy()
        for i, _ in enumerate(result):
            id = refl_nums[i]
            model_curve = prepared.curve_funcs[i](data['coords'][f'Qz_{id}'].values)
            new_data[f'R_{id}_model'] = sc.array(dims=[f'Qz_{id}'], values=model_curve)
            sld_profile = self.easy_science_multi_fitter._fit_objects[i].interface.sld_profile(self._models[i].unique_name)
            new_data[f'SLD_{id}'] = sc.array(dims=[f'z_{id}'], values=sld_profile[1] * 1e-6, unit=sc.Unit('1/angstrom') ** 2)
            if 'attrs' in new_data:
                new_data['attrs'][f'R_{id}_model'] = {'model': sc.scalar(self._models[i].as_dict())}
            new_data['coords'][f'z_{id}'] = sc.array(
                dims=[f'z_{id}'],
                values=sld_profile[0],
                unit=(1 / new_data['coords'][f'Qz_{id}'].unit).unit,
            )
        new_data['objective_chi2'] = self.objective_chi2
        new_data['objective_reduced_chi'] = self.objective_reduced_chi
        new_data['classical_chi2'] = self.classical_chi2
        new_data['classical_reduced_chi'] = self.classical_reduced_chi
        new_data['reduced_chi'] = self.objective_reduced_chi
        new_data['success'] = all(item.success for item in result)
        return new_data

    def fit_single_data_set_1d(
        self,
        data: DataSet1D,
        objective: str | None = None,
        constraints_factory: Callable | None = None,
    ) -> FitResults:
        """Perform fitting on a single 1D dataset.

        Parameters
        ----------
        data : DataSet1D
            The 1D dataset to fit. Note that ``data.ye`` stores
            variances (σ²), not standard deviations.
        objective : str | None, optional
            Per-call override for the zero-variance objective.
            If ``None``, uses the instance default set at construction. By default, None.
        constraints_factory : Callable | None, optional
            Inequality constraints to enforce (BUMPS engines only). Defaults
            to what :attr:`constraints_factory_provider` returns. By default, None.

        Returns
        -------
        FitResults
            Fit results from the minimizer.

        Note
        ----
        When ``data`` carries its own ``resolution_function`` the fit is
        smeared with it rather than with the model's.
        """
        item = _FitInput(
            x=np.asarray(data.x),
            y=np.asarray(data.y),
            variances=np.asarray(data.ye),
            model=self._models[0],
            label='single-dataset fit',
            resolution_function=getattr(data, 'resolution_function', None),
            fit_func=self._fit_func[0],
        )
        prepared = self.prepare([item], objective=objective)
        with self._constraints(constraints_factory, fitter=prepared.core_fitter):
            result = prepared.execute()[0]
        self.record_fit_results([result], prepared.finalize([result]))
        return result

    def fit_polarized(
        self,
        data: PolarizedDataSet,
        objective: str | None = None,
        constraints_factory: Callable | None = None,
    ) -> dict[str, FitResults]:
        """Fit all measured spin channels of a polarized experiment simultaneously.

        Each channel dataset gets its own fit function evaluating the
        corresponding spin cross-section, while every channel shares the single
        model — so structural parameters (thickness, roughness, nuclear SLD,
        scale, background) are common, and the magnetic parameters
        (`rho_m`/`theta_m`) are constrained by all channels at once. The refl1d
        backend computes all four cross-sections in one kernel evaluation and
        caches them per iteration, so fitting N channels costs about as much as
        fitting one.

        Parameters
        ----------
        data : PolarizedDataSet
            The polarized experiment (its model must be the model this fitter
            was constructed with). Note that per-channel ``ye`` stores
            variances (σ²), not standard deviations.
        objective : str | None, optional
            Per-call override for the zero-variance objective.
            If ``None``, uses the instance default set at construction. By default, None.
        constraints_factory : Callable | None, optional
            Inequality constraints to enforce (BUMPS engines only); see
            :mod:`easyreflectometry.inequality_constraints`. Defaults to what
            :attr:`constraints_factory_provider` returns. By default, None.

        Returns
        -------
        dict[str, FitResults]
            Fit results per channel, keyed 'pp', 'pm', 'mp', 'mm' (measured
            channels only, in that order).

        Note
        ----
        Unlike :meth:`for_experiments`, this method does not populate
        :attr:`fit_datasets` / :attr:`fit_channels` — those are set only by the
        caller-driven, `for_experiments`-built flow.
        """
        if len(self._models) != 1:
            raise ValueError('Polarized fitting requires a MultiFitter constructed with exactly one model.')
        model = self._models[0]
        if data.model is not model:
            raise ValueError('PolarizedDataSet.model must be the model this fitter was constructed with.')
        inputs = []
        for channel in data.available_channels:
            dataset = data[channel]
            if dataset.model is not model:
                raise ValueError(f"The '{channel.value}' channel dataset is bound to a different model than the fitter's.")
            inputs.append(
                _FitInput(
                    x=np.asarray(dataset.x),
                    y=np.asarray(dataset.y),
                    variances=np.asarray(dataset.ye),
                    model=model,
                    label=f'channel {channel.value}',
                    channel=channel,
                    resolution_function=getattr(dataset, 'resolution_function', None),
                )
            )
        prepared = self.prepare(inputs, objective=objective)
        with self._constraints(constraints_factory, fitter=prepared.core_fitter):
            results = prepared.execute()
        # All channels are fitted against one parameter vector (the shared model),
        # so `result.n_pars` is identical across `results`; `reduced_chi` and
        # `classical_reduced_chi` rely on that invariant.
        self.record_fit_results(results, prepared.finalize(results))
        return {channel.value: result for channel, result in zip(prepared.channels, results)}

    def mcmc_sample(
        self,
        data: sc.DataGroup,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: int | None = None,
        objective: str | None = None,
        initializer: str | None = None,
        progress_callback: Callable[..., Any] | None = None,
        abort_test: Callable[[], bool] | None = None,
        constraints_factory: Callable | None = None,
    ) -> dict:
        """Run Bayesian MCMC sampling on reflectometry data using the DREAM sampler.

        Requires that the minimizer is a BUMPS instance (i.e. the minimizer was
        switched to ``AvailableMinimizers.Bumps``).

        :param data: DataGroup with reflectivity data.
        :param samples: Number of retained DREAM samples requested from BUMPS.
        :param burn: Burn-in steps.
        :param thin: Thinning interval.
        :param population: BUMPS DREAM population count for advanced users.
        :param objective: Zero-variance handling strategy.
        :param initializer: DREAM population initializer. One of ``'eps'``,
            ``'cov'``, ``'lhs'``, or ``'random'``. By default, None (BUMPS
            uses ``'eps'``).
        :param progress_callback: Optional callback for progress updates during
            sampling.  Forwarded to the core MultiFitter.
        :param abort_test: Optional callable returning ``True`` to abort sampling.
        :param constraints_factory: Inequality constraints to enforce (see
            :mod:`easyreflectometry.inequality_constraints`); the posterior is
            penalised in the infeasible region. Defaults to what
            :attr:`constraints_factory_provider` returns.
        :return: Dictionary with keys ``'draws'``, ``'param_names'``, ``'state'``,
            and ``'logp'``.
        :raises RuntimeError: If the current minimizer is not a BUMPS instance.

        The sampler reads none of the generic settings (tolerance, budget):
        only the minimizer, which must be a BUMPS one.

        The underlying :class:`~easyscience.fitting.Sampler` is retained on
        :attr:`sampler`, so the chain can be continued without re-running the
        burn-in::

            fitter.mcmc_sample(data, samples=2000, burn=500, thin=10)
            extended = fitter.sampler.extend(additional_samples=8000, thin=10)
        """
        # Checked before preparing, so a wrong engine is reported before any data warnings.
        minimizer = self.settings.minimizer if self.settings is not None else self.easy_science_multi_fitter.minimizer
        if getattr(minimizer, 'package', None) != 'bumps':
            raise RuntimeError(
                'Bayesian sampling requires a BUMPS minimizer. Select one first, e.g. '
                '``project.minimizer = AvailableMinimizers.Bumps_simplex`` or '
                '``fitter.switch_minimizer(AvailableMinimizers.Bumps_simplex)``.'
            )
        obj = self._resolve_objective(objective, self.settings)
        refl_nums, inputs = self._datagroup_inputs(data)
        for i, item in zip(refl_nums, inputs):
            if obj != 'mighell' and np.all(np.asarray(item.variances) <= 0.0):
                raise ValueError(
                    f'Cannot run Bayesian sampling on reflectivity {i}: all points have zero variance. '
                    'The likelihood is undefined without measurement uncertainties. Supply uncertainties, '
                    "or explicitly opt in to the Mighell transform with objective='mighell' "
                    '(a chi-square bias correction, not a true likelihood).'
                )
        prepared = self.prepare(
            inputs,
            objective=obj,
            action='sampling',
            extra_warning=(
                ' The Mighell transform is a chi-square bias correction, not a true likelihood; '
                'posterior widths may be unreliable.'
            ),
        )
        core_fitter = prepared.core_fitter

        # Delegate the actual BUMPS/DREAM sampling to the core ``Sampler``,
        # which handles the multi-dataset reshaping internally.
        sampler_kwargs = {}
        if initializer is not None:
            sampler_kwargs['init'] = initializer

        # Resolved once and passed on as the explicit factory, so building it
        # (which resolves every constraint's parameter paths) happens once.
        factory = self._resolve_constraints_factory(constraints_factory)
        with self._constraints(factory, fitter=core_fitter):
            # The factory is current only while the problem is being built, so
            # `_keep_constraints_on_extend` covers a later continuation.
            sampler = Sampler(core_fitter, x=prepared.x, y=prepared.y, weights=prepared.weights)
            self._keep_constraints_on_extend(sampler, factory)
            # Retained so the chain can be continued afterwards via ``self.sampler.extend()``.
            self._sampler = sampler
            results = sampler.sample(
                samples=samples,
                burn=burn,
                thin=thin,
                population=population,
                sampler_kwargs=sampler_kwargs or None,
                progress_callback=progress_callback,
                abort_test=abort_test,
            )
        return {
            'draws': results.draws,
            'param_names': results.param_names,
            'state': results.state,
            'logp': results.logp,
        }

    @property
    def sampler(self) -> Sampler | None:
        """The ``Sampler`` behind the most recent :meth:`mcmc_sample` call, or None.

        Holds the live BUMPS chain state, so the sampling run can be continued
        with ``fitter.sampler.extend(additional_samples=...)`` instead of
        starting a fresh chain.
        """
        return self._sampler

    @property
    def chi2(self) -> float | None:
        """Total chi-squared across all fitted datasets, or None if no fit has been performed."""
        if self._fit_results is None:
            return None
        return sum(r.chi2 for r in self._fit_results)

    @property
    def reduced_chi(self) -> float | None:
        """Reduced chi-squared from the most recent fit, or None if no fit has been performed."""
        if self._fit_results is None:
            return None
        total_chi2 = sum(r.chi2 for r in self._fit_results)
        total_points = sum(np.size(r.x) for r in self._fit_results)
        n_params = self._fit_results[0].n_pars
        total_dof = total_points - n_params

        if total_dof <= 0:
            return None

        return total_chi2 / total_dof

    @property
    def classical_chi2(self) -> float | None:
        """Classical chi-squared using only points with positive variances."""
        if self._classical_fit_metrics is None:
            return None
        return float(sum(metric['classical_chi2'] for metric in self._classical_fit_metrics))

    @property
    def classical_reduced_chi(self) -> float | None:
        """Reduced classical chi-squared using only points with positive variances."""
        if self._classical_fit_metrics is None or self._fit_results is None:
            return None
        total_chi2 = self.classical_chi2
        total_points = sum(metric['n_classical_points'] for metric in self._classical_fit_metrics)
        n_params = self._fit_results[0].n_pars
        return _compute_reduced_chi2(total_chi2, total_points, n_params)

    @property
    def objective_chi2(self) -> float | None:
        """Objective-space chi-squared returned by the minimizer."""
        return self.chi2

    @property
    def objective_reduced_chi(self) -> float | None:
        """Objective-space reduced chi-squared returned by the minimizer."""
        return self.reduced_chi

    def record_fit_results(self, results: list[FitResults] | None, metrics: list[dict] | None = None) -> None:
        """Adopt fit results produced elsewhere, so this fitter reports on them.

        An application that executes a :class:`PreparedFit` itself — in a
        worker thread, for instance — leaves the ``MultiFitter`` that owns the
        goodness-of-fit properties none the wiser. Handing the results back
        here makes :attr:`chi2` / :attr:`reduced_chi` describe that fit.

        Parameters
        ----------
        results : list[FitResults] | None
            Results of the fit, one per fitted dataset. None clears them.
        metrics : list[dict] | None, optional
            :meth:`PreparedFit.finalize` of the same run; without it the
            classical metrics stay None, as they need the measured arrays.
        """
        self._fit_results = list(results) if results else None
        self._classical_fit_metrics = list(metrics) if (results and metrics) else None

    def switch_minimizer(self, minimizer: AvailableMinimizers) -> None:
        """Switch the minimizer for the fitting.

        A fitter with :attr:`settings` (one a project hands out) switches the
        settings' minimizer, as its runs are configured from them; for a
        project's fitter that is ``project.minimizer``.

        Parameters
        ----------
        minimizer : AvailableMinimizers
            Minimizer to be switched to.
        """
        if self.settings is None:
            self.easy_science_multi_fitter.switch_minimizer(minimizer)
            return
        self.settings.minimizer = minimizer
        self.settings.configure(self.easy_science_multi_fitter)


def _flatten_list(this_list: list) -> list:
    """Flatten nested lists.

    Parameters
    ----------
    this_list : list
        List to be flattened.

    Returns
    -------
    list
        Flattened list.
    """
    return np.array([item for sublist in this_list for item in sublist])
