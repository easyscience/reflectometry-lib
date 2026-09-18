# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import contextlib
import functools
import warnings
import weakref
from typing import Any
from typing import Callable

import numpy as np
import scipp as sc
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import FitResults
from easyscience.fitting import Sampler
from easyscience.fitting.multi_fitter import MultiFitter as EasyScienceMultiFitter

from easyreflectometry._bumps_constraints import NON_BUMPS_ERROR
from easyreflectometry._bumps_constraints import applied as _constraints_applied
from easyreflectometry._bumps_constraints import is_applied as _constraints_active
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.model import Model

_VALID_OBJECTIVES = ('legacy_mask', 'mighell', 'hybrid', 'auto')
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


def _validate_objective(objective: str) -> str:
    """Validate and resolve the objective string.

    Parameters
    ----------
    objective : str
        The objective mode string.

    Raises
    ------
    ValueError :
        If the objective is not one of the valid options.

    Returns
    -------
    str
        Resolved objective string ('auto' becomes 'hybrid').
    """
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(f'Unknown objective {objective!r}. Valid options: {_VALID_OBJECTIVES}')
    if objective == 'auto':
        return 'hybrid'
    return objective


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
    """Assemble the classical (positive-variance-only) and objective-space fit metrics.

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
        Number of points actually fitted, used as the ``reduced_chi``/``reduced_chi2``
        fallback's point count. If ``None``, derived from ``result.x``. By default, None.

    Returns
    -------
    dict
        Keys ``'classical_chi2'``, ``'classical_reduced_chi'``,
        ``'objective_chi2'``, ``'objective_reduced_chi'``, ``'n_classical_points'``.
    """
    sigma_classical = np.sqrt(np.clip(original['variances'], 0.0, None))
    n_classical_points = int(np.sum(original['variances'] > 0.0))
    classical_chi2 = _compute_weighted_chi2(original['y'], model_curve, sigma_classical)
    if n_points is None:
        n_points = np.size(result.x)
    return {
        'classical_chi2': classical_chi2,
        'classical_reduced_chi': _compute_reduced_chi2(classical_chi2, n_classical_points, result.n_pars),
        'objective_chi2': float(result.chi2),
        'objective_reduced_chi': _fit_result_reduced_chi(result, n_points),
        'n_classical_points': n_classical_points,
    }


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

    def _build_easy_science_fitter(self, models, fit_funcs) -> EasyScienceMultiFitter:
        """Build the EasyScience fitter, with its raw ``fit`` honouring constraints.

        :meth:`for_experiments` documents that the caller drives
        ``easy_science_multi_fitter.fit(...)`` directly (e.g. a GUI worker
        thread), which would bypass :meth:`_constraints` and silently fit an
        unconstrained problem. The returned fitter routes that path through
        the constraints machinery: :attr:`constraints_factory_provider` is
        resolved at call time (and in the calling thread — the shim's context
        variable is thread-local), and non-BUMPS engines are rejected rather
        than silently dropping the constraints.
        """
        fitter = _ConstrainedEasyScienceMultiFitter(models, fit_funcs)
        fitter._constraints_owner = weakref.ref(self)
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

        The resulting fitter is *not* run: the caller supplies the data arrays
        to ``easy_science_multi_fitter.fit(...)`` in the order given by
        :attr:`fit_datasets`, which lets a GUI drive it from a worker thread.
        That call resolves :attr:`constraints_factory_provider` at call time,
        so inequality constraints are applied on this path too — pass
        ``constraints_factory_provider`` (e.g.
        ``project.build_constraints_factory``) or set the attribute before
        fitting; with none set, no inequality constraints are enforced.

        Note
        ----
        Built via ``cls(*models, objective=objective)`` and then overwrites
        ``_fit_func`` / ``easy_science_multi_fitter`` with the per-channel
        versions — the ``__init__``-built pair is briefly constructed and
        discarded. Unlike :meth:`fit_polarized`, the minimizer selection and
        its ``tolerance`` / ``max_evaluations`` are *not* carried over: the
        returned fitter starts from ``easy_science_multi_fitter``'s defaults,
        so a caller that needs a specific minimizer must set it explicitly
        before calling ``.fit(...)``.

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

        fit_funcs = []
        for dataset, channel in zip(datasets, channels):
            model = dataset.model
            interface = model.interface
            func = interface.fit_func if channel is None else interface.fit_func_for_channel(channel)
            fit_funcs.append(_bind_fit_func(func, model.unique_name))

        fitter._fit_func = fit_funcs
        fitter.easy_science_multi_fitter = fitter._build_easy_science_fitter(models, fit_funcs)
        fitter.fit_datasets = datasets
        fitter.fit_channels = channels
        fitter.constraints_factory_provider = constraints_factory_provider
        return fitter

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
        obj = _validate_objective(objective) if objective is not None else self._objective

        refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
        x = []
        y = []
        dy = []
        original_arrays = []

        # Process each reflectivity dataset
        for i in refl_nums:
            x_vals = data['coords'][f'Qz_{i}'].values
            y_vals = data['data'][f'R_{i}'].values
            variances = data['data'][f'R_{i}'].variances

            x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)
            _emit_array_prep_warnings(stats, y_vals, f'reflectivity {i}')

            x.append(x_out)
            y.append(y_eff)
            dy.append(weights)
            original_arrays.append({'x': x_vals, 'y': y_vals, 'variances': variances})

        with self._constraints(constraints_factory):
            result = self.easy_science_multi_fitter.fit(x, y, weights=dy)
        self._fit_results = result
        self._classical_fit_metrics = []
        new_data = data.copy()
        for i, _ in enumerate(result):
            id = refl_nums[i]
            model_curve = self._fit_func[i](data['coords'][f'Qz_{id}'].values)
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
            metrics = _classical_metrics_for(original_arrays[i], model_curve, result[i])
            self._classical_fit_metrics.append(metrics)

            new_data['objective_chi2'] = metrics['objective_chi2']
            new_data['objective_reduced_chi'] = metrics['objective_reduced_chi']
            new_data['classical_chi2'] = metrics['classical_chi2']
            new_data['classical_reduced_chi'] = metrics['classical_reduced_chi']
            new_data['reduced_chi'] = metrics['objective_reduced_chi']
            new_data['success'] = result[i].success
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
        """
        obj = _validate_objective(objective) if objective is not None else self._objective

        x_vals = np.asarray(data.x)
        y_vals = np.asarray(data.y)
        variances = np.asarray(data.ye)

        x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)
        _emit_array_prep_warnings(stats, y_vals, 'single-dataset fit')

        if obj == 'legacy_mask' and len(x_out) == 0:
            raise ValueError('Cannot fit single dataset: all points have zero variance.')

        with self._constraints(constraints_factory):
            result = self.easy_science_multi_fitter.fit(x=[x_out], y=[y_eff], weights=[weights])[0]
        self._fit_results = [result]
        model_curve = self._fit_func[0](x_vals)
        self._classical_fit_metrics = [
            _classical_metrics_for({'y': y_vals, 'variances': variances}, model_curve, result, n_points=len(x_out))
        ]
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
        obj = _validate_objective(objective) if objective is not None else self._objective
        if len(self._models) != 1:
            raise ValueError('Polarized fitting requires a MultiFitter constructed with exactly one model.')
        model = self._models[0]
        if data.model is not model:
            raise ValueError('PolarizedDataSet.model must be the model this fitter was constructed with.')
        channels = data.available_channels
        for channel in channels:
            if data[channel].model is not model:
                raise ValueError(f"The '{channel.value}' channel dataset is bound to a different model than the fitter's.")

        channel_fit_funcs = [
            _bind_fit_func(model.interface.fit_func_for_channel(channel), model.unique_name) for channel in channels
        ]
        # One fit function per channel, all bound to the single model. Constructed
        # per call because the channel set comes from the data; the minimizer
        # selection and its generic settings (tolerance, max_evaluations) are
        # carried over. Engine-specific settings applied directly to the minimizer
        # instance of this fitter would not be.
        polarized_fitter = EasyScienceMultiFitter([model], channel_fit_funcs)
        polarized_fitter.switch_minimizer(self.easy_science_multi_fitter.minimizer.enum)
        polarized_fitter.tolerance = self.easy_science_multi_fitter.tolerance
        polarized_fitter.max_evaluations = self.easy_science_multi_fitter.max_evaluations

        x = []
        y = []
        dy = []
        original_arrays = []
        for channel in channels:
            dataset = data[channel]
            x_vals = np.asarray(dataset.x)
            y_vals = np.asarray(dataset.y)
            variances = np.asarray(dataset.ye)

            x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)
            _emit_array_prep_warnings(stats, y_vals, f'channel {channel.value}')
            if obj == 'legacy_mask' and len(x_out) == 0:
                raise ValueError(f'Cannot fit channel {channel.value}: all points have zero variance.')

            x.append(x_out)
            y.append(y_eff)
            dy.append(weights)
            original_arrays.append({'x': x_vals, 'y': y_vals, 'variances': variances})

        with self._constraints(constraints_factory, fitter=polarized_fitter):
            results = polarized_fitter.fit(x, y, weights=dy)
        # All channels are fitted against one parameter vector (the shared model),
        # so `result.n_pars` is identical across `results`; `reduced_chi` and
        # `classical_reduced_chi` below rely on that invariant.
        self._fit_results = list(results)

        self._classical_fit_metrics = []
        for index, (channel, result) in enumerate(zip(channels, results)):
            original = original_arrays[index]
            model_curve = channel_fit_funcs[index](original['x'])
            self._classical_fit_metrics.append(_classical_metrics_for(original, model_curve, result))

        return {channel.value: result for channel, result in zip(channels, results)}

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

        The underlying :class:`~easyscience.fitting.Sampler` is retained on
        :attr:`sampler`, so the chain can be continued without re-running the
        burn-in::

            fitter.mcmc_sample(data, samples=2000, burn=500, thin=10)
            extended = fitter.sampler.extend(additional_samples=8000, thin=10)
        """
        minimizer = self.easy_science_multi_fitter.minimizer
        if not (hasattr(minimizer, 'package') and minimizer.package == 'bumps'):
            raise RuntimeError(
                'Bayesian sampling requires a BUMPS minimizer. '
                'Use ``fitter.switch_minimizer(AvailableMinimizers.Bumps)`` first.'
            )

        obj = _validate_objective(objective) if objective is not None else self._objective

        refl_nums = [k[3:] for k in data['coords'].keys() if k.startswith('Qz_')]
        x = []
        y = []
        dy = []

        # Process each reflectivity dataset
        for i in refl_nums:
            x_vals = data['coords'][f'Qz_{i}'].values
            y_vals = data['data'][f'R_{i}'].values
            variances = data['data'][f'R_{i}'].variances

            if obj != 'mighell' and np.all(np.asarray(variances) <= 0.0):
                raise ValueError(
                    f'Cannot run Bayesian sampling on reflectivity {i}: all points have zero variance. '
                    'The likelihood is undefined without measurement uncertainties. Supply uncertainties, '
                    "or explicitly opt in to the Mighell transform with objective='mighell' "
                    '(a chi-square bias correction, not a true likelihood).'
                )

            x_out, y_eff, weights, stats = _prepare_fit_arrays(x_vals, y_vals, variances, obj)
            _emit_array_prep_warnings(
                stats,
                y_vals,
                f'reflectivity {i}',
                action='sampling',
                extra=(
                    ' The Mighell transform is a chi-square bias correction, not a true likelihood; '
                    'posterior widths may be unreliable.'
                ),
            )
            x.append(x_out)
            y.append(y_eff)
            dy.append(weights)

        # Delegate the actual BUMPS/DREAM sampling to the core ``Sampler``.
        # The core API moved from ``MultiFitter.mcmc_sample()`` to a dedicated
        # ``Sampler`` class: construct it with the configured fitter and the
        # bound data, then call ``sample()``. ``Sampler`` handles the
        # multi-dataset reshaping internally.
        sampler_kwargs = {}
        if initializer is not None:
            sampler_kwargs['init'] = initializer

        # Resolved once and passed on as the explicit factory, so building it
        # (which resolves every constraint's parameter paths) happens once.
        factory = self._resolve_constraints_factory(constraints_factory)
        with self._constraints(factory):
            # The factory is current only while the problem is being built, so
            # `_keep_constraints_on_extend` covers a later continuation.
            sampler = Sampler(self.easy_science_multi_fitter, x=x, y=y, weights=dy)
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

    def record_fit_results(self, results: list[FitResults] | None) -> None:
        """Adopt fit results produced elsewhere, so this fitter reports on them.

        An application that drives ``easy_science_multi_fitter.fit(...)`` itself
        — to run it in a worker thread, for instance — leaves the ``MultiFitter``
        that owns the goodness-of-fit properties none the wiser. Handing the
        results back here makes :attr:`chi2` / :attr:`reduced_chi` describe that
        fit instead of reporting "no fit performed".

        Only the minimizer-reported metrics are restored: the classical ones
        need the original data arrays, which are not part of ``FitResults``, so
        :attr:`classical_chi2` and :attr:`classical_reduced_chi` stay None.

        Parameters
        ----------
        results : list[FitResults] | None
            Results of the fit, one per fitted dataset. None clears them.
        """
        self._fit_results = list(results) if results else None
        self._classical_fit_metrics = None

    def switch_minimizer(self, minimizer: AvailableMinimizers) -> None:
        """Switch the minimizer for the fitting.

        Parameters
        ----------
        minimizer : AvailableMinimizers
            Minimizer to be switched to.
        """
        self.easy_science_multi_fitter.switch_minimizer(minimizer)


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
