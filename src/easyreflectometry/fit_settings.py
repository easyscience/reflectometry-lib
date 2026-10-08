# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""The one owner of a project's fit configuration.

Every fitter the project hands out is configured from a :class:`FitSettings`,
and a prepared fit works on a snapshot of it. Settings reach an engine on two
seams of the easyscience minimizers, as they exist in easyscience 2.5:

- *core fitter attributes* (``tolerance``, ``max_evaluations``), written by
  :meth:`FitSettings.configure`. ``None`` leaves them to the engine adapter:
  DFO-LS uses its own defaults, BUMPS resolves ``None`` to the smaller of its
  native ``ftol``/``xtol``;
- *per-call keyword arguments* (:meth:`FitSettings.engine_kwargs`): the
  method-specific options of the catalogue below, and LMFit's tolerance. The
  LMFit adapter of easyscience 2.5 chooses the tolerance keyword from the
  per-call method, which is always None, so a core-side tolerance reaches
  scipy's Powell and COBYLA as ``ftol`` and crashes. The keyword is therefore
  chosen here, by method, and the core fitter's tolerance stays unset.

The catalogue is deliberately small: the step size, population, strategy and
trust-region settings the requirements name. Options outside it can still be
passed per call by notebook users.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from easyscience.fitting import AvailableMinimizers

DEFAULT_MINIMIZER = AvailableMinimizers.LMFit_leastsq
OBJECTIVES = ('hybrid', 'mighell', 'legacy_mask')
MODES = ('minimize', 'sample')

#: Enum aliases a file may carry, resolved to the member the GUI lists
#: (same package and method).
_MINIMIZER_ALIASES = {'LMFit': 'LMFit_leastsq', 'Bumps': 'Bumps_simplex', 'DFO': 'DFO_leastsq'}
_OBJECTIVE_ALIASES = {'auto': 'hybrid'}

#: The scipy keyword each LMFit method takes its tolerance under.
_LMFIT_TOLERANCE_KEYWORD = {
    'leastsq': 'ftol',
    'least_squares': 'ftol',
    'powell': 'tol',
    'cobyla': 'tol',
    'differential_evolution': 'tol',
}


@dataclass(frozen=True)
class MinimizerOption:
    """A method-specific setting a minimizer accepts.

    Attributes
    ----------
    name : str
        Keyword understood by the engine.
    kind : str
        ``'float'``, ``'int'`` or ``'enum'``.
    doc : str
        One-line description for a user interface.
    choices : tuple[str, ...]
        Allowed values of an ``'enum'`` option.
    bounds : tuple[float | None, float | None]
        Inclusive numeric range; ``None`` is unbounded.
    """

    name: str
    kind: str
    doc: str
    choices: tuple = ()
    bounds: tuple = (None, None)

    def validate(self, value: Any) -> Any:
        """Return ``value`` if it is acceptable; raises ValueError otherwise."""
        if self.kind == 'enum':
            if value not in self.choices:
                raise ValueError(f'{self.name} must be one of {list(self.choices)}, got {value!r}.')
            return value
        numeric = (int,) if self.kind == 'int' else (int, float)
        if isinstance(value, bool) or not isinstance(value, numeric) or not math.isfinite(value):
            raise ValueError(f'{self.name} must be a finite {self.kind}, got {value!r}.')
        low, high = self.bounds
        if (low is not None and value < low) or (high is not None and value > high):
            raise ValueError(f'{self.name} must lie in [{low}, {high}], got {value!r}.')
        return value


_DE_STRATEGIES = (
    'best1bin',
    'best1exp',
    'rand1exp',
    'randtobest1exp',
    'currenttobest1exp',
    'best2exp',
    'rand2exp',
    'randtobest1bin',
    'currenttobest1bin',
    'best2bin',
    'rand2bin',
    'rand1bin',
)

#: Method-specific options, keyed by (enum package, enum method).
_OPTIONS: dict[tuple[str, str], tuple[MinimizerOption, ...]] = {
    ('lm', 'leastsq'): (
        MinimizerOption('epsfcn', 'float', 'Relative step for the finite-difference Jacobian.', bounds=(0.0, None)),
    ),
    ('lm', 'least_squares'): (
        MinimizerOption('diff_step', 'float', 'Relative step for the finite-difference Jacobian.', bounds=(0.0, None)),
    ),
    ('lm', 'differential_evolution'): (
        MinimizerOption('strategy', 'enum', 'Differential evolution strategy.', choices=_DE_STRATEGIES),
        MinimizerOption('popsize', 'int', 'Population size multiplier.', bounds=(1, None)),
        MinimizerOption('mutation', 'float', 'Mutation constant.', bounds=(0.0, 2.0)),
        MinimizerOption('recombination', 'float', 'Recombination (crossover) probability.', bounds=(0.0, 1.0)),
        MinimizerOption('seed', 'int', 'Random seed, for reproducible runs.', bounds=(0, None)),
    ),
    ('dfo', 'leastsq'): (MinimizerOption('rhobeg', 'float', 'Initial trust-region radius.', bounds=(0.0, None)),),
}


def option_schema(minimizer: AvailableMinimizers) -> list[MinimizerOption]:
    """The method-specific options ``minimizer`` accepts."""
    return list(_OPTIONS.get((minimizer.package, minimizer.method), ()))


def validate_options(minimizer: AvailableMinimizers, options: dict) -> dict:
    """``options`` if every one is known for ``minimizer`` and valid.

    Raises
    ------
    ValueError
        Naming the minimizer and the allowed options, before any engine call.
    """
    schema = {option.name: option for option in option_schema(minimizer)}
    unknown = sorted(set(options) - set(schema))
    if unknown:
        raise ValueError(f'{minimizer.name} does not accept {unknown}; allowed options: {sorted(schema)}.')
    return {name: schema[name].validate(value) for name, value in options.items()}


def requires_finite_bounds(minimizer: AvailableMinimizers) -> bool:
    """Whether ``minimizer`` needs a finite ``min`` and ``max`` on every free parameter."""
    return minimizer.package == 'lm' and minimizer.method == 'differential_evolution'


@dataclass
class FitSettings:
    """Minimizer choice and its settings, as saved in a project.

    Attributes
    ----------
    minimizer : AvailableMinimizers
        The engine and method.
    mode : str
        ``'minimize'`` or ``'sample'``. Persists an application's choice of
        Bayesian sampling; the library does not execute on it (classical fits
        and ``mcmc_sample`` stay separate entry points).
    tolerance : float | None
        Generic convergence tolerance; None leaves it to the engine.
    max_evaluations : int | None
        Generic budget (BUMPS: optimizer steps); None leaves it to the engine.
    objective : str
        Zero-variance handling: ``'hybrid'``, ``'mighell'`` or ``'legacy_mask'``.
    engine_options : dict[str, dict[str, Any]]
        Method-specific options keyed by minimizer name, then option name, so
        values of one method are neither lost nor applied when switching.
    """

    minimizer: AvailableMinimizers = field(default_factory=lambda: DEFAULT_MINIMIZER)
    mode: str = 'minimize'
    tolerance: float | None = None
    max_evaluations: int | None = None
    objective: str = 'hybrid'
    engine_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def active_options(self) -> dict[str, Any]:
        """Options of the selected minimizer.

        Only :meth:`validate` and :meth:`engine_kwargs` read these, so every
        option value reaching an engine has been validated.
        """
        return dict(self.engine_options.get(self.minimizer.name, {}))

    def engine_kwargs(self) -> dict:
        """The per-call keyword arguments for the selected minimizer.

        LMFit: ``minimizer_kwargs`` with the options and the tolerance under its
        method's keyword. DFO-LS: the options as plain keyword arguments.
        BUMPS: none (its tolerance and budget are core fitter attributes).
        """
        options = validate_options(self.minimizer, self.active_options())
        if self.minimizer.package == 'lm':
            keyword = _LMFIT_TOLERANCE_KEYWORD.get(self.minimizer.method)
            if self.tolerance is not None and keyword is not None:
                options.setdefault(keyword, self.tolerance)
            return {'minimizer_kwargs': options} if options else {}
        return options

    def set_option(self, name: str, value: Any) -> None:
        """Set (or, with None, clear) an option of the selected minimizer.

        Raises
        ------
        ValueError
            If the option is unknown or the value invalid; nothing is changed.
        """
        options = self.active_options()
        if value is None:
            options.pop(name, None)
        else:
            validate_options(self.minimizer, {name: value})
            options[name] = value
        if options:
            self.engine_options[self.minimizer.name] = options
        else:
            self.engine_options.pop(self.minimizer.name, None)

    def validate(self) -> None:
        """Check every field; raises ValueError naming the first offending one."""
        if not isinstance(self.minimizer, AvailableMinimizers):
            raise ValueError(f'minimizer must be an AvailableMinimizers member, got {self.minimizer!r}.')
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {self.mode!r}.')
        if self.mode == 'sample' and self.minimizer.package != 'bumps':
            raise ValueError('Bayesian sampling requires a BUMPS minimizer.')
        _check_tolerance(self.tolerance, self.minimizer)
        _check_budget(self.max_evaluations)
        if self.objective not in OBJECTIVES:
            raise ValueError(f'objective must be one of {OBJECTIVES}, got {self.objective!r}.')
        validate_options(self.minimizer, self.active_options())

    def configure(self, fitter) -> None:
        """Write the minimizer and generic settings onto a core fitter.

        The minimizer is switched only when it differs (core rebuilds it on
        every switch). Tolerance and budget are written unconditionally, so a
        field reset to None clears a previous override; LMFit's tolerance goes
        per call instead (:meth:`engine_kwargs`), so the core fitter's stays None.
        """
        if fitter.minimizer.enum is not self.minimizer:
            fitter.switch_minimizer(self.minimizer)
        fitter.tolerance = None if self.minimizer.package == 'lm' else self.tolerance
        fitter.max_evaluations = self.max_evaluations

    def to_dict(self) -> dict:
        """Serializable form, as stored in a project file."""
        return {
            # An alias is saved under the member it stands for, so a file round-trips.
            'minimizer': _MINIMIZER_ALIASES.get(self.minimizer.name, self.minimizer.name),
            'mode': self.mode,
            'tolerance': self.tolerance,
            'max_evaluations': self.max_evaluations,
            'objective': self.objective,
            'engine_options': copy.deepcopy(self.engine_options),
        }

    @classmethod
    def from_dict(cls, data: dict) -> tuple['FitSettings', list[str]]:
        """Settings from a project file, never raising on content.

        Aliases are normalised; an unknown or unavailable minimizer, an invalid
        value or a sampling mode without BUMPS falls back to the default, and
        unknown or invalid options are dropped; each is reported.

        Returns
        -------
        tuple[FitSettings, list[str]]
            The settings and the warnings to show the user.
        """
        report: list[str] = []
        settings = cls()
        settings.minimizer = _minimizer_from_name(data.get('minimizer'), report)

        mode = data.get('mode', 'minimize')
        if mode not in MODES:
            report.append(f'Unknown fit mode {mode!r}; using classical minimization.')
            mode = 'minimize'
        if mode == 'sample' and settings.minimizer.package != 'bumps':
            report.append('Bayesian sampling needs the BUMPS engine, which is not available; using classical minimization.')
            mode = 'minimize'
        settings.mode = mode

        objective = _OBJECTIVE_ALIASES.get(data.get('objective', 'hybrid'), data.get('objective', 'hybrid'))
        if objective not in OBJECTIVES:
            report.append(f'Unknown zero-variance objective {objective!r}; using hybrid.')
            objective = 'hybrid'
        settings.objective = objective

        for name, check in (
            ('tolerance', lambda v: _check_tolerance(v, settings.minimizer)),
            ('max_evaluations', _check_budget),
        ):
            value = data.get(name)
            try:
                check(value)
            except ValueError as error:
                report.append(f'{error} Using the engine default.')
                value = None
            setattr(settings, name, value)

        engine_options = data.get('engine_options')
        for minimizer_name, options in (engine_options if isinstance(engine_options, dict) else {}).items():
            minimizer = AvailableMinimizers.__members__.get(minimizer_name)
            try:
                if minimizer is None or not isinstance(options, dict):
                    raise ValueError(f'{minimizer_name} is not an available minimizer.')
                settings.engine_options[minimizer_name] = validate_options(minimizer, dict(options))
            except ValueError as error:
                report.append(f'Minimizer options dropped: {error}')
        return settings, report

    @classmethod
    def from_legacy(cls, minimizer_name: str | None) -> tuple['FitSettings', list[str]]:
        """Settings from a project file that only stored ``fitter_minimizer``."""
        return cls.from_dict({'minimizer': minimizer_name})


def _minimizer_from_name(name: str | None, report: list[str]) -> AvailableMinimizers:
    if name is None:
        return DEFAULT_MINIMIZER
    resolved = _MINIMIZER_ALIASES.get(name, name)
    minimizer = AvailableMinimizers.__members__.get(resolved)
    # AvailableMinimizers only defines members whose engine is installed.
    if minimizer is None:
        report.append(f'Minimizer {name!r} is not available; using {DEFAULT_MINIMIZER.name}.')
        return DEFAULT_MINIMIZER
    return minimizer


def _check_tolerance(value, minimizer: AvailableMinimizers) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Tolerance must be a finite positive number, got {value!r}.')
    if minimizer.package == 'dfo' and value > 0.1:
        raise ValueError('DFO-LS tolerance must be 0.1 or smaller.')


def _check_budget(value) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'Max evaluations must be a positive integer, got {value!r}.')
