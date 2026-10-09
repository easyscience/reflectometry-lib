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
        Numeric range; ``None`` is unbounded.
    exclusive : tuple[bool, bool]
        Whether the lower and the upper bound are themselves excluded.
    """

    name: str
    kind: str
    doc: str
    choices: tuple = ()
    bounds: tuple = (None, None)
    exclusive: tuple = (False, False)

    def validate(self, value: Any) -> Any:
        """Return ``value`` if it is acceptable; raises ValueError otherwise."""
        if self.kind == 'enum':
            if value not in self.choices:
                raise ValueError(f'{self.name} must be one of {list(self.choices)}, got {value!r}.')
            return value
        numeric = (int,) if self.kind == 'int' else (int, float)
        if isinstance(value, bool) or not isinstance(value, numeric) or not math.isfinite(value):
            raise ValueError(f'{self.name} must be a finite {self.kind}, got {value!r}.')
        (low, high), (low_open, high_open) = self.bounds, self.exclusive
        too_low = low is not None and (value <= low if low_open else value < low)
        too_high = high is not None and (value >= high if high_open else value > high)
        if too_low or too_high:
            interval = f'{"(" if low_open else "["}{low}, {high}{")" if high_open else "]"}'
            raise ValueError(f'{self.name} must lie in {interval}, got {value!r}.')
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
        MinimizerOption('mutation', 'float', 'Mutation constant.', bounds=(0.0, 2.0), exclusive=(False, True)),
        MinimizerOption('recombination', 'float', 'Recombination (crossover) probability.', bounds=(0.0, 1.0)),
        MinimizerOption('seed', 'int', 'Random seed, for reproducible runs.', bounds=(0, 2**32 - 1)),
    ),
    ('dfo', 'leastsq'): (
        MinimizerOption(
            'rhobeg', 'float', 'Initial trust-region radius; above the tolerance.', bounds=(0.0, None), exclusive=(True, False)
        ),
    ),
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


def canonical_minimizer(minimizer: Any) -> Any:
    """The member an alias stands for (``LMFit`` is ``LMFit_leastsq``); anything else unchanged."""
    if isinstance(minimizer, AvailableMinimizers) and minimizer.name in _MINIMIZER_ALIASES:
        return AvailableMinimizers[_MINIMIZER_ALIASES[minimizer.name]]
    return minimizer


def normalize_objective(objective: Any) -> str:
    """The zero-variance objective ``objective`` names (``'auto'`` is ``'hybrid'``).

    Raises
    ------
    ValueError
        If it names none.
    """
    if isinstance(objective, str):
        objective = _OBJECTIVE_ALIASES.get(objective, objective)
    if objective not in OBJECTIVES:
        raise ValueError(f'Unknown objective {objective!r}. Valid options: {OBJECTIVES + tuple(_OBJECTIVE_ALIASES)}.')
    return objective


def requires_finite_bounds(minimizer: AvailableMinimizers) -> bool:
    """Whether ``minimizer`` needs a finite ``min`` and ``max`` on every free parameter."""
    return minimizer.package == 'lm' and minimizer.method == 'differential_evolution'


@dataclass
class FitSettings:
    """Minimizer choice and its settings, as saved in a project.

    Attributes
    ----------
    minimizer : AvailableMinimizers
        The engine and method. An alias member (``LMFit``, ``Bumps``, ``DFO``)
        is stored as the member it stands for, so its options keep one key.
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'minimizer':
            value = canonical_minimizer(value)
        super().__setattr__(name, value)

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
                options[keyword] = self.tolerance
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
        _check_mode(self.mode, self.minimizer)
        _check_tolerance(self.tolerance, self.minimizer)
        _check_budget(self.max_evaluations)
        normalize_objective(self.objective)
        _check_trust_region(self.minimizer, self.tolerance, validate_options(self.minimizer, self.active_options()))

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
            'minimizer': self.minimizer.name,
            'mode': self.mode,
            'tolerance': self.tolerance,
            'max_evaluations': self.max_evaluations,
            'objective': self.objective,
            'engine_options': copy.deepcopy(self.engine_options),
        }

    @classmethod
    def from_dict(cls, data: dict) -> tuple['FitSettings', list[str]]:
        """Settings from a project file, never raising on content.

        Aliases are normalised, option keys included (the member's own key wins
        over its alias's). Content that is not a mapping, an unknown or
        unavailable minimizer, an invalid value or a sampling mode without
        BUMPS falls back to the default, and unknown or invalid options are
        dropped; each is reported.

        Returns
        -------
        tuple[FitSettings, list[str]]
            The settings and the warnings to show the user.
        """
        if not isinstance(data, dict):
            return cls(), [f'Fit settings must be a mapping, got {type(data).__name__}; using the defaults.']
        report: list[str] = []
        settings = cls(minimizer=_minimizer_from_name(data.get('minimizer'), report))

        for name, check, fallback in (
            ('mode', lambda v: _check_mode(v, settings.minimizer), 'Using classical minimization.'),
            ('objective', normalize_objective, 'Using hybrid.'),
            ('tolerance', lambda v: _check_tolerance(v, settings.minimizer), 'Using the engine default.'),
            ('max_evaluations', _check_budget, 'Using the engine default.'),
        ):
            if name not in data:
                continue
            try:
                setattr(settings, name, check(data[name]))
            except ValueError as error:
                report.append(f'{error} {fallback}')

        engine_options = data.get('engine_options')
        if engine_options is None:
            engine_options = {}
        elif not isinstance(engine_options, dict):
            report.append(f'Minimizer options must be a mapping, got {type(engine_options).__name__}; dropped.')
            engine_options = {}
        # Aliases first, so a member's own options win over its alias's.
        for name, options in sorted(engine_options.items(), key=lambda item: item[0] not in _MINIMIZER_ALIASES):
            key = _MINIMIZER_ALIASES.get(name, name) if isinstance(name, str) else name
            minimizer = AvailableMinimizers.__members__.get(key) if isinstance(key, str) else None
            try:
                if minimizer is None:
                    raise ValueError(f'{name!r} is not an available minimizer.')
                if not isinstance(options, dict):
                    raise ValueError(f'the options of {name} must be a mapping, got {type(options).__name__}.')
                valid = validate_options(minimizer, dict(options))
                if valid:
                    settings.engine_options.setdefault(key, {}).update(valid)
            except ValueError as error:
                report.append(f'Minimizer options dropped: {error}')

        try:
            _check_trust_region(settings.minimizer, settings.tolerance, settings.active_options())
        except ValueError as error:
            report.append(f'{error} Using the engine default.')
            settings.set_option('rhobeg', None)
        return settings, report


def _minimizer_from_name(name: Any, report: list[str]) -> AvailableMinimizers:
    if name is None:
        return DEFAULT_MINIMIZER
    # AvailableMinimizers only defines members whose engine is installed.
    minimizer = AvailableMinimizers.__members__.get(_MINIMIZER_ALIASES.get(name, name)) if isinstance(name, str) else None
    if minimizer is None:
        report.append(f'Minimizer {name!r} is not available; using {DEFAULT_MINIMIZER.name}.')
        return DEFAULT_MINIMIZER
    return minimizer


def _check_mode(value: Any, minimizer: AvailableMinimizers) -> str:
    if value not in MODES:
        raise ValueError(f'Unknown fit mode {value!r}; must be one of {MODES}.')
    if value == 'sample' and minimizer.package != 'bumps':
        raise ValueError('Bayesian sampling requires a BUMPS minimizer.')
    return value


def _check_tolerance(value: Any, minimizer: AvailableMinimizers) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Tolerance must be a finite positive number, got {value!r}.')
    if minimizer.package == 'dfo' and value > 0.1:
        raise ValueError('DFO-LS tolerance must be 0.1 or smaller.')
    return value


def _check_budget(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'Max evaluations must be a positive integer, got {value!r}.')
    return value


def _check_trust_region(minimizer: AvailableMinimizers, tolerance: float | None, options: dict) -> None:
    """DFO-LS takes the tolerance as its final trust-region radius, which must lie below the initial one."""
    rhobeg = options.get('rhobeg')
    if minimizer.package == 'dfo' and tolerance is not None and rhobeg is not None and rhobeg <= tolerance:
        raise ValueError(f'DFO-LS rhobeg ({rhobeg}) must exceed the tolerance ({tolerance}).')
