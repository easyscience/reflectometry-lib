# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-parameter inequality constraints enforced by the fit.

An :class:`InequalitySpec` is a purely declarative statement such as
``t_head < t_tail`` or ``t1 + t2 <= total``: two expressions over parameter
aliases and a relation. Unlike the equality constraints in
:mod:`easyreflectometry.constraints` it is *not* a parameter dependency —
no parameter is removed from the fit. Instead the specs are translated,
at the start of every fit, into penalty terms attached to the BUMPS
``FitProblem`` (``FitProblem(constraints=[...])``): while a constraint is
violated BUMPS skips the model, adds a large penalty and a term growing
with the violation, which steers the optimizer back into the feasible
region. Only the BUMPS engine family (``Bumps*`` minimizers and the DREAM
sampler) supports this; ``Fitter.fit`` rejects inequalities for LMFit and
DFO-LS.

Specs reference parameters by *structural path* (``models/0/sample/1/layers/0/thickness``,
see :meth:`easyreflectometry.Project.parameter_path`) rather than by
object or unique name, so they survive project save/load and can be
rebuilt against a reloaded object tree.

Design notes — why the translation happens inside the fit, and reads the
BUMPS parameters:

* BUMPS evaluates the constraints *before* the model and skips the model
  while any fails. The EasyScience parameter values are only written inside
  the model call, so at constraint-evaluation time they lag the optimizer's
  trial vector — and freeze entirely once a constraint fails. The operands
  built here therefore read ``bumps.Parameter.value`` of the live problem,
  which BUMPS sets for every trial point.
* Those BUMPS parameters only exist for the *free* EasyScience parameters
  and are rebuilt per fit, which is why a *factory* (``constraints_factory``)
  is passed down the fitting chain and invoked as each BUMPS problem is
  built (see :mod:`easyreflectometry._bumps_constraints`). Fixed parameters
  are frozen as constants;
  dependent (constrained or derived) parameters are expanded recursively
  into their independent leaves.
* Each penalty term returns the *linear* violation; BUMPS squares it once,
  giving a quadratic penalty.
"""

from __future__ import annotations

import keyword
import numbers
import re
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import numpy as np
from asteval import Interpreter
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

__all__ = [
    'RELATIONS',
    'InequalityEvaluation',
    'InequalitySpec',
    'UnitError',
    'build_constraints_factory',
    'check_units',
    'evaluate_spec',
]


class UnitError(ValueError):
    """A unit problem in an inequality constraint.

    Raised by :func:`check_units` when a side cannot be evaluated with its
    units or the two sides of a spec carry incompatible units. Subclasses
    ``ValueError`` so existing ``except ValueError`` callers keep working;
    new callers can catch this type instead of matching message substrings.
    """


RELATIONS = ('<', '<=', '>', '>=')
_RELATION_ALIASES = {'≤': '<=', '≥': '>=', '=<': '<=', '=>': '>='}

#: BUMPS prefixes every EasyScience parameter name; must match
#: ``easyscience.fitting.minimizers.minimizer_base.MINIMIZER_PARAMETER_PREFIX``.
_BUMPS_PREFIX = 'p'

_SAFE_SYMBOLS = {
    'pi': np.pi,
    'e': np.e,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'log': np.log,
    'log10': np.log10,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'abs': abs,
    'min': min,
    'max': max,
}

PathResolver = Callable[[str], DescriptorNumber]


def _normalize_relation(op: str) -> str:
    op = _RELATION_ALIASES.get(op.strip(), op.strip())
    if op not in RELATIONS:
        raise ValueError(f"Unsupported relation '{op}'. Use one of {', '.join(RELATIONS)}.")
    return op


def _new_interpreter() -> Interpreter:
    interpreter = Interpreter(minimal=True, use_numpy=False)
    for name, value in _SAFE_SYMBOLS.items():
        interpreter.symtable[name] = value
    return interpreter


def _evaluate(interpreter: Interpreter, expression: str, symbols: Dict[str, Any]) -> Any:
    interpreter.symtable.update(symbols)
    result = interpreter.eval(expression, raise_errors=True)
    return result


def _identifiers(expression: str) -> List[str]:
    """Identifiers used in `expression` that are not builtin math symbols."""
    names = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expression))
    return sorted(n for n in names if n not in _SAFE_SYMBOLS and not keyword.iskeyword(n))


@dataclass
class InequalitySpec:
    """Declarative cross-parameter inequality, e.g. ``t_head < t_tail``.

    Attributes
    ----------
    lhs_expression : str
        Expression over the aliases in `lhs_paths`, e.g. ``'a + b'``.
    op : str
        One of ``'<'``, ``'<='``, ``'>'``, ``'>='``.
    rhs_expression : str
        Expression over the aliases in `rhs_paths`; may also be a plain
        number (dimensionless or understood in the unit of the left side).
    lhs_paths, rhs_paths : dict[str, str]
        Alias to structural parameter path (``models/0/...``); no live objects
        are held so the spec is trivially serializable.
    name : str
        Optional user label.
    enabled : bool
        Disabled specs are kept but not applied to fits.
    """

    lhs_expression: str
    op: str
    rhs_expression: str
    lhs_paths: Dict[str, str] = field(default_factory=dict)
    rhs_paths: Dict[str, str] = field(default_factory=dict)
    name: str = ''
    enabled: bool = True

    def __post_init__(self) -> None:
        self.op = _normalize_relation(self.op)
        self.lhs_expression = str(self.lhs_expression).strip()
        self.rhs_expression = str(self.rhs_expression).strip()
        self.lhs_paths = dict(self.lhs_paths)
        self.rhs_paths = dict(self.rhs_paths)
        self.validate_syntax()

    # ----- validation -----

    def validate_syntax(self) -> None:
        """Check that both sides parse and every identifier has an alias mapping."""
        for side, expression, paths in (
            ('left', self.lhs_expression, self.lhs_paths),
            ('right', self.rhs_expression, self.rhs_paths),
        ):
            if not expression:
                raise ValueError(f'The {side}-hand side of an inequality cannot be empty.')
            for alias in paths:
                if not alias.isidentifier() or keyword.iskeyword(alias):
                    raise ValueError(f"Alias '{alias}' is not a valid identifier.")
            missing = [n for n in _identifiers(expression) if n not in paths]
            if missing:
                raise ValueError(f"The {side}-hand side '{expression}' references unmapped names: {', '.join(missing)}.")
            interpreter = _new_interpreter()
            try:
                _evaluate(interpreter, expression, {alias: 1.0 for alias in paths})
            except Exception as error:  # asteval raises its own hierarchy
                raise SyntaxError(f"Cannot evaluate the {side}-hand side '{expression}': {error}") from None
        # `paths` merges both sides, so one alias cannot mean two different
        # parameters — the right side would silently win.
        conflicting = sorted(
            alias for alias, path in self.lhs_paths.items() if alias in self.rhs_paths and self.rhs_paths[alias] != path
        )
        if conflicting:
            raise ValueError(
                f'Alias(es) {", ".join(repr(a) for a in conflicting)} map to different parameters on the '
                'two sides of the inequality. Use distinct alias names per side.'
            )

    @property
    def paths(self) -> Dict[str, str]:
        """All alias → path pairs of both sides."""
        merged = dict(self.lhs_paths)
        merged.update(self.rhs_paths)
        return merged

    # ----- serialization -----

    def to_dict(self) -> dict:
        return {
            'lhs_expression': self.lhs_expression,
            'op': self.op,
            'rhs_expression': self.rhs_expression,
            'lhs_paths': dict(self.lhs_paths),
            'rhs_paths': dict(self.rhs_paths),
            'name': self.name,
            'enabled': bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'InequalitySpec':
        return cls(
            lhs_expression=d['lhs_expression'],
            op=d['op'],
            rhs_expression=d['rhs_expression'],
            lhs_paths=d.get('lhs_paths', {}),
            rhs_paths=d.get('rhs_paths', {}),
            name=d.get('name', ''),
            enabled=d.get('enabled', True),
        )

    def __str__(self) -> str:
        return f'{self.lhs_expression} {self.op} {self.rhs_expression}'


@dataclass
class InequalityEvaluation:
    """Result of evaluating a spec against the current parameter values."""

    lhs: float
    rhs: float
    satisfied: bool
    violation: float

    @property
    def feasible(self) -> bool:
        return self.satisfied


# ----- value sources reading the live BUMPS trial vector -----


class _Constant:
    __slots__ = ('value',)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class _BumpsValue:
    """Reads a BUMPS parameter — i.e. the optimizer's current trial value."""

    __slots__ = ('parameter',)

    def __init__(self, bumps_parameter: Any) -> None:
        self.parameter = bumps_parameter

    def __call__(self) -> float:
        return float(self.parameter.value)


class _Expression:
    """Evaluates an expression whose symbols are value sources."""

    __slots__ = ('expression', 'sources', '_interpreter')

    def __init__(self, expression: str, sources: Dict[str, Callable[[], float]]) -> None:
        self.expression = expression
        self.sources = sources
        self._interpreter = _new_interpreter()

    def __call__(self) -> float:
        return float(_evaluate(self._interpreter, self.expression, {k: src() for k, src in self.sources.items()}))

    def __float__(self) -> float:
        return self()


class _Violation:
    """The BUMPS constraint object: ``float()`` is ``0`` when satisfied, else the violation."""

    __slots__ = ('lhs', 'op', 'rhs', 'label')

    def __init__(self, lhs: _Expression, op: str, rhs: _Expression, label: str) -> None:
        self.lhs, self.op, self.rhs, self.label = lhs, op, rhs, label

    def __float__(self) -> float:
        return _violation(self.lhs(), self.op, self.rhs())

    def __str__(self) -> str:
        return self.label

    def __bool__(self) -> bool:  # mirror bumps.Constraint: never silently truthy
        raise TypeError('Inequality constraints cannot be used as booleans')


def _violation(lhs: float, op: str, rhs: float) -> float:
    """Linear violation of ``lhs op rhs`` (``0`` when satisfied).

    Strict and non-strict relations are treated alike: a penalty of exactly
    zero at the boundary is the only continuous choice.
    """
    if op in ('<', '<='):
        diff = lhs - rhs
    else:
        diff = rhs - lhs
    return diff if diff > 0.0 else 0.0


def _source_for(parameter: DescriptorNumber, bumps_pars: Dict[str, Any], trail: tuple) -> Callable[[], float]:
    """Translate an EasyScience parameter into a source reading the BUMPS trial vector.

    * free parameter → the BUMPS parameter of the problem;
    * fixed parameter (or plain descriptor) → constant;
    * dependent parameter → its dependency expression, expanded recursively.
    """
    if id(parameter) in trail:
        raise ValueError(f"Circular dependency while expanding parameter '{parameter.name}'.")
    if isinstance(parameter, Parameter) and not parameter.independent:
        expression = getattr(parameter, '_clean_dependency_string', None)
        dependency_map = getattr(parameter, '_dependency_map', None) or {}
        if expression is None:
            # A dependent parameter always carries its dependency expression in
            # EasyScience; this is a defensive fallback for a foreign Parameter
            # subclass. Freezing silently would hide a bug, so say so.
            warnings.warn(
                f"Dependent parameter '{parameter.name}' has no dependency expression; "
                'its current value is frozen as a constant in the inequality constraint.',
                stacklevel=2,
            )
            return _Constant(parameter.value)
        sources = {alias: _source_for(dep, bumps_pars, trail + (id(parameter),)) for alias, dep in dependency_map.items()}
        return _Expression(expression, sources)
    key = _BUMPS_PREFIX + parameter.unique_name
    if key in bumps_pars:
        return _BumpsValue(bumps_pars[key])
    # Fixed, or not part of this fit (e.g. belongs to a model not being fitted).
    return _Constant(parameter.value)


def _resolve_all(spec: InequalitySpec, resolve: PathResolver) -> Dict[str, DescriptorNumber]:
    resolved = {}
    for alias, path in spec.paths.items():
        parameter = resolve(path)
        if not isinstance(parameter, DescriptorNumber):
            raise ValueError(f"Path '{path}' (alias '{alias}') does not point to a parameter.")
        resolved[alias] = parameter
    return resolved


def build_constraints_factory(specs: Iterable[InequalitySpec], resolve: PathResolver) -> Optional[Callable]:
    """Build the ``constraints_factory`` hook for the given specs.

    The returned callable is what ``easyscience``'s BUMPS engine invokes
    with the ``{prefixed unique name: bumps.Parameter}`` mapping of a
    freshly built problem; it returns one penalty object per enabled spec.
    Paths are resolved and dependent parameters expanded at that moment, so
    the factory always reflects the model as it is when the fit starts.

    Parameters
    ----------
    specs : Iterable[InequalitySpec]
        Specs to apply; disabled ones are skipped.
    resolve : Callable[[str], DescriptorNumber]
        Structural-path resolver, typically ``project.resolve_parameter_path``.

    Returns
    -------
    Callable | None
        The factory, or ``None`` when no spec is enabled (so callers can pass
        it straight through as ``constraints_factory=...``).
    """
    active = [spec for spec in specs if spec.enabled]
    if not active:
        return None

    def factory(bumps_pars: Dict[str, Any]) -> list:
        constraints = []
        for spec in active:
            parameters = _resolve_all(spec, resolve)
            lhs = _Expression(
                spec.lhs_expression,
                {alias: _source_for(parameters[alias], bumps_pars, ()) for alias in spec.lhs_paths},
            )
            rhs = _Expression(
                spec.rhs_expression,
                {alias: _source_for(parameters[alias], bumps_pars, ()) for alias in spec.rhs_paths},
            )
            constraints.append(_Violation(lhs, spec.op, rhs, spec.name or str(spec)))
        return constraints

    return factory


def evaluate_spec(spec: InequalitySpec, resolve: PathResolver) -> InequalityEvaluation:
    """Evaluate a spec against the *current* parameter values.

    Used for the start-point feasibility check before a fit is launched and
    for displaying the constraint state; dependent parameters contribute
    their current (already propagated) value.
    """
    parameters = _resolve_all(spec, resolve)
    interpreter = _new_interpreter()
    lhs = float(_evaluate(interpreter, spec.lhs_expression, {a: float(parameters[a].value) for a in spec.lhs_paths}))
    rhs = float(_evaluate(interpreter, spec.rhs_expression, {a: float(parameters[a].value) for a in spec.rhs_paths}))
    violation = _violation(lhs, spec.op, rhs)
    return InequalityEvaluation(lhs=lhs, rhs=rhs, satisfied=violation == 0.0, violation=violation)


def check_units(spec: InequalitySpec, resolve: PathResolver) -> None:
    """Raise :class:`UnitError` when the two sides of `spec` have incompatible units.

    Each side is evaluated with the unit-carrying ``DescriptorNumber``
    objects themselves (the same arithmetic the equality constraints use),
    so ``t1 + sld`` is rejected by EasyScience and ``t_head < t_tail``
    passes. A plain numeric side is accepted against any unit: it is read in
    the unit of the other side. The same applies to a side mixing literals
    with parameters (``90 - b``): it is checked numerically and its literals
    are read in the unit of the other side.
    """
    parameters = _resolve_all(spec, resolve)
    units = []
    for expression, paths in ((spec.lhs_expression, spec.lhs_paths), (spec.rhs_expression, spec.rhs_paths)):
        interpreter = _new_interpreter()
        try:
            result = _evaluate(interpreter, expression, {alias: parameters[alias] for alias in paths})
        except Exception as unit_error:
            # Mixed literal/parameter arithmetic such as ``90 - b`` cannot be
            # evaluated with unit-carrying objects (a bare number has no unit).
            # Fall back to a numeric evaluation and read the literals in the
            # unit of the other side; purely wrong mixes still fail here.
            try:
                numeric = _evaluate(_new_interpreter(), expression, {alias: float(parameters[alias].value) for alias in paths})
            except Exception:
                raise UnitError(f"Cannot evaluate '{expression}' with units: {unit_error}") from None
            if not isinstance(numeric, numbers.Number):
                raise ValueError(f"'{expression}' does not evaluate to a number.") from None
            units.append(None)
            continue
        if isinstance(result, DescriptorNumber):
            units.append(str(result.unit))
        elif isinstance(result, numbers.Number):
            units.append(None)
        else:
            raise ValueError(f"'{expression}' does not evaluate to a number.")
    lhs_unit, rhs_unit = units
    if lhs_unit is not None and rhs_unit is not None and lhs_unit != rhs_unit:
        raise UnitError(f"Incompatible units in '{spec}': left side is in '{lhs_unit}', right side in '{rhs_unit}'.")
