# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""User-facing helpers for constraining model parameters.

These functions are thin wrappers around the EasyScience
parameter-dependency mechanism (``Parameter.make_dependent_on`` /
``Parameter.make_independent``). The dependency graph is owned entirely
by EasyScience; constrained parameters are excluded from the free fit
parameters.

Constraints created through this module are preserved when a project is
saved and reloaded: ``Project.as_dict`` records the expression and the
structural paths of the parameters it refers to, and ``Project.from_dict``
re-applies them once every parameter exists again. Dependencies created by
calling ``make_dependent_on`` directly are outside that contract and are
not persisted.

Cross-parameter *inequalities* (``t_head < t_tail``) are not dependencies
but fit penalties; see :mod:`easyreflectometry.inequality_constraints`.
"""

from __future__ import annotations

import math
import numbers
import re
from typing import Iterable
from typing import Optional
from typing import Union

from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

__all__ = [
    'clamp_sum_partners',
    'constrain',
    'constrain_equal',
    'constrain_to_sum',
    'derived_parameter',
    'is_constrained_to_sum',
    'restore_sum_partners',
    'unconstrain',
]

#: Marks a dependency created through this module, so ``Project`` can persist
#: user constraints without also re-applying the internal ones that assemblies
#: and materials rebuild themselves. Read together with ``independent``: see
#: :func:`easyreflectometry.project.Project._user_constraints`.
USER_CONSTRAINT_FLAG = '_easyreflectometry_user_constraint'


def constrain_equal(parameter: Parameter, to: DescriptorNumber) -> None:
    """Tie `parameter` to always equal `to`.

    `parameter` becomes dependent: it is removed from the free fit
    parameters, immediately takes the value of `to`, and follows it from
    then on. Its value, unit, variance, min and max are replaced by
    `to`'s, and its `fixed` flag is cleared. While constrained, the
    parameter's value and bounds cannot be set directly.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make dependent (the follower).
    to : DescriptorNumber
        Parameter (or descriptor) to follow.
    """
    constrain(parameter, 'a', a=to)


def constrain(parameter: Parameter, expression: str, **parameters: DescriptorNumber) -> None:
    """Tie `parameter` to an arbitrary expression of other parameters.

    Placeholders in `expression` are supplied as keyword arguments:

    .. code-block:: python

        constrain(layer_b.thickness, '2 * t', t=layer_a.thickness)

    `parameter` becomes dependent; its value, unit, variance, min and max
    are replaced by the evaluated expression and its `fixed` flag is
    cleared. Placeholder names must be valid Python identifiers and not
    Python keywords. An unmapped name in the expression that matches a
    mathematical builtin (`e`, `pi`, `sin`, ...) evaluates silently
    instead of raising `NameError`, so prefer descriptive placeholder
    names.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make dependent (the follower).
    expression : str
        Mathematical expression to evaluate, e.g. ``'2 * t'``.
    parameters : DescriptorNumber
        Placeholder-name to parameter mapping for `expression`.
    """
    parameter.make_dependent_on(dependency_expression=expression, dependency_map=parameters)
    setattr(parameter, USER_CONSTRAINT_FLAG, True)


def unconstrain(parameter: Parameter) -> None:
    """Remove any constraint from `parameter`.

    Idempotent: calling it on an already-independent parameter is a
    no-op. The parameter keeps its current (last evaluated) value and
    becomes fittable again. Its bounds, unit, variance and `fixed` state
    are not restored to their pre-constraint values. Review and reset
    the bounds before fitting.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make independent again.
    """
    if not parameter.independent:
        parameter.make_independent()
    if hasattr(parameter, USER_CONSTRAINT_FLAG):
        delattr(parameter, USER_CONSTRAINT_FLAG)


def derived_parameter(
    name: str,
    expression: str,
    unit: Optional[str] = None,
    **parameters: DescriptorNumber,
) -> Parameter:
    """Create a standalone read-only parameter computed from other parameters.

    The returned parameter is *dependent*: it never enters a fit, it
    follows `expression` whenever any of `parameters` changes, and its
    value cannot be set directly — a live calculation to display or reuse
    inside another :func:`constrain` expression.

    .. code-block:: python

        total = derived_parameter('total', 't1 + t2', t1=layer_1.thickness, t2=layer_2.thickness)
        constrain(layer_3.thickness, 'T - t', T=total, t=layer_4.thickness)

    .. warning::

        A standalone derived parameter belongs to no model, so it has no
        structural path (:meth:`~easyreflectometry.Project.parameter_path`
        returns ``None``): it cannot be referenced from an inequality
        constraint, and a project holding a :func:`constrain` that depends
        on one cannot be saved (``Project.as_dict`` raises). It is a
        session-only convenience. For a derived value that must survive
        save/load or appear in inequalities, use a parameter owned by the
        model, such as :attr:`~easyreflectometry.model.Model.total_thickness`.

    Parameters
    ----------
    name : str
        Display name of the new parameter.
    expression : str
        Mathematical expression over the placeholder names in `parameters`.
    unit : Optional[str], optional
        Unit the result is converted to. By default the unit produced by
        the expression is kept.
    parameters : DescriptorNumber
        Placeholder-name to parameter mapping for `expression`.

    Returns
    -------
    Parameter
        A new dependent parameter.
    """
    if not parameters:
        raise ValueError('derived_parameter needs at least one parameter to depend on.')
    return Parameter.from_dependency(
        name=name,
        dependency_expression=expression,
        dependency_map=dict(parameters),
        desired_unit=unit,
    )


def constrain_to_sum(
    parameter: Parameter,
    of_parameters: Iterable[DescriptorNumber],
    *,
    total: Union[DescriptorNumber, numbers.Number, None] = None,
) -> None:
    """Constrain `parameter` so that the sum of `of_parameters` stays equal to `total`.

    `parameter` becomes dependent and takes the value
    ``total - sum(other parameters)``, where "other" means every entry of
    `of_parameters` except `parameter` itself (it may be listed or not).
    This is the "keep the total film thickness fixed while fitting how it
    is split" idiom:

    .. code-block:: python

        constrain_to_sum(layer_b.thickness, [layer_a.thickness, layer_b.thickness], total=120.0)

    Parameters
    ----------
    parameter : Parameter
        Parameter to make dependent (it absorbs the remainder).
    of_parameters : Iterable[DescriptorNumber]
        The parameters whose sum is constrained.
    total : Union[DescriptorNumber, numbers.Number, None], optional
        The target sum: a parameter/descriptor (e.g. a
        :func:`derived_parameter`) or a plain number in `parameter`'s
        unit. By default the current sum of `of_parameters` is frozen as
        a constant.
    """
    others = [p for p in of_parameters if p is not parameter]
    if not others and total is None:
        raise ValueError('constrain_to_sum needs at least one other parameter or an explicit total.')
    if total is None:
        total = float(parameter.value) + sum(float(p.value) for p in others)
    if isinstance(total, numbers.Number):
        total = DescriptorNumber(name=f'{parameter.name}_sum_total', value=float(total), unit=str(parameter.unit))
    elif not isinstance(total, DescriptorNumber):
        raise TypeError('total must be a number, a DescriptorNumber/Parameter or None.')

    dependency_map = {'total': total}
    terms = []
    for index, other in enumerate(others):
        alias = f'p{index}'
        dependency_map[alias] = other
        terms.append(alias)
    expression = 'total' if not terms else f'total - ({" + ".join(terms)})'
    constrain(parameter, expression, **dependency_map)


#: Aliases :func:`constrain_to_sum` builds: the frozen sum plus one ``p<i>``
#: per partner. :func:`is_constrained_to_sum` recognizes exactly this shape.
_SUM_TOTAL_ALIAS = 'total'
_SUM_PARTNER_ALIAS = re.compile(r'p\d+')


def is_constrained_to_sum(
    parameter: Parameter,
    of_parameters: Optional[Iterable[DescriptorNumber]] = None,
) -> bool:
    """Whether `parameter` currently carries a :func:`constrain_to_sum` dependency.

    Purely introspective — nothing is changed. Detection reads the public
    dependency map and recognizes the alias shape :func:`constrain_to_sum`
    builds (``'total'`` plus ``'p0'``, ``'p1'``, ...), so it also holds for a
    constraint restored from a project file.

    Parameters
    ----------
    parameter : Parameter
        The parameter that would absorb the remainder.
    of_parameters : Optional[Iterable[DescriptorNumber]], optional
        When given, every entry (except `parameter` itself) must additionally
        be among the partners of the sum. By default only the dependency shape
        is checked.
    """
    if getattr(parameter, 'independent', True):
        return False
    dependency_map = parameter.dependency_map or {}
    if _SUM_TOTAL_ALIAS not in dependency_map:
        return False
    partner_aliases = set(dependency_map) - {_SUM_TOTAL_ALIAS}
    if any(not _SUM_PARTNER_ALIAS.fullmatch(alias) for alias in partner_aliases):
        return False
    if of_parameters is not None:
        partners = {id(dependency_map[alias]) for alias in partner_aliases}
        required = {id(p) for p in of_parameters if p is not parameter}
        if not required <= partners:
            return False
    return True


# The parameter tied by `constrain_to_sum` takes whatever the partners leave
# over, so on its own the constraint lets a fit push the partners past the
# total and drive the remainder negative (e.g. a layer of negative thickness).
# `clamp_sum_partners` caps each partner at the headroom it actually leaves,
# sharing the slack in proportion to the current values. The original maxima
# are stashed on the parameters under this attribute so
# `restore_sum_partners` can give them back; `Project.as_dict`/`from_dict`
# persist the stash by structural path so the round trip survives save/load.
SUM_PARTNER_MAX_BACKUP = '_sum_partner_previous_max'


def clamp_sum_partners(partners: Iterable[Parameter], remainder: float) -> None:
    """Narrow the partners' maxima so a :func:`constrain_to_sum` remainder cannot go below zero.

    Each partner's ``max`` is capped at ``value * (1 + remainder / occupied)``
    — its share of the remaining budget, distributed in proportion to the
    current values. A partner whose ``max`` is already tighter is left alone,
    and so is one whose cap would collapse onto its ``min`` (a degenerate
    ``min == max`` bound is rejected by EasyScience). The original maxima are
    stashed on the parameters for :func:`restore_sum_partners`.

    Parameters
    ----------
    partners : Iterable[Parameter]
        The free parameters of the sum (everything except the tied remainder).
    remainder : float
        Current value of the tied remainder parameter.
    """
    partners = list(partners)
    occupied = sum(float(parameter.value) for parameter in partners)
    if occupied <= 0.0 or remainder < 0.0:
        # Nothing to share out, or the budget is already exhausted; leave the
        # bounds alone rather than pin every partner at its current value.
        return
    for parameter in partners:
        headroom = float(parameter.value) * (1.0 + remainder / occupied)
        if headroom >= float(parameter.max):
            continue
        if math.isclose(headroom, float(parameter.min), rel_tol=1e-9, abs_tol=0.0):
            continue
        if not hasattr(parameter, SUM_PARTNER_MAX_BACKUP):
            setattr(parameter, SUM_PARTNER_MAX_BACKUP, float(parameter.max))
        parameter.max = headroom


def restore_sum_partners(partners: Iterable[Parameter]) -> None:
    """Give back the maxima :func:`clamp_sum_partners` narrowed.

    Idempotent: a partner without a stashed backup (never clamped, or already
    restored) is left alone. A backup is only applied when it widens the
    current bound — restoring never narrows a maximum that was widened in the
    meantime.
    """
    for parameter in partners:
        previous = getattr(parameter, SUM_PARTNER_MAX_BACKUP, None)
        if previous is None:
            continue
        delattr(parameter, SUM_PARTNER_MAX_BACKUP)
        if float(previous) > float(parameter.max):
            parameter.max = float(previous)
