# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Inequality-constraint enforcement for the BUMPS engine.

EasyScience builds the ``FitProblem`` internally and offers no hook for
attaching inequality penalties to it, so the enforcement lives here
rather than in the core: :func:`install` wraps the ``FitDriver`` name in
the core's ``minimizer_bumps`` namespace and :func:`applied` makes a
factory current for the duration of a fit, attaching the constraints to
each problem as it is handed to the driver.

``FitProblem.constraints`` is a plain list read live by
``constraints_nllf()``, and the problem's single model is the ``Curve``
whose ``.pars`` is the very ``{prefixed name: BumpsParameter}`` mapping
the factory expects — so attaching before the driver runs is equivalent
to passing ``constraints=`` to the ``FitProblem`` constructor.

Both ``Bumps.fit`` and ``Bumps.mcmc_sample`` construct their driver via
the module-level ``FitDriver`` binding in ``minimizer_bumps``, so one
patched namespace covers classical fitting and DREAM sampling alike.
Only problems whose model exposes a ``.pars`` mapping are touched; any
other problem passes through the shim untouched.

Nothing here is passed to the core as a keyword: ``Bumps.fit`` accepts
arbitrary ``**kwargs`` and would swallow an unrecognised one without
complaint, which is exactly the silent-unconstrained-fit failure this
module exists to prevent.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable
from typing import Iterator
from typing import Optional

from easyscience.fitting.minimizers import minimizer_bumps

#: Raised for engines that cannot enforce inequality constraints. The wording is
#: matched by the test suite and printed in the constraints tutorial, so keep it
#: stable; ``constraints_factory`` names this library's own keyword.
NON_BUMPS_ERROR = (
    "Inequality constraints (constraints_factory) require the BUMPS engine; the selected minimizer uses '{package}'."
)

_active: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar(
    'easyreflectometry_constraints_factory', default=None
)


def _attach_constraints(problem) -> None:
    """Attach the active factory's constraints to a freshly built problem."""
    factory = _active.get()
    if factory is None or problem is None:
        return
    model = next(iter(problem.models), None)
    pars = getattr(model, 'pars', None)
    if pars is None:
        return
    # ``model.pars`` is empty when no parameter is free; the factory then
    # yields constant-only penalties, which is harmless.
    problem.constraints = list(factory(dict(pars)))
    # Only to get the warning BUMPS emits for an infeasible start point;
    # the penalties themselves are already live without this.
    problem.model_reset()


def _patch(module) -> None:
    """Wrap ``FitDriver`` in one consumer namespace."""
    original = module.FitDriver
    if getattr(original, '_easyreflectometry_shim', False):
        return

    def FitDriver(*args, **kwargs):
        # Both core call sites pass ``problem=`` by keyword; the positional
        # fallback covers a caller using BUMPS' (fitclass, problem) order.
        problem = kwargs.get('problem', args[1] if len(args) > 1 else None)
        _attach_constraints(problem)
        return original(*args, **kwargs)

    FitDriver.__name__ = getattr(original, '__name__', 'FitDriver')
    FitDriver.__qualname__ = FitDriver.__name__
    FitDriver.__doc__ = getattr(original, '__doc__', None)
    FitDriver.__wrapped__ = original
    FitDriver._easyreflectometry_shim = True
    module.FitDriver = FitDriver


def install() -> None:
    """Patch the BUMPS driver entry point.

    ``minimizer_bumps`` binds ``FitDriver`` at import time and both
    ``Bumps.fit`` and ``Bumps.mcmc_sample`` read the name from that
    namespace, so patching it covers fitting and sampling. Idempotent.
    """
    _patch(minimizer_bumps)


def is_applied() -> bool:
    """Whether an :func:`applied` block is active in the current context.

    Lets an inner wrapper (``MultiFitter`` routes the raw
    ``easy_science_multi_fitter.fit`` through the constraints machinery)
    detect that an outer block already attached a factory — possibly an
    explicit one that must not be overridden by re-resolving the provider.
    """
    return _active.get() is not None


@contextlib.contextmanager
def applied(factory: Optional[Callable]) -> Iterator[None]:
    """Attach `factory`'s constraints to problems built inside the block.

    A no-op when `factory` is ``None``.

    Parameters
    ----------
    factory : Optional[Callable]
        Receives the ``{prefixed name: BumpsParameter}`` mapping of a freshly
        built problem and returns the constraints to attach.
    """
    if factory is None:
        yield
        return
    install()
    token = _active.set(factory)
    try:
        yield
    finally:
        _active.reset(token)
