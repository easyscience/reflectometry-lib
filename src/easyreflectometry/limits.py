# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from easyscience.variable import Parameter

# Fixed-range limit definitions
SLD_LIMITS = (-1.0, 10.0)
SCALE_LIMITS = (0.0, 10.0)


def apply_default_limits(parameter: Parameter, kind: str) -> None:
    """Apply default min/max to a parameter if current bounds are infinite.

    Parameters
    ----------
    parameter : Parameter
        The parameter to adjust.
    kind : str
        One of 'thickness', 'roughness', 'sld', 'isld', 'rho_m', 'scale'.
    """
    if not parameter.independent:
        return

    if kind in ('thickness', 'roughness'):
        _apply_percentage_limits(parameter)
    elif kind in ('sld', 'isld', 'rho_m'):
        # A magnetic SLD is an SLD: same physical scale, same default window.
        _apply_fixed_limits(parameter, *SLD_LIMITS)
    elif kind == 'scale':
        _apply_fixed_limits(parameter, *SCALE_LIMITS)


def _apply_percentage_limits(parameter: Parameter) -> None:
    """Set min to 50% and max to 200% of the current value, only if current bounds are inf."""
    value = parameter.value
    if value == 0.0:
        return
    if np.isinf(parameter.min):
        parameter.min = 0.5 * value
    if np.isinf(parameter.max):
        parameter.max = 2.0 * value


def _apply_fixed_limits(parameter: Parameter, low: float, high: float) -> None:
    """Set fixed min/max, only if current bounds are inf."""
    if np.isinf(parameter.min) and low <= parameter.value:
        parameter.min = low
    if np.isinf(parameter.max) and high >= parameter.value:
        parameter.max = high
