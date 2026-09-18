# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import traceback

from .calculator_base import CalculatorBase
from .factory import CalculatorFactory
from .polarization import PolarizationChannel

imported_calculators = []

try:
    from .refnx.calculator import Refnx

    imported_calculators.append(Refnx)
except Exception:
    traceback.print_exc()
    print('Warning: refnx is not installed')

try:
    from .refl1d.calculator import Refl1d  # noqa: F401

    imported_calculators.append(Refl1d)
except Exception:
    traceback.print_exc()
    print('Warning: refl1d is not installed')

__all__ = ['CalculatorBase', 'CalculatorFactory', 'PolarizationChannel'] + [c.__name__ for c in imported_calculators]
