# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from ..calculator_base import CalculatorBase
from .wrapper import Refl1dWrapper


class Refl1d(CalculatorBase):
    """Calculator for refl1d."""

    name = 'refl1d'

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = Refl1dWrapper()
