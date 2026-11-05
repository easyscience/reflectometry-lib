# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from ..calculator_base import CalculatorBase
from .wrapper import RefnxWrapper


class Refnx(CalculatorBase):
    """Calculator for refnx."""

    name = 'refnx'

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = RefnxWrapper()
