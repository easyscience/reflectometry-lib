# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from ..storage_calculator_base import StorageCalculatorBase
from .wrapper import Refl1dWrapper


class Refl1d(StorageCalculatorBase):
    """Calculator for refl1."""

    name = 'refl1d'

    _material_link = {
        'sld': 'rho',
        'isld': 'irho',
    }

    _layer_link = {
        'thickness': 'thickness',
        'roughness': 'interface',
        'rho_m': 'magnetism_rhoM',
        'theta_m': 'magnetism_thetaM',
    }

    _item_link = {
        'repetitions': 'repeat',
    }

    _model_link = {
        'scale': 'scale',
        'background': 'bkg',
        'guide_field_angle': 'Aguide',
        'applied_field': 'H',
    }

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = Refl1dWrapper()
