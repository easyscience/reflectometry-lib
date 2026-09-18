# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""EasyReflectometry library."""

from importlib import metadata

from .analysis.bayesian import PosteriorResults
from .constraints import clamp_sum_partners
from .constraints import constrain
from .constraints import constrain_equal
from .constraints import constrain_to_sum
from .constraints import derived_parameter
from .constraints import is_constrained_to_sum
from .constraints import restore_sum_partners
from .constraints import unconstrain
from .inequality_constraints import InequalitySpec
from .inequality_constraints import UnitError
from .project import Project

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'InequalitySpec',
    'Project',
    'PosteriorResults',
    'UnitError',
    '__version__',
    'clamp_sum_partners',
    'constrain',
    'constrain_equal',
    'constrain_to_sum',
    'derived_parameter',
    'is_constrained_to_sum',
    'restore_sum_partners',
    'unconstrain',
]
