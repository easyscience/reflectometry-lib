# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from numbers import Number
from typing import Optional
from typing import Union

import yaml
from easyscience import global_object
from easyscience.variable import Parameter


def get_as_parameter(
    name: str,
    value: Union[Parameter, Number, None],
    default_dict: dict,
    unique_name_prefix: Optional[str] = None,
) -> Parameter:
    """This function creates a parameter for the variable `name`.

        A parameter has a value and metadata.
    If the value already is a parameter, it is returned.
        If the value is a number, a parameter is created with this value and metadata from the dictionary.
        If the value is None, a parameter is created with the default value and metadata from the dictionary.

        param value: The value to use for the parameter.  If None, the default value in the dictionary is used.
        param name: The name of the parameter
        param default_dict: Dictionary with entry for `name` containing the default value and metadata for the parameter
    """
    # This is a parameter, return it
    if isinstance(value, Parameter):
        return value

    # Ensure we got the dictionary for the parameter with the given name
    # Should leave the passed dictionary unchanged
    if name not in default_dict:
        parameter_dict = deepcopy(default_dict)
    else:
        parameter_dict = deepcopy(default_dict[name])

    # Add specific unique name prefix if requested
    if unique_name_prefix is not None:
        parameter_dict['unique_name'] = global_object.generate_unique_name(unique_name_prefix + 'Parameter')

    if value is None:
        # Create a default parameter using both value and metadata from dictionary
        return Parameter(name, **parameter_dict)
    elif isinstance(value, Number):
        # Create a parameter using provided value and metadata from dictionary
        del parameter_dict['value']
        return Parameter(name, value, **parameter_dict)

    raise ValueError(f'{name} must be a Parameter, a number, or None.')


def yaml_dump(dict_repr: dict) -> str:
    """Yaml dump."""
    return yaml.dump(dict_repr, sort_keys=False, allow_unicode=True)


def collect_unique_names_from_dict(structure_dict: dict, unique_names: Optional[list[str]] = None) -> list[str]:
    """This function returns a list with the 'unique_name' found the input dictionary."""
    if unique_names is None:
        unique_names = []

    def _collect(item):
        """Collect function."""
        if isinstance(item, dict):
            if 'unique_name' in item:
                unique_names.append(item['unique_name'])
            for value in item.values():
                _collect(value)
        elif isinstance(item, list):
            for element in item:
                _collect(element)

    _collect(structure_dict)
    return unique_names


def count_free_parameters(project) -> int:
    """Count free parameters.

    Dependent parameters (constrained or derived) are neither free nor fixed:
    they never enter a fit, whatever their ``free`` flag says.
    """
    return sum(1 for parameter in project.parameters if parameter.independent and parameter.free)


def count_fixed_parameters(project) -> int:
    """Count fixed parameters (independent parameters that are not free)."""
    return sum(1 for parameter in project.parameters if parameter.independent and not parameter.free)


def count_parameter_user_constraints(project) -> int:
    """Count the constraints created via :mod:`easyreflectometry.constraints`.

    Counts only parameters that are both marked as user-constrained and still
    dependent — the same test ``Project`` uses to decide what to persist.
    Internal dependencies (``Model.total_thickness``, conformal assembly ties,
    material mixtures) are not user constraints and are not counted.
    """
    from easyreflectometry.constraints import USER_CONSTRAINT_FLAG

    return sum(
        1 for parameter in project.parameters if getattr(parameter, USER_CONSTRAINT_FLAG, False) and not parameter.independent
    )
