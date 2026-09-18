# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyreflectometry import Project
from easyreflectometry.constraints import constrain
from easyreflectometry.constraints import unconstrain
from easyreflectometry.utils import count_fixed_parameters
from easyreflectometry.utils import count_free_parameters
from easyreflectometry.utils import count_parameter_user_constraints


def test_count_free_parameters():
    # When
    project = Project()
    project.default_model()
    project.parameters[0].free = True

    # Then
    count = count_free_parameters(project)

    # Expect
    assert count == 1


def test_count_fixed_parameters():
    # When
    project = Project()
    project.default_model()
    project.parameters[0].free = True

    # Then
    count = count_fixed_parameters(project)

    # Expect
    assert count == 13


def test_count_parameter_user_constraints_counts_only_user_constraints():
    # When
    project = Project()
    project.default_model()
    sample = project.models[0].sample
    follower = sample[2].layers[0].thickness

    # Then / Expect: internal dependents (e.g. total_thickness) are not counted
    assert count_parameter_user_constraints(project) == 0

    constrain(follower, '2 * t', t=sample[1].layers[0].thickness)
    assert count_parameter_user_constraints(project) == 1

    unconstrain(follower)
    assert count_parameter_user_constraints(project) == 0
