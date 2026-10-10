# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the summary goodness-of-fit computation and refinement section."""

import dataclasses

import pytest
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers

from easyreflectometry import Project
from easyreflectometry.fitting import FitRun
from easyreflectometry.summary import Summary


@pytest.fixture
def project() -> Project:
    global_object.map._clear()
    project = Project()
    project.default_model()
    return project


def _run(reduced_chi2) -> FitRun:
    return FitRun(
        status='completed',
        completed_at='2026-10-10T12:00:00',
        minimizer='LMFit_leastsq',
        objective='hybrid',
        inputs=((0, 'd', None),),
        n_free_parameters=2,
        pooled={'objective_reduced_chi2': reduced_chi2},
    )


class TestComputeGoodnessOfFit:
    def test_returns_na_when_no_fit_has_been_run(self, project: Project):
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == 'N/A'

    def test_uses_the_pooled_reduced_chi2_of_the_last_fit(self, project: Project):
        project._last_fit = _run(1.2345)
        assert Summary(project)._compute_goodness_of_fit() == '1.234'

    def test_returns_na_without_degrees_of_freedom(self, project: Project):
        project._last_fit = _run(None)
        assert Summary(project)._compute_goodness_of_fit() == 'N/A'

    def test_returns_na_for_a_failed_run(self, project: Project):
        project._last_fit = dataclasses.replace(_run(2.0), status='failed', pooled={})
        assert Summary(project)._compute_goodness_of_fit() == 'N/A'


class TestRefinementSection:
    def test_refinement_section_renders_counts_and_gof(self, project: Project):
        project._last_fit = _run(2.5)
        summary = Summary(project)

        html = summary._refinement_section()

        assert '2.5' in html
        # every placeholder must have been substituted with a number
        for placeholder in (
            'num_total_params',
            'num_free_params',
            'num_fixed_params',
            'num_constriants',
            'num_constraints',
            'goodness_of_fit',
        ):
            assert placeholder not in html

    def test_the_fit_rows_come_from_the_run_not_the_project(self, project: Project):
        project.models.duplicate_model(0)
        for model in project.models:
            model.sample[1].layers[0].thickness.fixed = False
        project.minimizer = AvailableMinimizers.Bumps_simplex
        project._last_fit = _run(1.5)  # LMFit_leastsq, two free parameters

        html = Summary(project)._refinement_section()

        assert 'LMFit_leastsq' in html and 'Bumps_simplex' not in html
        assert '<td>No. of free parameters in the fit:</td>\n    <td>2</td>' in html
        assert '<td>No. of free parameters (all models):</td>\n    <td>2</td>' in html
        project._last_fit = None
        assert '<td>No. of free parameters in the fit:</td>\n    <td>N/A</td>' in Summary(project)._refinement_section()

    def test_counts_cover_every_model(self, project: Project):
        single = Summary(project)._refinement_section()
        project.models.duplicate_model(0)
        both = Summary(project)._refinement_section()
        assert single != both
