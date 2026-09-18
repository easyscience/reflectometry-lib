# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.limits import SCALE_LIMITS
from easyreflectometry.limits import SLD_LIMITS
from easyreflectometry.limits import apply_default_limits


class TestApplyDefaultLimits:
    def test_sld_with_inf_bounds(self):
        param = Parameter('sld', 4.186, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'sld')
        assert param.min == SLD_LIMITS[0]
        assert param.max == SLD_LIMITS[1]

    def test_isld_with_inf_bounds(self):
        param = Parameter('isld', 0.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'isld')
        assert param.min == SLD_LIMITS[0]
        assert param.max == SLD_LIMITS[1]

    def test_sld_preserves_finite_bounds(self):
        param = Parameter('sld', 4.0, min=-2.0, max=8.0)
        apply_default_limits(param, 'sld')
        assert param.min == -2.0
        assert param.max == 8.0

    def test_scale_with_inf_bounds(self):
        param = Parameter('scale', 1.0, min=0, max=np.inf)
        apply_default_limits(param, 'scale')
        assert param.min == SCALE_LIMITS[0]
        assert param.max == SCALE_LIMITS[1]

    def test_scale_preserves_finite_bounds(self):
        param = Parameter('scale', 1.0, min=0.5, max=2.0)
        apply_default_limits(param, 'scale')
        assert param.min == 0.5
        assert param.max == 2.0

    def test_thickness_percentage_limits(self):
        param = Parameter('thickness', 10.0, min=0.0, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 0.0  # 0.0 is finite, not overwritten
        assert param.max == 20.0  # 2.0 * 10.0

    def test_thickness_both_inf(self):
        param = Parameter('thickness', 10.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 5.0  # 0.5 * 10.0
        assert param.max == 20.0  # 2.0 * 10.0

    def test_roughness_percentage_limits(self):
        param = Parameter('roughness', 3.3, min=0.0, max=np.inf)
        apply_default_limits(param, 'roughness')
        assert param.min == 0.0  # 0.0 is finite, not overwritten
        assert param.max == pytest.approx(6.6)  # 2.0 * 3.3

    def test_thickness_zero_value_unchanged(self):
        param = Parameter('thickness', 0.0, min=0.0, max=np.inf)
        apply_default_limits(param, 'thickness')
        assert param.min == 0.0
        assert param.max == np.inf  # unchanged, zero-valued skip

    def test_roughness_zero_value_unchanged(self):
        param = Parameter('roughness', 0.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'roughness')
        assert param.min == -np.inf
        assert param.max == np.inf

    def test_thickness_preserves_finite_bounds(self):
        param = Parameter('thickness', 10.0, min=2.0, max=25.0)
        apply_default_limits(param, 'thickness')
        assert param.min == 2.0
        assert param.max == 25.0

    def test_dependent_parameter_skipped(self):
        independent_param = Parameter('sld_main', 4.0, min=-np.inf, max=np.inf)
        dependent_param = Parameter('sld_dep', 4.0, min=-np.inf, max=np.inf)
        dependent_param.make_dependent_on(dependency_expression='a', dependency_map={'a': independent_param})
        apply_default_limits(dependent_param, 'sld')
        assert np.isinf(dependent_param.min)
        assert np.isinf(dependent_param.max)

    def test_unknown_kind_is_noop(self):
        param = Parameter('foo', 5.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'unknown')
        assert np.isinf(param.min)
        assert np.isinf(param.max)


class TestIntegrationWithConstructors:
    def setup_method(self):
        global_object.map._clear()

    def test_material_gets_sld_limits(self):
        from easyreflectometry.sample.elements.materials.material import Material

        mat = Material(sld=6.36, isld=0.0)
        assert mat.sld.min == SLD_LIMITS[0]
        assert mat.sld.max == SLD_LIMITS[1]
        assert mat.isld.min == SLD_LIMITS[0]
        assert mat.isld.max == SLD_LIMITS[1]

    def test_layer_gets_percentage_limits(self):
        from easyreflectometry.project import Project

        project = Project()
        project.default_model()
        layer = project.models[0].sample[1].layers[0]
        assert layer.thickness.min == 50.0
        assert layer.thickness.max == 200.0
        assert layer.roughness.min == 1.5
        assert layer.roughness.max == 6.0

    def test_layer_constructor_keeps_default_bounds_until_project_sync(self):
        from easyreflectometry.sample.elements.layers.layer import Layer

        layer = Layer(thickness=20.0, roughness=5.0)
        assert layer.thickness.min == 0.0
        assert layer.thickness.max == np.inf
        assert layer.roughness.min == 0.0
        assert layer.roughness.max == np.inf

    def test_layer_zero_thickness_unchanged(self):
        from easyreflectometry.project import Project

        project = Project()
        project.default_model()
        layer = project.models[0].sample[0].layers[0]
        assert layer.thickness.min == 0.0
        assert layer.thickness.max == np.inf
        assert layer.roughness.min == 0.0
        assert layer.roughness.max == np.inf

    def test_model_gets_scale_limits(self):
        from easyreflectometry.model.model import Model

        model = Model(scale=1.0)
        assert model.scale.min == SCALE_LIMITS[0]
        assert model.scale.max == SCALE_LIMITS[1]

    def test_existing_parameter_bounds_preserved(self):
        from easyreflectometry.sample.elements.materials.material import Material

        custom_sld = Parameter('sld', 4.0, min=-0.5, max=7.0)
        mat = Material(sld=custom_sld)
        assert mat.sld.min == -0.5
        assert mat.sld.max == 7.0


class TestMagneticParameterLimits:
    def setup_method(self):
        global_object.map._clear()

    def test_rho_m_uses_the_sld_window(self):
        param = Parameter('rho_m', 5.0, min=-np.inf, max=np.inf)
        apply_default_limits(param, 'rho_m')
        assert param.min == SLD_LIMITS[0]
        assert param.max == SLD_LIMITS[1]

    def test_magnetism_constructor_keeps_default_bounds_until_project_sync(self):
        from easyreflectometry.sample import LayerMagnetism

        magnetism = LayerMagnetism(rho_m=5.0)
        assert np.isinf(magnetism.rho_m.min)
        assert np.isinf(magnetism.rho_m.max)

    def test_project_sync_narrows_rho_m_and_leaves_theta_m(self):
        from easyreflectometry.project import Project
        from easyreflectometry.sample import LayerMagnetism

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        layer = project.models[0].sample[1].layers[0]
        layer.magnetism = LayerMagnetism(rho_m=5.0, theta_m=40.0)

        project._sync_parameter_states()

        assert layer.magnetism.rho_m.min == SLD_LIMITS[0]
        assert layer.magnetism.rho_m.max == SLD_LIMITS[1]
        # theta_m ships with explicit physical bounds; the sync must not touch them.
        assert layer.magnetism.theta_m.min == 0.0
        assert layer.magnetism.theta_m.max == 360.0

    def test_project_sync_keeps_explicit_rho_m_bounds(self):
        from easyreflectometry.project import Project
        from easyreflectometry.sample import LayerMagnetism

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        layer = project.models[0].sample[1].layers[0]
        layer.magnetism = LayerMagnetism(rho_m=Parameter('rho_m', 5.0, min=1.0, max=8.0))

        project._sync_parameter_states()

        assert layer.magnetism.rho_m.min == 1.0
        assert layer.magnetism.rho_m.max == 8.0
