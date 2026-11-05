# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""The model's guide field (`guide_field_angle`, `applied_field`) and its projection."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from refl1d.sample.reflectivity import magnetic_reflectivity
from refl1d.sample.reflectivity import reflectivity

from easyreflectometry.contrasts import derive_contrast
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.project import Project
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.special.calculations import field_frame_components

from .test_polarized_fitting import _magnetic_model
from .test_polarized_fitting import _refl1d_interface

Q = np.linspace(0.005, 0.2, 60)


def _bound_model(theta_m=300.0, guide_field_angle=270.0, applied_field=0.0) -> Model:
    model = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=theta_m))
    model.guide_field_angle = guide_field_angle
    model.applied_field = applied_field
    model.interface = _refl1d_interface()
    return model


class TestFieldFrameComponents:
    @pytest.mark.parametrize(
        ('theta_m', 'angle', 'parallel', 'perpendicular'),
        [
            (270.0, 270.0, 5.0, 0.0),  # moment along an in-plane field
            (270.0, -90.0, 5.0, 0.0),  # -90 is the same field as 270
            (270.0, 90.0, -5.0, 0.0),  # reversed in-plane field
            (90.0, 270.0, -5.0, 0.0),
            (0.0, 270.0, 0.0, 5.0),  # moment along the beam: fully transverse
            (270.0, 0.0, 0.0, 5.0),  # field along the normal
        ],
    )
    def test_axes(self, theta_m, angle, parallel, perpendicular):
        result = field_frame_components(5.0, theta_m, angle)
        assert_allclose(result, (parallel, perpendicular), atol=1e-12)

    @pytest.mark.parametrize('angle', [45.0, 135.0])
    def test_oblique_field(self, angle):
        theta = np.radians(300.0)
        tilt = np.radians(angle)
        parallel, perpendicular = field_frame_components(5.0, 300.0, angle)
        assert_allclose(parallel, 5.0 * np.sin(theta) * np.sin(tilt))
        # The moment's length is split between the two parts.
        assert_allclose(parallel**2 + perpendicular**2, 25.0)

    def test_in_plane_field_matches_the_former_cosine_projection(self):
        theta_m = np.linspace(0.0, 360.0, 13)
        parallel, _ = field_frame_components(2.0, theta_m, 270.0)
        assert_allclose(parallel, 2.0 * np.cos(np.radians(theta_m - 270.0)), atol=1e-12)

    @pytest.mark.parametrize('angle', [270.0, 90.0, -90.0])
    def test_sign_agrees_with_refl1d_critical_edges(self, angle):
        # A collinear moment on a semi-infinite medium: each NSF channel is a
        # scalar reflectivity at rho +/- (signed) longitudinal part.
        parallel, _ = field_frame_components(5.0, 270.0, angle)
        mm, _, _, pp = magnetic_reflectivity(
            Q / 2, depth=[0, 0], rho=[0, 8], rhoM=[0, 5], thetaM=[270, 270], sigma=0, Aguide=angle
        )
        assert_allclose(pp, reflectivity(Q / 2, depth=[0, 0], rho=[0, 8 + parallel], sigma=0), rtol=1e-9)
        assert_allclose(mm, reflectivity(Q / 2, depth=[0, 0], rho=[0, 8 - parallel], sigma=0), rtol=1e-9)


class TestRefl1dField:
    def setup_method(self):
        global_object.registry.clear()

    @pytest.mark.parametrize(('angle', 'field'), [(270.0, 0.0), (90.0, 0.0), (0.0, 0.0), (45.0, 1.0)])
    def test_matches_public_refl1d(self, angle, field):
        model = _bound_model(guide_field_angle=angle, applied_field=field)

        ours = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)
        mm, mp, pm, pp = magnetic_reflectivity(
            Q / 2,
            depth=[0, 100, 0],
            rho=[0, 4, 2.047],
            rhoM=[0, 2.5, 0],
            thetaM=[270, 300, 270],
            sigma=[0, 0],
            Aguide=angle,
            H=field,
        )

        for channel, expected in {'pp': pp, 'pm': pm, 'mp': mp, 'mm': mm}.items():
            assert_allclose(ours[channel], expected, rtol=1e-8, err_msg=channel)

    def test_field_along_the_normal_equalises_the_non_spin_flip_channels(self):
        model = _bound_model(theta_m=270.0, guide_field_angle=0.0)

        channels = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)

        assert_allclose(channels['pp'], channels['mm'], rtol=1e-10)
        assert channels['pm'].max() > 1e-6
        # ...but the moment still splits the eigenmodes: pp is not the
        # non-magnetic reflectivity, which rho +/- rho_m_parallel would predict.
        model.sample[1].layers[0].magnetism.rho_m = 0.0
        unmagnetised = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)['pp']
        assert np.abs(channels['pp'] - unmagnetised).max() > 1e-3

    def test_applied_field_breaks_the_equality(self):
        model = _bound_model(theta_m=270.0, guide_field_angle=0.0, applied_field=1.0)

        channels = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)

        assert np.abs(channels['pp'] - channels['mm']).max() > 1e-3

    def test_spin_flip_channels_differ_in_a_transverse_field(self):
        model = _bound_model(theta_m=0.0, applied_field=1.0)

        channels = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)

        assert np.abs(channels['pm'] - channels['mp']).max() > 1e-6

    @pytest.mark.parametrize(('attribute', 'value'), [('guide_field_angle', 0.0), ('applied_field', 1.0)])
    def test_field_change_invalidates_the_cache(self, attribute, value):
        model = _bound_model()
        before = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)['pp']

        setattr(model, attribute, value)
        after = model.interface().polarized_reflectivity_profiles(Q, model.unique_name)['pp']

        assert np.abs(before - after).max() > 1e-6

    def test_models_with_different_fields_share_one_calculator(self):
        interface = _refl1d_interface()
        in_plane = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        normal = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        normal.guide_field_angle = 0.0
        in_plane.interface = interface
        normal.interface = interface

        pp_in_plane = interface().polarized_reflectivity_profiles(Q, in_plane.unique_name)
        pp_normal = interface().polarized_reflectivity_profiles(Q, normal.unique_name)

        assert np.abs(pp_in_plane['pp'] - pp_in_plane['mm']).max() > 1e-3
        assert_allclose(pp_normal['pp'], pp_normal['mm'], rtol=1e-10)


class TestModelField:
    def setup_method(self):
        global_object.registry.clear()

    def test_defaults(self):
        model = Model()
        assert model.guide_field_angle.value == 270.0
        assert model.applied_field.value == 0.0
        assert model.guide_field_angle.fixed and model.applied_field.fixed

    def test_round_trip(self):
        model = Model(guide_field_angle=45.0, applied_field=0.5)
        model.applied_field.fixed = False

        restored = Model.from_dict(model.as_dict())

        assert restored.guide_field_angle.value == 45.0
        assert restored.applied_field.value == 0.5
        assert not restored.applied_field.fixed

    def test_older_dict_loads_the_defaults(self):
        model_dict = Model().as_dict()
        del model_dict['guide_field_angle'], model_dict['applied_field']

        restored = Model.from_dict(model_dict)

        assert restored.guide_field_angle.value == 270.0
        assert restored.applied_field.value == 0.0

    def test_contrast_keeps_the_field(self):
        reference = _magnetic_model(LayerMagnetism(rho_m=2.5))
        reference.guide_field_angle = 45.0
        reference.applied_field = 0.5
        reference.applied_field.fixed = False
        reference.applied_field.max = 2.0

        contrast = derive_contrast(reference, name='contrast')

        assert contrast.guide_field_angle.value == 45.0
        assert contrast.applied_field.value == 0.5
        assert not contrast.applied_field.fixed
        assert contrast.applied_field.max == 2.0
        assert contrast.applied_field is not reference.applied_field


class TestFieldEnablement:
    def setup_method(self):
        global_object.registry.clear()

    @staticmethod
    def _project(magnetism) -> Project:
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(magnetism))
        return project

    @staticmethod
    def _field(project):
        model = project.models[0]
        return model.guide_field_angle, model.applied_field

    def test_enabled_for_a_magnetic_model_on_refl1d(self):
        project = self._project(LayerMagnetism(rho_m=2.5))
        assert all(parameter.enabled for parameter in self._field(project))

    def test_disabled_for_a_non_magnetic_model(self):
        project = self._project(None)
        assert not any(parameter.enabled for parameter in self._field(project))

    def test_disabled_on_refnx_where_the_field_is_inert(self):
        project = self._project(None)
        project.models[0].guide_field_angle = 0.0
        reference = project.model_data_for_model_at_index(0, Q).y

        project.calculator = 'refnx'

        assert not any(parameter.enabled for parameter in self._field(project))
        assert_allclose(project.model_data_for_model_at_index(0, Q).y, reference, rtol=1e-3)


class TestInactiveParametersLeaveTheFit:
    """`enabled = False` is display; the fitter reads `fixed`."""

    def setup_method(self):
        global_object.registry.clear()

    @staticmethod
    def _in_fit(model, parameter):
        return any(candidate is parameter for candidate in model.get_free_parameters())

    def test_field_is_fixed_when_the_last_magnetic_layer_goes(self):
        project = TestFieldEnablement._project(LayerMagnetism(rho_m=2.5))
        model = project.models[0]
        model.applied_field.fixed = False
        model.guide_field_angle.fixed = False

        model.sample[1].layers[0].magnetism = None
        project._sync_parameter_states()

        assert not self._in_fit(model, model.applied_field)
        assert not self._in_fit(model, model.guide_field_angle)

    def test_field_is_fixed_on_a_calculator_without_magnetism(self):
        project = TestFieldEnablement._project(None)
        model = project.models[0]
        model.applied_field.fixed = False

        project.calculator = 'refnx'

        assert not self._in_fit(model, model.applied_field)

    def test_field_of_a_magnetic_model_stays_free(self):
        project = TestFieldEnablement._project(LayerMagnetism(rho_m=2.5))
        model = project.models[0]
        model.applied_field.fixed = False

        project._sync_parameter_states()

        assert self._in_fit(model, model.applied_field)


class TestUnusedMagneticMaterialInputs:
    def setup_method(self):
        global_object.registry.clear()

    @staticmethod
    def _project():
        from easyreflectometry.sample import MaterialMagnetic

        project = TestFieldEnablement._project(None)
        film = project.models[0].sample[1].layers[0]
        iron = MaterialMagnetic(name='Fe')
        film.material = iron
        film.rho_m_coupled = True
        project._sync_parameter_states()
        return project, film, iron

    def test_moment_is_fixed_when_the_last_layer_uncouples(self):
        project, film, iron = self._project()
        iron.magnetic_moment.fixed = False
        project._sync_parameter_states()
        assert not iron.magnetic_moment.fixed

        film.rho_m_coupled = False
        project._sync_parameter_states()

        assert iron.magnetic_moment.fixed

    def test_density_of_a_decoupled_unused_material_is_fixed(self):
        project, film, iron = self._project()
        iron.sld_coupled = False
        iron.density.fixed = False
        project._sync_parameter_states()
        assert not iron.density.fixed  # still drives the coupled layer's rho_m

        film.rho_m_coupled = False
        project._sync_parameter_states()

        assert iron.density.fixed

    def test_a_moment_a_user_constraint_uses_stays_free(self):
        from easyreflectometry.constraints import constrain

        project, film, iron = self._project()
        iron.magnetic_moment.fixed = False
        film.rho_m_coupled = False
        constrain(film.magnetism.rho_m, '2 * mu', mu=iron.magnetic_moment)

        project._sync_parameter_states()

        assert not iron.magnetic_moment.fixed
