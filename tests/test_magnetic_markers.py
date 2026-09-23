# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The in-plane moment vector contract and the per-layer markers built on it.

These are the numbers every arrow view draws, so the convention is pinned here
rather than in the rendering: all four cardinal directions, the canted example
from the demo notebook, and the 180 degree flip a negative rho_m causes.
"""

import numpy as np
import pytest
from easyscience import global_object

from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import GUIDE_FIELD_ANGLE
from easyreflectometry.project import Project
from easyreflectometry.project import _expanded_layers
from easyreflectometry.project import magnetic_vector_for_layer
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample


@pytest.fixture(autouse=True)
def _isolated_global_object():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _project(*assemblies) -> Project:
    model = Model(sample=Sample(*assemblies, name='Sample'), scale=1, background=0, name='Magnetic Model')
    model.resolution_function = PercentageFwhm(0)
    project = Project()
    project.calculator = 'refl1d'
    project.models = ModelCollection(model)
    return project


def _layer(name, sld=4.0, thickness=100.0, roughness=0.0, magnetism=None) -> Layer:
    return Layer(
        material=Material(sld=sld, isld=0, name=f'{name} Material'),
        thickness=thickness,
        roughness=roughness,
        magnetism=magnetism,
        name=name,
    )


def _simple_project(rho_m=2.5, theta_m=GUIDE_FIELD_ANGLE, roughness=(0.0, 0.0, 0.0)) -> Project:
    return _project(
        Multilayer(_layer('Superphase', sld=0.0, thickness=0.0, roughness=roughness[0]), name='Super'),
        Multilayer(
            _layer('Film', thickness=100.0, roughness=roughness[1], magnetism=LayerMagnetism(rho_m=rho_m, theta_m=theta_m)),
            name='Film',
        ),
        Multilayer(_layer('Substrate', sld=2.047, thickness=0.0, roughness=roughness[2]), name='Sub'),
    )


class TestMagneticVector:
    """The Section 2 contract: phi is measured from the guide field H."""

    @pytest.mark.parametrize(
        ('theta_m', 'phi'),
        [
            (270.0, 0.0),  # along H: collinear, no spin flip
            (90.0, 180.0),  # against H
            (0.0, 90.0),  # fully transverse
            (180.0, 270.0),  # fully transverse, the other way
            (40.0, 130.0),  # the canted example of the demo notebook
        ],
    )
    def test_all_four_cardinals_and_the_canted_example(self, theta_m, phi):
        vector = magnetic_vector_for_layer(LayerMagnetism(rho_m=2.0, theta_m=theta_m))

        assert vector['phi'] == pytest.approx(phi)
        assert vector['phi_param'] == pytest.approx(phi)
        assert vector['m'] == pytest.approx(2.0)

    def test_negative_rho_m_flips_the_drawn_direction_by_180_degrees(self):
        positive = magnetic_vector_for_layer(LayerMagnetism(rho_m=2.0, theta_m=40.0))
        negative = magnetic_vector_for_layer(LayerMagnetism(rho_m=-2.0, theta_m=40.0))

        # The parameter angle is a property of theta_m alone ...
        assert negative['phi_param'] == pytest.approx(positive['phi_param'])
        # ... the physical direction flips, with a positive magnitude.
        assert negative['phi'] == pytest.approx((positive['phi'] + 180.0) % 360.0)
        assert negative['m'] == pytest.approx(2.0)
        assert negative['rho_m'] == pytest.approx(-2.0)

    def test_the_sign_crossing_is_exactly_180_degrees(self):
        below = magnetic_vector_for_layer(LayerMagnetism(rho_m=-1e-9, theta_m=200.0))
        above = magnetic_vector_for_layer(LayerMagnetism(rho_m=+1e-9, theta_m=200.0))

        assert (below['phi'] - above['phi']) % 360.0 == pytest.approx(180.0)

    def test_zero_moment_keeps_the_parameter_angle_and_no_magnitude(self):
        vector = magnetic_vector_for_layer(LayerMagnetism(rho_m=0.0, theta_m=40.0))

        assert vector['m'] == 0.0
        assert vector['phi'] == pytest.approx(130.0)

    def test_components_split_the_moment_between_the_channel_types(self):
        vector = magnetic_vector_for_layer(LayerMagnetism(rho_m=2.0, theta_m=GUIDE_FIELD_ANGLE - 60.0))

        # 60 degrees off H: cos(60) along it, sin(60) across it.
        assert vector['m_par'] == pytest.approx(2.0 * 0.5)
        assert abs(vector['m_perp']) == pytest.approx(2.0 * np.sqrt(3) / 2)
        assert np.hypot(vector['m_par'], vector['m_perp']) == pytest.approx(vector['m'])


class TestMagneticLayerMarkers:
    def test_marker_keys_and_the_non_magnetic_layers_left_out(self):
        markers = _simple_project(theta_m=40.0).magnetic_layer_markers_for_model_at_index(0)

        assert [marker['label'] for marker in markers] == ['Film']
        assert set(markers[0]) == {
            'label',
            'z_min',
            'z_max',
            'z_center',
            'has_moment',
            'rho_m',
            'theta_m',
            'phi_param',
            'phi',
            'm',
            'm_par',
            'm_perp',
        }
        assert markers[0]['phi'] == pytest.approx(130.0)
        assert markers[0]['z_center'] == pytest.approx(0.5 * (markers[0]['z_min'] + markers[0]['z_max']))

    def test_a_non_magnetic_model_raises(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        with pytest.raises(ValueError, match='no magnetic layer'):
            project.magnetic_layer_markers_for_model_at_index(0)

    @staticmethod
    def _brackets_the_magnetic_step(project, marker) -> bool:
        """Whether the marker's extent really covers where rho_m(z) is large."""
        rho_m = project.magnetic_sld_data_for_model_at_index(0)['rho_m']
        inside = (rho_m.x >= marker['z_min']) & (rho_m.x <= marker['z_max'])
        # The moment lives inside the marker and (roughness tails aside) not far outside it.
        return np.abs(rho_m.y[inside]).max() == pytest.approx(np.abs(rho_m.y).max(), rel=1e-6)

    def test_extents_are_in_the_profile_frame_not_thickness_from_zero(self):
        # Strongly asymmetric roughness: refl1d pads the profile by 3 sigma at
        # each end, so the padding before the first interface differs from the
        # padding after the last and no fixed offset could absorb it.
        project = _simple_project(roughness=(0.5, 20.0, 2.0))
        profiles = project.magnetic_sld_data_for_model_at_index(0)
        z = profiles['rho_m'].x

        (marker,) = project.magnetic_layer_markers_for_model_at_index(0)

        assert marker['z_max'] - marker['z_min'] == pytest.approx(100.0)
        # Naive accumulation from zero would have put the layer at 0..100.
        assert marker['z_min'] != pytest.approx(0.0)
        assert z[0] <= marker['z_min'] < marker['z_max'] <= z[-1]
        assert self._brackets_the_magnetic_step(project, marker)

    def test_the_magnetic_step_of_a_buried_layer_falls_inside_its_marker(self):
        project = _project(
            Multilayer(_layer('Superphase', sld=0.0, thickness=0.0, roughness=1.0), name='Super'),
            Multilayer(_layer('Cap', sld=3.0, thickness=60.0, roughness=4.0), name='Cap'),
            Multilayer(
                _layer('Fe', thickness=45.0, roughness=6.0, magnetism=LayerMagnetism(rho_m=3.0, theta_m=40.0)),
                name='Fe',
            ),
            Multilayer(_layer('Substrate', sld=2.047, thickness=0.0, roughness=9.0), name='Sub'),
        )

        (marker,) = project.magnetic_layer_markers_for_model_at_index(0)

        assert marker['z_max'] - marker['z_min'] == pytest.approx(45.0)
        assert self._brackets_the_magnetic_step(project, marker)

    def test_every_repeat_of_a_multilayer_counts_as_a_layer(self):
        # A repeat contributes its layers once per repetition to the stack the
        # markers are anchored against. It cannot be exercised through a
        # profile: the refl1d calculator attaches a (zero) Magnetism to *every*
        # slab as soon as one layer is magnetic, and refl1d refuses to repeat
        # slabs that carry magnetism at all ('Repeated magnetic layers not
        # implemented'). A magnetic model with a repeat therefore has no
        # magnetic profile to annotate in the first place.
        sample = Sample(
            Multilayer(_layer('Superphase', sld=0.0, thickness=0.0), name='Super'),
            RepeatingMultilayer(
                [_layer('A', sld=3.0, thickness=20.0), _layer('B', sld=2.047, thickness=30.0)],
                repetitions=3,
                name='Rep',
            ),
            Multilayer(_layer('Substrate', sld=2.047, thickness=0.0), name='Sub'),
            name='Sample',
        )

        layers = _expanded_layers(sample)

        assert [layer.name for layer in layers] == ['Superphase'] + ['A', 'B'] * 3 + ['Substrate']

    def test_a_magnetic_substrate_and_superphase_are_clamped_to_the_profile(self):
        project = _project(
            Multilayer(
                _layer('Superphase', sld=0.0, thickness=0.0, magnetism=LayerMagnetism(rho_m=1.0, theta_m=0.0)),
                name='Super',
            ),
            Multilayer(_layer('Film', thickness=80.0, roughness=3.0), name='Film'),
            Multilayer(
                _layer('Substrate', sld=2.047, thickness=0.0, magnetism=LayerMagnetism(rho_m=2.0, theta_m=270.0)),
                name='Sub',
            ),
        )
        z = project.magnetic_sld_data_for_model_at_index(0)['rho_m'].x

        superphase, substrate = project.magnetic_layer_markers_for_model_at_index(0)

        # Semi-infinite media have no finite centre: they take the visible extent.
        assert superphase['z_min'] == pytest.approx(z[0])
        assert substrate['z_max'] == pytest.approx(z[-1])
        assert superphase['z_max'] == pytest.approx(substrate['z_min'] - 80.0, abs=1e-6)

    def test_roughness_padding_larger_than_the_thickness_still_returns_a_marker(self):
        project = _project(
            Multilayer(_layer('Superphase', sld=0.0, thickness=0.0, roughness=1.0), name='Super'),
            Multilayer(
                _layer('Thin', thickness=2.0, roughness=25.0, magnetism=LayerMagnetism(rho_m=3.0, theta_m=90.0)),
                name='Thin',
            ),
            Multilayer(_layer('Substrate', sld=2.047, thickness=0.0, roughness=1.0), name='Sub'),
        )
        z = project.magnetic_sld_data_for_model_at_index(0)['rho_m'].x

        (marker,) = project.magnetic_layer_markers_for_model_at_index(0)

        assert z[0] <= marker['z_min'] <= marker['z_max'] <= z[-1]
        assert marker['z_center'] == pytest.approx(0.5 * (marker['z_min'] + marker['z_max']))

    def test_a_negligible_moment_is_marked_as_having_none(self):
        project = _project(
            Multilayer(_layer('Superphase', sld=0.0, thickness=0.0), name='Super'),
            Multilayer(
                _layer('Strong', thickness=40.0, magnetism=LayerMagnetism(rho_m=4.0, theta_m=270.0)),
                name='Strong',
            ),
            Multilayer(
                _layer('Faint', sld=3.0, thickness=40.0, magnetism=LayerMagnetism(rho_m=0.01, theta_m=270.0)),
                name='Faint',
            ),
            Multilayer(_layer('Substrate', sld=2.047, thickness=0.0), name='Sub'),
        )

        strong, faint = project.magnetic_layer_markers_for_model_at_index(0)

        # 0.01 is below 1 % of 4.0, the same floor that masks the theta_m curve.
        assert strong['has_moment'] is True
        assert faint['has_moment'] is False
