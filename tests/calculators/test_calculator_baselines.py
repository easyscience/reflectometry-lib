# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Migration baselines for the calculator port to easyscience core 3.

The arrays in ``baselines/<engine>.npz`` were recorded with the storage-based
wrappers on ``develop`` (easyscience 2.5.1), before any wrapper was touched.
Every rewrite of the calculator layer must reproduce them for the same engine
and settings (CORE-3-DESIGN.md, 4.4.4). Each case also edits the model after a
first evaluation, so the path that brings new values to the backend is
compared, not only the first build.

Record (only on the old wrappers, never to make a failing port pass)::

    pixi run python tests/calculators/test_calculator_baselines.py --record
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer

BASELINE_DIR = Path(__file__).parent / 'baselines'
ENGINES = ('refnx', 'refl1d')
Q = np.linspace(0.005, 0.3, 60)


def _attach(model: Model, engine: str) -> Model:
    factory = CalculatorFactory()
    factory.switch(engine)
    model.interface = factory
    return model


def _reflectivity(model: Model) -> np.ndarray:
    return np.asarray(model.interface.fit_func(Q, model.unique_name))


def _profile(model: Model, prefix: str = '') -> dict[str, np.ndarray]:
    z, sld = model.interface.sld_profile(model.unique_name)
    return {f'{prefix}sld_z': np.asarray(z), f'{prefix}sld': np.asarray(sld)}


def _film_model(engine: str) -> Model:
    ambient = Material(0.0, 0.0, 'Ambient')
    film = Material(3.45, 0.0, 'Film')
    substrate = Material(2.07, 0.0, 'Substrate')
    sample = Sample(
        Multilayer(
            [
                Layer(ambient, 0.0, 0.0, 'AmbientLayer'),
                Layer(film, 100.0, 3.0, 'FilmLayer'),
                Layer(substrate, 0.0, 3.0, 'SubstrateLayer'),
            ],
            'Item',
        )
    )
    return _attach(Model(sample, 1.0, 0.0, PercentageFwhm(1.0), 'Film'), engine)


def case_film(engine: str) -> dict[str, np.ndarray]:
    model = _film_model(engine)
    out = {'R': _reflectivity(model), **_profile(model)}
    film_layer = model.sample[0].layers[1]
    film_layer.thickness.value = 150.0
    film_layer.material.sld.value = 4.0
    model.background.value = 1e-7
    out['R_edited'] = _reflectivity(model)
    return out


def case_five_layer(engine: str) -> dict[str, np.ndarray]:
    air = Layer(Material(0.0, 0.0, 'Air'), 0.0, 0.0, 'AirLayer')
    oxide = Layer(Material(3.47, 0.0, 'SiO2'), 15.0, 3.0, 'OxideLayer')
    nickel = Layer(Material(9.4, 0.05, 'Ni'), 80.0, 5.0, 'NickelLayer')
    titanium = Layer(Material(-1.9, 0.0, 'Ti'), 50.0, 4.0, 'TitaniumLayer')
    d2o = Layer(Material(6.36, 0.0, 'D2O'), 0.0, 4.0, 'D2OLayer')
    sample = Sample(Multilayer([air, oxide, nickel, titanium, d2o], 'Stack'))
    model = _attach(Model(sample, 0.9, 1e-6, PercentageFwhm(5.0), 'FiveLayer'), engine)
    out = {'R': _reflectivity(model), **_profile(model)}
    nickel.material.isld.value = 0.2
    titanium.roughness.value = 8.0
    model.scale.value = 1.1
    out['R_edited'] = _reflectivity(model)
    return out


def case_mixture(engine: str) -> dict[str, np.ndarray]:
    h2o = Material(-0.56, 0.0, 'H2O')
    d2o = Material(6.36, 0.0, 'D2O')
    mixture = MaterialMixture(h2o, d2o, 0.3, 'Water')
    sample = Sample(
        Multilayer(
            [
                Layer(Material(0.0, 0.0, 'Air'), 0.0, 0.0, 'AirLayer'),
                Layer(Material(2.0, 0.0, 'Polymer'), 60.0, 3.0, 'PolymerLayer'),
                Layer(mixture, 0.0, 3.0, 'WaterLayer'),
            ],
            'Stack',
        )
    )
    model = _attach(Model(sample, 1.0, 0.0, PercentageFwhm(2.0), 'Mixture'), engine)
    out = {'R': _reflectivity(model)}
    mixture.fraction.value = 0.6
    out['R_fraction'] = _reflectivity(model)
    d2o.sld.value = 5.9
    out['R_component'] = _reflectivity(model)
    return out


def case_repeated(engine: str) -> dict[str, np.ndarray]:
    nickel = Material(9.4, 0.0, 'Ni')
    titanium = Material(-1.9, 0.0, 'Ti')
    repeated = RepeatingMultilayer(
        [Layer(nickel, 40.0, 3.0, 'NiLayer'), Layer(titanium, 60.0, 3.0, 'TiLayer')],
        repetitions=5,
        name='NiTi',
    )
    sample = Sample(
        Multilayer(Layer(Material(0.0, 0.0, 'Air'), 0.0, 0.0, 'AirLayer'), 'Superphase'),
        repeated,
        Multilayer(Layer(Material(2.07, 0.0, 'Si'), 0.0, 3.0, 'SiLayer'), 'Subphase'),
    )
    model = _attach(Model(sample, 1.0, 0.0, PercentageFwhm(2.0), 'Repeated'), engine)
    out = {'R': _reflectivity(model), **_profile(model)}
    repeated.repetitions.value = 1
    out['R_single'] = _reflectivity(model)
    repeated.repetitions.value = 10
    repeated.layers[0].thickness.value = 30.0
    out['R_ten'] = _reflectivity(model)
    return out


def case_shared(engine: str) -> dict[str, np.ndarray]:
    """One material in two layers of one model and in a second model; one factory."""
    factory = CalculatorFactory()
    factory.switch(engine)
    air = Material(0.0, 0.0, 'Air')
    shared = Material(4.0, 0.0, 'Shared')
    spacer = Material(1.0, 0.0, 'Spacer')
    si = Material(2.07, 0.0, 'Si')

    def build(name: str, thickness: float) -> Model:
        sample = Sample(
            Multilayer(
                [
                    Layer(air, 0.0, 0.0, f'{name}Air'),
                    Layer(shared, thickness, 3.0, f'{name}SharedTop'),
                    Layer(spacer, 30.0, 3.0, f'{name}Spacer'),
                    Layer(shared, thickness, 3.0, f'{name}SharedBottom'),
                    Layer(si, 0.0, 3.0, f'{name}Si'),
                ],
                f'{name}Stack',
            )
        )
        model = Model(sample, 1.0, 0.0, PercentageFwhm(2.0), name)
        model.interface = factory
        return model

    first, second = build('First', 40.0), build('Second', 80.0)
    out = {'R_first': _reflectivity(first), 'R_second': _reflectivity(second)}
    shared.sld.value = 5.0
    out['R_first_edited'] = _reflectivity(first)
    out['R_second_edited'] = _reflectivity(second)
    return out


def case_resolution(engine: str) -> dict[str, np.ndarray]:
    """Model resolution versus a per-dataset resolution of the same model."""
    model = _film_model(engine)
    model.resolution_function = LinearSpline(np.array([0.0, 0.15, 0.3]), np.array([0.0005, 0.002, 0.006]))
    sigma_squared = (0.03 * Q / 2.3548) ** 2
    pointwise = Pointwise([Q, np.ones_like(Q), sigma_squared])
    return {
        'R_model_resolution': _reflectivity(model),
        'R_dataset_resolution': np.asarray(model.interface.fit_func_for(resolution_function=pointwise)(Q, model.unique_name)),
        'R_percentage': np.asarray(model.interface.fit_func_for(resolution_function=PercentageFwhm(3.0))(Q, model.unique_name)),
    }


def case_surfactant(engine: str) -> dict[str, np.ndarray]:
    surfactant = SurfactantLayer()
    sample = Sample(
        Multilayer(Layer(Material(0.0, 0.0, 'Air'), 0.0, 0.0, 'AirLayer'), 'Superphase'),
        surfactant,
        Multilayer(Layer(Material(6.36, 0.0, 'D2O'), 0.0, 3.0, 'D2OLayer'), 'Subphase'),
    )
    model = _attach(Model(sample, 1.0, 0.0, PercentageFwhm(2.0), 'Surfactant'), engine)
    out = {'R': _reflectivity(model)}
    surfactant.tail_layer.thickness.value = 20.0
    surfactant.tail_layer.area_per_molecule.value = 55.0
    out['R_edited'] = _reflectivity(model)
    return out


def case_magnetic(engine: str) -> dict[str, np.ndarray]:
    """All four spin channels, the magnetic profile and a per-dataset channel fit function."""
    layer = Layer(Material(4.0, 0.0, 'Sld 4'), 100.0, 3.0, 'MagneticLayer')
    sample = Sample(
        Multilayer(Layer(Material(0.0, 0.0, 'Vacuum'), 0.0, 0.0, 'VacuumLayer'), 'Superphase'),
        Multilayer(layer, 'Film'),
        Multilayer(Layer(Material(2.047, 0.0, 'Si'), 0.0, 3.0, 'SiLayer'), 'Subphase'),
    )
    model = _attach(Model(sample, 1.0, 0.0, PercentageFwhm(2.0), 'Magnetic'), engine)
    layer.magnetism = LayerMagnetism(rho_m=2.0, theta_m=45.0)

    def channels(prefix: str) -> dict[str, np.ndarray]:
        profiles = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        return {f'{prefix}{channel}': np.asarray(values) for channel, values in profiles.items()}

    out = channels('R_')
    z, sld, rho_m, theta_m = model.interface.magnetic_sld_profile(model.unique_name)
    out.update(mag_z=np.asarray(z), mag_sld=np.asarray(sld), mag_rho_m=np.asarray(rho_m), mag_theta_m=np.asarray(theta_m))
    pointwise = Pointwise([Q, np.ones_like(Q), (0.03 * Q / 2.3548) ** 2])
    out['R_mm_dataset_resolution'] = np.asarray(
        model.interface.fit_func_for(channel='mm', resolution_function=pointwise)(Q, model.unique_name)
    )
    layer.magnetism.rho_m.value = 3.0
    layer.magnetism.theta_m.value = 30.0
    out.update(channels('R_edited_'))
    model.guide_field_angle.value = 0.0
    out.update(channels('R_guide_'))
    return out


CASES = {
    'film': (case_film, ENGINES),
    'five_layer': (case_five_layer, ENGINES),
    'mixture': (case_mixture, ENGINES),
    'repeated': (case_repeated, ENGINES),
    'shared': (case_shared, ENGINES),
    'resolution': (case_resolution, ENGINES),
    'surfactant': (case_surfactant, ENGINES),
    'magnetic': (case_magnetic, ('refl1d',)),
}


def _baseline_path(engine: str) -> Path:
    return BASELINE_DIR / f'{engine}.npz'


@pytest.fixture
def clear():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


@pytest.mark.parametrize(
    ('engine', 'case'),
    [(engine, case) for case, (_, engines) in CASES.items() for engine in engines],
)
def test_matches_recorded_baseline(clear, engine, case):
    # Given
    recorded = np.load(_baseline_path(engine))
    expected = {key.split('.', 1)[1]: recorded[key] for key in recorded.files if key.startswith(f'{case}.')}

    # When
    actual = CASES[case][0](engine)

    # Then
    assert set(actual) == set(expected)
    for key, values in expected.items():
        assert_allclose(actual[key], values, rtol=1e-10, atol=1e-14, err_msg=f'{engine}/{case}/{key}')


def record() -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    for engine in ENGINES:
        arrays = {}
        for case, (build, engines) in CASES.items():
            if engine in engines:
                arrays.update({f'{case}.{key}': value for key, value in build(engine).items()})
        np.savez_compressed(_baseline_path(engine), **arrays)
        print(f'{_baseline_path(engine)}: {len(arrays)} arrays')


if __name__ == '__main__':
    if '--record' not in sys.argv:
        sys.exit(__doc__)
    record()
