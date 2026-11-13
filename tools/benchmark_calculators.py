# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Per-backend calculator benchmark for the easyscience core 3 port.

Run once on the storage-based wrappers ("before") and once on the rebuilt ones
("after"), on the same machine, and compare per backend::

    pixi run python tools/benchmark_calculators.py --label before --out benchmark-before.json

The report records the machine name and local paths, so keep it out of the
repository.

Only the public model, calculator and fitting API is used, so the same script
runs on both sides of the port. Acceptance: no end-to-end fit of a backend more
than 25% slower than before.
"""

import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Callable

import easyscience
import numpy as np

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

WARMUP = 3
REPEATS = 7
CALLS_PER_REPEAT = 20
FIT_REPEATS = 3


def _model(engine: str, repeated: bool = False, magnetic: bool = False, name: str = 'Model', d2o: float = 6.36) -> Model:
    """The 5-layer model, or a 10-layer unit repeated 20 times."""
    air = Multilayer(Layer(Material(0.0, 0.0, 'Air'), 0.0, 0.0, 'Air'), 'Superphase')
    sub = Multilayer(Layer(Material(d2o, 0.0, 'D2O'), 0.0, 4.0, 'D2O'), 'Subphase')
    if repeated:
        unit = [Layer(Material(9.4 - i, 0.0, f'M{i}'), 20.0 + i, 3.0, f'L{i}') for i in range(10)]
        middle = [RepeatingMultilayer(unit, repetitions=20, name='Unit')]
    else:
        layers = [
            Layer(Material(3.47, 0.0, 'SiO2'), 15.0, 3.0, 'Oxide'),
            Layer(Material(9.4, 0.05, 'Ni'), 80.0, 5.0, 'Nickel'),
            Layer(Material(-1.9, 0.0, 'Ti'), 50.0, 4.0, 'Titanium'),
        ]
        if magnetic:
            layers[1].magnetism = LayerMagnetism(rho_m=1.5, theta_m=90.0)
        middle = [Multilayer(layers, 'Stack')]
    model = Model(Sample(air, *middle, sub), 1.0, 1e-7, PercentageFwhm(5.0), name)
    factory = CalculatorFactory()
    factory.switch(engine)
    model.interface = factory
    return model


def _stats(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    return {
        'median_ms': 1e3 * median(ordered),
        'min_ms': 1e3 * ordered[0],
        'max_ms': 1e3 * ordered[-1],
        'n': len(ordered),
    }


def _time_calls(call: Callable[[], object]) -> dict:
    for _ in range(WARMUP):
        call()
    per_call = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(CALLS_PER_REPEAT):
            call()
        per_call.append((time.perf_counter() - start) / CALLS_PER_REPEAT)
    return _stats(per_call)


def _evaluation(engine: str, repeated: bool, n_q: int) -> dict:
    q = np.linspace(0.005, 0.3, n_q)
    start = time.perf_counter()
    model = _model(engine, repeated=repeated)
    model.interface.fit_func(q, model.unique_name)
    cold = time.perf_counter() - start
    parameter = model.sample[1].layers[0].thickness

    def edited_call():
        # A value change before every call, as in a fit iteration.
        parameter.value = parameter.value + 1e-6
        return model.interface.fit_func(q, model.unique_name)

    return {'cold_build_and_first_call_ms': 1e3 * cold, 'evaluation': _time_calls(edited_call)}


def _polarized_cache(n_q: int) -> dict:
    q = np.linspace(0.005, 0.3, n_q)
    model = _model('refl1d', magnetic=True)
    calculator = model.interface()
    parameter = model.sample[1].layers[1].thickness

    def miss():
        parameter.value = parameter.value + 1e-6
        return calculator.reflectivity_profile_channel(q, model.unique_name, 'pp')

    def hit():
        return calculator.reflectivity_profile_channel(q, model.unique_name, 'mm')

    return {'cache_miss': _time_calls(miss), 'cache_hit': _time_calls(hit)}


def _perturb(model: Model) -> None:
    """Free a few parameters and move them off the truth."""
    for layer, factor in zip(model.sample[1].layers, (1.1, 0.95, 1.08)):
        layer.thickness.fixed = False
        layer.thickness.value = layer.thickness.value * factor
        layer.roughness.fixed = False
    model.scale.fixed = False


def _fit(build_experiments: Callable[[], list]) -> dict:
    seconds, evaluations = [], []
    for _ in range(FIT_REPEATS):
        experiments = build_experiments()
        fitter = MultiFitter.for_experiments(experiments)
        start = time.perf_counter()
        results = fitter.prepare().execute()
        seconds.append(time.perf_counter() - start)
        evaluations.append(getattr(results[0], 'n_evaluations', None))
    return {'fit': _stats(seconds), 'n_evaluations': evaluations}


def _contrast_experiments(engine: str, repeated: bool = False) -> list:
    q = np.linspace(0.005, 0.3, 300)
    experiments = []
    for i, d2o in enumerate((6.36, 2.0, -0.56)):
        model = _model(engine, repeated=repeated, name=f'Contrast{i}', d2o=d2o)
        y = np.asarray(model.interface.fit_func(q, model.unique_name))
        if not repeated:
            _perturb(model)
        else:
            unit = model.sample[1].layers
            unit[0].thickness.fixed = False
            unit[0].thickness.value = 22.0
        experiments.append(DataSet1D(name=f'D{i}', x=q, y=y, ye=(0.02 * y) ** 2, model=model))
    return experiments


def _polarized_experiments() -> list:
    q = np.linspace(0.005, 0.3, 300)
    model = _model('refl1d', magnetic=True, name='Polarized')
    channels = model.interface.polarized_reflectivity_profiles(q, model.unique_name)
    _perturb(model)
    datasets = {c: DataSet1D(name=c, x=q, y=y, ye=(0.02 * y) ** 2) for c, y in channels.items()}
    return [PolarizedDataSet(name='Polarized', channels=datasets, model=model)]


def _environment(label: str) -> dict:
    lib = Path(__file__).resolve().parents[1]
    core = Path(easyscience.__file__).resolve().parent

    def rev(checkout: Path) -> str:
        # An installed core lives inside the lib's .pixi env: only a source tree has its own .git.
        if not (checkout / '.git').exists():
            return 'installed distribution'
        result = subprocess.run(['git', '-C', str(checkout), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        return result.stdout.strip()

    return {
        'label': label,
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'machine': platform.node(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'refnx': version('refnx'),
        'refl1d': version('refl1d'),
        'easyscience_distribution': version('easyscience'),
        'easyscience_source': str(core),
        'easyscience_revision': rev(core.parents[1]),
        'lib_revision': rev(lib),
        'policy': f'{WARMUP} warm-up calls; {REPEATS} repeats of {CALLS_PER_REPEAT} calls; fits repeated {FIT_REPEATS}x',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for engine in ('refnx', 'refl1d'):
        rows[f'{engine}/five_layer/300q'] = _evaluation(engine, repeated=False, n_q=300)
        rows[f'{engine}/repeated_10x20/1000q'] = _evaluation(engine, repeated=True, n_q=1000)
        rows[f'{engine}/fit/five_layer_3_contrasts/300q'] = _fit(lambda: _contrast_experiments(engine))
        rows[f'{engine}/fit/repeated_10x20_3_contrasts/300q'] = _fit(lambda: _contrast_experiments(engine, True))
    rows['refl1d/polarized_cache/300q'] = _polarized_cache(300)
    rows['refl1d/fit/polarized_4_channels/300q'] = _fit(_polarized_experiments)

    report = {'environment': _environment(args.label), 'rows': rows}
    args.out.write_text(json.dumps(report, indent=2))
    for name, row in rows.items():
        timing = row.get('fit') or row.get('evaluation') or row.get('cache_miss')
        print(f'{name:45s} median {timing["median_ms"]:10.2f} ms')


if __name__ == '__main__':
    main()
