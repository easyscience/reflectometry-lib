# Technology Stack

**Analysis Date:** 2026-10-15 **Branch analysed:** `develop` @
`66adaf54` (tag `v1.7.0-51`; last release `v1.8.0` on 2026-09-18,
develop carries the unreleased 1.9 work listed under "Unreleased" in
`CHANGELOG.md`)

## Languages

**Primary:**

- Python 3.11–3.13 - All source code under `src/easyreflectometry/`
  (~18.5k lines) and `tests/` (~21k lines)

**Secondary:**

- Jupyter Notebook (.ipynb) - Tutorials in `docs/docs/tutorials/`
  (simulation, fitting, advancedfitting) and standalone notebooks in
  `notebooks/`
- Markdown - MkDocs site under `docs/docs/`, `CHANGELOG.md`

## Runtime

**Environment:**

- CPython 3.11, 3.12, 3.13 (`requires-python = '>=3.11'`)
- The pixi `default` environment is Python 3.13 with the `dev` extra
  installed editable; `py-311-env` / `py-313-env` exist for CI matrices.

**Package Manager:**

- Pixi (`pixi.toml` + `pixi.lock`; conda-forge channel only) is the
  canonical developer environment and what CI uses (`pixi run ...`).
- pip for end users (`pip install easyreflectometry[dev]`).
- A bare `python` on this machine is a conda environment with an old
  easyscience; always use `pixi run python` for tests.

## Frameworks

**Core:**

- EasyScience `easyscience>=2.5.1,<3` (PyPI; pixi resolves 2.5.1) -
  Parameter model (`Parameter`, `DescriptorNumber`), dependency
  resolver, `EasyList`/`ModelBase` base classes, serializer,
  `Fitter`/`MultiFitter`, `Sampler`/`SamplingResults`, minimizer
  wrappers (LMFit, Bumps, DFO), `InterfaceFactoryTemplate`
- scipp 26.3 - `sc.DataGroup`/`sc.array` with variances as the fit I/O
  container
- refnx 0.1.63 - Reflectivity engine (unpolarized only)
- refl1d >= 1.0.0 (1.0.1 installed) - Reflectivity engine, including
  magnetism / polarized channels (`pp`, `pm`, `mp`, `mm`)
- bumps 1.0.4 - DREAM MCMC sampler and `Bumps_*` minimizers (via core);
  `FitProblem.constraints` is also the carrier for inequality penalties
- lmfit 1.3.4 - Default minimizer `LMFit_leastsq` (via core)
- orsopy >= 1.2 (1.2.3) - ORSO `.ort` text and `.orb` nexus I/O, model
  language (`resolve_stack`) for sample import/export
- asteval - Safe expression evaluation for inequality constraints
- periodictable - Scattering lengths / formulas (via `special/`,
  `contrasts.py`)
- scipy 1.17 - `erf` smearing in volume-fraction profiles, Bayesian
  helpers
- plotly - Interactive corner/trace/distribution plots and interactive
  HTML summary figures
- matplotlib - Static SLD / fit plots in `plot.py` and the PDF summary
- xhtml2pdf (+ `svglib<1.6` on Linux/macOS) - PDF export of the HTML
  summary
- h5py, pooch, pyyaml - Nexus reading, example-data download, YAML
  `repr`

**Testing:**

- pytest with `--import-mode=importlib` and `-m "not slow"` in
  `addopts`; markers `fast` and `slow`
- pytest-cov, pytest-xdist, nbmake (notebook tests)

**Build/Dev:**

- hatchling build backend; versioningit derives the version from `v*`
  tags (`default-tag = 'v999.0.0'` when untagged)
- Ruff (`line-length = 128`, single quotes, rules E/W/F/I/S, isort
  `force-single-line`)
- pre-commit, prettier (non-Python formatting), pydoclint /
  format-docstring (configured but disabled via `exclude = '\.'`),
  interrogate (`fail-under = 0`), radon, spdx-headers,
  validate-pyproject
- copier (`.copier-answers.yml`) - repo scaffold synced from
  `gh:easyscience/templates` (`template_type=lib`)
- MkDocs Material + mkdocstrings + mkdocs-jupyter + mike (versioned docs
  on `gh-pages`)

## Key Dependencies

**Critical (fitting pipeline):**

- `easyscience` - `MultiFitter`, `Sampler`, `AvailableMinimizers`,
  `FitResults`, `MINIMIZER_PARAMETER_PREFIX`; the library subclasses the
  core `MultiFitter` and monkeypatches `minimizer_bumps.FitDriver` (see
  `_bumps_constraints.py`)
- `bumps` - DREAM, `FitProblem`, `Parameter`; required for inequality
  constraints and MCMC
- `numpy` - Array math everywhere

**Infrastructure:**

- `scipp` - `DataGroup` is the public fit I/O format
- `orsopy` - Only supported structured data/model exchange format

## Configuration

**Environment:**

- No `.env`; `PYTHONIOENCODING=utf-8` set by pixi activation
- `JUPYTER_PLATFORM_DIRS=1` and `PYTHONWARNINGS` set by the `docs-*`
  pixi tasks

**Build:**

- `pyproject.toml` - dependencies, pytest, ruff, coverage, versioningit
- `pixi.toml` - environments and the task vocabulary CI calls
  (`unit-tests`, `unit-tests-all`, `slow-tests`, `integration-tests`,
  `functional-tests`, `notebook-tests`, `py-lint-check`, `docs-build`,
  `dist-build`, ...)
- `.pre-commit-config.yaml`, `prettierrc.toml`, `.badgery.yaml`,
  `.codecov.yml`

## Platform Requirements

**Development:**

- Windows (win-64), Linux (linux-64), macOS 14+ (osx-arm64) per
  `pixi.toml` platforms; CI matrices run all three
- `svglib` is only installed on Linux/macOS (pin `<1.6`)

**Production:**

- Pure-Python wheel on PyPI; engines (refnx, refl1d) ship compiled
  extensions via their own wheels

---

_Stack analysis: 2026-10-15 (develop @ 66adaf54)_
