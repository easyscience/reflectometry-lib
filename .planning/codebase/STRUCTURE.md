# Codebase Structure

**Analysis Date:** 2026-10-15 **Branch analysed:** `develop` @
`66adaf54`

## Directory Layout

```
reflectometry-lib/
├── src/easyreflectometry/              # Main package (~18.5k lines)
│   ├── __init__.py                     # Exports Project, PosteriorResults, constraint helpers, InequalitySpec, UnitError
│   ├── project.py                      # Project (3211 lines): experiments, models, persistence, fit orchestration
│   ├── fitting.py                      # MultiFitter, PreparedFit, FitRun, FitScopeError, FitPreconditionError (1428 lines)
│   ├── fit_settings.py                 # FitSettings dataclass, MinimizerOption schema, objective/minimizer aliases
│   ├── _bumps_constraints.py           # FitDriver shim attaching inequality penalties to BUMPS problems
│   ├── constraints.py                  # constrain / constrain_equal / unconstrain / constrain_to_sum / derived_parameter
│   ├── inequality_constraints.py       # InequalitySpec, asteval evaluation, build_constraints_factory, UnitError
│   ├── contrasts.py                    # derive_contrast, ReplaceMaterial/ReplaceFormula, plan_link/LinkPlan/LinkRecord
│   ├── limits.py                       # apply_default_limits (SLD, scale, percentage bounds)
│   ├── orso_utils.py                   # ORSO .ort/.orb read, model-language import/export (891 lines)
│   ├── plot.py                         # matplotlib plot(DataGroup), plot_volume_fraction
│   ├── utils.py                        # get_as_parameter, yaml_dump, parameter counting helpers
│   ├── main.py                         # Prints available calculators (no console_scripts entry)
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── bayesian.py                 # PosteriorResults, plot_corner/trace/distribution (plotly), credible_intervals,
│   │                                   #   posterior predictive, save_posterior/load_posterior (1412 lines)
│   ├── calculators/
│   │   ├── __init__.py                 # CalculatorBase, CalculatorFactory, PolarizationChannel
│   │   ├── calculator_base.py          # Abstract calculator: create/claim items, per-model resolution, magnetism toggles
│   │   ├── wrapper_base.py             # Abstract engine wrapper incl. calculate_channel / calculate_polarized
│   │   ├── factory.py                  # CalculatorFactory: fit_func, fit_func_for(channel, resolution_function)
│   │   ├── polarization.py             # PolarizationChannel enum (pp, pm, mp, mm) + index map
│   │   ├── refnx/{calculator,wrapper}.py
│   │   └── refl1d/{calculator,wrapper}.py   # magnetism, polarized probes, oversampled resolution
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_store.py               # DataSet1D (x, y, ye, xe, model, resolution_function, include_in_fit)
│   │   ├── measurement.py              # load / load_as_dataset / merge_datasets / resolution_from_dataset
│   │   └── polarized.py                # PolarizedDataSet, channel detection from ORSO header or filename
│   ├── model/
│   │   ├── model.py                    # Model (scale, background, resolution_function, total_thickness, has_magnetism)
│   │   ├── model_collection.py         # ModelCollection with colour palette
│   │   └── resolution_functions.py     # ResolutionFunction, PercentageFwhm, LinearSpline, Pointwise
│   ├── sample/
│   │   ├── __init__.py
│   │   ├── base_core.py                # BaseCore(ModelBase): interface binding, to_dict/from_dict
│   │   ├── paths.py                    # walk(): structural paths like models/0/sample/1/layers/0/thickness
│   │   ├── references.py               # @ref / @ref_id shared-object serialization (writing/reading contexts)
│   │   ├── volume_fraction.py          # VolumeFractionProfile, volume_fraction_profile (erf-smeared occupancy)
│   │   ├── assemblies/                 # BaseAssembly, Multilayer, RepeatingMultilayer, SurfactantLayer, Bilayer, GradientLayer
│   │   ├── collections/                # BaseCollection(EasyList), LayerCollection, MaterialCollection, Sample
│   │   └── elements/
│   │       ├── layers/                 # Layer, LayerAreaPerMolecule, LayerMagnetism (rho_m, theta_m)
│   │       └── materials/             # Material, MaterialDensity (sld_coupled), MaterialMixture, MaterialSolvated
│   ├── special/                        # parse_formula, scattering-length and SLD calculations
│   └── summary/                        # Summary: HTML/PDF report with parameter tables, goodness of fit, plotly figures
├── tests/                              # ~21k lines, 1410 tests (34 marked slow)
│   ├── _static/                        # example.ort, example.orb, e361r.txt (+ PROVENANCE.md)
│   ├── calculators/                    # refnx/refl1d wrappers, polarization, resolution conventions, recycled names
│   ├── data/                           # DataSet1D, PolarizedDataSet
│   ├── model/, sample/, special/, summary/
│   ├── unit/                           # constraints, inequality constraints, bumps shim, fit settings, project core, derived params
│   ├── functional/                     # placeholder (test_dummy.py)
│   ├── integration/                    # convergence/ (reference costs), cross-engine resolution, .ort fitting
│   ├── test_fitting.py                 # MultiFitter fit + MCMC (TestMCMCSample*)
│   ├── test_project.py, test_shared_persistence.py, test_contrasts.py, test_prepared_fit.py,
│   ├── test_per_dataset_resolution.py, test_polarized_fitting.py, test_bayesian.py, test_orso_*.py, ...
│   └── package_test.py
├── docs/
│   ├── mkdocs.yml                      # MkDocs Material + mkdocstrings + mkdocs-jupyter + mike
│   └── docs/
│       ├── user-guide/resolution.md    # which resolution a model/dataset/fit uses
│       ├── tutorials/{basic,simulation,fitting,advancedfitting}/   # notebooks incl. bayesian_bumps, constraints, contrasts, polarized_fitting
│       └── api-reference/              # one page per module (constraints, inequality_constraints, contrasts, volume_fraction, ...)
├── notebooks/                          # ess_hercules_2026, polarized_fitting, zero_variance_fitting (not in docs build)
├── tools/                              # license_headers.py, update_docs_assets.py, update_github_labels.py
├── .github/workflows/                  # 24 workflows (see INTEGRATIONS.md)
├── pyproject.toml, pixi.toml, pixi.lock
├── CHANGELOG.md                        # "Unreleased" section is the develop delta since 1.8.0
├── .copier-answers.yml                 # scaffold from gh:easyscience/templates
└── Makefile, .coveragerc               # stale (reference setup.py / EasyReflectometry/ dir)
```

Untracked in the working copy (not on `develop`): `.planning/`,
`.claude/`, `MD/`, `examples/`, `integration/`.

## Directory Purposes

**`src/easyreflectometry/` (root modules):**

- `project.py` is the application-facing facade used by the GUI:
  experiments dict, models, materials palette, persistence, fit/sample
  preparation, contrasts/links, inequality constraints, ORSO import,
  spin asymmetry, derived data for plots.
- `fitting.py` holds the fitting pipeline; `fit_settings.py` the
  serializable settings it runs from; `_bumps_constraints.py` the engine
  shim. These three plus `inequality_constraints.py` form the fit layer.
- `constraints.py` and `contrasts.py` build on EasyScience parameter
  dependencies: equality constraints persisted by structural path, and
  contrast derivation / linking between models.

**`src/easyreflectometry/sample/`:**

- Object tree (`Sample` → assemblies → layers → materials). `paths.py`
  gives stable structural paths; `references.py` lets shared objects
  serialize once (`file_format=3`); `volume_fraction.py` derives
  occupancy profiles from the tree alone.

**`src/easyreflectometry/calculators/`:**

- Engine abstraction. `CalculatorBase` owns item creation and the
  per-model resolution registry; `WrapperBase` is the per-engine
  translation layer; `CalculatorFactory` hands out bound fit functions
  (`fit_func_for(channel, resolution_function)`).

**`src/easyreflectometry/data/`:**

- `DataSet1D` carries its measured resolution and `include_in_fit`;
  `PolarizedDataSet` groups channel datasets sharing one model;
  `measurement.py` loads text/ORSO files and merges datasets of one
  contrast.

**`tests/`:**

- Flat `test_*.py` for cross-cutting features, mirrored sub-packages for
  modules, `unit/` for newer focused suites, `integration/` for real
  minimizer runs (convergence references in `reference.md`). `slow`
  marks guard sampler and long fits; default `pytest` deselects them.

## Key File Locations

**Entry Points:**

- `src/easyreflectometry/project.py`: `Project.prepare_fit`,
  `prepare_sample`, `record_fit`, `fitter`, `add_contrast`,
  `load_experiment_from_files`, `save_as_json` / `load_from_json`
- `src/easyreflectometry/fitting.py`: `MultiFitter.fit`,
  `fit_single_data_set_1d`, `fit_polarized`, `for_experiments`,
  `prepare`, `mcmc_sample`
- `src/easyreflectometry/analysis/bayesian.py`: `PosteriorResults`,
  `save_posterior` / `load_posterior`

**Configuration:**

- `pyproject.toml` (deps `easyscience>=2.5.1,<3`, pytest `addopts`,
  ruff), `pixi.toml` (tasks), `docs/mkdocs.yml`

**Core Logic (fitting pipeline):**

- `fitting.py`: `_prepare_fit_arrays` (zero-variance objectives),
  `PreparedFit`, `FitRun`, `MultiFitter._constraints`,
  `_check_fit_preconditions`
- `fit_settings.py`: `FitSettings.engine_kwargs`, `configure`,
  `option_schema`
- `_bumps_constraints.py`: `install`, `applied`, `_attach_constraints`
- Installed core (pixi env): `easyscience/fitting/multi_fitter.py`,
  `sampler.py`, `minimizers/minimizer_bumps.py`

**Persistence:**

- `project.py`: `as_dict` / `from_dict`, `_needs_new_format`,
  `_user_constraints`, `_as_dict_add_contrasts`; `sample/references.py`;
  `sample/collections/base_collection.py` `to_dict` / `from_dict`

## Naming Conventions

**Files:** snake*case modules; tests `test*<module>.py`; private module
`\_bumps_constraints.py`.

**Classes:** PascalCase (`MultiFitter`, `PreparedFit`, `FitRun`,
`InequalitySpec`, `PolarizedDataSet`); core import aliased as
`EasyScienceMultiFitter`; private subclasses prefixed `_`.

**Functions:** snake*case; private helpers `*`-prefixed;
`Project`exposes`\*\_at_index` accessors for GUI use
(`model_data_for_experiment_at_index`).

**Constants:** UPPER_SNAKE (`DEFAULT_MINIMIZER`, `GUIDE_FIELD_ANGLE`,
`DEFAULT_THETA_M`, `SIGMA_TO_FWHM`, `USER_CONSTRAINT_FLAG`).

**Parameter paths:** `models/<i>/sample/<j>/layers/<k>/<attr>` (stable
across save/load; used by inequality constraints and constraint
records).

## Where to Add New Code

**New fit option or minimizer setting:**

- `fit_settings.py` (`MinimizerOption`, `option_schema`,
  `engine_kwargs`) → consumed by `MultiFitter.prepare`; tests in
  `tests/test_fit_settings.py`, `tests/test_prepared_fit.py`.

**New constraint kind:**

- Equality: `constraints.py` + persistence in
  `Project._user_constraints` / `_restore_user_constraints`. Inequality:
  `inequality_constraints.py`
  - `Project.add_inequality_constraint`. Tests under `tests/unit/`.

**New calculator backend:**

- `calculators/<name>/calculator.py` + `wrapper.py`; registration is
  automatic via `CalculatorBase.__init_subclass__`; mirror
  `tests/calculators/<name>/` and add to
  `tests/calculators/test_resolution_conventions.py` /
  `tests/integration/test_cross_engine_resolution.py`.

**New sample element or assembly:**

- `sample/elements/...` or `sample/assemblies/...`, export from
  `sample/__init__.py`, handle in `orso_utils.py` conversion if it maps
  to ORSO, add an `api-reference` page and `mkdocs.yml` nav entry.

**New Bayesian analysis:**

- `analysis/bayesian.py`, export in `analysis/__init__.py`, tests in
  `tests/test_bayesian.py`.

**New Project behaviour that must persist:**

- `Project.as_dict` / `from_dict` plus `_needs_new_format` if older
  readers would misread it; tests in `tests/test_project.py` /
  `tests/test_shared_persistence.py`; document in `CHANGELOG.md`
  "Unreleased".

## Special Directories

**`.planning/codebase/`:** GSD analysis documents; untracked, not
gitignored (no `.gitignore` entry).

**`docs/`:** MkDocs source; tutorials are executed by `notebook-exec` in
CI before the site is built. API pages use mkdocstrings over `../src`.

**`notebooks/`:** Standalone notebooks, committed, not part of the docs
build.

**`tests/_static/`:** Third-party and example fixtures with
`PROVENANCE.md`.

---

_Structure analysis: 2026-10-15 (develop @ 66adaf54)_
