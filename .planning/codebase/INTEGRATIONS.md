# External Integrations

**Analysis Date:** 2026-10-15 **Branch analysed:** `develop` @
`66adaf54`

## APIs & External Services

**Fitting Engine (EasyScience corelib, PyPI `easyscience>=2.5.1,<3`):**

- `easyscience.fitting.multi_fitter.MultiFitter` - Subclassed as
  `_ConstrainedEasyScienceMultiFitter` in `fitting.py` so the raw
  `fit()` path still applies inequality constraints
- `easyscience.fitting.Sampler` / `SamplingResults` - Owns the DREAM
  chain; `MultiFitter.mcmc_sample()` builds one and keeps it on
  `fitter.sampler` for `extend()`, `save()`, `load_state()`
- `easyscience.fitting.AvailableMinimizers`, `FitResults`
- `easyscience.fitting.minimizers.minimizer_bumps` - Its module-level
  `FitDriver` name is wrapped by `_bumps_constraints.install()` to
  attach `FitProblem.constraints`
- `easyscience.fitting.minimizers.minimizer_base.MINIMIZER_PARAMETER_PREFIX` -
  Stripped from parameter names in `analysis/bayesian.py`
- `easyscience.variable.Parameter` / `DescriptorNumber` and
  `parameter_dependency_resolver.resolve_all_parameter_dependencies` -
  Equality constraints (`constraints.py`), contrasts, project load
- `easyscience.base_classes.ModelBase` / `EasyList`,
  `easyscience.io.SerializerComponent` / `SerializerBase` - Base classes
  for `BaseCore`, `BaseCollection`, `DataSet1D`, `CalculatorBase`
- `easyscience.global_object` - Unique-name registry and undo stack
- `easyscience.fitting.calculators.interface_factory.InterfaceFactoryTemplate`
  / `ItemContainer` - `CalculatorFactory` base
- Auth: None (Python import)

**Reflectivity Calculators (pluggable via `CalculatorFactory`):**

- refnx (`refnx.reflect`) - `src/easyreflectometry/calculators/refnx/`;
  unpolarized only (`supports_magnetism()` is False)
- refl1d (`refl1d.names`, `refl1d.profile.build_profile`,
  `refl1d.sample.layers.Repeat`) -
  `src/easyreflectometry/calculators/refl1d/`; magnetism via
  `rho_m`/`theta_m` per layer, polarized cross-sections, magnetic SLD
  profile. `DEFAULT_THETA_M = 270.0`, resolution oversampling factor 21.
- BornAgain was removed in 1.8.0.

**BUMPS (direct dependency and via core):**

- DREAM MCMC through the core `Sampler`
- `FitProblem.constraints` list receives penalty callables built by
  `inequality_constraints.build_constraints_factory`; penalties are
  attached by the shim as the driver is built, not passed as kwargs
- `Bumps_*` minimizers are the only engines that can enforce inequality
  constraints; `fit_settings.requires_finite_bounds` guards engines that
  need finite bounds

**ORSO (orsopy):**

- Read: `.ort` text and `.orb` nexus via `orso_utils._load_orso_any`;
  columns validated and unit-scaled (`_Q_UNIT_TO_INV_ANGSTROM`,
  `_SLD_UNIT_TO_INV_ANGSTROM_SQUARED`), sQz cleaned to a `Pointwise`
  resolution, polarization read from the header
  (`data/polarized.channel_from_orso_polarization`)
- Model import: `load_orso_model` converts the ORSO model language
  (`resolve_stack`, substacks → `RepeatingMultilayer`) into a `Sample`
- Write: `save_orso_experiment` / `sample_to_orso_model` export
  experiments with the model embedded in the header

**Chemistry (periodictable):**

- `special/calculations.py` scattering lengths, molecular weights,
  density → SLD; `contrasts.ReplaceFormula` validates formulas

**Plotting / Reports:**

- plotly - Corner, trace, distribution plots (`analysis/bayesian.py`)
  and interactive summary figures; plotly is a hard dependency
- matplotlib - `plot.py`, static summary figures, `plot_volume_fraction`
- xhtml2pdf (`pisa`) - `Summary.save_pdf_summary`; stderr silenced via
  `_silence_pdf_converter`
- arviz - Optional (`dev` extra), required only by `plot_trace` and
  `gelman_rubin` (`_require_arviz()` raises a helpful ImportError)

## Data Storage

**Databases:**

- None

**File Storage:**

- Local filesystem only
- Project files: `project.json` written atomically by
  `Project.save_as_json` (temp file + fsync + replace, permission
  preserving). `file_format` 2 by default, 3 when the file uses shared
  objects (`@ref`/`@ref_id` from `sample/references.py`), excludes an
  experiment from the fit, or has ambiguous model names.
- Posterior chains: `save_posterior` / `load_posterior` write the BUMPS
  state plus a JSON sidecar (`_SIDECAR_SCHEMA_VERSION = 1`) carrying a
  data fingerprint and the easyreflectometry version
- Example data fetched with pooch in tutorials; test fixtures under
  `tests/_static/` (provenance documented in `PROVENANCE.md`)

**Caching:**

- `Project._fitter` cached per current model index, invalidated when
  models change
- `Refl1dWrapper._model_state_token` caches polarized reflectivities per
  model state

## Authentication & Identity

- Not applicable (scientific library)

## Monitoring & Observability

**Error Tracking:**

- `warnings.warn()` for data quality and recoverable issues: zero
  variance handling (`fitting.py`), ORSO column problems
  (`orso_utils.py`, 14 sites), unreadable dependencies on load, volume
  fractions below zero (`VolumeFractionWarning`), duplicate collection
  items
- Domain exceptions: `FitScopeError`, `FitPreconditionError`
  (`fitting.py`), `UnitError` (`inequality_constraints.py`),
  `UnsupportedSubstitution`, `LayoutMismatch` (`contrasts.py`),
  `VolumeFractionWarning`; file errors from `Project` are raised, not
  printed (since 1.8.0)
- `Project.load_report` collects every warning of a load

**Logs:**

- `logging.getLogger` in `project.py`, `orso_utils.py`, `summary.py`,
  `sample/paths.py`, `calculators/refl1d/wrapper.py`; no handler is
  configured by the library

## CI/CD & Deployment

**Hosting:**

- PyPI (`pypi-publish.yml` on tag push, via `pixi run default-build`;
  legacy `python-publish.yml` also present)
- GitHub Pages via mike (`docs.yml`): `dev` alias from develop,
  versioned
  - `latest` on release; a quality dashboard is published by
    `dashboard.yml` (badgery + radon + interrogate)

**CI Pipeline (`.github/workflows/`, 24 workflows):**

- `test.yml` - Main matrix (3 OS × py-311/py-313 pixi envs): unit,
  functional, integration tests, wheel build, then a package-test job.
  Pull requests, master/develop and manual dispatch run the full set
  including `slow`; other pushes run `-m fast` / the default fast suite.
- `lint-format.yml` - pyproject validity, SPDX headers, ruff lint +
  format, pydoclint, prettier, nbqa on notebooks
- `coverage.yml` - docstring coverage, unit + integration coverage to
  Codecov (`unit-tests-all`, `integration-tests-all`)
- `tutorial-tests.yml` - nbmake over `docs/docs/tutorials/`; scheduled
  via `tutorial-tests-trigger.yml`; `test-trigger.yml` schedules code
  tests
- `pypi-test.yml` - Scheduled install-from-PyPI smoke test
- `docs.yml`, `dashboard.yml` - site and metrics publishing
- `security.yml` / `codeql-analysis.yml` / `ossar-analysis.yml` - CodeQL
  and OSSAR scans
- `release-drafter.yml`, `release-notes.yml`, `release-pr.yml` (develop
  → master), `backmerge.yml` (master → develop),
  `release-drafter-verify-pr-labels.yml`, `pr-labels.yml`,
  `issues-labels.yml`, `cleanup.yml`
- `python-ci.yml`, `python-package.yml` - Older pip-based matrices that
  still run on every push/PR alongside `test.yml`

## Environment Configuration

**Required env vars:**

- None for library use

**Secrets location:**

- GitHub Secrets / `easyscience[bot]` app token used by release,
  backmerge, docs and dashboard workflows; PyPI trusted publishing

## Webhooks & Callbacks

**Incoming:**

- None

**Outgoing (in-process callbacks):**

- `progress_callback` on `MultiFitter.fit`, `fit_polarized`,
  `mcmc_sample` and `PreparedFit.execute(**kwargs)` → forwarded to the
  core fitter / `Sampler.sample`
- `abort_test` on `mcmc_sample` → `Sampler.sample`
- `constraints_factory_provider` - zero-argument callable (normally a
  weak reference to `Project.build_constraints_factory`) resolved at fit
  time, in the calling thread
- `MultiFitter.for_experiments` / `Project.prepare_fit` are designed for
  a GUI worker thread: the caller executes `prepared.core_fitter` and
  reports back through `Project.record_fit`

---

_Integration audit: 2026-10-15 (develop @ 66adaf54)_
