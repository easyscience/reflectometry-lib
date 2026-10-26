# Codebase Concerns

**Analysis Date:** 2026-10-15 **Branch analysed:** `develop` @
`66adaf54`

## Summary

`develop` is 51 commits past `v1.7.0` and carries the unreleased 1.9
work (contrasts, shared-object persistence, per-dataset resolution,
prepare/record fit split, volume fractions, inequality constraints on
BUMPS). The earlier concerns about the `bayesian_mp` branch
(multiprocessing, chains/seed, git-branch core pin) are gone: the core
dependency is a PyPI range, `n_workers`/`chains`/`seed` and
`CalculatorFactory.__reduce__` no longer exist in the library, and MCMC
goes through the core `Sampler`. The concerns below are what remains.

---

## 1. Coupling to EasyScience internals

### Monkeypatch of `minimizer_bumps.FitDriver`

**Issue:** Inequality constraints work only because
`_bumps_constraints.install()` rebinds the module-level `FitDriver` name
inside the core's `minimizer_bumps` namespace and inspects
`problem.models[0].pars`. Any core refactor (importing `FitDriver`
differently, building the problem elsewhere, renaming `.pars`) silently
disables every inequality constraint; the shim passes untouched problems
through without complaint.

**Files:** `src/easyreflectometry/_bumps_constraints.py`,
`src/easyreflectometry/fitting.py` (`_constraints`,
`_keep_constraints_on_extend`),
`tests/unit/test_bumps_constraints_shim.py`

**Impact:** High. Physics silently dropped on a core upgrade.

**Fix approach:** Keep the shim test that asserts penalties reach the
problem on both `fit` and `mcmc_sample`; add a start-up assertion that
the patched attribute exists; pursue an upstream hook
(`constraints=`/callback) in corelib and gate `<3` → `<2.6` until it
lands.

### Pin `easyscience>=2.5.1,<3` vs. local core on `release-3.0`

**Issue:** The local `C:\projects\easy\ERA\core` checkout is on
`release-3.0` (60 commits past 2.5.1) while pixi resolves 2.5.1 from
PyPI. Changes developed against the local core (e.g. the `_update`/fset
dependency fix, `update_from_calculator`) are not what tests run
against, and 3.0 is excluded by the pin.

**Files:** `pyproject.toml`, `pixi.lock`

**Impact:** Medium. Divergence between what is tested and what the next
core release ships.

**Fix approach:** Add a CI job (allowed to fail) installing core from
`release-3.0`; list the library's core API surface (`Sampler`,
`MultiFitter._precompute_reshaping`, `MINIMIZER_PARAMETER_PREFIX`,
`InterfaceFactoryTemplate`, `minimizer_bumps.FitDriver`) and check it
against the branch.

### Private core imports

**Issue:** `MINIMIZER_PARAMETER_PREFIX` from `minimizer_base`,
`resolve_all_parameter_dependencies` from
`parameter_dependency_resolver`, `NewBase` from `base_classes.new_base`,
`ItemContainer` from `interface_factory` are not public core API.

**Files:** `analysis/bayesian.py`, `project.py`,
`sample/collections/base_collection.py`,
`calculators/calculator_base.py`

**Impact:** Medium.

---

## 2. Size and complexity hotspots

### `project.py` is a 3211-line god object

**Issue:** `Project` mixes session state, persistence (two file formats,
reference resolution, constraint records), fit orchestration, contrasts
and links, ORSO import, polarized loading, spin asymmetry, derived data
for plots, and GUI index bookkeeping. Most behaviours are tested through
it, so refactors are expensive.

**Files:** `src/easyreflectometry/project.py`

**Impact:** Medium-High for maintainability; the radon dashboard
publishes the complexity.

**Fix approach:** Extract cohesive modules that `Project` delegates to:
`project_io.py` (as_dict/from_dict helpers, `_needs_new_format`),
`project_experiments.py` (loading, polarized, merge),
`spin_asymmetry.py` (`_spin_asymmetry`, `_unwrapped_angle`, channel
helpers). Keep `Project` as the facade.

### `fitting.py` (1428 lines) and `analysis/bayesian.py` (1412 lines)

**Issue:** `bayesian.py` is mostly plotly styling constants and
hand-rolled contour/density code for corner plots; `fitting.py` combines
data preparation, settings, constraints wiring and five public fit
paths.

**Impact:** Low-Medium.

**Fix approach:** Split `bayesian.py` into `posterior.py` (results,
persistence, predictive) and `posterior_plots.py`; consider moving
`PreparedFit`/`FitRun` to `fit_run.py`.

### `Project.__init__` and `from_dict` ordering

**Issue:** Loading depends on a specific order (materials → models →
experiments → constraints → contrasts → links) and on `references`
contexts being active. Several `_from_dict_*` helpers append to a shared
`report` list. Easy to break when adding a persisted field.

**Files:** `project.py` lines ~2845-3000

**Impact:** Medium.

---

## 3. Persistence format risks

### `file_format=3` is refused by older readers

**Issue:** By design (`_needs_new_format`), but the triggers are subtle:
shared objects, an experiment left out of the fit, duplicate model
names. A user may save a file in 1.9 that 1.8 refuses without an obvious
reason.

**Files:** `project.py` (`_needs_new_format`,
`_as_dict_add_experiments`)

**Impact:** Medium. Document in release notes; already in CHANGELOG.

### Constraint records keyed by structural path

**Issue:** `_user_constraints` refuses (raises `ValueError`) when a
constrained parameter is not reachable from the models (e.g. a
constraint on a palette material not used by any model). Saving a
project then fails late, at save time.

**Files:** `project.py` (`_constraint_candidates`, `_constraint_record`)

**Impact:** Medium. Surfaced as an exception, which is the intended
behaviour, but the GUI must handle it.

### `derived_parameter` is session-only

**Issue:** Documented, but a user who builds a derived parameter and
saves loses it silently.

**Files:** `constraints.py`

**Impact:** Low.

---

## 4. Testing gaps

### Thin integration/functional layers

**Issue:** `tests/functional/test_dummy.py` asserts `2 + 2 == 4` and is
the whole functional suite; `tests/integration/fitting/test_dummy.py`
and `tests/integration/scipp-analysis/test_dummy.py` are `slow`-marked
placeholders. Real integration coverage is `integration/convergence/`,
`test_cross_engine_resolution.py`, `test_ort_file_fitting.py`.

**Impact:** Medium. CI jobs named "functional tests" pass vacuously.

**Fix approach:** Either delete the placeholders or move the end-to-end
project tests (`test_project.py` fit paths, `test_polarized_fitting.py`
slow cases) under `integration/`.

### Sampler continuation and persistence

**Issue:** `fitter.sampler.extend()` after constraints, and
`save_posterior`/`load_posterior` fingerprint mismatch handling, have
limited coverage (`TestMCMCSample*` in `test_fitting.py`,
`test_bayesian.py`); all real DREAM runs are `slow` and only run on
PRs/develop.

**Impact:** Medium.

### Default `pytest` skips `slow`

**Issue:** `addopts = -m "not slow"` means local runs and some CI pushes
never execute a real minimizer or sampler. Regressions in engine wiring
surface only on PR CI.

**Impact:** Medium; deliberate trade-off (see `pyproject.toml` comment).
34 of 1410 tests are `slow`.

### refnx-only assumptions

**Issue:** Several fits and convergence references are refnx-based;
refl1d is exercised mainly through polarized/magnetism tests and
`test_cross_engine_resolution.py`.

**Impact:** Low-Medium.

---

## 5. Repository hygiene

### Stale build files

**Issue:** `Makefile` references `setup.py`, `flake8`, Sphinx and an
`EasyReflectometry/` directory; `.coveragerc` points at
`EasyReflectometry/`. Neither reflects hatchling/ruff/MkDocs/`src/`.

**Files:** `Makefile`, `.coveragerc`

**Impact:** Low. Confusing for new contributors.

**Fix approach:** Delete both (pixi tasks cover everything) or rewrite
to call `pixi run`.

### Duplicate CI matrices

**Issue:** `python-ci.yml` and `python-package.yml` (pip-based, run on
every push/PR) duplicate `test.yml`; `python-publish.yml` duplicates
`pypi-publish.yml`.

**Impact:** Low-Medium (CI minutes, confusing failures).

### Untracked planning/notes directories

**Issue:** `.planning/`, `.claude/`, `MD/`, `examples/`, `integration/`
are untracked and not in `.gitignore`, so `git status` is always noisy
and an accidental `git add .` would commit them.

**Impact:** Low.

**Fix approach:** Add `.planning/`, `.claude/`, `MD/` to `.gitignore`
(or a global excludes file); decide whether `examples/` and
`integration/` belong in the repo.

### Docstring tooling disabled

**Issue:** `pydoclint` and `format-docstring` are configured with
`exclude = '\.'` (everything), `interrogate --fail-under=0`. The
`docstring-lint-check` CI step therefore cannot fail.

**Impact:** Low.

### `main.py` has no console entry

**Issue:** `main()` prints available calculators but `pyproject.toml`
declares no `[project.scripts]`.

**Impact:** Low; remove or register.

---

## 6. Numerical and physics conventions to keep in view

- **SLD unit convention:** values are stored in units of 1e-6 Å^-2 while
  the declared unit is `1/angstrom^2`; EasyScience strips numeric unit
  factors so this cannot be fixed without a breaking migration (issue
  #377 left open on purpose).
- **Resolution convention:** `SIGMA_TO_FWHM` conversions in
  `resolution_functions.py`, `orso_utils.py` and the refnx wrapper must
  stay consistent; `tests/calculators/test_resolution_conventions.py`
  and `tests/integration/test_cross_engine_resolution.py` pin them.
- **Weights convention:** core weights are 1/σ (issue #370); keep in
  mind when touching `_prepare_fit_arrays`.
- **refl1d magnetism angle:** `DEFAULT_THETA_M = 270.0` and
  `GUIDE_FIELD_ANGLE = 270.0`; thetaM=90 is parallel to the guide field
  in refl1d. Past reviews got this backwards.
- **Mighell objective:** `hybrid` default substitutes Mighell for zero
  variance points; sampling with `mighell` is a bias correction, not a
  likelihood (warned in `mcmc_sample`).
- **Spin asymmetry heuristics:** `SPIN_ASYMMETRY_SIGNIFICANCE = 3.0`,
  `SPIN_ASYMMETRY_CANCELLATION_FRACTION = 1e-3`,
  `MAGNETIC_MOMENT_FLOOR_FRACTION = 0.01` in `project.py` are tuned
  constants without a documented derivation.

---

## 7. Work in flight (not on develop)

- Branch `425-magnetism` (3 commits ahead: magnetism fixes for issue
  #425 point 3, review fixes, ruff) is the current working branch.
- Uncommitted app-side work for per-dataset resolution (A1-A5) and the
  magnetism/fitting gaps lives in the GUI repository, not here.
- Local corelib changes (`update_from_calculator`, dependent-parameter
  callback fix) are uncommitted in `C:\projects\easy\ERA\core` and not
  part of the 2.5.1 the tests use.

---

_Concerns audit: 2026-10-15 (develop @ 66adaf54)_
